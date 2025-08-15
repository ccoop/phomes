import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import config
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from shared import catalog, registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

demographics_df = None
valid_zipcodes = None
demographics_median_row = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global demographics_df, valid_zipcodes, demographics_median_row
    logger.info("Starting up API server...")

    demographics_df = catalog.load_source("demographics", validate=True)
    logger.info(f"Loaded demographics for {len(demographics_df)} zipcodes")

    # Create efficient zipcode lookup and median fallback
    valid_zipcodes = set(demographics_df.zipcode.values)
    numeric_cols = demographics_df.select_dtypes(include=['number']).columns.tolist()
    medians = demographics_df[numeric_cols].median()
    demographics_median_row = pd.DataFrame([medians.to_dict()])
    logger.info(f"Prepared efficient lookup for {len(valid_zipcodes)} zipcodes")

    logger.info("API startup complete")
    yield
    logger.info("API shutdown")


app = FastAPI(
    title="phData Home Price Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms")
    sqft_living: int = Field(..., gt=0, le=50_000, description="Square feet of living space")
    sqft_lot: int = Field(..., gt=0, description="Square feet of lot")
    floors: float = Field(..., ge=1, description="Number of floors")
    waterfront: int = Field(..., ge=0, le=1, description="Waterfront property (0 or 1)")
    view: int = Field(..., ge=0, le=4, description="View rating (0-4)")
    condition: int = Field(..., ge=1, le=5, description="Overall condition (1-5)")
    grade: int = Field(..., ge=1, le=13, description="Overall grade (1-13)")
    sqft_above: int = Field(..., ge=0, description="Square feet above ground")
    sqft_basement: int = Field(..., ge=0, description="Square feet of basement")
    yr_built: int = Field(..., ge=1900, le=2022, description="Year built")
    yr_renovated: int = Field(..., ge=0, le=2022, description="Year renovated (0 if never)")
    zipcode: str = Field(..., min_length=5, max_length=5, description="Zipcode of the property")
    lat: float = Field(..., description="Latitude")
    long: float = Field(..., description="Longitude")
    sqft_living15: int = Field(..., ge=0, description="Living space of nearest 15 neighbors")
    sqft_lot15: int = Field(..., ge=0, description="Lot size of nearest 15 neighbors")


class ModelResponse(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float
    model_version: str
    request_id: str
    timestamp: str
    response_time_ms: float


def get_active_model():
    """Determine which model to use - env override, production, or best model."""
    model_id = os.getenv("ACTIVE_MODEL_ID")
    if model_id:
        return model_id

    production_model = registry.get_production_model()
    if production_model:
        return production_model["id"]

    registry_data = registry.load_registry()
    return registry_data["best_model"]["id"]


def try_auto_promotion(model_id):
    """Try to auto-promote model if enabled."""
    if not config.PROMOTION["auto_promote"]:
        return

    try:
        gate_results, gates_passed = registry.evaluate_quality_gates(model_id)
        if gates_passed:
            result = registry.promote_to_production(model_id)
            logger.info(f"Auto-promoted model {model_id}: {json.dumps(result)}")
        else:
            logger.info(f"Model {model_id} failed quality gates: {gate_results}")
    except Exception as e:
        logger.warning(f"Auto-promotion failed for {model_id}: {e}")


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=ModelResponse)
async def predict(request: PredictRequest):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        model_id = get_active_model()
        production_model = registry.get_production_model()
        if not production_model or model_id != production_model["id"]:
            # Promote new model if it passes quality gates
            try_auto_promotion(model_id)

        model_data = registry.get_model(model_id)
        logger.info(f"Request {request_id}: Predicting for zipcode {request.zipcode}")

        # Merge demographics, fallback to median if not in dataset
        request_df = pd.DataFrame([request.dict()])
        if request.zipcode in valid_zipcodes:
            merged_df = request_df.merge(demographics_df, on="zipcode", how="left")
        else:
            logger.info(f"Request {request_id}: Unknown zipcode {request.zipcode}, using median demographics")
            merged_df = pd.concat([request_df, demographics_median_row], axis=1)

        # Align features and predict
        feature_df = merged_df[model_data["features"]]
        prediction = model_data["pipeline"].predict(feature_df)[0]

        price_range_low = prediction * 0.85
        price_range_high = prediction * 1.15
        response_time = (time.perf_counter() - start_time) * 1000
        response = ModelResponse(
            predicted_price=float(prediction),
            price_range_low=float(price_range_low),
            price_range_high=float(price_range_high),
            model_version=model_id,
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            response_time_ms=response_time,
        )

        logger.info(f"Request {request_id}: Predicted ${prediction:,.0f} (range: ${price_range_low:,.0f}-${price_range_high:,.0f}) in {response_time:.1f}ms")
        return response

    except Exception as e:
        response_time = (time.perf_counter() - start_time) * 1000
        logger.error(f"Request {request_id}: Prediction error after {response_time:.1f}ms: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
