import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

import config
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from shared import registry, catalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load demographics and prepare efficient lookup once at startup
demographics_df = None
valid_zipcodes = None
demographics_median_row = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global demographics_df, valid_zipcodes, demographics_median_row
    logger.info("Starting up API server...")

    # Load demographics data once using DataCatalog for consistency
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
    description="ML API for predicting home prices using King County housing data",
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
    model_version: str
    timestamp: str
    features_used: Dict[str, Any]


def get_model_id():
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
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=ModelResponse)
async def predict(request: PredictRequest):
    try:
        # Get model and check if auto-promotion needed
        model_id = get_model_id()
        production_model = registry.get_production_model()
        if not production_model or model_id != production_model["id"]:
            try_auto_promotion(model_id)
        
        model_data = registry.get_model(model_id)
        logger.info(f"Prediction request for zipcode {request.zipcode}")
        
        # Efficient zipcode handling with median fallback
        request_df = pd.DataFrame([request.dict()])
        if request.zipcode in valid_zipcodes:
            merged_df = request_df.merge(demographics_df, on="zipcode", how="left")
        else:
            logger.info(f"Unknown zipcode {request.zipcode}, using median demographics")
            merged_df = pd.concat([request_df, demographics_median_row], axis=1)

        # Align features and predict
        feature_df = merged_df[model_data["features"]]
        prediction = model_data["pipeline"].predict(feature_df)[0]

        response = ModelResponse(
            predicted_price=float(prediction),
            model_version=model_id,
            timestamp=datetime.now().isoformat(),
            features_used=request.dict(),
        )

        logger.info(f"Prediction: ${prediction:,.0f} using model {model_id}")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
