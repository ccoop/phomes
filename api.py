import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import config
from shared import registry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load demographics once at startup
demographics_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global demographics_df
    logger.info("Starting up API server...")
    
    # Load demographics data once
    demographics_df = pd.read_csv(config.DATA_SOURCES["demographics_path"], dtype={"zipcode": str})
    logger.info(f"Loaded demographics for {len(demographics_df)} zipcodes")
    
    logger.info("API startup complete")
    yield
    logger.info("API shutdown")

app = FastAPI(
    title="phData Home Price Prediction API",
    description="ML API for predicting home prices using King County housing data",
    version="0.1.0",
    lifespan=lifespan
)

class PredictRequest(BaseModel):
    id: int = Field(..., description="Unique identifier")
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, description="Number of bathrooms")
    sqft_living: int = Field(..., gt=0, le=50_000, description="Square feet of living space")
    sqft_lot: int = Field(..., gt=0, le=1_000_000, description="Square feet of lot")
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


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=ModelResponse)
async def predict(request: PredictRequest):
    try:
        model_id = os.getenv("ACTIVE_MODEL_ID")
        
        if model_id:
            production_model = registry.get_production_model()
            if config.PROMOTION["auto_promote"] and (not production_model or model_id != production_model["id"]):
                try:
                    gate_results, gates_passed = registry.evaluate_quality_gates(model_id)
                    if gates_passed:
                        result = registry.promote_to_production(model_id)
                        logger.info(f"Auto-promoted model {model_id} to production: {json.dumps(result)}")
                    else:
                        logger.info(f"Model {model_id} failed quality gates, not auto-promoted: {gate_results}")
                except Exception as e:
                    logger.warning(f"Auto-promotion failed for {model_id}: {e}")
        
        if not model_id:
            production_model = registry.get_production_model()
            if production_model:
                model_id = production_model["id"]
            else:
                registry_data = registry.load_registry()
                model_id = registry_data["best_model"]["id"]
        
        model_data = registry.get_model(model_id)
        
        # Merge request data with demographics
        request_df = pd.DataFrame([request.dict()])
        merged_df = request_df.merge(demographics_df, on='zipcode', how='left')
        
        # Align features and predict
        feature_df = merged_df[model_data['features']]
        prediction = model_data['pipeline'].predict(feature_df)[0]

        response = ModelResponse(
            predicted_price=float(prediction),
            model_version=model_id,
            timestamp=datetime.now().isoformat(),
            features_used=request.dict()
        )

        log_entry = {
            "endpoint": "/predict",
            "input": request.dict(),
            "prediction": float(prediction),
            "model_id": model_id,
            "timestamp": response.timestamp
        }
        logger.info(json.dumps(log_entry))

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

