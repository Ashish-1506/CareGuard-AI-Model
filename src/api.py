from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Import custom modules
from features import FeatureEngineer
from explain import ModelExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CareGuard AI API",
    description="AI-Driven Risk Prediction Engine for Chronic Care Patients",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and explainer
model_bundle = None
explainer = None

# Pydantic models for API
class PatientFeatures(BaseModel):
    """Input features for a single patient"""
    age: float = Field(..., ge=18, le=120, description="Patient age in years")
    sex: str = Field(..., description="Patient sex (M/F)")
    condition_primary: str = Field(..., description="Primary chronic condition")
    hba1c_last: float = Field(..., ge=4.0, le=15.0, description="Latest HbA1c percentage")
    weight_trend_30d: float = Field(..., ge=-10.0, le=10.0, description="Weight change in kg over 30 days")
    adherence_mean: float = Field(..., ge=0.0, le=1.0, description="Mean medication adherence (0-1)")
    bnp_last: float = Field(..., ge=0, le=5000, description="Latest BNP in pg/mL")
    egfr_trend_90d: float = Field(..., ge=-50, le=50, description="eGFR trend over 90 days")
    sbp_last: float = Field(..., ge=70, le=250, description="Latest systolic blood pressure")
    bmi: float = Field(..., ge=15, le=60, description="Body mass index")
    days_since_last_lab: int = Field(..., ge=0, le=730, description="Days since last lab work")
    smoker: int = Field(..., ge=0, le=1, description="Smoking status (0/1)")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    patient_id: Optional[str] = None
    risk_probability: float
    risk_band: str
    threshold_used: float
    prediction_binary: int
    confidence_interval: Optional[List[float]] = None
    timestamp: str

class ExplanationResponse(BaseModel):
    """Response model for explanations"""
    patient_id: Optional[str] = None
    risk_probability: float
    risk_band: str
    clinical_summary: str
    top_drivers: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    patients: List[PatientFeatures]
    include_explanations: bool = False

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    version: str

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model_bundle, explainer

    try:
        logger.info("Loading model bundle...")
        model_bundle = joblib.load("../models/model.pkl")

        logger.info("Initializing explainer...")
        explainer = ModelExplainer("../models/model.pkl")

        logger.info("CareGuard AI API startup complete!")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_bundle = None
        explainer = None

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy" if model_bundle is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_bundle is not None,
        version="1.0.0"
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CareGuard AI - Chronic Care Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

def prepare_patient_data(patient: PatientFeatures) -> pd.DataFrame:
    """Convert patient features to DataFrame"""

    patient_dict = patient.dict()
    df = pd.DataFrame([patient_dict])

    # Add required columns that might be missing
    if 'patient_id' not in df.columns:
        df['patient_id'] = 'temp_' + str(np.random.randint(10000, 99999))
    if 'patient_name' not in df.columns:
        df['patient_name'] = 'Patient ' + df['patient_id'].astype(str)
    if 'last_updated' not in df.columns:
        df['last_updated'] = datetime.now()

    return df

def get_risk_band(probability: float, threshold: float = 0.25) -> str:
    """Convert probability to risk band"""
    if probability >= 0.25:
        return 'High'
    elif probability >= 0.10:
        return 'Medium'
    else:
        return 'Low'

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(patient: PatientFeatures):
    """Predict risk for a single patient"""

    if model_bundle is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check service health."
        )

    try:
        # Prepare data
        df = prepare_patient_data(patient)

        # Engineer features
        feature_engineer = model_bundle['feature_engineer']
        scaler = model_bundle['scaler']
        model = model_bundle['calibrated_model']
        threshold = model_bundle['threshold']

        df_features = feature_engineer.create_features(df)
        X = feature_engineer.prepare_model_features(df_features)
        X_scaled = scaler.transform(X)

        # Make prediction
        prob = float(model.predict_proba(X_scaled)[0, 1])
        risk_band = get_risk_band(prob, threshold)
        prediction_binary = int(prob >= threshold)

        return PredictionResponse(
            risk_probability=prob,
            risk_band=risk_band,
            threshold_used=threshold,
            prediction_binary=prediction_binary,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def predict_batch(request: BatchPredictionRequest):
    """Predict risk for multiple patients"""

    if model_bundle is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check service health."
        )

    try:
        predictions = []

        for i, patient in enumerate(request.patients):
            patient_pred = await predict_single(patient)
            patient_pred.patient_id = f"batch_{i}"
            predictions.append(patient_pred)

        return {
            "predictions": predictions,
            "total_patients": len(predictions),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(patient: PatientFeatures, patient_id: Optional[str] = None):
    """Get explanation for a patient's risk prediction"""

    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Explainer not loaded. Please check service health."
        )

    try:
        # Prepare data
        df = prepare_patient_data(patient)

        # Get explanation
        explanation = explainer.explain_patient(df, patient_id)

        return ExplanationResponse(
            patient_id=explanation['patient_id'],
            risk_probability=explanation['risk_probability'],
            risk_band=explanation['risk_band'],
            clinical_summary=explanation['clinical_summary'],
            top_drivers=explanation['top_drivers'],
            recommendations=explanation['recommendations'],
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get model information"""

    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check service health."
        )

    try:
        return {
            "model_type": "XGBoost with Isotonic Calibration",
            "features": model_bundle['features'],
            "threshold": model_bundle['threshold'],
            "metrics": model_bundle.get('metrics', {}),
            "training_timestamp": model_bundle.get('timestamp', 'Unknown'),
            "feature_count": len(model_bundle['features'])
        }

    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/features/descriptions")
async def get_feature_descriptions():
    """Get human-readable feature descriptions"""

    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check service health."
        )

    try:
        feature_engineer = model_bundle['feature_engineer']
        descriptions = feature_engineer.get_feature_descriptions()

        return {
            "descriptions": descriptions,
            "total_features": len(descriptions)
        }

    except Exception as e:
        logger.error(f"Feature descriptions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature descriptions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
