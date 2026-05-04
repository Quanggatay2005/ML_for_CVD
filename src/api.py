"""
src/api.py -- FastAPI Backend for BRFSS CVD Risk Prediction
============================================================
Endpoints:
  GET  /            -- health check + model status
  GET  /features    -- returns feature list + metadata
  GET  /model-info  -- returns training metrics for all models
  POST /predict     -- accepts patient data, returns CVD risk score
  GET  /metrics     -- returns best model test metrics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import os
import numpy as np
import pandas as pd
import uvicorn
from datetime import datetime
try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None

# ── Artifact paths (relative to project root) ─────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_HERE)
MODELS_DIR  = os.path.join(_ROOT, "models")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")
IMPUTER_PATH    = os.path.join(MODELS_DIR, "imputer.pkl")
FEATURES_PATH   = os.path.join(MODELS_DIR, "features.json")
MODEL_INFO_PATH = os.path.join(MODELS_DIR, "model_info.json")

# ── Load artifacts ────────────────────────────────────────────────────────────
def _load_artifacts():
    model = scaler = imputer = None
    feature_names = []
    model_info = {}

    try:
        if all(os.path.exists(p) for p in [BEST_MODEL_PATH, SCALER_PATH,
                                             IMPUTER_PATH, FEATURES_PATH]):
            model   = joblib.load(BEST_MODEL_PATH)
            scaler  = joblib.load(SCALER_PATH)
            imputer = joblib.load(IMPUTER_PATH)
            with open(FEATURES_PATH) as f:
                feature_names = json.load(f)
            if os.path.exists(MODEL_INFO_PATH):
                with open(MODEL_INFO_PATH) as f:
                    model_info = json.load(f)
            print("[API] Model artifacts loaded successfully.")
            print(f"[API] Best model : {model_info.get('best_model_name', 'unknown')}")
            print(f"[API] Features   : {len(feature_names)}")
        else:
            print("[API] WARNING: Model artifacts not found. "
                  "Run  python main.py --no-api  to train first.")
    except Exception as e:
        print(f"[API] ERROR loading artifacts: {e}")

    return model, scaler, imputer, feature_names, model_info


model, scaler, imputer, feature_names, model_info = _load_artifacts()

# ── Kafka Producer Initialization ─────────────────────────────────────────────
producer = None
if KafkaProducer is not None:
    try:
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            retries=1,
            request_timeout_ms=2000,
            max_block_ms=2000
        )
        print("[API] Kafka Producer connected successfully to localhost:9092.")
    except Exception as e:
        print(f"[API] WARNING: Could not connect to Kafka: {e}. Shadow streaming disabled.")
else:
    print("[API] WARNING: kafka-python not installed. Shadow streaming disabled.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CVD Risk Prediction API",
    description=(
        "RESTful API for predicting Cardiovascular Disease (heart attack) risk "
        "based on BRFSS health survey indicators. "
        "Backed by the best of LightGBM / Random Forest / SVM trained on ~100k rows."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    """
    Patient feature vector — all values should be numeric (float).
    Field names match the columns of new_brfss.csv (excluding CVDINFR4 target).

    Common fields (see /features for full list):
      _AGEG5YR  : Age group (1-13, where 13 = 80+)
      SEXVAR    : Sex (1=Male, 2=Female)
      GENHLTH   : General health (1=Excellent ... 5=Poor)
      CVDCRHD4  : Diagnosed with coronary heart disease (1=Yes, 2=No)
      DIABETE4  : Diabetes (1=Yes, 2=Prediabetes, 3=No)
      _BMI5     : BMI * 100
    """
    # Dynamic fields are validated at runtime against feature_names
    model_config = {"extra": "allow"}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def health_check():
    """Returns server status and model readiness."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "best_model": model_info.get("best_model_name", "none"),
        "feature_count": len(feature_names),
        "target": "CVDINFR4 (1 = heart attack history, 0 = no)",
    }


@app.get("/features", summary="Feature metadata")
def get_features():
    """Returns the ordered list of features the model expects."""
    return {
        "feature_count": len(feature_names),
        "features": feature_names,
    }


@app.get("/metrics", summary="Best model metrics")
def get_metrics():
    """Returns the test-set performance metrics for the best model."""
    if not model_info:
        raise HTTPException(status_code=404, detail="No model info found.")
    return {
        "best_model": model_info.get("best_model_name"),
        "metrics": model_info.get("test_metrics", {}),
        "all_results": model_info.get("all_results", []),
    }


@app.get("/model-info", summary="Full training metadata")
def get_model_info():
    """Returns full model info including all model comparison results."""
    if not model_info:
        raise HTTPException(status_code=404, detail="No model info found.")
    return model_info


@app.post("/predict", summary="Predict CVD risk")
def predict_cvd_risk(data: PatientData):
    """
    Accepts a patient's health indicator values and returns:
    - risk_percentage : 0-100
    - risk_level      : 'High' (>=50%) or 'Low'
    - interpretation  : human-readable summary

    All features listed at GET /features must be provided.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run  python main.py --no-api  to train first.",
        )

    try:
        # Build ordered feature vector from request payload
        payload = data.model_dump()

        # Check for missing features
        missing = [f for f in feature_names if f not in payload]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing features in request: {missing}",
            )

        row = pd.DataFrame([[payload[f] for f in feature_names]], columns=feature_names)

        # Apply the same imputer + scaler as training
        row_imputed = imputer.transform(row)
        row_scaled  = scaler.transform(row_imputed)

        # Predict
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(row_scaled)[0][1])
        else:
            # LinearSVC — use decision_function, clip to [0,1]
            score = float(model.decision_function(row_scaled)[0])
            # Sigmoid-like normalisation
            probability = float(1 / (1 + np.exp(-score)))

        risk_percentage = round(probability * 100, 2)
        risk_level = "High" if probability >= 0.45 else "Low"

        # ── Send to Kafka (Shadow Stream) ─────────────────────────────────────
        if producer is not None:
            try:
                event = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "features": payload,
                    "prediction": {
                        "probability": probability,
                        "risk_percentage": risk_percentage,
                        "risk_level": risk_level
                    }
                }
                producer.send('inference_logs', value=event)
            except Exception as e:
                print(f"[API] Error sending inference to Kafka: {e}")

        return {
            "risk_percentage": risk_percentage,
            "risk_level": risk_level,
            "probability": round(probability, 4),
            "interpretation": (
                f"The patient has a {risk_percentage}% estimated risk of "
                f"having experienced a heart attack."
            ),
            "message": "Prediction successful.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
