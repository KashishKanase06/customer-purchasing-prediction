"""
app.py
======
FastAPI backend for the Customer Purchase Prediction service.

Endpoints:
    GET  /              → health check
    GET  /model-info    → model metadata
    POST /predict       → single session prediction
    POST /predict-batch → batch predictions (list of sessions)

Run:
    uvicorn app:app --reload --port 8000
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# ── Load artifacts ────────────────────────────────────────────────────────────
MODELS_DIR = "models"

def _load(fname):
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artifact '{fname}' not found. Run `python train_model.py` first."
        )
    return joblib.load(path)

model        = _load("model.pkl")
scaler       = _load("scaler.pkl")
encoders     = _load("encoders.pkl")
feature_cols = _load("feature_cols.pkl")
metadata     = _load("metadata.pkl")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Purchase Prediction API",
    description="Predicts whether an online shopping session will result in a purchase.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
VALID_MONTHS  = ["Jan", "Feb", "Mar", "Apr", "May", "June",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VALID_VISITOR = ["Returning_Visitor", "New_Visitor", "Other"]

class SessionInput(BaseModel):
    Administrative:            int   = Field(0,   ge=0,  description="Number of administrative pages visited")
    Administrative_Duration:   float = Field(0.0, ge=0.0)
    Informational:             int   = Field(0,   ge=0)
    Informational_Duration:    float = Field(0.0, ge=0.0)
    ProductRelated:            int   = Field(1,   ge=0)
    ProductRelated_Duration:   float = Field(0.0, ge=0.0)
    BounceRates:               float = Field(0.0, ge=0.0, le=1.0)
    ExitRates:                 float = Field(0.1, ge=0.0, le=1.0)
    PageValues:                float = Field(0.0, ge=0.0)
    SpecialDay:                float = Field(0.0, ge=0.0, le=1.0)
    Month:                     str   = Field("Feb")
    OperatingSystems:          int   = Field(2,   ge=1)
    Browser:                   int   = Field(2,   ge=1)
    Region:                    int   = Field(1,   ge=1)
    TrafficType:               int   = Field(3,   ge=1)
    VisitorType:               str   = Field("Returning_Visitor")
    Weekend:                   int   = Field(0,   ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "Administrative": 2,
                "Administrative_Duration": 50.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 15,
                "ProductRelated_Duration": 600.0,
                "BounceRates": 0.01,
                "ExitRates": 0.03,
                "PageValues": 25.0,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": 0,
            }
        }

class PredictionResult(BaseModel):
    will_purchase:   bool
    purchase_label:  str
    confidence:      float
    purchase_prob:   float
    no_purchase_prob: float

class BatchInput(BaseModel):
    sessions: List[SessionInput]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _preprocess(session: SessionInput) -> np.ndarray:
    """Convert a SessionInput into a model-ready numpy row."""
    if session.Month not in VALID_MONTHS:
        raise HTTPException(400, f"Month must be one of {VALID_MONTHS}")
    if session.VisitorType not in VALID_VISITOR:
        raise HTTPException(400, f"VisitorType must be one of {VALID_VISITOR}")

    month_enc   = encoders["Month"].transform([session.Month])[0]
    visitor_enc = encoders["VisitorType"].transform([session.VisitorType])[0]

    row = [
        session.Administrative,
        session.Administrative_Duration,
        session.Informational,
        session.Informational_Duration,
        session.ProductRelated,
        session.ProductRelated_Duration,
        session.BounceRates,
        session.ExitRates,
        session.PageValues,
        session.SpecialDay,
        month_enc,
        session.OperatingSystems,
        session.Browser,
        session.Region,
        session.TrafficType,
        visitor_enc,
        session.Weekend,
    ]
    return np.array(row).reshape(1, -1)

def _predict_one(row: np.ndarray) -> PredictionResult:
    """Run inference on a single preprocessed row."""
    model_name = metadata.get("model_name", "")
    if model_name == "LogisticRegression":
        row_input = scaler.transform(row)
    else:
        row_input = row

    pred  = int(model.predict(row_input)[0])
    proba = model.predict_proba(row_input)[0]

    return PredictionResult(
        will_purchase    = bool(pred),
        purchase_label   = "Will Purchase ✓" if pred else "Will NOT Purchase ✗",
        confidence       = round(float(max(proba)) * 100, 2),
        purchase_prob    = round(float(proba[1]) * 100, 2),
        no_purchase_prob = round(float(proba[0]) * 100, 2),
    )

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Customer Purchase Prediction API"}

@app.get("/model-info", tags=["Model"])
def model_info():
    return metadata

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(session: SessionInput):
    """Predict purchase intent for a single browsing session."""
    row = _preprocess(session)
    return _predict_one(row)

@app.post("/predict-batch", tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """Predict purchase intent for multiple sessions at once."""
    results = []
    for i, session in enumerate(batch.sessions):
        try:
            row    = _preprocess(session)
            result = _predict_one(row)
            results.append({"index": i, "result": result.dict(), "error": None})
        except Exception as e:
            results.append({"index": i, "result": None, "error": str(e)})
    return {"predictions": results, "total": len(results)}
