"""Simple FastAPI service for bank churn prediction."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH  = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# ---------------------------------------------------------------------------
# Lifespan — load artifacts once at startup (modern FastAPI pattern)
# ---------------------------------------------------------------------------
artifacts: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            "Model artifacts not found. Run `python main.py` first."
        )
    artifacts["model"]  = joblib.load(MODEL_PATH)
    artifacts["scaler"] = joblib.load(SCALER_PATH)
    yield
    artifacts.clear()

app = FastAPI(title="Bank Churn Predictor API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class CustomerFeatures(BaseModel):
    credit_score: float
    country: int          # 0=France, 1=Germany, 2=Spain
    gender: int           # 0=Female, 1=Male
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int      # 0 or 1
    active_member: int    # 0 or 1
    estimated_salary: float
    # engineered features — computed automatically if omitted
    balance_salary_ratio: float | None = None
    age_group: int | None = None
    zero_balance: int | None = None


class PredictionResponse(BaseModel):
    churn: int
    churn_probability: float
    risk_level: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Bank Churn Predictor API is running. Visit /docs for usage."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in artifacts}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    model  = artifacts.get("model")
    scaler = artifacts.get("scaler")

    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Auto-compute engineered features if not supplied
    bsr = customer.balance_salary_ratio
    if bsr is None:
        bsr = customer.balance / (customer.estimated_salary + 1)

    ag = customer.age_group
    if ag is None:
        if customer.age <= 30:
            ag = 0
        elif customer.age <= 45:
            ag = 1
        elif customer.age <= 60:
            ag = 2
        else:
            ag = 3

    zb = customer.zero_balance
    if zb is None:
        zb = int(customer.balance == 0)

    FEATURE_NAMES = [
        "credit_score", "country", "gender", "age", "tenure",
        "balance", "products_number", "credit_card", "active_member",
        "estimated_salary", "balance_salary_ratio", "age_group", "zero_balance",
    ]

    features = pd.DataFrame([[
        customer.credit_score,
        customer.country,
        customer.gender,
        customer.age,
        customer.tenure,
        customer.balance,
        customer.products_number,
        customer.credit_card,
        customer.active_member,
        customer.estimated_salary,
        bsr, ag, zb,
    ]], columns=FEATURE_NAMES)

    features_scaled = scaler.transform(features)
    prediction  = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])

    if probability >= 0.7:
        risk = "HIGH"
    elif probability >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return PredictionResponse(
        churn=prediction,
        churn_probability=round(probability, 4),
        risk_level=risk,
    )
