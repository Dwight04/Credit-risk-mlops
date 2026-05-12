import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Paths
MODEL_DIR = Path(__file__).parent.parent / "models"

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="Real-time credit risk scoring using XGBoost model trained on Give Me Some Credit dataset",
    version="1.0.0"
)

# Load model at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

# Input schema — mirrors Give Me Some Credit features
class CreditRiskInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0, le=1,
        description="Total balance on credit cards / credit limits (0-1)"
    )
    age: int = Field(
        ..., ge=18, le=100,
        description="Age of borrower in years"
    )
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., ge=0,
        description="Number of times 30-59 days past due in last 2 years",
        alias="NumberOfTime30-59DaysPastDueNotWorse"
    )
    DebtRatio: float = Field(
        ..., ge=0,
        description="Monthly debt payments / monthly gross income"
    )
    MonthlyIncome: float = Field(
        ..., ge=0,
        description="Monthly income in dollars"
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ..., ge=0,
        description="Number of open loans and lines of credit"
    )
    NumberOfTimes90DaysLate: int = Field(
        ..., ge=0,
        description="Number of times 90+ days past due"
    )
    NumberRealEstateLoansOrLines: int = Field(
        ..., ge=0,
        description="Number of mortgage and real estate loans"
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., ge=0,
        description="Number of times 60-89 days past due in last 2 years",
        alias="NumberOfTime60-89DaysPastDueNotWorse"
    )
    NumberOfDependents: int = Field(
        ..., ge=0,
        description="Number of dependents in family"
    )

    class Config:
        populate_by_name = True

# Output schema
class CreditRiskOutput(BaseModel):
    probability_of_default: float = Field(
        ..., description="Predicted probability of default (0-1)"
    )
    risk_score: int = Field(
        ..., description="Credit risk score (300-850, higher = lower risk)"
    )
    risk_band: str = Field(
        ..., description="Risk band: LOW / MEDIUM / HIGH"
    )
    recommendation: str = Field(
        ..., description="Approve / Review / Decline"
    )
    timestamp: str = Field(
        ..., description="Prediction timestamp"
    )

def compute_risk_score(probability):
    """
    Convert probability of default to credit score (300-850)
    Higher score = lower risk — mirrors industry standard scoring
    """
    score = int(850 - (probability * 550))
    return max(300, min(850, score))

def compute_risk_band(probability):
    """
    Assign risk band based on probability of default
    """
    if probability < 0.1:
        return "LOW", "Approve"
    elif probability < 0.3:
        return "MEDIUM", "Review"
    else:
        return "HIGH", "Decline"

def engineer_features(data: dict) -> pd.DataFrame:
    """
    Apply same feature engineering as train.py
    """
    df = pd.DataFrame([data])

    # Rename aliased columns
    df.rename(columns={
        "NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60_89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse"
    }, inplace=True)

    # Cap outliers
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(0, 1)
    df["DebtRatio"] = df["DebtRatio"].clip(0, 1)

    # Feature engineering — must match train.py exactly
    df["TotalLate"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["IncomePerDependent"] = df["MonthlyIncome"] / (
        df["NumberOfDependents"] + 1
    )

    return df

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

# Prediction endpoint
@app.post("/predict", response_model=CreditRiskOutput)
def predict(input_data: CreditRiskInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    try:
        # Prepare features
        input_dict = input_data.dict(by_alias=True)
        df = engineer_features(input_dict)

        # Score
        probability = round(
            float(model.predict_proba(df)[:, 1][0]), 4
        )

        # Derive outputs
        risk_score = compute_risk_score(probability)
        risk_band, recommendation = compute_risk_band(probability)

        return CreditRiskOutput(
            probability_of_default=probability,
            risk_score=risk_score,
            risk_band=risk_band,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# Batch prediction endpoint
@app.post("/predict/batch")
def predict_batch(inputs: list[CreditRiskInput]):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    results = []
    for input_data in inputs:
        result = predict(input_data)
        results.append(result)
    return results

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
