"""FastAPI service serving the demand forecasting and fraud detection models."""

from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path("./models")
DEMAND_MODEL_PATH = MODEL_DIR / "supply_chain_xgb.joblib"
FRAUD_MODEL_PATH = MODEL_DIR / "fraud_iforest.joblib"

# Global model holders
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup."""
    print("Loading models...")
    if DEMAND_MODEL_PATH.exists():
        models["demand"] = joblib.load(DEMAND_MODEL_PATH)
        print("Demand model loaded")
    else:
        print("Warning: demand model not found")
    if FRAUD_MODEL_PATH.exists():
        models["fraud"] = joblib.load(FRAUD_MODEL_PATH)
        print("Fraud model loaded")
    else:
        print("Warning: fraud model not found")
    yield
    models.clear()

app = FastAPI(title="Deloitte AI Portfolio API", lifespan=lifespan)

# ---------- Schemas ----------
class DemandRequest(BaseModel):
    lag_1_orders: float
    lag_2_orders: float
    lag_4_orders: float
    rolling_4_mean: float
    rolling_4_std: float

class DemandResponse(BaseModel):
    forecasted_demand: float

class FraudRequest(BaseModel):
    amount: float
    delta_orig: float = Field(0.0)
    delta_dest: float = Field(0.0)
    amount_to_orig: float = Field(0.0)
    amount_to_dest: float = Field(0.0)
    oldbalanceOrg: float = Field(0.0)
    newbalanceOrig: float = Field(0.0)
    oldbalanceDest: float = Field(0.0)
    newbalanceDest: float = Field(0.0)

class FraudResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "demand_model_loaded": "demand" in models,
        "fraud_model_loaded": "fraud" in models,
    }

@app.post("/predict/demand", response_model=DemandResponse)
def predict_demand(req: DemandRequest):
    if "demand" not in models:
        raise HTTPException(status_code=503, detail="Demand model not loaded")
    artifact = models["demand"]
    model = artifact["model"]
    features = artifact["feature_cols"]
    df = pd.DataFrame([req.model_dump()])[features]
    pred = float(model.predict(df)[0])
    return DemandResponse(forecasted_demand=pred)

@app.post("/predict/fraud", response_model=FraudResponse)
def predict_fraud(req: FraudRequest):
    if "fraud" not in models:
        raise HTTPException(status_code=503, detail="Fraud model not loaded")
    artifact = models["fraud"]
    model = artifact["model"]
    features = artifact["feature_cols"]
    data = req.model_dump()
    df = pd.DataFrame([data])[features].fillna(0)
    # decision_function: lower = more anomalous
    score = float(model.decision_function(df)[0])
    raw_pred = int(model.predict(df)[0])
    is_anomaly = raw_pred == -1
    return FraudResponse(anomaly_score=score, is_anomaly=is_anomaly)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
