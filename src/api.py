"""FastAPI service serving the demand forecasting, fraud detection models, and multi-agent document analyst."""

from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# LangGraph agent
from src.agent_analyst import app as document_analyst

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

class DocumentRequest(BaseModel):
    document_text: str

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "demand_model_loaded": "demand" in models,
        "fraud_model_loaded": "fraud" in models,
        "document_analyst_ready": True,
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
    score = float(model.decision_function(df)[0])
    raw_pred = int(model.predict(df)[0])
    is_anomaly = raw_pred == -1
    return FraudResponse(anomaly_score=score, is_anomaly=is_anomaly)

@app.post("/analyze/document")
def analyze_document(req: DocumentRequest):
    """Run the multi-agent LangGraph document analyst."""
    initial_state = {
        "document_text": req.document_text,
        "doc_type": "unknown",
        "extracted_entities": {},
        "risk_findings": [],
        "compliance_status": "",
        "final_report": "",
        "audit_trail": [],
    }
    result = document_analyst.invoke(initial_state)
    return {
        "doc_type": result.get("doc_type"),
        "extracted_entities": result.get("extracted_entities"),
        "risk_findings": result.get("risk_findings"),
        "compliance_status": result.get("compliance_status"),
        "final_report": result.get("final_report"),
        "audit_trail": result.get("audit_trail"),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
