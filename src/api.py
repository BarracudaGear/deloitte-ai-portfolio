"""FastAPI service for enterprise AI portfolio: forecasting, fraud detection, document analysis, and security gateway."""

from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Existing components
from src.agent_analyst import app as document_analyst
from src.guardrails import EnterpriseSecurityGateway

MODEL_DIR = Path("./models")
DEMAND_MODEL_PATH = MODEL_DIR / "supply_chain_xgb.joblib"
FRAUD_MODEL_PATH = MODEL_DIR / "fraud_iforest.joblib"

# Global holders
models = {}
gateway = EnterpriseSecurityGateway()

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

app = FastAPI(title="Enterprise AI Portfolio API", lifespan=lifespan)

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

class ChatRequest(BaseModel):
    message: str

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "demand_model_loaded": "demand" in models,
        "fraud_model_loaded": "fraud" in models,
        "document_analyst_ready": True,
        "security_gateway_ready": True,
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
    """Run the multi-agent document analyst."""
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

@app.post("/gateway/chat")
def gateway_chat(req: ChatRequest):
    """Security gateway: input guardrails → safe routing → output redaction."""
    input_check = gateway.check_input(req.message)

    if input_check["blocked"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Request blocked by enterprise security gateway",
                "reasons": input_check["reasons"],
                "action": "BLOCKED",
                "latency_ms": input_check["latency_ms"],
            },
        )

    # Safe path: simple echo / placeholder routing to existing analysis logic
    # In production this would call the appropriate model or agent
    raw_response = (
        f"Processed enterprise query safely. "
        f"Message length: {len(req.message)}. "
        f"No unauthorized topics or injection patterns detected."
    )

    redacted_response, pii_types = gateway.redact_pii(raw_response)
    # Also redact the original message for the audit log
    _, input_pii = gateway.redact_pii(req.message)

    return {
        "response": redacted_response,
        "security_audit": {
            "input_action": input_check["action"],
            "input_latency_ms": input_check["latency_ms"],
            "output_pii_redacted": pii_types,
            "input_pii_detected": input_pii,
            "status": "ALLOWED",
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
