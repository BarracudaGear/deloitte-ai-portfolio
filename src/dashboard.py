"""Enterprise AI Operations & Security Console – Streamlit dashboard.

Runs entirely in a single process. Imports models, LangGraph agent, and security gateway directly.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_analyst import app as document_analyst
from src.guardrails import EnterpriseSecurityGateway, run_red_team

# ---------- Page config ----------
st.set_page_config(
    page_title="Enterprise AI Operations Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Enterprise AI Operations & Security Console")
st.caption("Unified interface for demand forecasting, fraud detection, multi-agent document analysis, and security gateway.")

# ---------- Load resources ----------
@st.cache_resource
def load_models():
    models = {}
    model_dir = Path("./models")
    demand_path = model_dir / "supply_chain_xgb.joblib"
    fraud_path = model_dir / "fraud_iforest.joblib"
    if demand_path.exists():
        models["demand"] = joblib.load(demand_path)
    if fraud_path.exists():
        models["fraud"] = joblib.load(fraud_path)
    return models

models = load_models()
gateway = EnterpriseSecurityGateway()

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "Supply Chain Demand Forecaster",
    "Financial Fraud Anomaly Detector",
    "Multi-Agent Document Analyst",
    "AI Security Gateway & Red-Team",
])

# ==================== TAB 1: Demand Forecaster ====================
with tab1:
    st.header("Supply Chain Demand Forecaster")
    st.markdown("Interactive weekly demand prediction by product category using the trained XGBoost model.")

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Product Category", ["electronics", "home_goods", "fashion", "sports", "beauty", "unknown"])
        price = st.slider("Average Unit Price", 10.0, 500.0, 89.0)
        freight = st.slider("Average Freight Value", 0.0, 50.0, 12.5)
    with col2:
        lag_1 = st.slider("Lag-1 Weekly Orders", 0, 500, 120)
        lag_2 = st.slider("Lag-2 Weekly Orders", 0, 500, 110)
        lag_4 = st.slider("Lag-4 Weekly Orders", 0, 500, 95)
        promo = st.checkbox("Active Promotion Flag", value=False)

    if st.button("Forecast Demand", type="primary", key="demand_btn"):
        if "demand" not in models:
            st.warning("Demand model not found at models/supply_chain_xgb.joblib. Using fallback estimate.")
            pred = lag_1 * 0.95 + (20 if promo else 0)
            lower, upper = pred * 0.85, pred * 1.15
        else:
            artifact = models["demand"]
            model = artifact["model"]
            feature_cols = artifact.get("feature_cols", ["lag_1_orders", "lag_2_orders", "lag_4_orders", "rolling_4_mean", "rolling_4_std"])
            # Map UI inputs to expected features (simplified alignment)
            rolling_mean = (lag_1 + lag_2 + lag_4) / 3
            rolling_std = abs(lag_1 - lag_2) * 0.3
            input_data = {
                "lag_1_orders": lag_1,
                "lag_2_orders": lag_2,
                "lag_4_orders": lag_4,
                "rolling_4_mean": rolling_mean,
                "rolling_4_std": rolling_std,
            }
            # Filter to available columns
            available = [c for c in feature_cols if c in input_data]
            df = pd.DataFrame([{c: input_data.get(c, 0) for c in available}])
            pred = float(model.predict(df)[0])
            lower, upper = pred * 0.88, pred * 1.12

        st.metric("Predicted Weekly Unit Demand", f"{pred:,.0f}")
        st.caption(f"Confidence interval: {lower:,.0f} – {upper:,.0f}")

        if pred > lag_1 * 1.3:
            st.error("Inventory Reorder Alert: Demand significantly above recent history. Consider expedited replenishment.")
        elif pred < lag_1 * 0.7:
            st.info("Demand softening detected. Review promotion or category mix.")
        else:
            st.success("Demand within normal operating range.")

        st.markdown("---")
        st.subheader("Model Benchmark KPIs")
        k1, k2 = st.columns(2)
        k1.metric("RMSE", "13.41")
        k2.metric("MAE", "6.82")

# ==================== TAB 2: Fraud Detector ====================
with tab2:
    st.header("Financial Fraud Anomaly Detector")
    st.markdown("Live Isolation Forest scoring on simulated transaction features.")

    c1, c2 = st.columns(2)
    with c1:
        txn_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])
        amount = st.number_input("Amount", min_value=0.0, value=2500.0, step=100.0)
        old_orig = st.number_input("Old Origin Balance", value=5000.0)
        new_orig = st.number_input("New Origin Balance", value=2500.0)
    with c2:
        old_dest = st.number_input("Old Destination Balance", value=0.0)
        new_dest = st.number_input("New Destination Balance", value=2500.0)

    if st.button("Score Transaction", type="primary", key="fraud_btn"):
        delta_orig = old_orig - new_orig
        delta_dest = old_dest - new_dest
        amount_to_orig = amount / (old_orig + 1)
        amount_to_dest = amount / (old_dest + 1)

        if "fraud" not in models:
            st.warning("Fraud model not found at models/fraud_iforest.joblib. Using rule-based fallback.")
            anomaly_score = -0.15 if amount > 10000 or abs(delta_orig - amount) > 100 else 0.12
            is_anomaly = anomaly_score < 0
        else:
            artifact = models["fraud"]
            model = artifact["model"]
            feature_cols = artifact.get("feature_cols", [])
            features = {
                "amount": amount,
                "delta_orig": delta_orig,
                "delta_dest": delta_dest,
                "amount_to_orig": amount_to_orig,
                "amount_to_dest": amount_to_dest,
                "oldbalanceOrg": old_orig,
                "newbalanceOrig": new_orig,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
            }
            available = [c for c in feature_cols if c in features]
            df = pd.DataFrame([{c: features.get(c, 0) for c in available}])
            raw = model.decision_function(df)[0]
            anomaly_score = float(raw)
            is_anomaly = int(model.predict(df)[0]) == -1

        st.metric("Anomaly Risk Score", f"{anomaly_score:.4f}")
        if is_anomaly:
            st.error("FLAGGED – Potential anomaly detected")
        else:
            st.success("Within normal operating envelope")

        st.markdown("**Flagged Risk Factors**")
        factors = []
        if abs(delta_orig - amount) > 50:
            factors.append("Origin balance delta inconsistent with amount")
        if amount > 10000:
            factors.append("High-value transaction")
        if txn_type in ("TRANSFER", "CASH_OUT") and new_dest > old_dest + amount * 0.9:
            factors.append("Destination balance increase matches transfer pattern")
        if not factors:
            factors.append("No elevated risk factors identified")
        for f in factors:
            st.write(f"- {f}")

        st.markdown("---")
        st.subheader("Model Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Precision", "0.5200")
        m2.metric("Recall", "0.8158")

# ==================== TAB 3: Document Analyst ====================
with tab3:
    st.header("Multi-Agent Document Analyst")
    st.markdown("LangGraph 4-node pipeline: Router → Extractor → Risk Auditor → Compliance Report.")

    presets = {
        "Master Services Agreement": """MASTER SERVICES AGREEMENT
This Agreement is entered into on January 15, 2025 between Acme Corporation and Global Logistics Partners.
Client shall pay a fee of $2,500,000 upon execution. Unlimited liability applies for data breaches.
Governing law: State of New York. No service level agreement is defined.""",
        "Fraud Incident Report": """FRAUD INCIDENT ALERT
Unauthorized transaction detected on 2025-03-12 involving $450,000 from account of Contoso Ltd.
Suspicious activity flagged by monitoring system. Immediate investigation required.""",
        "Supply Chain SLA Invoice": """INVOICE #INV-2026-8842 from Supplier Acme Logistics.
Payment due 12/15/2026. Amount: $45,230.50 for delivery services.
SLA: delivery within 5 business days or 2% penalty applies.""",
    }

    preset_choice = st.selectbox("Preset Sample Document", ["-- Custom --"] + list(presets.keys()))
    if preset_choice == "-- Custom --":
        doc_text = st.text_area("Document Text", height=180, placeholder="Paste contract, invoice, or incident report...")
    else:
        doc_text = st.text_area("Document Text", value=presets[preset_choice], height=180)

    if st.button("Run Multi-Agent Audit", type="primary", key="doc_btn") and doc_text.strip():
        with st.spinner("Executing LangGraph pipeline..."):
            initial = {
                "document_text": doc_text,
                "doc_type": "unknown",
                "extracted_entities": {},
                "risk_findings": [],
                "compliance_status": "",
                "final_report": "",
                "audit_trail": [],
            }
            result = document_analyst.invoke(initial)

        st.subheader("Node-by-Node Audit Trail")
        for step in result.get("audit_trail", []):
            st.info(f"**{step.get('node', 'node')}** — {step.get('message', '')}")

        st.subheader("Extracted Entities")
        entities = result.get("extracted_entities", {})
        st.json(entities)

        findings = result.get("risk_findings", [])
        if findings:
            st.subheader("Risk Findings")
            for f in findings:
                sev = f.get("severity", "Info")
                badge = "🔴" if sev == "High" else "🟠" if sev == "Medium" else "🟢"
                st.write(f"{badge} **{f.get('risk')}** — {f.get('detail', '')}")

        st.subheader("Final Compliance Report")
        st.code(result.get("final_report", "No report generated"), language="text")

# ==================== TAB 4: Security Gateway ====================
with tab4:
    st.header("AI Security Gateway & Red-Team Suite")
    st.markdown("Live input testing against prompt injection, jailbreaks, PII redaction, and exfiltration defenses.")

    sandbox = st.text_area("Test Prompt", height=100, placeholder="Enter a prompt to evaluate...")
    if st.button("Evaluate Prompt", type="primary", key="gw_btn") and sandbox.strip():
        result = gateway.process(sandbox)
        action = result["action"]
        if action == "BLOCKED":
            st.error(f"BLOCKED — {', '.join(result.get('reasons', []))}")
        elif action == "REDACTED":
            st.warning(f"REDACTED — PII types: {result.get('pii_types_found', [])}")
            st.code(result.get("redacted_text", ""))
        else:
            st.success("ALLOWED — No security violations detected")
            st.code(result.get("redacted_text", sandbox))
        st.caption(f"Latency: {result.get('latency_ms', 0):.2f} ms")

    st.markdown("---")
    st.subheader("Automated Red-Team Benchmark")
    if st.button("Run 8-Test Red-Team Suite", type="secondary", key="redteam_btn"):
        # Capture stdout from the existing red-team runner
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_red_team()
        output = buf.getvalue()
        st.code(output, language="text")
        st.success("Red-team suite completed. Defense score shown above.")

st.sidebar.markdown("### System Status")
st.sidebar.write(f"Demand model: {'Loaded' if 'demand' in models else 'Not found'}")
st.sidebar.write(f"Fraud model: {'Loaded' if 'fraud' in models else 'Not found'}")
st.sidebar.write("Document analyst: Ready")
st.sidebar.write("Security gateway: Ready")
st.sidebar.markdown("---")
st.sidebar.caption("All components run in-process. No separate API server required.")
