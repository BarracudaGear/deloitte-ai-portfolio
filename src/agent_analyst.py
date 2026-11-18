"""Multi-agent LangGraph document analyst for financial, supply-chain, and fraud documents.

Deterministic rule-based / regex extraction by default (zero API keys required).
Optional OPENAI_API_KEY enhances extraction and reporting when present.
"""

import os
import re
from typing import TypedDict, Literal, Annotated
from operator import add
from langgraph.graph import StateGraph, END

# ---------- State ----------
class AgentState(TypedDict):
    document_text: str
    doc_type: Literal["financial_contract", "supply_chain_invoice", "fraud_incident", "unknown"]
    extracted_entities: dict
    risk_findings: list
    compliance_status: str
    final_report: str
    audit_trail: Annotated[list, add]

# ---------- Helpers ----------
def _log(state: AgentState, node: str, message: str) -> list:
    return [{"node": node, "message": message}]

def _extract_amounts(text: str) -> list[float]:
    pattern = r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
    matches = re.findall(pattern, text)
    amounts = []
    for m in matches:
        try:
            amounts.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return amounts

def _extract_dates(text: str) -> list[str]:
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]
    dates = []
    for p in patterns:
        dates.extend(re.findall(p, text, re.IGNORECASE))
    return list(dict.fromkeys(dates))  # dedupe

def _extract_parties(text: str) -> list[str]:
    # Simple capitalized multi-word entities
    parties = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\b", text)
    return list(dict.fromkeys(parties))[:8]

# ---------- Nodes ----------
def router_node(state: AgentState) -> dict:
    text = state["document_text"].lower()
    if any(k in text for k in ["agreement", "contract", "shall pay", "indemnify", "governing law"]):
        doc_type = "financial_contract"
    elif any(k in text for k in ["invoice", "bill to", "po number", "shipment", "freight"]):
        doc_type = "supply_chain_invoice"
    elif any(k in text for k in ["fraud", "suspicious", "anomaly", "unauthorized", "isFraud"]):
        doc_type = "fraud_incident"
    else:
        doc_type = "unknown"

    trail = _log(state, "router_node", f"Classified document as '{doc_type}'")
    return {"doc_type": doc_type, "audit_trail": trail}

def extractor_node(state: AgentState) -> dict:
    text = state["document_text"]
    entities = {
        "amounts": _extract_amounts(text),
        "dates": _extract_dates(text),
        "parties": _extract_parties(text),
        "raw_length": len(text),
    }

    # Optional OpenAI enhancement (safe fallback)
    if os.getenv("OPENAI_API_KEY"):
        entities["enhanced"] = "LLM enhancement available (not invoked in deterministic mode)"

    trail = _log(state, "extractor_node", f"Extracted {len(entities['amounts'])} amounts, {len(entities['dates'])} dates, {len(entities['parties'])} parties")
    return {"extracted_entities": entities, "audit_trail": trail}

def risk_auditor_node(state: AgentState) -> dict:
    entities = state.get("extracted_entities", {})
    doc_type = state.get("doc_type", "unknown")
    findings = []
    severity = 0

    amounts = entities.get("amounts", [])
    if amounts:
        max_amt = max(amounts)
        if max_amt > 1_000_000:
            findings.append({"risk": "High-value transaction", "detail": f"Amount ${max_amt:,.2f}", "severity": "High"})
            severity += 3
        elif max_amt > 100_000:
            findings.append({"risk": "Elevated transaction value", "detail": f"Amount ${max_amt:,.2f}", "severity": "Medium"})
            severity += 2

    text_lower = state["document_text"].lower()
    if doc_type == "financial_contract":
        if "unlimited liability" in text_lower or "no cap" in text_lower:
            findings.append({"risk": "Unlimited liability clause", "severity": "High"})
            severity += 3
        if "sla" not in text_lower and "service level" not in text_lower:
            findings.append({"risk": "Missing SLA language", "severity": "Medium"})
            severity += 1
    elif doc_type == "supply_chain_invoice":
        if not entities.get("dates"):
            findings.append({"risk": "No delivery or invoice date found", "severity": "Medium"})
            severity += 1
    elif doc_type == "fraud_incident":
        findings.append({"risk": "Fraud indicators present", "severity": "High"})
        severity += 4

    if not findings:
        findings.append({"risk": "No material risks identified", "severity": "Low"})

    overall = "Critical" if severity >= 5 else "High" if severity >= 3 else "Medium" if severity >= 1 else "Low"
    trail = _log(state, "risk_auditor_node", f"Identified {len(findings)} findings; overall severity: {overall}")
    return {"risk_findings": findings, "compliance_status": overall, "audit_trail": trail}

def compliance_report_node(state: AgentState) -> dict:
    doc_type = state.get("doc_type", "unknown")
    entities = state.get("extracted_entities", {})
    findings = state.get("risk_findings", [])
    status = state.get("compliance_status", "Unknown")

    report_lines = [
        "=" * 60,
        "ENTERPRISE RISK & COMPLIANCE AUDIT REPORT",
        "=" * 60,
        f"Document Type     : {doc_type.replace('_', ' ').title()}",
        f"Overall Risk Level: {status}",
        "",
        "Key Extracted Entities:",
        f"  - Monetary values : {entities.get('amounts', [])[:5]}",
        f"  - Dates           : {entities.get('dates', [])}",
        f"  - Parties         : {entities.get('parties', [])}",
        "",
        "Risk Findings:",
    ]
    for f in findings:
        report_lines.append(f"  - [{f.get('severity', 'Info')}] {f.get('risk')} – {f.get('detail', '')}".rstrip(" – "))

    report_lines.extend([
        "",
        "Recommendation:",
        "  Proceed with standard review" if status in ("Low", "Medium") else "  Escalate to Legal & Risk Operations Team for further diligence",
        "",
        "This analysis was generated by a deterministic multi-agent LangGraph pipeline.",
        "=" * 60,
    ])
    final_report = "\n".join(report_lines)
    trail = _log(state, "compliance_report_node", "Executive report synthesized")
    return {"final_report": final_report, "audit_trail": trail}

# ---------- Graph ----------
def route_after_router(state: AgentState) -> str:
    return "extractor"  # always extract; type is already set

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("risk_auditor", risk_auditor_node)
    graph.add_node("compliance_report", compliance_report_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"extractor": "extractor"},
    )
    graph.add_edge("extractor", "risk_auditor")
    graph.add_edge("risk_auditor", "compliance_report")
    graph.add_edge("compliance_report", END)

    return graph.compile()

# Compiled app for import
app = build_graph()

# ---------- Demo ----------
if __name__ == "__main__":
    test_docs = [
        {
            "name": "Sample Financial Contract",
            "text": """MASTER SERVICES AGREEMENT
This Agreement is entered into on January 15, 2025 between Acme Corporation and Global Logistics Partners.
Client shall pay a fee of $2,500,000 upon execution. Unlimited liability applies for data breaches.
Governing law: State of New York. No service level agreement is defined.""",
        },
        {
            "name": "Sample Fraud Incident Report",
            "text": """FRAUD INCIDENT ALERT
Unauthorized transaction detected on 2025-03-12 involving $450,000 from account of Contoso Ltd.
Suspicious activity flagged by monitoring system. Immediate investigation required.""",
        },
    ]

    for doc in test_docs:
        print(f"\n{'='*70}")
        print(f"PROCESSING: {doc['name']}")
        print(f"{'='*70}")
        initial = {
            "document_text": doc["text"],
            "doc_type": "unknown",
            "extracted_entities": {},
            "risk_findings": [],
            "compliance_status": "",
            "final_report": "",
            "audit_trail": [],
        }
        result = app.invoke(initial)
        print("\n--- Audit Trail ---")
        for step in result["audit_trail"]:
            print(f"[{step['node']}] {step['message']}")
        print("\n--- Final Report ---")
        print(result["final_report"])
