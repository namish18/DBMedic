"""
dashboard/app.py
================
Optional DBMedic Phase 6 Streamlit Dashboard.

Displays actual backend observations, clearly separates simulated vs live runs,
enforces interactive human approval confirmation, and never leaks credentials.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.audit_logger import AuditLogger
from agent.core.environment_check import generate_readiness_report
from agent.core.incident_orchestrator import IncidentOrchestrator
from dashboard.components.approval_panel import render_approval_panel
from dashboard.components.audit_panel import render_audit_panel
from dashboard.components.benchmark_panel import render_benchmark_panel
from dashboard.components.diagnostic_panel import render_diagnostic_panel
from dashboard.components.incident_panel import render_incident_panel


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="DBMedic — PostgreSQL Self-Healing Agent",
        layout="wide",
    )
    st.title("DBMedic — Autonomous PostgreSQL Diagnostic & Safe Self-Healing")
    st.caption("CLI (`python scripts/dbmedic.py`) is primary; this dashboard reflects actual backend files & observations.")

    if st.sidebar.button("Refresh Environment Readiness"):
        generate_readiness_report(save=True)

    orchestrator = IncidentOrchestrator()
    audit_logger = orchestrator.audit_logger

    readiness = _load_json(PROJECT_ROOT / "reports" / "environment_readiness.json")
    comparison = _load_json(PROJECT_ROOT / "benchmarks" / "final_demo" / "remediation_comparison.json")
    final_metrics = _load_json(PROJECT_ROOT / "benchmarks" / "final_demo" / "final_metrics.json")
    incidents = orchestrator.list_incidents()

    selected = render_incident_panel(st, readiness, incidents)
    render_diagnostic_panel(st, selected)
    render_approval_panel(st, selected, orchestrator)
    render_benchmark_panel(st, comparison, final_metrics)
    render_audit_panel(st, audit_logger, incident_id=selected.get("incident_id") if selected else None)


if __name__ == "__main__":
    main()

