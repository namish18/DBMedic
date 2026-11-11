"""
dashboard/components/incident_panel.py
======================================
Database health and active incident overview panel for DBMedic dashboard.
"""

from __future__ import annotations

from typing import Any

from agent.core.audit_logger import redact_secrets


def summarize_database_health(readiness: dict[str, Any] | None) -> dict[str, Any]:
    """Produce a sanitized summary dict of database health from environment readiness."""
    if not readiness:
        return {
            "status": "NOT_CHECKED",
            "database": "dbmedic_lab",
            "connected": False,
            "schema_valid": False,
            "orders_estimated_rows": None,
            "idx_exists": None,
            "blocker": "Run 'python scripts/dbmedic.py doctor' first.",
        }
    checks = readiness.get("checks", {})
    db = checks.get("database_connection", {})
    schema = checks.get("lab_schema", {})
    target = db.get("target_database", {})
    return redact_secrets(
        {
            "status": db.get("status", "UNKNOWN"),
            "database": target.get("database") or "dbmedic_lab",
            "host": target.get("host"),
            "port": target.get("port"),
            "connected": bool(db.get("connected", False)),
            "schema_valid": bool(schema.get("schema_valid", False)),
            "orders_estimated_rows": schema.get("orders_estimated_rows"),
            "idx_exists": schema.get("idx_orders_customer_id_exists"),
            "blocker": db.get("blocker") or schema.get("blocker"),
        }
    )


def render_incident_panel(st: Any, readiness: dict[str, Any] | None, incidents: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Render Database Health & Active Incidents sections, returning selected incident."""
    st.subheader("1. Database Health")
    db_info = summarize_database_health(readiness)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target DB", str(db_info["database"]))
    c2.metric("DB Status", str(db_info["status"]))
    c3.metric("Orders Est. Rows", str(db_info["orders_estimated_rows"] or "N/A"))
    c4.metric("customer_id Index", "Present" if db_info["idx_exists"] else ("Absent" if db_info["idx_exists"] is False else "Unknown"))
    if db_info.get("blocker"):
        st.warning(f"Database status: {db_info['blocker']}")

    st.subheader("2. Active Incidents")
    if not incidents:
        st.info("No incident runs found yet. Run `python scripts/dbmedic.py diagnose --incident slow_query_001`.")
        return None

    options = {
        f"{inc['incident_id']} [{inc['status']} | mode={inc['execution_mode']}{' | SIMULATED' if inc.get('simulated') else ''}]": inc
        for inc in incidents
    }
    selected_label = st.selectbox("Select Incident", list(options.keys()))
    selected = options[selected_label]
    if selected.get("simulated"):
        st.warning("SIMULATED RUN: Observations below come from simulated mode, not live PostgreSQL or Tiny Aya inference.")
    return selected

