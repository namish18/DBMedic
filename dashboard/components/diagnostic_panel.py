"""
dashboard/components/diagnostic_panel.py
========================================
AI Diagnosis, tool calls, and PostgreSQL evidence panel for DBMedic dashboard.
"""

from __future__ import annotations

from typing import Any


def render_diagnostic_panel(st: Any, incident: dict[str, Any] | None) -> None:
    """Render AI/Deterministic Diagnosis and Evidence & Execution Plan sections."""
    st.subheader("3. AI / Diagnostic Analysis")
    if not incident:
        st.info("Select or run an incident to inspect diagnosis.")
        return

    diag = incident.get("diagnosis")
    if not diag:
        st.warning(f"Incident '{incident['incident_id']}' has no diagnosis yet (status={incident['status']}).")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Execution Mode", str(incident.get("execution_mode")))
    col2.metric("Root Cause", str((diag.get("root_cause") or {}).get("type", "unknown")))
    score = diag.get("diagnostic_score")
    col3.metric("Diagnostic Score", f"{score:.2f}" if isinstance(score, (int, float)) else "N/A")

    st.write(f"**Summary:** {diag.get('summary', '')}")
    st.write(f"**Model / Engine:** `{incident.get('model_id')}`")
    tools_called = diag.get("tools_called") or []
    if tools_called:
        st.write("**Selected Tools:** " + ", ".join(f"`{t}`" for t in tools_called))

    st.subheader("4. Evidence & Execution Plan Telemetry")
    pre = incident.get("pre_remediation_metrics") or {}
    ev = incident.get("evidence") or {}
    t_stats = ev.get("table_statistics") or {}
    idx = ev.get("index_inspection") or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pre-Remediation Scan", str(pre.get("scan_type", "N/A")))
    m2.metric("Median Latency", f"{pre.get('median_execution_time_ms', 'N/A')} ms")
    m3.metric("Table Est. Rows", str(t_stats.get("estimated_row_count", "N/A")))
    m4.metric("Suitable Index", str(idx.get("is_suitable_for_filter", "N/A")))

    with st.expander("Inspect Raw Diagnostic Evidence JSON"):
        st.json(ev)

