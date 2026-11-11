"""
dashboard/components/audit_panel.py
===================================
Append-only audit trail viewer and hash-chain integrity display.
"""

from __future__ import annotations

from typing import Any

from agent.core.audit_logger import AuditLogger


def render_audit_panel(st: Any, audit_logger: AuditLogger, incident_id: str | None = None) -> None:
    """Render Audit Trail table & hash chain verification section."""
    st.subheader("8. Audit Trail & Hash-Chain Verification")
    integrity = audit_logger.verify_integrity()
    events = audit_logger.read_events(incident_id=incident_id) if incident_id else audit_logger.read_events()

    c1, c2 = st.columns(2)
    c1.metric("Hash Chain Valid", str(integrity.get("valid")))
    c2.metric("Total Recorded Events", str(integrity.get("total_events")))
    st.caption(str(integrity.get("limitations")))

    if not events:
        st.info("No audit trail events recorded yet.")
        return

    rows = [
        {
            "seq": e.get("sequence"),
            "timestamp": e.get("timestamp"),
            "incident_id": e.get("incident_id"),
            "event_type": e.get("event_type"),
            "mode": e.get("execution_mode"),
            "actor": e.get("actor"),
            "hash_prefix": str(e.get("event_hash", ""))[:12],
        }
        for e in reversed(events[-50:])
    ]
    st.dataframe(rows, use_container_width=True)

