"""
dashboard/components/approval_panel.py
======================================
Human-in-the-loop remediation approval panel.

Displaying a recommendation NEVER executes DDL. Explicit human review inputs
and button clicks are strictly required.
"""

from __future__ import annotations

from typing import Any

from agent.core.approval_manager import ApprovalError
from agent.core.incident_orchestrator import (
    STATE_APPROVED,
    STATE_AWAITING_APPROVAL,
    STATE_VERIFYING,
    IncidentOrchestrator,
)


def render_approval_panel(st: Any, incident: dict[str, Any] | None, orchestrator: IncidentOrchestrator) -> None:
    """Render Remediation Approval controls requiring explicit human interaction."""
    st.subheader("5. Remediation Approval & Safe Execution")
    if not incident:
        st.info("No incident selected.")
        return

    req = incident.get("approval_request")
    if not req:
        st.info(f"No pending DDL approval request for '{incident['incident_id']}' (status={incident['status']}).")
        return

    st.write(f"**Current Lifecycle Status:** `{incident['status']}`")
    st.write(f"**Action ID:** `{req.get('action_id')}`")
    st.write(f"**Target Database:** `{(req.get('target_database') or {}).get('database')}`")
    st.write(f"**SQL SHA-256:** `{req.get('sql_sha256')}`")
    st.code(f"{req.get('proposed_sql')};", language="sql")

    status = incident.get("status")
    if status == STATE_AWAITING_APPROVAL:
        st.info("No database modification has occurred. Review the exact SQL above before deciding.")
        operator_name = st.text_input("Human Operator Name (model names rejected)", key=f"op_{incident['incident_id']}")
        reviewed_chk = st.checkbox("I have reviewed the exact SQL and target database above.", key=f"chk_{incident['incident_id']}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Approve SQL Action", key=f"app_btn_{incident['incident_id']}", type="primary"):
                if not operator_name.strip() or not reviewed_chk:
                    st.error("Enter a valid human operator name and check the SQL review confirmation box.")
                else:
                    try:
                        orchestrator.submit_human_decision(
                            incident,
                            approved=True,
                            operator=operator_name.strip(),
                            confirmation_method="dashboard_human_confirm",
                        )
                        st.success("Approval recorded! Incident moved to APPROVED.")
                    except ApprovalError as exc:
                        st.error(f"Approval blocked: {exc}")
        with c2:
            reject_reason = st.text_input("Rejection Reason (optional)", key=f"rej_rsn_{incident['incident_id']}")
            if st.button("Reject Action", key=f"rej_btn_{incident['incident_id']}"):
                if not operator_name.strip():
                    st.error("Enter human operator name to record rejection.")
                else:
                    orchestrator.submit_human_decision(
                        incident,
                        approved=False,
                        operator=operator_name.strip(),
                        reason=reject_reason or "Rejected from dashboard",
                    )
                    orchestrator.close_incident(incident)
                    st.warning("Action rejected; incident closed without database changes.")
    elif status == STATE_APPROVED:
        st.success(f"Approved by `{req.get('approved_by')}` via `{req.get('confirmation_method')}`.")
        if st.button("Execute Approved Remediation & Verify", key=f"exec_btn_{incident['incident_id']}"):
            try:
                updated = orchestrator.execute_approved_remediation(incident)
                if updated["status"] == STATE_VERIFYING:
                    updated = orchestrator.verify_remediation(updated)
                    updated = orchestrator.close_incident(updated)
                st.success(f"Remediation & verification complete! Final status: {updated['status']}")
            except Exception as exc:
                st.error(f"Execution failed cleanly: {exc}")
    else:
        st.write(f"Approval state: `{req.get('approval_status')}` (operator: `{req.get('approved_by') or req.get('rejected_by') or 'N/A'}`).")

