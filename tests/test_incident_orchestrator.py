"""
tests/test_incident_orchestrator.py
===================================
Tests for Phase 6 end-to-end incident lifecycle orchestrator:
  - Valid full lifecycle progression (CREATED -> ... -> RESOLVED)
  - Strict invalid state transition rejection
  - Simulated vs deterministic vs live_tiny_aya separation
  - No silent fallback from tiny-aya to deterministic
  - Mocked live Tiny Aya execution lifecycle
  - Rejection path closure
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.approval_manager import ApprovalManager
from agent.core.audit_logger import AuditLogger
from agent.core.incident_orchestrator import (
    EXECUTION_MODE_LIVE_TINY_AYA,
    EXECUTION_MODE_SIMULATED,
    STATE_APPROVED,
    STATE_AWAITING_APPROVAL,
    STATE_BLOCKED,
    STATE_CREATED,
    STATE_DIAGNOSED,
    STATE_REJECTED,
    STATE_REMEDIATING,
    STATE_RESOLVED,
    IncidentOrchestrator,
    InvalidStateTransitionError,
    OrchestrationBlockedError,
)
from agent.core.remediation_manager import RemediationManager


def _make_orchestrator(tmp_path: Path, *, agent_factory=None) -> IncidentOrchestrator:
    audit = AuditLogger(tmp_path / "audit.jsonl")
    app = ApprovalManager(store_path=tmp_path / "approvals.json", audit_logger=audit)
    rem = RemediationManager(approval_manager=app, audit_logger=audit)
    return IncidentOrchestrator(
        incidents_dir=tmp_path / "incidents",
        audit_logger=audit,
        approval_manager=app,
        remediation_manager=rem,
        agent_factory=agent_factory,
        comparison_output_path=tmp_path / "remediation_comparison.json",
        env={"LAB_DATABASE_URL": "postgresql://dbmedic:pw@localhost:5432/dbmedic_lab"},
    )


@pytest.mark.unit
def test_complete_simulated_lifecycle_reaches_resolved(tmp_path: Path):
    orch = _make_orchestrator(tmp_path)

    inc = orch.create_incident(
        incident_id="slow_query_001",
        execution_mode="simulated",
    )
    assert inc["status"] == STATE_CREATED
    assert inc["simulated"] is True

    inc = orch.collect_incident_evidence(inc, n_benchmark_runs=5)
    inc = orch.run_agent_diagnosis(inc)
    assert inc["status"] == STATE_DIAGNOSED

    inc = orch.validate_diagnosis(inc)
    assert inc["diagnosis_validation"]["valid"] is True

    inc = orch.prepare_remediation(inc)
    inc = orch.request_human_approval(inc)
    assert inc["status"] == STATE_AWAITING_APPROVAL

    inc = orch.submit_human_decision(
        inc,
        approved=True,
        operator="alice_engineer",
        confirmation_method="interactive_cli",
    )
    assert inc["status"] == STATE_APPROVED

    inc = orch.execute_approved_remediation(inc)
    inc = orch.verify_remediation(inc, n_benchmark_runs=5)
    inc = orch.close_incident(inc)

    assert inc["status"] == STATE_RESOLVED
    assert inc["execution_mode"] == EXECUTION_MODE_SIMULATED
    assert inc["verification_report"]["verification_outcome"]["success"] is True

    integrity = orch.audit_logger.verify_integrity()
    assert integrity["valid"] is True
    assert integrity["total_events"] >= 10


@pytest.mark.unit
def test_invalid_state_transitions_rejected(tmp_path: Path):
    orch = _make_orchestrator(tmp_path)
    inc = orch.create_incident(incident_id="slow_query_invalid", execution_mode="simulated")

    # Cannot skip CREATED -> APPROVED or REMEDIATING or RESOLVED
    with pytest.raises(InvalidStateTransitionError):
        orch.transition_state(inc, STATE_APPROVED)
    with pytest.raises(InvalidStateTransitionError):
        orch.transition_state(inc, STATE_REMEDIATING)
    with pytest.raises(InvalidStateTransitionError):
        orch.transition_state(inc, STATE_RESOLVED)

    # Verify audit trail logged the rejected transition attempts
    rejected_events = orch.audit_logger.read_events(event_type="STATE_TRANSITION_REJECTED")
    assert len(rejected_events) == 3

    # Cannot execute remediation while still AWAITING_APPROVAL
    inc = orch.collect_incident_evidence(inc, n_benchmark_runs=3)
    inc = orch.run_agent_diagnosis(inc)
    inc = orch.validate_diagnosis(inc)
    inc = orch.prepare_remediation(inc)
    inc = orch.request_human_approval(inc)
    assert inc["status"] == STATE_AWAITING_APPROVAL

    with pytest.raises(InvalidStateTransitionError):
        orch.execute_approved_remediation(inc)


@pytest.mark.unit
def test_rejected_incident_cannot_remediate_and_closes_cleanly(tmp_path: Path):
    orch = _make_orchestrator(tmp_path)
    inc = orch.create_incident(incident_id="inc_rej", execution_mode="simulated")
    inc = orch.collect_incident_evidence(inc, n_benchmark_runs=3)
    inc = orch.run_agent_diagnosis(inc)
    inc = orch.validate_diagnosis(inc)
    inc = orch.prepare_remediation(inc)
    inc = orch.request_human_approval(inc)

    inc = orch.submit_human_decision(
        inc,
        approved=False,
        operator="alice_engineer",
        reason="Wait until off-peak",
    )
    assert inc["status"] == STATE_REJECTED

    with pytest.raises(InvalidStateTransitionError):
        orch.execute_approved_remediation(inc)

    inc = orch.close_incident(inc)
    assert inc["status"] == STATE_RESOLVED

    # RESOLVED is terminal
    with pytest.raises(InvalidStateTransitionError):
        orch.transition_state(inc, STATE_REMEDIATING)


@pytest.mark.unit
def test_tiny_aya_mode_never_silently_falls_back_when_unavailable(tmp_path: Path):
    orch = _make_orchestrator(tmp_path)
    inc = orch.create_incident(incident_id="slow_query_blocked", execution_mode="tiny-aya")
    assert inc["execution_mode"] == EXECUTION_MODE_LIVE_TINY_AYA

    # Pretend DB evidence succeeded so we specifically test run_agent_diagnosis blocking
    inc["status"] = "INVESTIGATING"
    inc["evidence"] = {
        "target_database": {"database": "dbmedic_lab", "host": "localhost", "port": 5432},
        "index_inspection": {"is_suitable_for_filter": False},
    }

    with patch(
        "agent.core.incident_orchestrator.check_tiny_aya_availability",
        return_value={"live_inference_ready": False, "blockers": ["No HF_TOKEN or GPU"]},
    ):
        with pytest.raises(OrchestrationBlockedError, match="no silent fallback allowed"):
            orch.run_agent_diagnosis(inc)

    loaded = orch.load_incident("slow_query_blocked")
    assert loaded["status"] == STATE_BLOCKED
    assert loaded["diagnosis"] is None


@pytest.mark.unit
def test_mocked_live_tiny_aya_diagnosis_records_live_mode_and_tools(tmp_path: Path):
    class _MockTinyAyaAgent:
        def run(self, query: str, incident_id: str | None = None, log_trajectory: bool = True):
            return SimpleNamespace(
                success=True,
                model_id="CohereLabs/tiny-aya-global",
                model_revision="rev123",
                tools_called=["get_query_execution_plan", "inspect_customer_id_index"],
                num_steps=3,
                diagnosis="Missing index on orders.customer_id",
                root_cause="missing_suitable_index",
                evidence=["Seq Scan on orders (18.4 ms)"],
                recommended_action="CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
                confidence=0.95,
                error=None,
            )

    orch = _make_orchestrator(tmp_path, agent_factory=lambda model_id: _MockTinyAyaAgent())
    inc = orch.create_incident(incident_id="slow_query_live_mock", execution_mode="tiny-aya")
    inc["status"] = "INVESTIGATING"
    inc["evidence"] = {
        "target_database": {"database": "dbmedic_lab", "host": "localhost", "port": 5432},
        "index_inspection": {"is_suitable_for_filter": False},
        "table_statistics": {"estimated_row_count": 200_000},
    }
    inc["pre_remediation_metrics"] = {"scan_type": "Sequential Scan", "median_execution_time_ms": 22.1}

    inc = orch.run_agent_diagnosis(inc)
    assert inc["status"] == STATE_DIAGNOSED
    assert inc["diagnosis"]["execution_mode"] == EXECUTION_MODE_LIVE_TINY_AYA
    assert inc["diagnosis"]["simulated"] is False
    assert "get_query_execution_plan" in inc["diagnosis"]["tools_called"]

    inc = orch.validate_diagnosis(inc)
    assert inc["diagnosis_validation"]["valid"] is True
    assert inc["diagnosis_validation"]["requires_remediation"] is True

