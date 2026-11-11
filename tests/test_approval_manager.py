"""
tests/test_approval_manager.py
==============================
Security tests for human approval workflow, tamper protection, expiration,
duplicate execution blocking, and database identity verification.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.approval_manager import (
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_REJECTED,
    ApprovalError,
    ApprovalExpiredError,
    ApprovalManager,
    DatabaseIdentityMismatchError,
    DuplicateApprovalError,
    ModelSelfApprovalError,
    SQLTamperingError,
)
from agent.core.audit_logger import AuditLogger

SAMPLE_SQL = "CREATE INDEX idx_orders_customer_id ON orders(customer_id)"
LAB_DB = {"database": "dbmedic_lab", "host": "localhost", "port": 5432}


@pytest.fixture
def mgr(tmp_path: Path) -> ApprovalManager:
    audit = AuditLogger(tmp_path / "audit.jsonl")
    return ApprovalManager(store_path=tmp_path / "approvals.json", audit_logger=audit)


@pytest.mark.unit
def test_forged_raw_model_boolean_rejected(mgr: ApprovalManager):
    mgr.create_approval_request(
        incident_id="slow_query_001",
        action_id="act_1",
        target_database=LAB_DB,
        proposed_sql=SAMPLE_SQL,
        action_description="Create index",
        evidence={"scan_type": "Sequential Scan"},
    )
    # Even if an agent fabricates human_approved=True, verification MUST fail
    with pytest.raises(ModelSelfApprovalError):
        mgr.verify_approval_authorization(
            incident_id="slow_query_001",
            action_id="act_1",
            proposed_sql=SAMPLE_SQL,
            current_database_identity=LAB_DB,
            approval_payload={"human_approved": True},
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_actor",
    ["tiny_aya", "CohereLabs/tiny-aya-global", "model", "agent", "llm", "autonomous"],
)
def test_model_cannot_approve_its_own_action(mgr: ApprovalManager, bad_actor: str):
    mgr.create_approval_request(
        incident_id="slow_query_001",
        action_id="act_1",
        target_database=LAB_DB,
        proposed_sql=SAMPLE_SQL,
        action_description="Create index",
        evidence={"scan_type": "Sequential Scan"},
    )
    with pytest.raises(ModelSelfApprovalError):
        mgr.record_human_approval(
            incident_id="slow_query_001",
            action_id="act_1",
            approved_by=bad_actor,
            confirmation_method="interactive_cli",
        )


@pytest.mark.unit
def test_approval_expiration_rejected(mgr: ApprovalManager):
    t0 = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    mgr.create_approval_request(
        incident_id="slow_query_001",
        action_id="act_exp",
        target_database=LAB_DB,
        proposed_sql=SAMPLE_SQL,
        action_description="Create index",
        evidence={"scan_type": "Sequential Scan"},
        ttl_seconds=60,
        now=t0,
    )
    expired_time = t0 + timedelta(seconds=120)
    with pytest.raises(ApprovalExpiredError):
        mgr.record_human_approval(
            incident_id="slow_query_001",
            action_id="act_exp",
            approved_by="operator_alice",
            confirmation_method="interactive_cli",
            now=expired_time,
        )


@pytest.mark.unit
def test_sql_tampering_invalidates_approval(mgr: ApprovalManager):
    mgr.create_approval_request(
        incident_id="slow_query_001",
        action_id="act_tamper",
        target_database=LAB_DB,
        proposed_sql=SAMPLE_SQL,
        action_description="Create index",
        evidence={"scan_type": "Sequential Scan"},
    )
    rec = mgr.record_human_approval(
        incident_id="slow_query_001",
        action_id="act_tamper",
        approved_by="operator_alice",
        confirmation_method="interactive_cli",
    )
    assert rec["approval_status"] == APPROVAL_STATUS_APPROVED

    with pytest.raises(SQLTamperingError):
        mgr.verify_approval_authorization(
            incident_id="slow_query_001",
            action_id="act_tamper",
            proposed_sql="DROP TABLE orders;",
            current_database_identity=LAB_DB,
        )


@pytest.mark.unit
def test_database_identity_mismatch_rejected(mgr: ApprovalManager):
    mgr.create_approval_request(
        incident_id="slow_query_001",
        action_id="act_db",
        target_database=LAB_DB,
        proposed_sql=SAMPLE_SQL,
        action_description="Create index",
        evidence={"scan_type": "Sequential Scan"},
    )
    mgr.record_human_approval(
        incident_id="slow_query_001",
        action_id="act_db",
        approved_by="operator_bob",
        confirmation_method="interactive_cli",
    )
    with pytest.raises(DatabaseIdentityMismatchError):
        mgr.verify_approval_authorization(
            incident_id="slow_query_001",
            action_id="act_db",
            proposed_sql=SAMPLE_SQL,
            current_database_identity={"database": "production_db", "host": "localhost", "port": 5432},
        )


@pytest.mark.unit
def test_rejected_and_duplicate_executions_blocked(mgr: ApprovalManager):
    mgr.create_approval_request(
        incident_id="inc_rej",
        action_id="act_rej",
        target_database=LAB_DB,
        proposed_sql=SAMPLE_SQL,
        action_description="Create index",
        evidence={"scan_type": "Sequential Scan"},
    )
    rej = mgr.record_human_rejection(
        incident_id="inc_rej",
        action_id="act_rej",
        rejected_by="operator_alice",
        reason="Maintenance window closed",
    )
    assert rej["approval_status"] == APPROVAL_STATUS_REJECTED
    with pytest.raises(ApprovalError):
        mgr.verify_approval_authorization(
            incident_id="inc_rej",
            action_id="act_rej",
            proposed_sql=SAMPLE_SQL,
            current_database_identity=LAB_DB,
        )

    # Now test duplicate execution guard
    mgr.create_approval_request(
        incident_id="inc_dup",
        action_id="act_dup",
        target_database=LAB_DB,
        proposed_sql=SAMPLE_SQL,
        action_description="Create index",
        evidence={"scan_type": "Sequential Scan"},
    )
    mgr.record_human_approval(
        incident_id="inc_dup",
        action_id="act_dup",
        approved_by="operator_alice",
        confirmation_method="interactive_cli",
    )
    mgr.mark_executed(incident_id="inc_dup", action_id="act_dup", execution_result={"success": True})

    with pytest.raises(DuplicateApprovalError):
        mgr.verify_approval_authorization(
            incident_id="inc_dup",
            action_id="act_dup",
            proposed_sql=SAMPLE_SQL,
            current_database_identity=LAB_DB,
        )

