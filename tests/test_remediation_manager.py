"""
tests/test_remediation_manager.py
=================================
Tests for Phase 6 allowlisted remediation enforcement, duplicate index safety,
database identity revalidation, and Phase 1 apply_remediation integration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.approval_manager import ApprovalError, ApprovalManager, DatabaseIdentityMismatchError
from agent.core.audit_logger import AuditLogger
from agent.core.remediation_manager import (
    DisallowedRemediationError,
    RemediationExecutionError,
    RemediationManager,
)

ALLOWED_SQL = "CREATE INDEX idx_orders_customer_id ON orders(customer_id)"
LAB_DB = {"database": "dbmedic_lab", "host": "localhost", "port": 5432}


class _DummyCursor:
    def execute(self, *args, **kwargs):
        pass

    def fetchone(self):
        return ("dbmedic_lab", "127.0.0.1", 5432)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _DummyConn:
    def cursor(self):
        return _DummyCursor()

    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_key,bad_sql",
    [
        ("drop_table_orders", "DROP TABLE orders"),
        ("truncate_orders", "TRUNCATE TABLE orders"),
        ("create_idx_orders_customer_id", "DROP TABLE orders;"),
        ("create_idx_orders_customer_id", "CREATE INDEX idx_orders_customer_id ON orders(customer_id); DROP TABLE users;"),
        ("create_idx_arbitrary", "CREATE INDEX idx_evil ON customers(email)"),
    ],
)
def test_allowlist_blocks_arbitrary_and_destructive_sql(tmp_path: Path, bad_key: str, bad_sql: str):
    audit = AuditLogger(tmp_path / "audit.jsonl")
    app = ApprovalManager(store_path=tmp_path / "approvals.json", audit_logger=audit)
    rem = RemediationManager(approval_manager=app, audit_logger=audit)

    with pytest.raises(DisallowedRemediationError):
        rem.execute_remediation(
            incident_id="slow_query_001",
            action_id="act_1",
            remediation_key=bad_key,
            proposed_sql=bad_sql,
            conn=_DummyConn(),
        )


@pytest.mark.unit
def test_remediation_blocks_without_human_approval(tmp_path: Path):
    audit = AuditLogger(tmp_path / "audit.jsonl")
    app = ApprovalManager(store_path=tmp_path / "approvals.json", audit_logger=audit)
    rem = RemediationManager(approval_manager=app, audit_logger=audit)

    app.create_approval_request(
        incident_id="slow_query_001",
        action_id="act_pending",
        target_database=LAB_DB,
        proposed_sql=ALLOWED_SQL,
        action_description="Create index",
        evidence={"scan_type": "Sequential Scan"},
    )
    with pytest.raises(ApprovalError):
        rem.execute_remediation(
            incident_id="slow_query_001",
            action_id="act_pending",
            remediation_key="create_idx_orders_customer_id",
            proposed_sql=ALLOWED_SQL,
            conn=_DummyConn(),
            current_database_identity=LAB_DB,
        )


@pytest.mark.unit
def test_remediation_detects_already_existing_index_idempotently(tmp_path: Path):
    audit = AuditLogger(tmp_path / "audit.jsonl")
    app = ApprovalManager(store_path=tmp_path / "approvals.json", audit_logger=audit)
    rem = RemediationManager(approval_manager=app, audit_logger=audit)

    app.create_approval_request(
        incident_id="slow_query_001",
        action_id="act_exists",
        target_database=LAB_DB,
        proposed_sql=ALLOWED_SQL,
        action_description="Create index",
        evidence={"scan_type": "Sequential Scan"},
    )
    app.record_human_approval(
        incident_id="slow_query_001",
        action_id="act_exists",
        approved_by="operator_alice",
        confirmation_method="interactive_cli",
    )

    with (
        patch("agent.core.remediation_manager.index_exists", return_value=True),
        patch(
            "agent.core.remediation_manager.get_table_indexes",
            return_value=[
                {
                    "index_name": "idx_orders_customer_id",
                    "index_definition": "CREATE INDEX idx_orders_customer_id ON public.orders USING btree (customer_id)",
                }
            ],
        ),
        patch("agent.core.remediation_manager.apply_remediation") as mock_apply,
    ):
        res = rem.execute_remediation(
            incident_id="slow_query_001",
            action_id="act_exists",
            remediation_key="create_idx_orders_customer_id",
            proposed_sql=ALLOWED_SQL,
            conn=_DummyConn(),
            current_database_identity=LAB_DB,
        )

    assert res["success"] is True
    assert res["index_already_existed"] is True
    mock_apply.assert_not_called()


@pytest.mark.unit
def test_remediation_never_auto_retries_on_ddl_failure(tmp_path: Path):
    audit = AuditLogger(tmp_path / "audit.jsonl")
    app = ApprovalManager(store_path=tmp_path / "approvals.json", audit_logger=audit)
    rem = RemediationManager(approval_manager=app, audit_logger=audit)

    app.create_approval_request(
        incident_id="slow_query_001",
        action_id="act_fail",
        target_database=LAB_DB,
        proposed_sql=ALLOWED_SQL,
        action_description="Create index",
        evidence={"scan_type": "Sequential Scan"},
    )
    app.record_human_approval(
        incident_id="slow_query_001",
        action_id="act_fail",
        approved_by="operator_alice",
        confirmation_method="interactive_cli",
    )

    with (
        patch("agent.core.remediation_manager.index_exists", return_value=False),
        patch("agent.core.remediation_manager.apply_remediation", side_effect=RuntimeError("lock timeout")) as mock_apply,
    ):
        with pytest.raises(RemediationExecutionError, match="no auto-retry"):
            rem.execute_remediation(
                incident_id="slow_query_001",
                action_id="act_fail",
                remediation_key="create_idx_orders_customer_id",
                proposed_sql=ALLOWED_SQL,
                conn=_DummyConn(),
                current_database_identity=LAB_DB,
            )

    assert mock_apply.call_count == 1


@pytest.mark.unit
def test_remediation_blocks_on_db_identity_mismatch(tmp_path: Path):
    audit = AuditLogger(tmp_path / "audit.jsonl")
    app = ApprovalManager(store_path=tmp_path / "approvals.json", audit_logger=audit)
    rem = RemediationManager(approval_manager=app, audit_logger=audit)

    app.create_approval_request(
        incident_id="slow_query_001",
        action_id="act_mismatch",
        target_database=LAB_DB,
        proposed_sql=ALLOWED_SQL,
        action_description="Create index",
        evidence={"scan_type": "Sequential Scan"},
    )
    app.record_human_approval(
        incident_id="slow_query_001",
        action_id="act_mismatch",
        approved_by="operator_alice",
        confirmation_method="interactive_cli",
    )

    with pytest.raises(DatabaseIdentityMismatchError):
        rem.execute_remediation(
            incident_id="slow_query_001",
            action_id="act_mismatch",
            remediation_key="create_idx_orders_customer_id",
            proposed_sql=ALLOWED_SQL,
            conn=_DummyConn(),
            current_database_identity={"database": "wrong_db", "host": "localhost", "port": 5432},
        )

