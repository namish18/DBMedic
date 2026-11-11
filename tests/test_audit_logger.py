"""
tests/test_audit_logger.py
==========================
Tests for append-only SHA-256 hash-chained audit logging and credential redaction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.audit_logger import (
    GENESIS_HASH,
    AuditLogger,
    redact_database_url,
    redact_secrets,
)


@pytest.mark.unit
def test_redact_database_url_and_secrets():
    url = "postgresql://dbmedic:super_secret_pw@localhost:5432/dbmedic_lab"
    assert redact_database_url(url) == "postgresql://dbmedic:****@localhost:5432/dbmedic_lab"

    payload = {
        "hf_token": "hf_abc1234567890XYZ",
        "postgres_password": "pw",
        "token_present": False,
        "lab_database_url": url,
        "safe_note": f"Failed connecting to {url} using hf_abc1234567890XYZ",
    }
    cleaned = redact_secrets(payload)
    dumped = json.dumps(cleaned)
    assert "super_secret_pw" not in dumped
    assert "hf_abc1234567890XYZ" not in dumped
    assert cleaned["hf_token"] == "***REDACTED***"
    assert cleaned["postgres_password"] == "***REDACTED***"
    assert cleaned["token_present"] is False


@pytest.mark.unit
def test_audit_logger_appends_valid_chain_and_orders_events(tmp_path: Path):
    log_file = tmp_path / "audit_trail.jsonl"
    audit = AuditLogger(log_path=log_file)

    e1 = audit.log_event(
        incident_id="slow_query_001",
        event_type="INCIDENT_CREATED",
        payload={"query": "SELECT COUNT(*) FROM orders", "password": "leak"},
        execution_mode="deterministic",
    )
    e2 = audit.log_event(
        incident_id="slow_query_001",
        event_type="APPROVAL_REQUESTED",
        payload={"proposed_sql": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)"},
        execution_mode="deterministic",
    )
    e3 = audit.log_event(
        incident_id="other_incident",
        event_type="INCIDENT_CLOSED",
        payload={"status": "RESOLVED"},
        execution_mode="simulated",
    )

    assert e1["sequence"] == 1
    assert e1["prev_event_hash"] == GENESIS_HASH
    assert e2["sequence"] == 2
    assert e2["prev_event_hash"] == e1["event_hash"]
    assert e3["sequence"] == 3
    assert e3["prev_event_hash"] == e2["event_hash"]
    assert "leak" not in log_file.read_text(encoding="utf-8")

    integrity = audit.verify_integrity()
    assert integrity["valid"] is True
    assert integrity["total_events"] == 3
    assert "not cryptographically tamper-proof" in integrity["limitations"].lower()

    filtered = audit.read_events(incident_id="slow_query_001")
    assert len(filtered) == 2
    assert [ev["event_type"] for ev in filtered] == [
        "INCIDENT_CREATED",
        "APPROVAL_REQUESTED",
    ]


@pytest.mark.unit
def test_audit_logger_detects_tampering(tmp_path: Path):
    log_file = tmp_path / "audit_trail.jsonl"
    audit = AuditLogger(log_path=log_file)

    audit.log_event("inc_1", "INCIDENT_CREATED", {"detail": "initial"})
    audit.log_event("inc_1", "REMEDIATION_EXECUTED", {"detail": "created index"})

    lines = log_file.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["payload"]["detail"] = "forged"
    lines[0] = json.dumps(tampered)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    integrity = audit.verify_integrity()
    assert integrity["valid"] is False
    assert len(integrity["hash_mismatches"]) == 1
