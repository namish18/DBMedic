"""
tests/test_environment_check.py
===============================
Tests for Phase 6 environment readiness inspection and credential protection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.environment_check import (
    STATUS_BLOCKED,
    STATUS_OPTIONAL,
    STATUS_READY,
    check_agenttune_availability,
    check_auditkit_availability,
    check_database_connection,
    check_gpu_availability,
    check_huggingface_access,
    check_lab_schema,
    check_python_environment,
    check_tiny_aya_availability,
    generate_readiness_report,
)


class _FakeCursor:
    def __init__(self):
        self._sql = ""

    def execute(self, sql, params=None):
        self._sql = str(sql)

    def fetchone(self):
        if "current_database" in self._sql:
            return ("dbmedic_lab", "dbmedic", "127.0.0.1", 5432, "PostgreSQL 17.0")
        if "reltuples" in self._sql:
            return (200000,)
        if "pg_indexes" in self._sql:
            return (False,)
        return None

    def fetchall(self):
        if "pg_tables" in self._sql:
            return [("customers",), ("order_items",), ("orders",), ("products",)]
        return []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeConn:
    def cursor(self):
        return _FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.mark.unit
def test_check_python_environment_detects_missing_env_vars_and_redacts():
    res = check_python_environment(
        env={
            "HF_TOKEN": "hf_SUPERSECRETTOKENVALUE123456",
            "POSTGRES_PASSWORD": "my_secret_pw",
        }
    )
    assert res["status"] == STATUS_BLOCKED
    assert "LAB_DATABASE_URL" in res["missing_required_env"]
    serialized = json.dumps(res)
    assert "hf_SUPERSECRETTOKENVALUE123456" not in serialized
    assert "my_secret_pw" not in serialized


@pytest.mark.unit
def test_check_huggingface_access_never_exposes_token():
    secret = "hf_ABC123456789SecretToken"
    res = check_huggingface_access(env={"HF_TOKEN": secret})
    assert res["token_present"] is True
    assert res["token_env_var"] == "HF_TOKEN"
    assert secret not in json.dumps(res)


@pytest.mark.unit
def test_check_auditkit_distinguishes_stub():
    res = check_auditkit_availability()
    assert "using_stub" in res
    assert "license" in res
    assert "LSAL-1.2" in res["license"]
    if res["using_stub"]:
        assert res["installed"] is False
        assert res["stub_name"] == "AuditKITStub"
        assert "NOT independent AuditKIT validation" in res["note"]


@pytest.mark.unit
def test_check_agenttune_and_gpu_and_tiny_aya():
    at = check_agenttune_availability()
    assert "installed" in at
    assert "dataset_manifest_exists" in at

    gpu = check_gpu_availability()
    assert "cuda_available" in gpu
    assert "total_ram_gb" in gpu

    ta = check_tiny_aya_availability(env={})
    assert ta["deterministic_fallback_supported"] is True
    assert ta["status"] in {STATUS_READY, STATUS_OPTIONAL}


@pytest.mark.unit
def test_check_database_connection_missing_url():
    res = check_database_connection(env={})
    assert res["status"] == STATUS_BLOCKED
    assert res["connected"] is False


@pytest.mark.unit
def test_check_database_connection_and_schema_mocked_ready():
    secret_url = "postgresql://dbmedic:TopSecretPass99@localhost:5432/dbmedic_lab"

    conn_res = check_database_connection(
        env={"LAB_DATABASE_URL": secret_url},
        conn_factory=_FakeConn,
    )
    assert conn_res["status"] == STATUS_READY
    assert conn_res["connected"] is True
    assert "TopSecretPass99" not in json.dumps(conn_res)

    schema_res = check_lab_schema(
        env={"LAB_DATABASE_URL": secret_url},
        conn_factory=_FakeConn,
    )
    assert schema_res["status"] == STATUS_READY
    assert schema_res["schema_valid"] is True
    assert schema_res["orders_estimated_rows"] == 200000
    assert schema_res["idx_orders_customer_id_exists"] is False


@pytest.mark.unit
def test_generate_readiness_report_writes_sanitized_json(tmp_path: Path):
    out_file = tmp_path / "environment_readiness.json"
    secret_url = "postgresql://dbmedic:VerySecretPW@localhost:5432/dbmedic_lab"

    report = generate_readiness_report(
        output_path=out_file,
        env={
            "LAB_DATABASE_URL": secret_url,
            "HF_TOKEN": "hf_HiddenToken1234567890",
        },
        conn_factory=lambda: (_ for _ in ()).throw(ConnectionRefusedError("offline")),
        save=True,
    )
    assert out_file.exists()
    raw = out_file.read_text(encoding="utf-8")
    assert "VerySecretPW" not in raw
    assert "hf_HiddenToken1234567890" not in raw
    assert report["capabilities"]["offline_mode_ready"] is True
    assert report["checks"]["database_connection"]["status"] == STATUS_BLOCKED
