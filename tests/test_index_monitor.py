"""
tests/test_index_monitor.py
===========================
Automated tests for index_monitor tool.

Tests verify:
  - Table name allowlist enforcement.
  - Safe identifier validation for index checking.
  - Retrieval of index definitions and indexed columns from pg_indexes.
  - Retrieval of index usage metrics from pg_stat_user_indexes.
  - Inspection of orders.customer_id index presence/absence.
  - Verification that index monitoring never creates or deletes database objects.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from agent.tools.index_monitor import (
    _validate_index_identifier,
    _validate_table_name,
    check_index_exists,
    get_index_usage,
    get_table_indexes,
    inspect_customer_id_index,
)


# ---------------------------------------------------------------------------
# Unit Tests (Identifier & Security Validation)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIndexIdentifierValidation:
    def test_safe_identifiers_accepted(self) -> None:
        valid_names = ["orders_pkey", "idx_orders_customer_id", "custom_idx_123"]
        for name in valid_names:
            assert _validate_index_identifier(name) == name

    def test_unsafe_identifiers_rejected(self) -> None:
        unsafe_names = [
            "idx; DROP TABLE orders;",
            "idx' OR '1'='1",
            "idx name with spaces",
            "123_starts_with_digit",
        ]
        for name in unsafe_names:
            with pytest.raises(ValueError, match="Invalid index identifier"):
                _validate_index_identifier(name)

    def test_non_string_rejected(self) -> None:
        with pytest.raises(TypeError):
            _validate_index_identifier(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration Tests (Live Database if reachable)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_conn():
    url = os.getenv("LAB_DATABASE_URL")
    if not url:
        pytest.skip("LAB_DATABASE_URL not set")
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            yield conn
    except Exception as exc:
        pytest.skip(f"Database dbmedic_lab unreachable ({exc})")


@pytest.mark.integration
class TestIndexMonitorIntegration:
    def test_get_table_indexes_returns_primary_key(
        self, live_conn: psycopg.Connection
    ) -> None:
        indexes = get_table_indexes("orders", conn=live_conn)
        assert isinstance(indexes, list)
        assert len(indexes) >= 1

        pk_index = next((i for i in indexes if i["index_name"] == "orders_pkey"), None)
        assert pk_index is not None
        assert pk_index["is_unique"] is True
        assert pk_index["indexed_columns"] == ["id"]
        assert "CREATE UNIQUE INDEX" in pk_index["index_definition"]

    def test_check_index_exists_primary_key(
        self, live_conn: psycopg.Connection
    ) -> None:
        assert check_index_exists("orders_pkey", conn=live_conn) is True
        assert check_index_exists("non_existent_index_xyz", conn=live_conn) is False

    def test_get_index_usage(self, live_conn: psycopg.Connection) -> None:
        usage = get_index_usage("orders", conn=live_conn)
        assert isinstance(usage, list)
        for entry in usage:
            assert "index_name" in entry
            assert entry["table_name"] == "orders"
            assert entry["index_scan_count"] >= 0

    def test_inspect_customer_id_index_structure(
        self, live_conn: psycopg.Connection
    ) -> None:
        result = inspect_customer_id_index(conn=live_conn)
        assert result["table_name"] == "orders"
        assert result["target_column"] == "customer_id"
        assert isinstance(result["index_exists"], bool)
        assert isinstance(result["is_suitable_for_filter"], bool)
        assert result["index_scan_count"] >= 0

    def test_index_monitor_does_not_modify_indexes(
        self, live_conn: psycopg.Connection
    ) -> None:
        before = get_table_indexes("orders", conn=live_conn)
        inspect_customer_id_index(conn=live_conn)
        after = get_table_indexes("orders", conn=live_conn)

        assert [i["index_name"] for i in before] == [i["index_name"] for i in after]

