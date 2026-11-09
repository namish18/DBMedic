"""
tests/test_table_monitor.py
===========================
Automated tests for table_monitor tool.

Tests verify:
  - Allowlist enforcement for table names (rejecting invalid / arbitrary names).
  - Retrieval of row count estimates, table sizes, and sequential scan counts.
  - Calculation of sequential scan ratio.
  - Ensuring table monitoring operations are strictly read-only.
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

from agent.tools.table_monitor import (
    ALLOWED_TABLES,
    _validate_table_name,
    get_sequential_scan_statistics,
    get_table_row_estimates,
    get_table_size,
    get_table_statistics,
)


# ---------------------------------------------------------------------------
# Unit Tests (Security & Allowlist)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTableAllowlist:
    def test_permitted_tables_accepted(self) -> None:
        for tbl in ALLOWED_TABLES:
            assert _validate_table_name(tbl) == tbl
            assert _validate_table_name(tbl.upper()) == tbl

    def test_disallowed_table_rejected(self) -> None:
        disallowed = ["users", "pg_shadow", "passwords", "orders; DROP TABLE orders;"]
        for tbl in disallowed:
            with pytest.raises(ValueError, match="not in permitted allowlist"):
                _validate_table_name(tbl)

    def test_non_string_type_rejected(self) -> None:
        with pytest.raises(TypeError):
            _validate_table_name(1234)  # type: ignore[arg-type]


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
class TestTableMonitorIntegration:
    def test_get_table_statistics(self, live_conn: psycopg.Connection) -> None:
        stats = get_table_statistics("orders", conn=live_conn)
        assert stats["table_name"] == "orders"
        assert isinstance(stats["estimated_row_count"], int)
        assert isinstance(stats["table_size_bytes"], int)
        assert stats["table_size_bytes"] > 0
        assert isinstance(stats["sequential_scan_count"], int)
        assert "table_size_pretty" in stats
        assert "dead_tuple_estimate" in stats

    def test_get_table_size(self, live_conn: psycopg.Connection) -> None:
        size = get_table_size("orders", conn=live_conn)
        assert size["table_name"] == "orders"
        assert size["total_size_bytes"] >= size["table_size_bytes"]
        assert isinstance(size["total_size_pretty"], str)
        assert isinstance(size["index_size_bytes"], int)

    def test_get_sequential_scan_statistics(self, live_conn: psycopg.Connection) -> None:
        stats = get_sequential_scan_statistics("orders", conn=live_conn)
        assert stats["table_name"] == "orders"
        assert stats["sequential_scan_count"] >= 0
        assert stats["sequential_tuples_read"] >= 0
        assert 0.0 <= stats["seq_scan_ratio"] <= 1.0

    def test_get_table_row_estimates(self, live_conn: psycopg.Connection) -> None:
        estimates = get_table_row_estimates("orders", conn=live_conn)
        assert estimates["table_name"] == "orders"
        assert estimates["estimated_row_count"] >= 0
        assert estimates["pages"] >= 0

    def test_invalid_table_raises_in_get_table_statistics(
        self, live_conn: psycopg.Connection
    ) -> None:
        with pytest.raises(ValueError, match="not in permitted allowlist"):
            get_table_statistics("malicious_table", conn=live_conn)

    def test_table_monitor_does_not_modify_data(
        self, live_conn: psycopg.Connection
    ) -> None:
        with live_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM orders")
            before = cur.fetchone()[0]

        get_table_statistics("orders", conn=live_conn)
        get_table_size("orders", conn=live_conn)
        get_sequential_scan_statistics("orders", conn=live_conn)

        with live_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM orders")
            after = cur.fetchone()[0]

        assert before == after

