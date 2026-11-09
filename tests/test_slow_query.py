"""
tests/test_slow_query.py
==========================
Automated tests for the slow-query incident and remediation workflow.

Tests verify:
  - The target query executes correctly.
  - A sequential scan is detected when no index is present.
  - Remediation gating: apply_remediation raises PermissionError without approval.
  - Remediation gating: unknown keys raise KeyError.
  - BenchmarkResult serialisation is correct.
  - compare_results computes improvement correctly.

NOTE: Tests that mutate the database (applying the index) are marked with
``pytest.mark.mutating`` so they can be excluded in CI runs that only verify
read-only behaviour:

    pytest tests/test_slow_query.py -v -m "not mutating"

Run all (including mutating tests):
    pytest tests/test_slow_query.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg
import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from agent.tools.query_analyzer import apply_remediation, index_exists
from database.scenarios.slow_query import (
    BenchmarkResult,
    compare_results,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lab_url() -> str:
    url = os.getenv("LAB_DATABASE_URL")
    if not url:
        pytest.skip("LAB_DATABASE_URL not set — skipping slow-query tests")
    return url


@pytest.fixture(scope="module")
def conn(lab_url: str):
    try:
        with psycopg.connect(lab_url, connect_timeout=3) as connection:
            yield connection
    except Exception as exc:
        pytest.skip(f"Database dbmedic_lab unreachable ({exc})")


# ---------------------------------------------------------------------------
# Target query correctness
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTargetQuery:
    def test_query_executes_without_error(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM orders WHERE customer_id = %s", (4242,)
            )
            result = cur.fetchone()
        assert result is not None
        assert isinstance(result[0], int)
        assert result[0] >= 0

    def test_query_returns_non_negative_count(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM orders WHERE customer_id = %s", (1,)
            )
            count = cur.fetchone()[0]
        assert count >= 0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBenchmarkRunner:
    def test_run_benchmark_returns_result(self, conn: psycopg.Connection) -> None:
        result = run_benchmark(conn, n_runs=3)
        assert isinstance(result, BenchmarkResult)

    def test_benchmark_has_correct_run_count(self, conn: psycopg.Connection) -> None:
        result = run_benchmark(conn, n_runs=3)
        assert len(result.runs) == 3

    def test_benchmark_median_is_positive(self, conn: psycopg.Connection) -> None:
        result = run_benchmark(conn, n_runs=3)
        assert result.median_execution_time_ms > 0

    def test_benchmark_detects_scan_type(self, conn: psycopg.Connection) -> None:
        result = run_benchmark(conn, n_runs=3)
        assert result.scan_type != "unknown"

    def test_benchmark_serialisation(self, conn: psycopg.Connection) -> None:
        result = run_benchmark(conn, n_runs=3)
        d = result.to_dict()
        assert "incident_id" in d
        assert "median_execution_time_ms" in d
        assert "scan_type" in d
        assert "runs" in d
        assert isinstance(d["runs"], list)


# ---------------------------------------------------------------------------
# Sequential scan detection (pre-remediation)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSeqScanDetection:
    def test_seq_scan_occurs_without_index(self, conn: psycopg.Connection) -> None:
        """
        When idx_orders_customer_id does not exist the planner should choose
        a sequential scan on the orders table.
        """
        if index_exists(conn, "idx_orders_customer_id"):
            pytest.skip("Index already exists — incident state has been remediated")

        result = run_benchmark(conn, n_runs=3)
        # The plan should contain a Seq Scan node somewhere on orders
        assert "Seq Scan" in result.scan_type, (
            f"Expected 'Seq Scan' but got '{result.scan_type}'. "
            "The planner may have chosen a different plan — check table statistics."
        )


# ---------------------------------------------------------------------------
# Remediation gating
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRemediationGating:
    def test_remediation_requires_approval(self, conn: psycopg.Connection) -> None:
        """apply_remediation must raise PermissionError when approved=False."""
        with pytest.raises(PermissionError, match="explicit approval"):
            apply_remediation(conn, "create_idx_orders_customer_id", approved=False)

    def test_remediation_rejects_unknown_key(self, conn: psycopg.Connection) -> None:
        """Unknown remediation keys must raise KeyError."""
        with pytest.raises(KeyError):
            apply_remediation(conn, "drop_everything", approved=True)

    def test_remediation_default_is_not_approved(self, conn: psycopg.Connection) -> None:
        """The default value of approved must be False."""
        with pytest.raises(PermissionError):
            apply_remediation(conn, "create_idx_orders_customer_id")


# ---------------------------------------------------------------------------
# compare_results unit tests (no DB needed)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCompareResults:
    def test_improvement_detected(self) -> None:
        before = {"median_execution_time_ms": 200.0, "scan_type": "Seq Scan"}
        after  = {"median_execution_time_ms": 5.0,   "scan_type": "Index Scan"}
        result = compare_results(before, after)
        assert result["performance_improved"] is True
        assert result["improvement_pct"] == pytest.approx(97.5, abs=0.1)

    def test_no_improvement_detected(self) -> None:
        before = {"median_execution_time_ms": 5.0,   "scan_type": "Index Scan"}
        after  = {"median_execution_time_ms": 10.0,  "scan_type": "Index Scan"}
        result = compare_results(before, after)
        assert result["performance_improved"] is False

    def test_zero_baseline_handled(self) -> None:
        before = {"median_execution_time_ms": 0.0, "scan_type": "unknown"}
        after  = {"median_execution_time_ms": 1.0, "scan_type": "Index Scan"}
        result = compare_results(before, after)
        assert result["improvement_pct"] == 0.0

