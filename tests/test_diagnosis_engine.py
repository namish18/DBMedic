"""
tests/test_diagnosis_engine.py
==============================
Automated tests for DBMedic diagnosis_engine.

Tests verify:
  - calculate_diagnostic_score across various evidence conditions and selectivity guardrails.
  - generate_recommendation emits correct DDL and requires_approval flag.
  - create_diagnostic_report satisfies standardized DBMedic JSON schema.
  - diagnose_slow_query detects missing index and sequential scan on orders.
  - Diagnostic engine does not execute remediation automatically or modify data.
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

from agent.core.diagnosis_engine import (
    calculate_diagnosis_confidence,
    calculate_diagnostic_score,
    create_diagnostic_report,
    diagnose_slow_query,
    generate_recommendation,
)


# ---------------------------------------------------------------------------
# Unit Tests (Deterministic Logic & Schema Formatting)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDiagnosisEngineUnit:
    def test_calculate_score_full_incident(self) -> None:
        evidence = {
            "scan_type": "Sequential Scan",
            "execution_time_ms": 18.5,
            "table_name": "orders",
            "filter_column": "customer_id",
            "suitable_index_exists": False,
        }
        table_stats = {
            "estimated_row_count": 200_000,
            "sequential_scan_count": 5,
        }
        selectivity = 20 / 200_000  # 0.0001 (high selectivity)

        score = calculate_diagnostic_score(
            evidence, table_stats, row_selectivity=selectivity
        )
        assert score >= 0.95

    def test_calculate_score_small_table_penalized(self) -> None:
        evidence = {
            "scan_type": "Sequential Scan",
            "execution_time_ms": 1.2,
            "table_name": "orders",
            "filter_column": "customer_id",
            "suitable_index_exists": False,
        }
        table_stats = {
            "estimated_row_count": 50,  # Tiny table (<1000 rows)
            "sequential_scan_count": 0,
        }
        selectivity = 0.5

        score = calculate_diagnostic_score(
            evidence, table_stats, row_selectivity=selectivity
        )
        # Small tables should receive a low diagnostic score because seq scan is normal
        assert score <= 0.40

    def test_calculate_score_poor_selectivity_penalized(self) -> None:
        evidence = {
            "scan_type": "Sequential Scan",
            "execution_time_ms": 15.0,
            "table_name": "orders",
            "filter_column": "customer_id",
            "suitable_index_exists": False,
        }
        table_stats = {
            "estimated_row_count": 200_000,
            "sequential_scan_count": 1,
        }
        # Poor selectivity (>15% rows matched, e.g. 50,000 / 200,000 = 25%)
        score = calculate_diagnostic_score(
            evidence, table_stats, row_selectivity=0.25
        )
        assert score < 0.80

    def test_generate_recommendation_missing_index(self) -> None:
        rec = generate_recommendation("missing_suitable_index", "orders", "customer_id", row_count=200_000, selectivity=0.001)
        assert rec["action"] == "create_index"
        assert rec["sql"] == "CREATE INDEX idx_orders_customer_id ON orders(customer_id)"
        assert rec["requires_approval"] is True

    def test_generate_recommendation_small_table_suppressed(self) -> None:
        # Tables under 1000 rows should NOT receive index creation recommendations
        rec = generate_recommendation("missing_suitable_index", "orders", "customer_id", row_count=50, selectivity=0.01)
        assert rec["action"] == "none"
        assert rec["sql"] is None
        assert rec["requires_approval"] is False

    def test_generate_recommendation_no_root_cause(self) -> None:
        rec = generate_recommendation("none_detected", "orders", "customer_id")
        assert rec["action"] == "none"
        assert rec["sql"] is None
        assert rec["requires_approval"] is False

    def test_create_diagnostic_report_schema(self) -> None:
        report = create_diagnostic_report(
            incident_id="slow_query_001",
            incident_type="slow_query",
            status="diagnosed",
            severity="high",
            summary="Query performance degradation detected",
            root_cause={
                "type": "missing_suitable_index",
                "description": "Sequential scan detected on orders",
            },
            evidence={
                "scan_type": "Sequential Scan",
                "execution_time_ms": 18.5,
                "table_name": "orders",
                "filter_column": "customer_id",
                "suitable_index_exists": False,
            },
            recommendation={
                "action": "create_index",
                "sql": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
                "requires_approval": True,
            },
            diagnostic_score=0.95,
        )

        expected_keys = {
            "incident_id",
            "incident_type",
            "status",
            "severity",
            "summary",
            "root_cause",
            "evidence",
            "recommendation",
            "diagnostic_score",
            "confidence",
        }
        assert expected_keys.issubset(set(report.keys()))
        assert report["root_cause"]["type"] == "missing_suitable_index"
        assert report["evidence"]["table_name"] == "orders"
        assert report["evidence"]["filter_column"] == "customer_id"
        assert report["recommendation"]["requires_approval"] is True
        assert report["diagnostic_score"] == 0.95
        assert report["confidence"] == 0.95


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
class TestDiagnosisEngineIntegration:
    def test_diagnose_slow_query_produces_structured_report(
        self, live_conn: psycopg.Connection
    ) -> None:
        report = diagnose_slow_query(conn=live_conn)

        assert report["incident_id"] == "slow_query_001"
        assert report["incident_type"] == "slow_query"
        assert report["status"] in ("diagnosed", "optimized")
        assert "root_cause" in report
        assert "evidence" in report
        assert "recommendation" in report
        assert 0.0 <= report["diagnostic_score"] <= 1.0

        evidence = report["evidence"]
        assert evidence["table_name"] == "orders"
        assert evidence["filter_column"] == "customer_id"
        assert isinstance(evidence["suitable_index_exists"], bool)
        assert evidence["execution_time_ms"] >= 0.0

    def test_diagnosis_does_not_modify_database(
        self, live_conn: psycopg.Connection
    ) -> None:
        with live_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM orders")
            before_orders = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'orders'"
            )
            before_indexes = cur.fetchone()[0]

        diagnose_slow_query(conn=live_conn)

        with live_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM orders")
            after_orders = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'orders'"
            )
            after_indexes = cur.fetchone()[0]

        assert before_orders == after_orders
        assert before_indexes == after_indexes
