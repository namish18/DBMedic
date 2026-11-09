"""
tests/test_query_analyzer.py
============================
Automated tests for query_analyzer tool.

Tests verify:
  - Plan retrieval for registered templates.
  - Rejection of unregistered templates (preventing arbitrary SQL).
  - Extraction of planning time, execution time, buffer hits, and scan types.
  - Detection of Sequential Scan vs Index Scan from execution plans.
  - Read-only guarantees (no database modification during plan retrieval).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import psycopg
import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from agent.tools.query_analyzer import (
    DEFAULT_CUSTOMER_ID,
    DEFAULT_QUERY_KEY,
    REGISTERED_QUERY_TEMPLATES,
    analyze_query_performance,
    detect_sequential_scan,
    extract_plan_metrics,
    get_query_execution_plan,
)


# ---------------------------------------------------------------------------
# Sample Mock Plans
# ---------------------------------------------------------------------------

MOCK_SEQ_SCAN_PLAN = [
    {
        "Plan": {
            "Node Type": "Aggregate",
            "Strategy": "Plain",
            "Partial Mode": "Simple",
            "Parallel Aware": False,
            "Startup Cost": 4200.0,
            "Total Cost": 4200.01,
            "Plan Rows": 1,
            "Plan Width": 8,
            "Actual Startup Time": 18.2,
            "Actual Total Time": 18.25,
            "Actual Rows": 1,
            "Actual Loops": 1,
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": False,
                    "Relation Name": "orders",
                    "Alias": "orders",
                    "Startup Cost": 0.0,
                    "Total Cost": 4150.0,
                    "Plan Rows": 20,
                    "Plan Width": 0,
                    "Actual Startup Time": 0.05,
                    "Actual Total Time": 18.1,
                    "Actual Rows": 22,
                    "Actual Loops": 1,
                    "Filter": "(customer_id = 4242)",
                    "Rows Removed by Filter": 199978,
                    "Shared Hit Blocks": 1250,
                    "Shared Read Blocks": 45,
                }
            ],
        },
        "Planning Time": 0.15,
        "Execution Time": 18.42,
        "Shared Hit Blocks": 1250,
        "Shared Read Blocks": 45,
    }
]

MOCK_INDEX_SCAN_PLAN = [
    {
        "Plan": {
            "Node Type": "Aggregate",
            "Strategy": "Plain",
            "Partial Mode": "Simple",
            "Parallel Aware": False,
            "Startup Cost": 8.5,
            "Total Cost": 8.51,
            "Plan Rows": 1,
            "Plan Width": 8,
            "Actual Startup Time": 0.35,
            "Actual Total Time": 0.36,
            "Actual Rows": 1,
            "Actual Loops": 1,
            "Plans": [
                {
                    "Node Type": "Index Scan",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": False,
                    "Scan Direction": "Forward",
                    "Index Name": "idx_orders_customer_id",
                    "Relation Name": "orders",
                    "Alias": "orders",
                    "Startup Cost": 0.42,
                    "Total Cost": 8.45,
                    "Plan Rows": 20,
                    "Plan Width": 0,
                    "Actual Startup Time": 0.04,
                    "Actual Total Time": 0.32,
                    "Actual Rows": 22,
                    "Actual Loops": 1,
                    "Index Cond": "(customer_id = 4242)",
                    "Shared Hit Blocks": 4,
                    "Shared Read Blocks": 0,
                }
            ],
        },
        "Planning Time": 0.22,
        "Execution Time": 0.45,
        "Shared Hit Blocks": 4,
        "Shared Read Blocks": 0,
    }
]


# ---------------------------------------------------------------------------
# Unit Tests (Deterministic Parser Logic)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPlanParsers:
    def test_detect_sequential_scan_positive(self) -> None:
        analysis = detect_sequential_scan(MOCK_SEQ_SCAN_PLAN)
        assert analysis["has_sequential_scan"] is True
        assert analysis["primary_scan_type"] == "Sequential Scan"
        assert analysis["raw_node_type"] == "Seq Scan"
        assert analysis["scanned_relation"] == "orders"
        assert analysis["scanned_index"] is None
        assert len(analysis["all_scans"]) == 1

    def test_detect_sequential_scan_negative_on_index_scan(self) -> None:
        analysis = detect_sequential_scan(MOCK_INDEX_SCAN_PLAN)
        assert analysis["has_sequential_scan"] is False
        assert analysis["primary_scan_type"] == "Index Scan"
        assert analysis["raw_node_type"] == "Index Scan"
        assert analysis["scanned_relation"] == "orders"
        assert analysis["scanned_index"] == "idx_orders_customer_id"

    def test_extract_plan_metrics_seq_scan(self) -> None:
        metrics = extract_plan_metrics(MOCK_SEQ_SCAN_PLAN)
        assert metrics["planning_time_ms"] == pytest.approx(0.15, abs=0.01)
        assert metrics["execution_time_ms"] == pytest.approx(18.42, abs=0.01)
        assert metrics["actual_rows"] == 22
        assert metrics["estimated_rows"] == 20
        assert metrics["shared_hit_blocks"] == 1250
        assert metrics["shared_read_blocks"] == 45
        assert metrics["relation_name"] == "orders"
        assert metrics["scan_type"] == "Sequential Scan"

    def test_extract_plan_metrics_index_scan(self) -> None:
        metrics = extract_plan_metrics(MOCK_INDEX_SCAN_PLAN)
        assert metrics["planning_time_ms"] == pytest.approx(0.22, abs=0.01)
        assert metrics["execution_time_ms"] == pytest.approx(0.45, abs=0.01)
        assert metrics["actual_rows"] == 22
        assert metrics["shared_hit_blocks"] == 4
        assert metrics["shared_read_blocks"] == 0
        assert metrics["index_name"] == "idx_orders_customer_id"
        assert metrics["scan_type"] == "Index Scan"

    def test_rejection_of_unregistered_template(self) -> None:
        with pytest.raises(ValueError, match="not registered"):
            get_query_execution_plan(template_key="drop_tables_hack")


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
class TestQueryAnalyzerIntegration:
    def test_get_query_execution_plan_live(self, live_conn: psycopg.Connection) -> None:
        plan = get_query_execution_plan(live_conn, template_key=DEFAULT_QUERY_KEY)
        assert isinstance(plan, dict)
        assert "Plan" in plan
        assert "Node Type" in plan["Plan"]

    def test_analyze_query_performance_live(self, live_conn: psycopg.Connection) -> None:
        result = analyze_query_performance(live_conn, customer_id=DEFAULT_CUSTOMER_ID)
        assert "metrics" in result
        assert "scan_analysis" in result
        assert result["metrics"]["execution_time_ms"] >= 0.0
        assert result["metrics"]["planning_time_ms"] >= 0.0
        assert result["scan_analysis"]["scanned_relation"] == "orders"

    def test_plan_retrieval_does_not_modify_database(self, live_conn: psycopg.Connection) -> None:
        with live_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM orders")
            count_before = cur.fetchone()[0]

        analyze_query_performance(live_conn)

        with live_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM orders")
            count_after = cur.fetchone()[0]

        assert count_before == count_after

