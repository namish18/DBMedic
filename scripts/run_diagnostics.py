"""
scripts/run_diagnostics.py
===========================
DBMedic Phase 2 Diagnostic Runner.

Executes end-to-end diagnostic investigation on dbmedic_lab:
  1. Verifies connectivity to dbmedic_lab.
  2. Runs query plan analysis on the slow-query incident.
  3. Collects table statistics on the orders table.
  4. Inspects index status on orders.customer_id.
  5. Invokes the diagnosis engine to generate a structured diagnostic report.
  6. Prints the formatted JSON report to stdout.
  7. Persists the report to benchmarks/slow_query/diagnostic_report.json.

Usage:
    python scripts/run_diagnostics.py
    python scripts/run_diagnostics.py --mock   # Run offline with mock incident observations
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg

from agent.core.diagnosis_engine import (
    calculate_diagnostic_score,
    create_diagnostic_report,
    diagnose_slow_query,
    generate_recommendation,
)
from agent.tools import get_lab_connection
from agent.tools.index_monitor import inspect_customer_id_index
from agent.tools.query_analyzer import (
    analyze_query_performance,
    detect_sequential_scan,
    extract_plan_metrics,
)
from agent.tools.table_monitor import get_table_statistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_diagnostics")

REPORT_FILE = PROJECT_ROOT / "benchmarks" / "slow_query" / "diagnostic_report.json"

MOCK_PLAN_INCIDENT = [
    {
        "Plan": {
            "Node Type": "Aggregate",
            "Strategy": "Plain",
            "Plan Rows": 1,
            "Actual Rows": 1,
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "orders",
                    "Plan Rows": 20,
                    "Actual Rows": 22,
                    "Filter": "(customer_id = 4242)",
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


def run_mock_diagnostics() -> dict:
    """Simulate the diagnostic pipeline using representative incident telemetry."""
    logger.info("[MOCK MODE] Simulating Phase 2 diagnostic investigation …")
    plan_metrics = extract_plan_metrics(MOCK_PLAN_INCIDENT)
    scan_analysis = detect_sequential_scan(MOCK_PLAN_INCIDENT)

    table_name = "orders"
    filter_column = "customer_id"
    table_stats = {
        "table_name": table_name,
        "estimated_row_count": 200_000,
        "table_size_bytes": 35_651_584,
        "table_size_pretty": "34 MB",
        "sequential_scan_count": 14,
        "sequential_tuples_read": 2_800_000,
        "index_scan_count": 0,
        "index_tuples_fetched": 0,
        "dead_tuple_estimate": 0,
        "live_tuple_estimate": 200_000,
    }
    index_info = {
        "table_name": table_name,
        "target_column": filter_column,
        "index_exists": False,
        "index_name": None,
        "index_definition": None,
        "is_suitable_for_filter": False,
        "index_scan_count": 0,
        "index_tuples_read": 0,
        "index_tuples_fetched": 0,
    }

    evidence = {
        "scan_type": "Sequential Scan",
        "execution_time_ms": plan_metrics["execution_time_ms"],
        "table_name": table_name,
        "filter_column": filter_column,
        "suitable_index_exists": False,
    }

    diagnostic_score = calculate_diagnostic_score(
        evidence, table_stats, row_selectivity=22 / 200_000
    )
    rec = generate_recommendation("missing_suitable_index", table_name, filter_column)

    return create_diagnostic_report(
        incident_id="slow_query_001",
        incident_type="slow_query",
        status="diagnosed",
        severity="high",
        summary="Query performance degradation detected",
        root_cause={
            "type": "missing_suitable_index",
            "description": f"Sequential scan detected on {table_name}",
        },
        evidence=evidence,
        recommendation=rec,
        diagnostic_score=diagnostic_score,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DBMedic Diagnostic Runner")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline diagnosis with synthetic incident telemetry (no live DB required)",
    )
    args = parser.parse_args()

    if args.mock:
        report = run_mock_diagnostics()
    else:
        logger.info("Connecting to dbmedic_lab for diagnostic investigation …")
        try:
            with get_lab_connection(read_only=True) as conn:
                logger.info("Phase 2 Diagnostic Investigation starting …")

                # 1. Query performance analysis
                logger.info("Step 1/4: Analyzing query execution plan & buffer metrics …")
                perf = analyze_query_performance(conn=conn)
                scan_type = perf["scan_analysis"]["primary_scan_type"]
                exec_time = perf["metrics"]["execution_time_ms"]
                logger.info(
                    "Execution plan: scan_type='%s', execution_time=%.3f ms",
                    scan_type,
                    exec_time,
                )

                # 2. Table statistics collection
                logger.info("Step 2/4: Collecting table statistics on 'orders' …")
                t_stats = get_table_statistics("orders", conn=conn)
                logger.info(
                    "Table stats: rows=%d, total_size=%s, seq_scans=%d",
                    t_stats["estimated_row_count"],
                    t_stats["table_size_pretty"],
                    t_stats["sequential_scan_count"],
                )

                # 3. Index inspection
                logger.info("Step 3/4: Inspecting index coverage on 'orders.customer_id' …")
                idx_stats = inspect_customer_id_index(conn=conn)
                logger.info(
                    "Index coverage: exists=%s, suitable=%s",
                    idx_stats["index_exists"],
                    idx_stats["is_suitable_for_filter"],
                )

                # 4. Synthesize diagnosis
                logger.info("Step 4/4: Generating structured diagnostic report …")
                report = diagnose_slow_query(conn=conn)

        except (psycopg.OperationalError, psycopg.errors.ConnectionTimeout, EnvironmentError) as exc:
            logger.error(
                "\n" + "!" * 70 + "\n"
                "DATABASE CONNECTION ERROR:\n"
                "Unable to connect to dbmedic_lab at localhost:5432.\n"
                "To start your PostgreSQL 17 Docker container, run:\n"
                "    docker compose up -d\n"
                "To test the diagnostic pipeline offline without a live DB, run:\n"
                "    python scripts/run_diagnostics.py --mock\n"
                + "!" * 70 + "\n"
            )
            sys.exit(1)
        except Exception as exc:
            logger.error("Diagnostic execution failed: %s", exc, exc_info=True)
            sys.exit(1)

    # Output formatted JSON report to stdout
    json_output = json.dumps(report, indent=2)
    print("\n" + "=" * 60)
    print("  DBMedic Diagnostic Report")
    print("=" * 60)
    print(json_output)
    print("=" * 60 + "\n")

    # Persist report
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json_output, encoding="utf-8")
    logger.info("Report successfully saved to %s", REPORT_FILE)


if __name__ == "__main__":
    main()
