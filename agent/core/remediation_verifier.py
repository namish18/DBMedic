"""
agent/core/remediation_verifier.py
==================================
Before/after PostgreSQL benchmark and execution-plan verifier for DBMedic.

Captures comparable pre- and post-remediation measurements:
  - Query execution times (per-run wall-clock & EXPLAIN ANALYZE execution_time_ms)
  - Median execution time
  - Planning time
  - Scan type
  - Actual rows and estimated rows
  - Shared buffer hits and shared buffer reads
  - Index existence and exact index definition

Verification rules:
  - Multiple identical iterations are run with the same customer_id parameter.
  - No planner hints (never sets ``enable_seqscan = off``).
  - Remediation is NEVER declared successful solely because an index exists.
  - Legitimate planner Sequential Scan choices (small table or low selectivity)
    are detected and explained rather than blindly flagged or mislabeled.
"""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.tools import get_lab_connection
from agent.tools.index_monitor import get_table_indexes
from agent.tools.query_analyzer import (
    DEFAULT_CUSTOMER_ID,
    DEFAULT_QUERY_KEY,
    detect_sequential_scan,
    extract_plan_metrics,
    get_query_execution_plan,
    index_exists,
)
from agent.tools.table_monitor import get_table_statistics
from database.scenarios.slow_query import (
    BENCHMARK_RUNS,
    INCIDENT_QUERY,
    compare_results,
    run_benchmark,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_COMPARISON_OUTPUT_PATH = (
    PROJECT_ROOT / "benchmarks" / "final_demo" / "remediation_comparison.json"
)

SMALL_TABLE_THRESHOLD = 1_000
LOW_SELECTIVITY_THRESHOLD = 0.15


def _capture_live_metrics(
    conn: Any = None,
    *,
    customer_id: int = DEFAULT_CUSTOMER_ID,
    n_runs: int = BENCHMARK_RUNS,
    phase_label: str = "baseline",
    remediation_status: str = "not_applied",
) -> dict[str, Any]:
    """Capture live benchmark + EXPLAIN (ANALYZE, BUFFERS) + index + table metadata."""
    with get_lab_connection(conn, read_only=True) as db_conn:
        bench = run_benchmark(
            db_conn,
            customer_id=customer_id,
            n_runs=n_runs,
            remediation_status=remediation_status,
        )
        plan_res = get_query_execution_plan(
            db_conn,
            template_key=DEFAULT_QUERY_KEY,
            params={"customer_id": customer_id},
            analyze=True,
            buffers=True,
        )
        raw_plan = plan_res.get("raw_plan", [])
        plan_metrics = extract_plan_metrics(raw_plan)
        scan_analysis = detect_sequential_scan(raw_plan)

        idx_present = bool(index_exists(db_conn, "idx_orders_customer_id"))
        idx_def: str | None = None
        if idx_present:
            for item in get_table_indexes("orders", conn=db_conn):
                if item.get("index_name") == "idx_orders_customer_id":
                    idx_def = item.get("index_definition")
                    break

        table_stats = get_table_statistics("orders", conn=db_conn)

    # Prefer scan-node estimated/actual rows if root node is plain Aggregate COUNT(*)
    all_scans = scan_analysis.get("all_scans", [])
    scan_node = all_scans[0] if all_scans else {}
    estimated_rows = int(scan_node.get("plan_rows") or plan_metrics.get("estimated_rows", 0))
    actual_rows = int(
        bench.runs[0]["rows_returned"]
        if bench.runs
        else (scan_node.get("actual_rows") or plan_metrics.get("actual_rows", 0))
    )

    timings = [float(r["wall_clock_ms"]) for r in bench.runs] if bench.runs else []
    mean_ms = statistics.mean(timings) if timings else float(bench.median_execution_time_ms)

    return {
        "phase": phase_label,
        "simulated": False,
        "execution_mode": "live_postgres",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "query": INCIDENT_QUERY,
        "customer_id": customer_id,
        "n_runs": n_runs,
        "runs": bench.runs,
        "execution_time_ms": round(float(plan_metrics.get("execution_time_ms", bench.median_execution_time_ms)), 3),
        "median_execution_time_ms": round(float(bench.median_execution_time_ms), 3),
        "mean_execution_time_ms": round(float(mean_ms), 3),
        "min_execution_time_ms": round(min(timings), 3) if timings else 0.0,
        "max_execution_time_ms": round(max(timings), 3) if timings else 0.0,
        "planning_time_ms": round(float(plan_metrics.get("planning_time_ms", 0.0)), 3),
        "scan_type": scan_analysis.get("primary_scan_type") or bench.scan_type,
        "raw_scan_type": scan_analysis.get("raw_node_type") or bench.scan_type,
        "has_sequential_scan": bool(scan_analysis.get("has_sequential_scan", False)),
        "actual_rows": actual_rows,
        "estimated_rows": estimated_rows,
        "shared_hit_blocks": int(plan_metrics.get("shared_hit_blocks") or bench.shared_hit_blocks),
        "shared_read_blocks": int(plan_metrics.get("shared_read_blocks") or bench.shared_read_blocks),
        "index_name": "idx_orders_customer_id",
        "index_exists": idx_present,
        "index_definition": idx_def,
        "table_row_count": int(table_stats.get("estimated_row_count", 0)),
        "raw_plan": raw_plan,
    }


def _build_simulated_metrics(
    *,
    phase_label: str,
    customer_id: int,
    n_runs: int,
    index_present: bool,
) -> dict[str, Any]:
    """Build explicitly tagged simulated benchmark metrics for offline demos/tests only."""
    if not index_present:
        timings = [26.4, 24.8, 24.1, 24.5, 23.9, 25.1, 24.3][:n_runs]
        median_ms = round(statistics.median(timings), 3)
        return {
            "phase": phase_label,
            "simulated": True,
            "execution_mode": "simulated",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "query": INCIDENT_QUERY,
            "customer_id": customer_id,
            "n_runs": len(timings),
            "runs": [{"run": i + 1, "wall_clock_ms": t, "rows_returned": 20} for i, t in enumerate(timings)],
            "execution_time_ms": median_ms,
            "median_execution_time_ms": median_ms,
            "mean_execution_time_ms": round(statistics.mean(timings), 3),
            "min_execution_time_ms": min(timings),
            "max_execution_time_ms": max(timings),
            "planning_time_ms": 0.16,
            "scan_type": "Sequential Scan",
            "raw_scan_type": "Seq Scan",
            "has_sequential_scan": True,
            "actual_rows": 20,
            "estimated_rows": 20,
            "shared_hit_blocks": 1250,
            "shared_read_blocks": 45,
            "index_name": "idx_orders_customer_id",
            "index_exists": False,
            "index_definition": None,
            "table_row_count": 200_000,
        }

    timings = [1.35, 1.18, 1.21, 1.16, 1.15, 1.24, 1.19][:n_runs]
    median_ms = round(statistics.median(timings), 3)
    return {
        "phase": phase_label,
        "simulated": True,
        "execution_mode": "simulated",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "query": INCIDENT_QUERY,
        "customer_id": customer_id,
        "n_runs": len(timings),
        "runs": [{"run": i + 1, "wall_clock_ms": t, "rows_returned": 20} for i, t in enumerate(timings)],
        "execution_time_ms": median_ms,
        "median_execution_time_ms": median_ms,
        "mean_execution_time_ms": round(statistics.mean(timings), 3),
        "min_execution_time_ms": min(timings),
        "max_execution_time_ms": max(timings),
        "planning_time_ms": 0.19,
        "scan_type": "Index Only Scan",
        "raw_scan_type": "Index Only Scan",
        "has_sequential_scan": False,
        "actual_rows": 20,
        "estimated_rows": 20,
        "shared_hit_blocks": 45,
        "shared_read_blocks": 1,
        "index_name": "idx_orders_customer_id",
        "index_exists": True,
        "index_definition": "CREATE INDEX idx_orders_customer_id ON public.orders USING btree (customer_id)",
        "table_row_count": 200_000,
    }


def capture_pre_remediation_metrics(
    conn: Any = None,
    *,
    customer_id: int = DEFAULT_CUSTOMER_ID,
    n_runs: int = BENCHMARK_RUNS,
    simulated: bool = False,
    simulated_index_present: bool = False,
) -> dict[str, Any]:
    """Capture comparable baseline query execution, plan, and index state before remediation."""
    if simulated:
        return _build_simulated_metrics(
            phase_label="pre_remediation",
            customer_id=customer_id,
            n_runs=n_runs,
            index_present=simulated_index_present,
        )
    return _capture_live_metrics(
        conn,
        customer_id=customer_id,
        n_runs=n_runs,
        phase_label="pre_remediation",
        remediation_status="not_applied",
    )


def capture_post_remediation_metrics(
    conn: Any = None,
    *,
    customer_id: int = DEFAULT_CUSTOMER_ID,
    n_runs: int = BENCHMARK_RUNS,
    simulated: bool = False,
) -> dict[str, Any]:
    """Capture post-remediation query execution, plan, and index state after remediation."""
    if simulated:
        return _build_simulated_metrics(
            phase_label="post_remediation",
            customer_id=customer_id,
            n_runs=n_runs,
            index_present=True,
        )
    return _capture_live_metrics(
        conn,
        customer_id=customer_id,
        n_runs=n_runs,
        phase_label="post_remediation",
        remediation_status="applied",
    )


def verify_index_created(
    conn: Any = None,
    *,
    table_name: str = "orders",
    index_name: str = "idx_orders_customer_id",
    expected_column: str = "customer_id",
    simulated: bool = False,
) -> dict[str, Any]:
    """Inspect PostgreSQL catalogs to verify expected index exists with right column."""
    if simulated:
        return {
            "index_name": index_name,
            "table_name": table_name,
            "expected_column": expected_column,
            "index_exists": True,
            "covers_expected_column": True,
            "indexed_columns": [expected_column],
            "index_type": "btree",
            "index_definition": f"CREATE INDEX {index_name} ON public.{table_name} USING btree ({expected_column})",
            "simulated": True,
        }

    with get_lab_connection(conn, read_only=True) as db_conn:
        exists = bool(index_exists(db_conn, index_name))
        matched: dict[str, Any] | None = None
        if exists:
            for idx in get_table_indexes(table_name, conn=db_conn):
                if idx.get("index_name") == index_name:
                    matched = idx
                    break

    cols = list(matched.get("indexed_columns", [])) if matched else []
    covers_col = expected_column in cols
    return {
        "index_name": index_name,
        "table_name": table_name,
        "expected_column": expected_column,
        "index_exists": exists,
        "covers_expected_column": covers_col,
        "indexed_columns": cols,
        "index_type": matched.get("index_type") if matched else None,
        "index_definition": matched.get("index_definition") if matched else None,
        "simulated": False,
    }


def compare_execution_plans(
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compare execution plan characteristics before and after remediation."""
    before_scan = str(before_metrics.get("scan_type", "unknown"))
    after_scan = str(after_metrics.get("scan_type", "unknown"))

    before_blocks = int(before_metrics.get("shared_hit_blocks", 0)) + int(
        before_metrics.get("shared_read_blocks", 0)
    )
    after_blocks = int(after_metrics.get("shared_hit_blocks", 0)) + int(
        after_metrics.get("shared_read_blocks", 0)
    )
    block_reduction = before_blocks - after_blocks
    block_reduction_pct = (
        round((block_reduction / before_blocks) * 100.0, 2) if before_blocks > 0 else 0.0
    )

    is_index_backed_after = any(
        tok in after_scan.lower() for tok in ("index scan", "index only scan", "bitmap index")
    )
    switched_from_seq_to_index = ("seq" in before_scan.lower()) and is_index_backed_after

    comparable_workload = (
        before_metrics.get("query") == after_metrics.get("query")
        and before_metrics.get("customer_id") == after_metrics.get("customer_id")
    )

    return {
        "before_scan_type": before_scan,
        "after_scan_type": after_scan,
        "before_has_sequential_scan": bool(before_metrics.get("has_sequential_scan", "seq" in before_scan.lower())),
        "after_has_sequential_scan": bool(after_metrics.get("has_sequential_scan", "seq" in after_scan.lower())),
        "switched_from_seq_to_index": switched_from_seq_to_index,
        "uses_index_scan_after": is_index_backed_after,
        "before_planning_time_ms": before_metrics.get("planning_time_ms"),
        "after_planning_time_ms": after_metrics.get("planning_time_ms"),
        "before_total_buffer_blocks": before_blocks,
        "after_total_buffer_blocks": after_blocks,
        "buffer_block_reduction": block_reduction,
        "buffer_block_reduction_pct": block_reduction_pct,
        "before_actual_rows": before_metrics.get("actual_rows"),
        "after_actual_rows": after_metrics.get("actual_rows"),
        "comparable_workload": comparable_workload,
    }


def calculate_performance_improvement(
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Calculate execution-time improvement building on Phase 1 ``compare_results``."""
    core_cmp = compare_results(before_metrics, after_metrics)
    before_ms = float(before_metrics["median_execution_time_ms"])
    after_ms = float(after_metrics["median_execution_time_ms"])
    speedup_factor = round(before_ms / after_ms, 2) if after_ms > 0 else None

    return {
        **core_cmp,
        "before_mean_ms": before_metrics.get("mean_execution_time_ms"),
        "after_mean_ms": after_metrics.get("mean_execution_time_ms"),
        "speedup_factor": speedup_factor,
        "buffer_hits_before": before_metrics.get("shared_hit_blocks"),
        "buffer_hits_after": after_metrics.get("shared_hit_blocks"),
        "buffer_reads_before": before_metrics.get("shared_read_blocks"),
        "buffer_reads_after": after_metrics.get("shared_read_blocks"),
    }


def determine_remediation_success(
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
    index_verification: dict[str, Any],
    plan_comparison: dict[str, Any],
    performance_comparison: dict[str, Any],
) -> dict[str, Any]:
    """
    Determine genuine remediation outcome from BOTH execution plan & measured speed.

    Never marks success solely because the index exists, and explains legitimate
    planner sequential scan choices when selectivity or table size warrants it.
    """
    index_ok = bool(
        index_verification.get("index_exists")
        and index_verification.get("covers_expected_column", True)
    )
    uses_index = bool(plan_comparison.get("uses_index_scan_after"))
    perf_improved = bool(performance_comparison.get("performance_improved"))
    improvement_pct = float(performance_comparison.get("improvement_pct", 0.0))
    comparable = bool(plan_comparison.get("comparable_workload", True))

    if not comparable:
        return {
            "success": False,
            "verdict": "INCOMPARABLE_WORKLOAD",
            "index_created": index_ok,
            "plan_improved": False,
            "performance_improved": False,
            "planner_choice_explanation": "Before and after workloads did not share identical query/customer_id parameters.",
            "reason": "Cannot verify remediation because before/after benchmark workloads differ.",
        }

    if not index_ok:
        return {
            "success": False,
            "verdict": "FAILED_INDEX_MISSING",
            "index_created": False,
            "plan_improved": uses_index,
            "performance_improved": perf_improved,
            "planner_choice_explanation": "Expected index is absent or does not cover customer_id.",
            "reason": "Remediation index idx_orders_customer_id was not verified in database catalogs.",
        }

    # Index exists — inspect whether planner actually chose it and latency improved
    if not uses_index:
        table_rows = int(after_metrics.get("table_row_count") or before_metrics.get("table_row_count") or 0)
        actual_rows = int(after_metrics.get("actual_rows") or 0)
        selectivity = (actual_rows / table_rows) if table_rows > 0 else 0.0

        if 0 < table_rows < SMALL_TABLE_THRESHOLD:
            explanation = (
                f"Index exists, but PostgreSQL legitimately chose '{plan_comparison['after_scan_type']}' "
                f"because table has only ~{table_rows} rows (< {SMALL_TABLE_THRESHOLD}); "
                "scanning few heap pages directly costs less than index traversal."
            )
            verdict = "PLANNER_LEGITIMATE_SEQ_SCAN_SMALL_TABLE"
            legitimate = True
        elif selectivity >= LOW_SELECTIVITY_THRESHOLD:
            explanation = (
                f"Index exists, but PostgreSQL legitimately chose '{plan_comparison['after_scan_type']}' "
                f"because filter matched {actual_rows}/{table_rows} rows ({selectivity:.1%} selectivity); "
                "high fraction fetches make sequential I/O cheaper than random index lookups."
            )
            verdict = "PLANNER_LEGITIMATE_SEQ_SCAN_LOW_SELECTIVITY"
            legitimate = True
        else:
            explanation = (
                f"Index exists for a selective predicate ({actual_rows}/{max(table_rows, 1)} rows), "
                f"yet planner still selected '{plan_comparison['after_scan_type']}'. "
                "Run ANALYZE orders or inspect column types/operator classes."
            )
            verdict = "REMEDIATION_PLAN_UNCHANGED"
            legitimate = False

        return {
            "success": False,
            "verdict": verdict,
            "index_created": True,
            "plan_improved": False,
            "performance_improved": perf_improved,
            "planner_choice_legitimate": legitimate,
            "planner_choice_explanation": explanation,
            "reason": explanation,
        }

    if not perf_improved:
        explanation = (
            f"Index exists and planner switched to '{plan_comparison['after_scan_type']}', "
            f"but median latency did not decrease ({performance_comparison['baseline_median_ms']} ms -> "
            f"{performance_comparison['after_remediation_median_ms']} ms)."
        )
        return {
            "success": False,
            "verdict": "INDEX_USED_NO_LATENCY_GAIN",
            "index_created": True,
            "plan_improved": True,
            "performance_improved": False,
            "planner_choice_legitimate": True,
            "planner_choice_explanation": explanation,
            "reason": explanation,
        }

    explanation = (
        f"PostgreSQL planner switched from '{plan_comparison['before_scan_type']}' to "
        f"'{plan_comparison['after_scan_type']}' after index creation, reducing median latency from "
        f"{performance_comparison['baseline_median_ms']} ms to "
        f"{performance_comparison['after_remediation_median_ms']} ms "
        f"({improvement_pct:.2f}% faster)."
    )
    return {
        "success": True,
        "verdict": "VERIFIED_IMPROVED",
        "index_created": True,
        "plan_improved": True,
        "performance_improved": True,
        "planner_choice_legitimate": True,
        "planner_choice_explanation": explanation,
        "reason": explanation,
    }


def verify_and_save_comparison(
    *,
    incident_id: str,
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
    index_verification: dict[str, Any],
    execution_mode: str = "deterministic",
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Assemble full before/after verification report and save to JSON."""
    plan_cmp = compare_execution_plans(before_metrics, after_metrics)
    perf_cmp = calculate_performance_improvement(before_metrics, after_metrics)
    outcome = determine_remediation_success(
        before_metrics,
        after_metrics,
        index_verification,
        plan_cmp,
        perf_cmp,
    )

    is_sim = bool(
        before_metrics.get("simulated")
        or after_metrics.get("simulated")
        or execution_mode == "simulated"
    )

    report = {
        "schema_version": "dbmedic.remediation_comparison/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "incident_id": incident_id,
        "execution_mode": "simulated" if is_sim else execution_mode,
        "simulated": is_sim,
        "before": before_metrics,
        "after": after_metrics,
        "index_verification": index_verification,
        "plan_comparison": plan_cmp,
        "performance_comparison": perf_cmp,
        "verification_outcome": outcome,
    }

    dest = Path(output_path) if output_path is not None else DEFAULT_COMPARISON_OUTPUT_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Strip raw_plan noise when persisting summary if very large, keeping core fields intact
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["output_path"] = str(dest)
    return report
