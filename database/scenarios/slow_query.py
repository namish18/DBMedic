"""
database/scenarios/slow_query.py
=================================
Slow-query incident definition and benchmark runner for DBMedic.

Incident: orders.customer_id has no secondary index, causing a full sequential
scan when querying a single customer's orders.

Design rules:
  - Diagnostic operations are strictly read-only.
  - Remediation is separated from diagnosis and requires explicit caller approval.
  - No planner hints (enable_seqscan = off) are used to force plans artificially.
  - Credentials are never logged.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INCIDENT_QUERY = "SELECT COUNT(*) FROM orders WHERE customer_id = %(customer_id)s"
TARGET_CUSTOMER_ID = 4242

BENCHMARK_RUNS = 7          # Odd number → clean median
BENCHMARK_DIR = Path("benchmarks/slow_query")

BASELINE_FILE = BENCHMARK_DIR / "baseline.json"
REMEDIATION_FILE = BENCHMARK_DIR / "after_remediation.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Captures one complete benchmark session (multiple runs)."""

    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    customer_id: int = TARGET_CUSTOMER_ID
    runs: list[dict[str, Any]] = field(default_factory=list)
    median_execution_time_ms: float = 0.0
    scan_type: str = "unknown"
    rows_returned: int = 0
    shared_hit_blocks: int = 0
    shared_read_blocks: int = 0
    remediation_status: str = "not_applied"
    performance_improvement_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "query": self.query,
            "customer_id": self.customer_id,
            "runs": self.runs,
            "median_execution_time_ms": round(self.median_execution_time_ms, 3),
            "scan_type": self.scan_type,
            "rows_returned": self.rows_returned,
            "shared_hit_blocks": self.shared_hit_blocks,
            "shared_read_blocks": self.shared_read_blocks,
            "remediation_status": self.remediation_status,
            "performance_improvement_pct": (
                round(self.performance_improvement_pct, 2)
                if self.performance_improvement_pct is not None
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_plan_info(plan_json: list[dict]) -> dict[str, Any]:
    """
    Pull relevant fields from an EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) result.

    Returns a dict with:
      - execution_time_ms
      - scan_type   ('Seq Scan' | 'Index Scan' | 'Index Only Scan' | …)
      - rows_returned
      - shared_hit_blocks
      - shared_read_blocks
    """
    top = plan_json[0]
    plan_node = top["Plan"]

    execution_time_ms: float = top.get("Execution Time", 0.0)

    # Walk the plan tree to find the first scan node on the orders table
    scan_type = _find_scan_type(plan_node)

    rows_returned: int = plan_node.get("Plan Rows", 0)
    shared_hit: int = top.get("Planning", {}).get("Shared Hit Blocks", 0)
    shared_read: int = top.get("Planning", {}).get("Shared Read Blocks", 0)

    # Actual totals live at the top-level too
    if "Shared Hit Blocks" in top:
        shared_hit = top["Shared Hit Blocks"]
        shared_read = top["Shared Read Blocks"]

    return {
        "execution_time_ms": execution_time_ms,
        "scan_type": scan_type,
        "rows_returned": rows_returned,
        "shared_hit_blocks": shared_hit,
        "shared_read_blocks": shared_read,
    }


def _find_scan_type(node: dict) -> str:
    """Recursively locate the first scan node in the plan tree."""
    node_type: str = node.get("Node Type", "")
    if "Scan" in node_type:
        return node_type
    for child in node.get("Plans", []):
        result = _find_scan_type(child)
        if result != "unknown":
            return result
    return "unknown"


def _wall_clock_query(
    conn: psycopg.Connection,
    customer_id: int,
) -> tuple[float, int]:
    """Execute the target query and return (wall_clock_ms, row_count)."""
    start = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(INCIDENT_QUERY, {"customer_id": customer_id})
        row_count: int = cur.fetchone()[0]  # type: ignore[index]
    elapsed_ms = (time.perf_counter() - start) * 1_000
    return elapsed_ms, row_count


def _explain_query(
    conn: psycopg.Connection,
    customer_id: int,
) -> list[dict]:
    """
    Run EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) for the target query.
    Returns the parsed JSON plan list.
    """
    explain_sql = (
        "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
        + INCIDENT_QUERY
    )
    with conn.cursor() as cur:
        cur.execute(explain_sql, {"customer_id": customer_id})
        rows = cur.fetchone()
    if rows is None:
        raise RuntimeError("EXPLAIN returned no rows")
    return rows[0]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_benchmark(
    conn: psycopg.Connection,
    *,
    customer_id: int = TARGET_CUSTOMER_ID,
    n_runs: int = BENCHMARK_RUNS,
    remediation_status: str = "not_applied",
) -> BenchmarkResult:
    """
    Execute the slow-query benchmark.

    Runs the target query *n_runs* times, records each execution plan and
    wall-clock time, then aggregates into a BenchmarkResult.

    Parameters
    ----------
    conn:
        An open psycopg connection to dbmedic_lab.
    customer_id:
        The customer whose orders are queried.  Defaults to TARGET_CUSTOMER_ID.
    n_runs:
        Number of measurement iterations (default 7 for a clean median).
    remediation_status:
        Label stored in the result JSON ("not_applied" | "applied").

    Returns
    -------
    BenchmarkResult populated with per-run timings, scan type, and buffer stats.
    """
    result = BenchmarkResult(
        query=INCIDENT_QUERY,
        customer_id=customer_id,
        remediation_status=remediation_status,
    )

    logger.info(
        "Starting benchmark: %d runs, customer_id=%d", n_runs, customer_id
    )

    timings: list[float] = []
    plan_info: dict[str, Any] = {}

    for run_index in range(n_runs):
        # Obtain the execution plan on the first run only (plan is stable)
        if run_index == 0:
            plan_json = _explain_query(conn, customer_id)
            plan_info = _extract_plan_info(plan_json)
            logger.info("Scan type detected: %s", plan_info["scan_type"])

        wall_ms, rows = _wall_clock_query(conn, customer_id)
        timings.append(wall_ms)

        result.runs.append(
            {
                "run": run_index + 1,
                "wall_clock_ms": round(wall_ms, 3),
                "rows_returned": rows,
            }
        )
        logger.debug("Run %d: %.3f ms", run_index + 1, wall_ms)

    result.median_execution_time_ms = statistics.median(timings)
    result.scan_type = plan_info.get("scan_type", "unknown")
    result.rows_returned = plan_info.get("rows_returned", 0)
    result.shared_hit_blocks = plan_info.get("shared_hit_blocks", 0)
    result.shared_read_blocks = plan_info.get("shared_read_blocks", 0)

    logger.info(
        "Benchmark complete — median: %.3f ms, scan: %s",
        result.median_execution_time_ms,
        result.scan_type,
    )
    return result


def save_result(result: BenchmarkResult, path: Path) -> None:
    """Serialise a BenchmarkResult to JSON on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    logger.info("Benchmark result written to %s", path)


def load_result(path: Path) -> dict[str, Any]:
    """Load a previously saved benchmark result from disk."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def compare_results(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare baseline and post-remediation benchmark results.

    Returns a comparison dict containing absolute and percentage improvement.
    """
    before_ms: float = before["median_execution_time_ms"]
    after_ms: float = after["median_execution_time_ms"]

    if before_ms > 0:
        improvement_pct = ((before_ms - after_ms) / before_ms) * 100
    else:
        improvement_pct = 0.0

    improved = after_ms < before_ms

    return {
        "baseline_median_ms": before_ms,
        "after_remediation_median_ms": after_ms,
        "improvement_ms": round(before_ms - after_ms, 3),
        "improvement_pct": round(improvement_pct, 2),
        "baseline_scan": before.get("scan_type"),
        "after_scan": after.get("scan_type"),
        "performance_improved": improved,
    }

