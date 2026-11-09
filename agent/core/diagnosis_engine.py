"""
agent/core/diagnosis_engine.py
==============================
Deterministic PostgreSQL Root-Cause Diagnostic Engine for DBMedic.

Combines observational evidence from:
  - agent.tools.query_analyzer (execution plan, scan type, timings, buffer stats)
  - agent.tools.table_monitor  (table size, row estimates, historical scan counts)
  - agent.tools.index_monitor  (index existence, leading column definitions, usage)

Computes a deterministic root-cause diagnosis, weighted diagnostic score,
and remediation recommendation without calling any LLM.

Security constraints:
  - Diagnosis is strictly read-only.
  - Remediation is never executed automatically; recommendations require explicit approval.
  - Standardized JSON report adheres strictly to the DBMedic incident format.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg

from agent.tools import get_lab_connection
from agent.tools.index_monitor import inspect_customer_id_index
from agent.tools.query_analyzer import (
    DEFAULT_CUSTOMER_ID,
    DEFAULT_QUERY_KEY,
    analyze_query_performance,
)
from agent.tools.table_monitor import get_table_statistics

logger = logging.getLogger(__name__)

# Incident identifiers
INCIDENT_ID = "slow_query_001"
INCIDENT_TYPE = "slow_query"

# Diagnostic heuristics
MIN_ROWS_FOR_INDEX_ADVANTAGE = 1_000   # Sequential scans on small tables (<1,000 rows) are expected
DEFAULT_LATENCY_THRESHOLD_MS = 5.0     # Queries exceeding threshold with point filter warrant indexing
HIGH_SELECTIVITY_THRESHOLD = 0.05      # Filter matches < 5% of total table rows
POOR_SELECTIVITY_THRESHOLD = 0.15      # Filter matches > 15% of table (index scan unlikely to help)


# ---------------------------------------------------------------------------
# Diagnostic Scoring & Recommendation Functions
# ---------------------------------------------------------------------------

def calculate_diagnostic_score(
    evidence: dict[str, Any],
    table_stats: dict[str, Any],
    *,
    row_selectivity: float,
    latency_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
) -> float:
    """
    Calculate a deterministic heuristic diagnostic score (0.0 to 1.0).

    NOTE: This is a weighted heuristic score reflecting evidentiary strength,
    not a calibrated probability.

    Evidentiary Weights:
      + 0.30: Execution plan uses a Sequential Scan on the filtered table.
      + 0.30: No suitable index exists covering the filter column.
      + 0.25: High selectivity (<5% rows) on a non-trivial table (>1,000 rows).
      + 0.10: Observed execution latency exceeds benchmark threshold (default 5.0 ms).
      + 0.05: Historical sequential scan activity observed on the table.

    Penalties / Guardrails:
      - 0.30: Table has fewer than 1,000 rows (sequential scan is planner-optimal).
      - 0.20: Poor selectivity (>15% rows matched, where seq scan is often faster).
    """
    score = 0.0

    # 1. Scan type is Sequential Scan
    scan_type = evidence.get("scan_type", "")
    if scan_type in ("Sequential Scan", "Seq Scan"):
        score += 0.30

    # 2. Missing suitable index on filter column
    if not evidence.get("suitable_index_exists", False):
        score += 0.30

    # 3. Selectivity & Table Volume Guardrails
    row_count = table_stats.get("estimated_row_count", 0)
    if row_count < MIN_ROWS_FOR_INDEX_ADVANTAGE:
        # Small table: sequential scan is expected and efficient
        score -= 0.30
    elif row_selectivity <= HIGH_SELECTIVITY_THRESHOLD:
        # High selectivity on large table: prime candidate for index
        score += 0.25
    elif row_selectivity > POOR_SELECTIVITY_THRESHOLD:
        # Poor selectivity: full scan is often optimal
        score -= 0.20

    # 4. Latency threshold check
    exec_time = evidence.get("execution_time_ms", 0.0)
    if exec_time >= latency_threshold_ms:
        score += 0.10

    # 5. Historical sequential scan activity
    if table_stats.get("sequential_scan_count", 0) > 0:
        score += 0.05

    return round(max(0.0, min(1.0, score)), 2)


# Backwards compatibility alias
calculate_diagnosis_confidence = calculate_diagnostic_score


def generate_recommendation(
    root_cause_type: str,
    table_name: str,
    column_name: str,
    *,
    row_count: int = 10_000,
    selectivity: float = 0.01,
) -> dict[str, Any]:
    """
    Generate a structured remediation recommendation based on root cause.

    Guardrails:
      - Small tables (<1000 rows) do NOT receive index creation recommendations.
      - Poor selectivity queries (>15% rows) do NOT receive index creation recommendations.
      - Remediation is never executed automatically; requires_approval is always True
        for actionable index creation.
    """
    if root_cause_type == "missing_suitable_index":
        if row_count >= MIN_ROWS_FOR_INDEX_ADVANTAGE and selectivity <= POOR_SELECTIVITY_THRESHOLD:
            index_name = f"idx_{table_name}_{column_name}"
            sql = f"CREATE INDEX {index_name} ON {table_name}({column_name})"
            return {
                "action": "create_index",
                "sql": sql,
                "requires_approval": True,
            }

    return {
        "action": "none",
        "sql": None,
        "requires_approval": False,
    }


def create_diagnostic_report(
    incident_id: str,
    incident_type: str,
    status: str,
    severity: str,
    summary: str,
    root_cause: dict[str, Any],
    evidence: dict[str, Any],
    recommendation: dict[str, Any],
    diagnostic_score: float,
) -> dict[str, Any]:
    """
    Assemble the standardized DBMedic diagnostic report dictionary.

    Provides both 'diagnostic_score' (clarified heuristic metric) and
    'confidence' (backwards-compatible alias).
    """
    return {
        "incident_id": incident_id,
        "incident_type": incident_type,
        "status": status,
        "severity": severity,
        "summary": summary,
        "root_cause": root_cause,
        "evidence": evidence,
        "recommendation": recommendation,
        "diagnostic_score": diagnostic_score,
        "confidence": diagnostic_score,
    }


# ---------------------------------------------------------------------------
# Public Engine API
# ---------------------------------------------------------------------------

def diagnose_slow_query(
    conn: psycopg.Connection | None = None,
    *,
    customer_id: int = DEFAULT_CUSTOMER_ID,
    latency_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
) -> dict[str, Any]:
    """
    Investigate the slow query incident by correlating plan, table, and index metrics.

    Steps:
      1. Connects to dbmedic_lab (read-only default).
      2. Analyzes execution plan and execution metrics for target query.
      3. Inspects orders table statistics and volume.
      4. Inspects index status for customer_id on orders.
      5. Evaluates selectivity, scan type, and table size.
      6. Synthesizes evidence into a standardized JSON report.
    """
    logger.info("Initiating slow-query diagnosis for customer_id=%d", customer_id)

    with get_lab_connection(conn, read_only=True) as db_conn:
        # Step 1: Query analysis
        perf = analyze_query_performance(
            conn=db_conn,
            customer_id=customer_id,
        )
        plan_metrics = perf["metrics"]
        scan_analysis = perf["scan_analysis"]

        # Step 2: Table statistics
        table_name = "orders"
        filter_column = "customer_id"
        table_stats = get_table_statistics(table_name, conn=db_conn)

        # Step 3: Index inspection
        index_info = inspect_customer_id_index(conn=db_conn)

    # Step 4: Evaluate evidence
    scan_type = plan_metrics.get("scan_type", "unknown")
    execution_time_ms = plan_metrics.get("execution_time_ms", 0.0)
    has_seq_scan = scan_analysis.get("has_sequential_scan", False)
    suitable_index_exists = index_info.get("index_exists", False)

    # Selectivity computation
    total_rows = max(1, table_stats.get("estimated_row_count", 1))
    matched_rows = max(1, plan_metrics.get("actual_rows", 1))
    row_selectivity = matched_rows / total_rows

    evidence = {
        "scan_type": "Sequential Scan" if has_seq_scan else scan_type,
        "execution_time_ms": execution_time_ms,
        "table_name": table_name,
        "filter_column": filter_column,
        "suitable_index_exists": suitable_index_exists,
    }

    # Step 5: Root-cause identification
    if has_seq_scan and not suitable_index_exists:
        if total_rows >= MIN_ROWS_FOR_INDEX_ADVANTAGE:
            root_cause = {
                "type": "missing_suitable_index",
                "description": f"Sequential scan detected on {table_name}",
            }
            severity = "high"
            summary = "Query performance degradation detected"
            status = "diagnosed"
        else:
            root_cause = {
                "type": "small_table_scan",
                "description": f"Sequential scan chosen due to small table size ({total_rows} rows)",
            }
            severity = "low"
            summary = "Sequential scan is expected for small table volume"
            status = "diagnosed"
    elif suitable_index_exists:
        root_cause = {
            "type": "none_detected",
            "description": f"Suitable index exists on {table_name}({filter_column}); query utilizes index scan",
        }
        severity = "low"
        summary = "Query is operating within normal performance parameters"
        status = "optimized"
    else:
        root_cause = {
            "type": "suboptimal_plan",
            "description": f"Query did not use an index scan on {table_name}",
        }
        severity = "medium"
        summary = "Potential plan inefficiency detected"
        status = "diagnosed"

    # Step 6: Diagnostic Score & Recommendation
    score = calculate_diagnostic_score(
        evidence=evidence,
        table_stats=table_stats,
        row_selectivity=row_selectivity,
        latency_threshold_ms=latency_threshold_ms,
    )

    recommendation = generate_recommendation(
        root_cause_type=root_cause["type"],
        table_name=table_name,
        column_name=filter_column,
        row_count=total_rows,
        selectivity=row_selectivity,
    )

    report = create_diagnostic_report(
        incident_id=INCIDENT_ID,
        incident_type=INCIDENT_TYPE,
        status=status,
        severity=severity,
        summary=summary,
        root_cause=root_cause,
        evidence=evidence,
        recommendation=recommendation,
        diagnostic_score=score,
    )

    logger.info(
        "Diagnosis complete: root_cause=%s, diagnostic_score=%.2f",
        root_cause["type"],
        score,
    )
    return report
