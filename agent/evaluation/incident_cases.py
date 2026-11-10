"""
agent/evaluation/incident_cases.py
====================================
Reproducible slow-query incident evaluation dataset.

Defines at least 10 reproducible slow-query variations covering:
  - Different customer IDs (varying rows returned)
  - Different query selectivity (high vs moderate)
  - Existing vs missing indexes (incident vs post-remediation states)
  - Different query wordings (paraphrase robustness)

Split into:
  - BASELINE_EVAL_CASES: Run for baseline evaluation (ground truth known)
  - TRAINING_CANDIDATES: Candidates for AgentTune training data
  - HIDDEN_EVAL_CASES: Reserved; expected answers NOT exposed to the agent

Ground truth uses deterministic PostgreSQL observations, not model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IncidentCase:
    """One reproducible evaluation incident."""
    incident_id: str
    query: str
    description: str
    # Ground truth — not passed to the model
    expected_root_cause: str
    expected_tools_to_call: list[str]
    expected_diagnosis_keywords: list[str]
    # Optional parameters for incident setup
    customer_id: int = 4242
    split: str = "baseline"       # baseline | training | hidden
    requires_index_absent: bool = True  # True = incident state (no index on customer_id)
    notes: str = ""


# ---------------------------------------------------------------------------
# Ground truth for the standard slow-query incident
# ---------------------------------------------------------------------------

# Deterministic PostgreSQL observations used as evaluation ground truth:
#   - Sequential scan on orders (from EXPLAIN output)
#   - No idx_orders_customer_id index exists (from pg_indexes)
#   - ~200,000 rows in orders table
#   - customer_id filter is high-selectivity (matches ~20 rows / 200k = 0.01%)

STANDARD_ROOT_CAUSE = "missing_suitable_index"
STANDARD_EVIDENCE_KEYWORDS = [
    "sequential scan",
    "seq scan",
    "customer_id",
    "orders",
    "index",
]

# ---------------------------------------------------------------------------
# Baseline Evaluation Cases (10 cases)
# ---------------------------------------------------------------------------

BASELINE_EVAL_CASES: list[IncidentCase] = [

    # --- Case 1: Standard incident, canonical question ---
    IncidentCase(
        incident_id="slow_query_001",
        query="Why is my orders query taking so long?",
        description="Standard slow-query incident — canonical diagnostic question",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=[
            "get_query_execution_plan",
            "analyze_query_performance",
            "inspect_customer_id_index",
        ],
        expected_diagnosis_keywords=STANDARD_EVIDENCE_KEYWORDS,
        customer_id=4242,
        split="baseline",
        requires_index_absent=True,
    ),

    # --- Case 2: Different customer ID, same question ---
    IncidentCase(
        incident_id="slow_query_002",
        query="The query for customer 1234 on the orders table is very slow. Diagnose it.",
        description="Slow-query with different customer ID, same underlying cause",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=[
            "analyze_query_performance",
            "inspect_customer_id_index",
        ],
        expected_diagnosis_keywords=STANDARD_EVIDENCE_KEYWORDS,
        customer_id=1234,
        split="baseline",
        requires_index_absent=True,
    ),

    # --- Case 3: Direct sequential scan question ---
    IncidentCase(
        incident_id="slow_query_003",
        query="Why does querying orders by customer_id perform a full table scan instead of using an index?",
        description="Direct question about sequential scan on orders.customer_id",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=[
            "get_query_execution_plan",
            "get_table_indexes",
        ],
        expected_diagnosis_keywords=["sequential scan", "index", "customer_id"],
        customer_id=4242,
        split="baseline",
        requires_index_absent=True,
    ),

    # --- Case 4: Index inspection focus ---
    IncidentCase(
        incident_id="slow_query_004",
        query="Does the orders table have an index on customer_id? Is that why queries are slow?",
        description="Index-focused question probing inspect_customer_id_index tool selection",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=[
            "inspect_customer_id_index",
            "get_table_indexes",
        ],
        expected_diagnosis_keywords=["index", "customer_id", "orders"],
        customer_id=4242,
        split="baseline",
        requires_index_absent=True,
    ),

    # --- Case 5: Table size / volume framing ---
    IncidentCase(
        incident_id="slow_query_005",
        query="How large is the orders table and could its size be causing query performance issues?",
        description="Table-size focused question testing table monitor tool selection",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=[
            "get_table_statistics",
            "get_table_size",
            "analyze_query_performance",
        ],
        expected_diagnosis_keywords=["sequential scan", "orders", "row"],
        customer_id=4242,
        split="baseline",
        requires_index_absent=True,
    ),

    # --- Case 6: Explicit performance measurement request ---
    IncidentCase(
        incident_id="slow_query_006",
        query="Measure the execution time and execution plan for the slow orders query on customer 999.",
        description="Performance measurement request using analyze_query_performance",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=[
            "analyze_query_performance",
            "get_query_execution_plan",
        ],
        expected_diagnosis_keywords=["execution time", "sequential scan", "orders"],
        customer_id=999,
        split="baseline",
        requires_index_absent=True,
    ),

    # --- Case 7: Full diagnosis request ---
    IncidentCase(
        incident_id="slow_query_007",
        query="Run a full diagnosis on the slow query incident for the orders table.",
        description="Explicit full diagnosis request — tests diagnose_slow_query tool selection",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=["diagnose_slow_query"],
        expected_diagnosis_keywords=STANDARD_EVIDENCE_KEYWORDS,
        customer_id=4242,
        split="baseline",
        requires_index_absent=True,
    ),

    # --- Case 8: Index usage statistics focus ---
    IncidentCase(
        incident_id="slow_query_008",
        query="Are the indexes on the orders table being used? Why might queries be scanning all rows?",
        description="Index usage statistics question",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=[
            "get_index_usage",
            "get_table_indexes",
            "inspect_customer_id_index",
        ],
        expected_diagnosis_keywords=["index", "scan", "orders", "customer_id"],
        customer_id=4242,
        split="baseline",
        requires_index_absent=True,
    ),

    # --- Case 9: High-selectivity framing ---
    IncidentCase(
        incident_id="slow_query_009",
        query="A query filtering orders by a specific customer ID reads thousands of rows unnecessarily. What is wrong?",
        description="High-selectivity framing — 200k rows read for single customer match",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=[
            "analyze_query_performance",
            "inspect_customer_id_index",
        ],
        expected_diagnosis_keywords=["sequential scan", "index", "customer_id"],
        customer_id=4242,
        split="baseline",
        requires_index_absent=True,
    ),

    # --- Case 10: Broad performance review question ---
    IncidentCase(
        incident_id="slow_query_010",
        query="Review the performance of the orders table and identify any structural issues causing slow queries.",
        description="Broad performance review — tests multi-tool orchestration",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=[
            "get_table_statistics",
            "get_table_indexes",
            "analyze_query_performance",
        ],
        expected_diagnosis_keywords=STANDARD_EVIDENCE_KEYWORDS,
        customer_id=4242,
        split="baseline",
        requires_index_absent=True,
    ),
]

# ---------------------------------------------------------------------------
# Training Candidates (not used for evaluation; for AgentTune later)
# ---------------------------------------------------------------------------

TRAINING_CANDIDATES: list[IncidentCase] = [
    IncidentCase(
        incident_id="train_001",
        query="Investigate the slow query on the orders table for customer 7777.",
        description="Training variant — different customer ID",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=["analyze_query_performance", "inspect_customer_id_index"],
        expected_diagnosis_keywords=STANDARD_EVIDENCE_KEYWORDS,
        customer_id=7777,
        split="training",
    ),
    IncidentCase(
        incident_id="train_002",
        query="What index would help the query SELECT COUNT(*) FROM orders WHERE customer_id = 100?",
        description="Training variant — explicit SQL context",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=["get_query_execution_plan", "get_table_indexes"],
        expected_diagnosis_keywords=["index", "customer_id", "orders"],
        customer_id=100,
        split="training",
    ),
]

# ---------------------------------------------------------------------------
# Hidden Evaluation Cases (expected answers NOT exposed to the model)
# Ground truth is known but NOT included in what the model sees.
# ---------------------------------------------------------------------------

HIDDEN_EVAL_CASES: list[IncidentCase] = [
    IncidentCase(
        incident_id="hidden_001",
        query="Something is wrong with orders queries. Find the root cause.",
        description="Hidden eval — vague symptom description",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=["diagnose_slow_query"],
        expected_diagnosis_keywords=STANDARD_EVIDENCE_KEYWORDS,
        customer_id=4242,
        split="hidden",
        notes="Expected answers hidden from agent; used for unbiased final evaluation.",
    ),
    IncidentCase(
        incident_id="hidden_002",
        query="Database is slow. What tool should I investigate first?",
        description="Hidden eval — extremely vague question testing tool selection",
        expected_root_cause=STANDARD_ROOT_CAUSE,
        expected_tools_to_call=["diagnose_slow_query", "analyze_query_performance"],
        expected_diagnosis_keywords=["orders", "scan"],
        customer_id=4242,
        split="hidden",
    ),
]

# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

ALL_CASES: list[IncidentCase] = (
    BASELINE_EVAL_CASES + TRAINING_CANDIDATES + HIDDEN_EVAL_CASES
)


def get_case_by_id(incident_id: str) -> IncidentCase | None:
    """Return an incident case by its ID."""
    for case in ALL_CASES:
        if case.incident_id == incident_id:
            return case
    return None


def get_baseline_cases() -> list[IncidentCase]:
    """Return the 10 baseline evaluation cases."""
    return [c for c in ALL_CASES if c.split == "baseline"]


def get_training_candidates() -> list[IncidentCase]:
    """Return training candidate cases."""
    return [c for c in ALL_CASES if c.split == "training"]

