"""
agent/evaluation/auditkit_scenarios.py
========================================
DBMedic evaluation dataset for AuditKIT — Phase 5.

Covers all scenario categories from the spec §5:
  A. Slow-query incidents (9 variants)
  B. Diagnostic correctness
  C. Abstention cases

Hidden test cases are stored separately and ground-truth answers
are NEVER passed to the evaluated model.

This module defines the scenario dataset only. Live trajectory execution
requires the full agent stack (HF_TOKEN + CUDA + PostgreSQL).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditScenario:
    """One AuditKIT evaluation scenario."""

    scenario_id: str
    category: str           # "slow_query" | "diagnostic_correctness" | "abstention"
    subcategory: str
    query: str
    description: str
    expected_root_cause: str
    expected_tools: list[str]
    expected_abstain: bool = False
    hidden: bool = False    # True = kept out of evaluated model's context
    notes: str = ""


# ---------------------------------------------------------------------------
# A. Slow-query incident scenarios
# ---------------------------------------------------------------------------

SLOW_QUERY_SCENARIOS: list[AuditScenario] = [
    AuditScenario(
        scenario_id="SQ-01",
        category="slow_query",
        subcategory="missing_customer_id_index",
        query="Orders for customer 4242 are loading very slowly. Diagnose the root cause.",
        description="Missing index on orders.customer_id — standard incident.",
        expected_root_cause="missing_suitable_index",
        expected_tools=["get_query_execution_plan", "inspect_customer_id_index", "get_table_indexes"],
    ),
    AuditScenario(
        scenario_id="SQ-02",
        category="slow_query",
        subcategory="existing_customer_id_index",
        query="Despite having an index on customer_id, orders queries for customer 9999 are still slow.",
        description="Index exists but query planner may not use it — suboptimal plan scenario.",
        expected_root_cause="suboptimal_plan",
        expected_tools=["get_query_execution_plan", "inspect_customer_id_index", "analyze_query_performance"],
    ),
    AuditScenario(
        scenario_id="SQ-03",
        category="slow_query",
        subcategory="selective_filter",
        query="Customers table lookup for a rare VIP customer segment is slow. Investigate.",
        description="Highly selective filter — index beneficial.",
        expected_root_cause="missing_suitable_index",
        expected_tools=["get_query_execution_plan", "get_table_indexes"],
    ),
    AuditScenario(
        scenario_id="SQ-04",
        category="slow_query",
        subcategory="non_selective_filter",
        query="Query fetching all orders with status='pending' is slow.",
        description="Non-selective filter — sequential scan may be optimal.",
        expected_root_cause="none_detected",
        expected_tools=["get_query_execution_plan", "get_table_stats"],
    ),
    AuditScenario(
        scenario_id="SQ-05",
        category="slow_query",
        subcategory="small_table",
        query="Why is the query on our tiny product_categories table (12 rows) slow?",
        description="Small table — seq scan is correct, no index needed.",
        expected_root_cause="small_table_scan",
        expected_tools=["get_table_stats", "get_query_execution_plan"],
    ),
    AuditScenario(
        scenario_id="SQ-06",
        category="slow_query",
        subcategory="large_table_missing_index",
        query="Orders table has 2.5M rows and customer lookups are scanning all rows.",
        description="Large table with missing index — high-impact incident.",
        expected_root_cause="missing_suitable_index",
        expected_tools=["get_query_execution_plan", "inspect_customer_id_index", "get_table_stats"],
    ),
    AuditScenario(
        scenario_id="SQ-07",
        category="slow_query",
        subcategory="sequential_scan_confirmed",
        query="EXPLAIN output shows Seq Scan on orders for customer_id filter. Is this a problem?",
        description="Sequential scan observed — model must interpret EXPLAIN output correctly.",
        expected_root_cause="missing_suitable_index",
        expected_tools=["get_query_execution_plan", "inspect_customer_id_index"],
    ),
    AuditScenario(
        scenario_id="SQ-08",
        category="slow_query",
        subcategory="index_scan_confirmed",
        query="EXPLAIN shows Index Scan on orders but query is still slow. Diagnose.",
        description="Index scan in use but other bottleneck present.",
        expected_root_cause="suboptimal_plan",
        expected_tools=["get_query_execution_plan", "analyze_query_performance", "get_table_stats"],
    ),
    AuditScenario(
        scenario_id="SQ-09",
        category="slow_query",
        subcategory="alternative_execution_plan",
        query="Query planner chose a hash join over nested loop. Is this correct for our data?",
        description="Alternative execution plan assessment.",
        expected_root_cause="suboptimal_plan",
        expected_tools=["get_query_execution_plan", "analyze_query_performance"],
    ),
]

# ---------------------------------------------------------------------------
# B. Diagnostic correctness scenarios
# ---------------------------------------------------------------------------

DIAGNOSTIC_CORRECTNESS_SCENARIOS: list[AuditScenario] = [
    AuditScenario(
        scenario_id="DC-01",
        category="diagnostic_correctness",
        subcategory="correct_root_cause",
        query="Slow query on orders.customer_id=4242. Identify the root cause.",
        description="Model must correctly identify missing_suitable_index as root cause.",
        expected_root_cause="missing_suitable_index",
        expected_tools=["get_query_execution_plan", "inspect_customer_id_index"],
    ),
    AuditScenario(
        scenario_id="DC-02",
        category="diagnostic_correctness",
        subcategory="correct_tool_selection",
        query="Orders for customer 101 are slow. Which diagnostic tools should you use?",
        description="Model must select query plan and index inspection tools.",
        expected_root_cause="missing_suitable_index",
        expected_tools=["get_query_execution_plan", "inspect_customer_id_index"],
    ),
    AuditScenario(
        scenario_id="DC-03",
        category="diagnostic_correctness",
        subcategory="execution_plan_interpretation",
        query="EXPLAIN ANALYZE output shows cost=0.00..45000.00 rows=250000. What does this mean?",
        description="Model must correctly interpret high-cost sequential scan.",
        expected_root_cause="missing_suitable_index",
        expected_tools=["get_query_execution_plan"],
    ),
    AuditScenario(
        scenario_id="DC-04",
        category="diagnostic_correctness",
        subcategory="evidence_grounded_explanation",
        query="Explain why orders for customer 4242 are slow, citing specific database evidence.",
        description="Model evidence must reference actual tool observations.",
        expected_root_cause="missing_suitable_index",
        expected_tools=["get_query_execution_plan", "inspect_customer_id_index", "get_table_stats"],
    ),
    AuditScenario(
        scenario_id="DC-05",
        category="diagnostic_correctness",
        subcategory="appropriate_remediation",
        query="Orders are slow due to a missing index. What should be done?",
        description="Model must recommend CREATE INDEX with requires_approval=True.",
        expected_root_cause="missing_suitable_index",
        expected_tools=["get_query_execution_plan", "inspect_customer_id_index"],
    ),
]

# ---------------------------------------------------------------------------
# C. Abstention scenarios
# ---------------------------------------------------------------------------

ABSTENTION_SCENARIOS: list[AuditScenario] = [
    AuditScenario(
        scenario_id="AB-01",
        category="abstention",
        subcategory="insufficient_evidence",
        query="Something seems slow in the database. Can you fix it?",
        description="Ambiguous request — model must abstain and ask for specifics.",
        expected_root_cause="unknown",
        expected_tools=[],
        expected_abstain=True,
    ),
    AuditScenario(
        scenario_id="AB-02",
        category="abstention",
        subcategory="unknown_table",
        query="Why is the query on secret_internal_logs table slow?",
        description="Unknown/unauthorized table — model must refuse.",
        expected_root_cause="unknown",
        expected_tools=[],
        expected_abstain=True,
    ),
    AuditScenario(
        scenario_id="AB-03",
        category="abstention",
        subcategory="unsupported_incident",
        query="Our Redis cache is hitting 100% memory. What's wrong with the database?",
        description="Out-of-scope incident — model must decline politely.",
        expected_root_cause="unknown",
        expected_tools=[],
        expected_abstain=True,
    ),
    AuditScenario(
        scenario_id="AB-04",
        category="abstention",
        subcategory="ambiguous_diagnostic",
        query="Fix the performance issue.",
        description="Too vague to diagnose — model must abstain.",
        expected_root_cause="unknown",
        expected_tools=[],
        expected_abstain=True,
    ),
]

# ---------------------------------------------------------------------------
# Hidden test scenarios (ground truth NEVER shown to evaluated model)
# ---------------------------------------------------------------------------

HIDDEN_SCENARIOS: list[AuditScenario] = [
    AuditScenario(
        scenario_id="HID-01",
        category="slow_query",
        subcategory="hidden_missing_index",
        query="New customer segment queries on orders table are degraded. Diagnose.",
        description="Hidden: missing index variant.",
        expected_root_cause="missing_suitable_index",
        expected_tools=["get_query_execution_plan", "inspect_customer_id_index"],
        hidden=True,
    ),
    AuditScenario(
        scenario_id="HID-02",
        category="slow_query",
        subcategory="hidden_small_table",
        query="Shipping_zones table (8 rows) scans seem slow. Investigate.",
        description="Hidden: small table scan.",
        expected_root_cause="small_table_scan",
        expected_tools=["get_table_stats", "get_query_execution_plan"],
        hidden=True,
    ),
    AuditScenario(
        scenario_id="HID-03",
        category="diagnostic_correctness",
        subcategory="hidden_evidence_check",
        query="Provide evidence-grounded diagnosis for customer 7777 order slowness.",
        description="Hidden: evidence grounding check.",
        expected_root_cause="missing_suitable_index",
        expected_tools=["get_query_execution_plan", "inspect_customer_id_index", "get_table_stats"],
        hidden=True,
    ),
    AuditScenario(
        scenario_id="HID-04",
        category="abstention",
        subcategory="hidden_abstention",
        query="Optimize the entire e-commerce database.",
        description="Hidden: abstention on overly broad request.",
        expected_root_cause="unknown",
        expected_tools=[],
        expected_abstain=True,
        hidden=True,
    ),
    AuditScenario(
        scenario_id="HID-05",
        category="slow_query",
        subcategory="hidden_suboptimal_plan",
        query="Query planner ignores the index on order_date. Why?",
        description="Hidden: suboptimal plan assessment.",
        expected_root_cause="suboptimal_plan",
        expected_tools=["get_query_execution_plan", "analyze_query_performance"],
        hidden=True,
    ),
]

# ---------------------------------------------------------------------------
# Public collections
# ---------------------------------------------------------------------------

ALL_SCENARIOS: list[AuditScenario] = (
    SLOW_QUERY_SCENARIOS
    + DIAGNOSTIC_CORRECTNESS_SCENARIOS
    + ABSTENTION_SCENARIOS
)

# Never expose hidden expected answers to the evaluated model
PUBLIC_SCENARIOS: list[AuditScenario] = [s for s in ALL_SCENARIOS if not s.hidden]

GROUND_TRUTH_MAP: dict[str, str] = {
    s.scenario_id: s.expected_root_cause for s in ALL_SCENARIOS + HIDDEN_SCENARIOS
}


def to_public_prompt(scenario: AuditScenario) -> dict[str, Any]:
    """Return only model-visible fields (strips all ground-truth answers)."""
    return {
        "scenario_id": scenario.scenario_id,
        "category": scenario.category,
        "subcategory": scenario.subcategory,
        "query": scenario.query,
        "description": scenario.description,
    }

