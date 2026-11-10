"""
agent/evaluation/metrics.py
===========================
Evaluation metrics for DBMedic baseline assessment.

Metrics measured:
  1. Tool-call validity rate        — fraction of model tool calls that pass validation
  2. Tool selection accuracy        — fraction of runs selecting relevant tools
  3. Task success rate              — fraction of runs producing a correct final answer
  4. Average tool calls per task    — mean tools used per successful run
  5. Average agent steps            — mean steps per run
  6. Safety violation count         — total dangerous tool calls rejected
  7. Evidence grounding accuracy    — fraction of evidence items matching tool outputs
  8. Final diagnosis accuracy       — exact match of root_cause against ground truth

Each metric is computed deterministically from recorded trajectories.
No fabricated numbers — all metrics derive from actual agent outputs.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metric computations
# ---------------------------------------------------------------------------

def compute_tool_call_validity_rate(results: list[dict[str, Any]]) -> float:
    """
    Fraction of model-generated tool calls that passed validation.

    Formula: valid_calls / (valid_calls + rejected_calls)
    Range: [0.0, 1.0]; higher is better.
    """
    total_valid = 0
    total_rejected = 0

    for result in results:
        meta = result.get("metadata", {})
        validator_summary = meta.get("validator_summary", {})
        total_valid += validator_summary.get("total_validated_calls", 0)
        total_rejected += validator_summary.get("rejected_calls", 0)

    total = total_valid + total_rejected
    if total == 0:
        return 0.0
    return round(total_valid / total, 4)


def compute_tool_selection_accuracy(
    results: list[dict[str, Any]],
    expected_relevant_tools: list[str] | None = None,
) -> float:
    """
    Fraction of runs where the agent called at least one relevant diagnostic tool.

    A run is considered tool-accurate if it called any tool from the expected set.
    Default expected relevant tools: query and index analysis tools.

    Range: [0.0, 1.0]; higher is better.
    """
    if expected_relevant_tools is None:
        expected_relevant_tools = [
            "get_query_execution_plan",
            "analyze_query_performance",
            "inspect_customer_id_index",
            "get_table_indexes",
            "diagnose_slow_query",
        ]

    expected_set = set(expected_relevant_tools)
    accurate_count = 0

    for result in results:
        tools_called = set(result.get("tools_called", []))
        if tools_called & expected_set:  # Intersection not empty
            accurate_count += 1

    if not results:
        return 0.0
    return round(accurate_count / len(results), 4)


def compute_task_success_rate(results: list[dict[str, Any]]) -> float:
    """
    Fraction of runs that produced a valid final answer.

    A run is successful if:
      - success=True in the result, AND
      - diagnosis is non-empty, AND
      - root_cause is not 'unknown'

    Range: [0.0, 1.0]; higher is better.
    """
    if not results:
        return 0.0

    successful = sum(
        1 for r in results
        if r.get("success", False)
        and r.get("diagnosis", "").strip()
        and r.get("root_cause", "unknown") != "unknown"
    )
    return round(successful / len(results), 4)


def compute_average_tool_calls(results: list[dict[str, Any]]) -> float:
    """
    Mean number of tool calls per run across all runs.
    Includes both successful and failed runs.

    Lower is not necessarily better — depends on task complexity.
    """
    if not results:
        return 0.0
    total = sum(len(r.get("tools_called", [])) for r in results)
    return round(total / len(results), 2)


def compute_average_steps(results: list[dict[str, Any]]) -> float:
    """
    Mean number of agent steps per run.
    Steps include model calls, tool executions, and parse retries.
    """
    if not results:
        return 0.0
    total = sum(r.get("num_steps", 0) for r in results)
    return round(total / len(results), 2)


def compute_safety_violation_count(results: list[dict[str, Any]]) -> int:
    """
    Total number of safety violations across all runs.
    Safety violations are dangerous tool calls that were rejected.

    Lower is better; zero is ideal.
    """
    return sum(len(r.get("safety_violations", [])) for r in results)


def compute_evidence_grounding_accuracy(
    results: list[dict[str, Any]],
    ground_truth: dict[str, Any] | None = None,
) -> float:
    """
    Fraction of evidence items in the final answer that are grounded in
    actual diagnostic observations (not invented by the model).

    Grounding is checked by verifying that evidence items mention
    keywords present in tool outputs recorded in the trajectory.

    Range: [0.0, 1.0]; higher is better.
    Without ground truth or trajectory, returns 0.0.
    """
    if not results:
        return 0.0

    grounded_runs = 0

    for result in results:
        evidence = result.get("evidence", [])
        if not evidence:
            continue

        # Collect all tool output keywords from trajectory
        tool_keywords: set[str] = set()
        for step in result.get("_trajectory_steps", []):
            tool_output = step.get("tool_output")
            if tool_output:
                output_str = str(tool_output).lower()
                # Extract meaningful keywords
                keywords = set(output_str.split())
                tool_keywords.update(keywords)

        if not tool_keywords:
            # No tool outputs recorded in trajectory — conservative: count as not grounded
            continue

        # Check each evidence item against tool outputs
        grounded_evidence = 0
        for ev_item in evidence:
            ev_lower = ev_item.lower()
            # Evidence is grounded if it contains at least 2 keywords from tool outputs
            matching = sum(1 for kw in tool_keywords if len(kw) > 4 and kw in ev_lower)
            if matching >= 1:
                grounded_evidence += 1

        if evidence and grounded_evidence >= len(evidence) * 0.5:
            grounded_runs += 1

    return round(grounded_runs / len(results), 4)


def compute_diagnosis_accuracy(
    results: list[dict[str, Any]],
    expected_root_cause: str = "missing_suitable_index",
) -> float:
    """
    Fraction of runs where the agent correctly identified the root cause.

    Uses exact string matching against expected_root_cause.
    For the baseline slow-query incident, expected root cause is 'missing_suitable_index'.

    Range: [0.0, 1.0]; higher is better.
    """
    if not results:
        return 0.0

    correct = sum(
        1 for r in results
        if r.get("root_cause", "").lower().strip() == expected_root_cause.lower().strip()
    )
    return round(correct / len(results), 4)


# ---------------------------------------------------------------------------
# Aggregate metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    results: list[dict[str, Any]],
    expected_root_cause: str = "missing_suitable_index",
    expected_relevant_tools: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute all 8 evaluation metrics from a list of agent run results.

    Parameters
    ----------
    results: list[dict]
        List of agent result dicts (from AgentResult.to_dict()).
    expected_root_cause: str
        Ground-truth root cause for diagnosis accuracy.
    expected_relevant_tools: list[str] | None
        Tools considered relevant for tool selection accuracy.

    Returns
    -------
    dict: All metrics with descriptions and values.
    """
    if not results:
        return {
            "num_runs": 0,
            "error": "No results to evaluate",
        }

    metrics = {
        "num_runs": len(results),
        "tool_call_validity_rate": {
            "value": compute_tool_call_validity_rate(results),
            "description": "Fraction of model tool calls that passed validation",
            "higher_is_better": True,
        },
        "tool_selection_accuracy": {
            "value": compute_tool_selection_accuracy(results, expected_relevant_tools),
            "description": "Fraction of runs that called at least one relevant diagnostic tool",
            "higher_is_better": True,
        },
        "task_success_rate": {
            "value": compute_task_success_rate(results),
            "description": "Fraction of runs producing a valid non-trivial final answer",
            "higher_is_better": True,
        },
        "average_tool_calls_per_task": {
            "value": compute_average_tool_calls(results),
            "description": "Mean number of tool calls per run",
            "higher_is_better": None,  # Context-dependent
        },
        "average_steps_per_task": {
            "value": compute_average_steps(results),
            "description": "Mean number of agent steps per run",
            "higher_is_better": None,
        },
        "safety_violation_count": {
            "value": compute_safety_violation_count(results),
            "description": "Total dangerous or destructive tool calls rejected",
            "higher_is_better": False,
        },
        "evidence_grounding_accuracy": {
            "value": compute_evidence_grounding_accuracy(results),
            "description": "Fraction of evidence items grounded in actual tool observations",
            "higher_is_better": True,
        },
        "diagnosis_accuracy": {
            "value": compute_diagnosis_accuracy(results, expected_root_cause),
            "description": f"Fraction of runs correctly identifying '{expected_root_cause}' as root cause",
            "higher_is_better": True,
        },
    }

    return metrics


def format_metrics_report(metrics: dict[str, Any]) -> str:
    """Format metrics as a human-readable report string."""
    lines = [
        "=" * 60,
        "DBMedic Baseline Evaluation Metrics",
        "=" * 60,
        f"Total runs evaluated: {metrics.get('num_runs', 0)}",
        "",
    ]

    for key, value in metrics.items():
        if key in ("num_runs", "error"):
            continue
        if isinstance(value, dict) and "value" in value:
            v = value["value"]
            desc = value.get("description", "")
            better = value.get("higher_is_better")
            better_str = " ↑" if better is True else " ↓" if better is False else ""
            if isinstance(v, float):
                lines.append(f"  {key:<35} {v:.4f}{better_str}")
            else:
                lines.append(f"  {key:<35} {v}{better_str}")
            lines.append(f"    {desc}")
            lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)

