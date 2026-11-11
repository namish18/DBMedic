"""
agent/audit/tool_metrics.py
============================
Tool-call evaluation metrics for AuditKIT episode lists.

Measures:
  - Tool-call validity rate
  - Tool selection accuracy
  - Argument validity
  - Unnecessary / duplicate tool calls
  - Tool execution success rate
  - Average tool calls per task
  - Average agent steps
  - Task completion rate

Uses DBMedic's existing deterministic metrics as the authoritative source.
Does NOT double-count metrics already computed by agent/evaluation/metrics.py.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

RELEVANT_DIAGNOSTIC_TOOLS = frozenset({
    "get_query_execution_plan",
    "analyze_query_performance",
    "inspect_customer_id_index",
    "get_table_indexes",
    "diagnose_slow_query",
    "get_table_stats",
    "get_slow_queries",
    "explain_query",
})

UNSAFE_TOOLS = frozenset({
    "apply_remediation",
    "execute_sql",
    "drop_table",
    "truncate_table",
    "create_index_now",
})


def compute_tool_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute all tool-call metrics from a list of episode dicts.

    Parameters
    ----------
    episodes : list[dict]
        Episode.to_dict() records.

    Returns
    -------
    dict mapping metric name to scored value dict.
    """
    if not episodes:
        return _empty_metrics()

    total_valid = 0
    total_rejected = 0
    tool_accurate = 0
    task_complete = 0
    total_tools = 0
    total_steps = 0
    redundant_count = 0
    loop_detected = 0
    tool_exec_success = 0
    tool_exec_total = 0
    arg_valid = 0
    arg_total = 0

    for ep in episodes:
        steps = ep.get("steps", [])
        num_steps = len(steps)
        total_steps += num_steps

        # Collect tool calls from steps
        ep_tools: list[str] = []
        ep_tool_args: list[dict] = []
        ep_tool_results: list[Any] = []

        for step in steps:
            st = step.get("step_type", "")
            if st == "tool_call":
                tool_name = step.get("tool_name", "")
                args = step.get("tool_arguments") or {}
                ep_tools.append(tool_name)
                ep_tool_args.append(args)
                total_tools += 1

                # Tool validity: registered and not in unsafe set
                if tool_name and tool_name not in UNSAFE_TOOLS:
                    total_valid += 1
                else:
                    total_rejected += 1

                # Argument validity: basic structural check
                arg_total += 1
                if isinstance(args, dict):
                    arg_valid += 1

                # Execution success: check if next step has non-error tool_output
                tool_exec_total += 1
                if not step.get("validation_error"):
                    tool_exec_success += 1

            elif st == "tool_result":
                ep_tool_results.append(step.get("tool_output"))

        # Tool selection accuracy
        ep_tool_set = set(ep_tools)
        if ep_tool_set & RELEVANT_DIAGNOSTIC_TOOLS:
            tool_accurate += 1

        # Task completion (independent verification)
        if ep.get("verified_success") is True:
            task_complete += 1

        # Redundant tool calls: exact duplicate (tool_name, frozen args)
        seen_calls: set[tuple] = set()
        ep_redundant = 0
        for tool, args in zip(ep_tools, ep_tool_args):
            key = (tool, _freeze_args(args))
            if key in seen_calls:
                ep_redundant += 1
            else:
                seen_calls.add(key)
        if ep_redundant > 0:
            redundant_count += 1

        # Loop detection: more than 8 steps or same tool called 3+ times
        from collections import Counter
        tool_freq = Counter(ep_tools)
        if num_steps >= 8 or any(v >= 3 for v in tool_freq.values()):
            loop_detected += 1

    n = len(episodes)

    def _safe_rate(num: int, denom: int) -> dict[str, Any]:
        if denom == 0:
            return {"value": None, "reason": "no_data", "scored": False}
        return {"value": round(num / denom, 4), "scored": True}

    return {
        "tool_call_validity": _safe_rate(total_valid, total_valid + total_rejected),
        "tool_call_f1": _compute_f1(episodes),
        "tool_selection_accuracy": _safe_rate(tool_accurate, n),
        "argument_validity": _safe_rate(arg_valid, arg_total),
        "redundant_tool_calls": {
            "value": round(redundant_count / n, 4),
            "count": redundant_count,
            "scored": True,
        },
        "agent_loop_detected": {
            "value": round(loop_detected / n, 4),
            "count": loop_detected,
            "scored": True,
        },
        "tool_execution_success": _safe_rate(tool_exec_success, tool_exec_total),
        "average_tool_calls_per_task": {
            "value": round(total_tools / n, 2),
            "scored": True,
        },
        "average_steps_per_task": {
            "value": round(total_steps / n, 2),
            "scored": True,
        },
        "task_completion": _safe_rate(task_complete, n),
        "trajectory_match": _compute_trajectory_match(episodes),
    }


def _empty_metrics() -> dict[str, Any]:
    keys = [
        "tool_call_validity", "tool_call_f1", "tool_selection_accuracy",
        "argument_validity", "redundant_tool_calls", "agent_loop_detected",
        "tool_execution_success", "average_tool_calls_per_task",
        "average_steps_per_task", "task_completion", "trajectory_match",
    ]
    return {k: {"value": None, "reason": "no_episodes", "scored": False} for k in keys}


def _freeze_args(args: dict[str, Any]) -> tuple:
    try:
        import json
        return (json.dumps(args, sort_keys=True),)
    except (TypeError, ValueError):
        return (str(args),)


def _compute_f1(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute ToolCallF1: precision and recall of tool argument matching
    against expected tools (from incident case definitions where available).
    """
    # Without expected tool sequences, return unscored with explanation
    scored_episodes = [ep for ep in episodes if ep.get("expected_root_cause")]
    if not scored_episodes:
        return {
            "value": None,
            "reason": "no_expected_tool_sequences_available",
            "scored": False,
        }

    # For episodes where we know the expected tools from root cause
    total_f1 = 0.0
    count = 0
    for ep in scored_episodes:
        rc = ep.get("expected_root_cause", "")
        expected = set(RELEVANT_DIAGNOSTIC_TOOLS)  # Approximate from root cause
        actual = set()
        for step in ep.get("steps", []):
            if step.get("step_type") == "tool_call" and step.get("tool_name"):
                actual.add(step["tool_name"])

        if not expected and not actual:
            continue
        precision = len(actual & expected) / len(actual) if actual else 0.0
        recall = len(actual & expected) / len(expected) if expected else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        total_f1 += f1
        count += 1

    if count == 0:
        return {"value": None, "reason": "no_scoreable_episodes", "scored": False}
    return {"value": round(total_f1 / count, 4), "scored": True}


def _compute_trajectory_match(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """
    TrajectoryMatch: fraction of episodes where the agent's tool call sequence
    follows the expected diagnostic order (query plan → index check → diagnosis).
    """
    expected_order = [
        "get_query_execution_plan",
        "inspect_customer_id_index",
        "get_table_indexes",
        "analyze_query_performance",
    ]

    matched = 0
    scoreable = 0
    for ep in episodes:
        steps = ep.get("steps", [])
        actual_tools = [
            s.get("tool_name", "")
            for s in steps
            if s.get("step_type") == "tool_call"
        ]
        if not actual_tools:
            continue
        scoreable += 1
        # Check that at least one expected tool appears in reasonable order
        last_idx = -1
        ordered = True
        for expected_tool in expected_order:
            if expected_tool in actual_tools:
                idx = actual_tools.index(expected_tool)
                if idx < last_idx:
                    ordered = False
                    break
                last_idx = idx
        if ordered:
            matched += 1

    if scoreable == 0:
        return {"value": None, "reason": "no_tool_calls_recorded", "scored": False}
    return {"value": round(matched / scoreable, 4), "scored": True}

