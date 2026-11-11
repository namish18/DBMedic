"""
training/agenttune/trajectory_adapter.py
========================================
Bidirectional adapter between DBMedic trajectories / Phase 2 tools and
AgentTune's native structures (`EventLog`, `Trajectory`, `Project`,
`DictToolHarness`, `Failure`, `ClassifiedFailure`, `TrainingExample`).

Preserves:
  - Strict read-only tool boundaries (never exposes apply_remediation or arbitrary SQL).
  - DBMedic Phase 3 JSON protocol (`tool_call` / `final_answer`).
  - Tiny Aya strict alternating `user` / `assistant` conversation turns.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.harness import DictToolHarness
from agenttune.agentic.project import Project
from agenttune.agentic.trajectory.dataset import Step, Trajectory
from agenttune.decide.closed_loop.contracts import (
    ClassifiedFailure,
    Failure,
    TrainingExample,
)

from training.data.formatter import (
    DEFAULT_TOOL_PROMPT,
    enforce_strict_alternation,
)
from training.failure_analysis.classifier import (
    APPROVED_DIAGNOSTIC_TOOLS,
    FailureRecord,
    _normalize_trajectory_dict,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_default_system_prompt() -> str:
    path = PROJECT_ROOT / "agent" / "prompts" / "system_prompt.txt"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "You are DBMedic, a PostgreSQL diagnostic assistant. "
            "Respond only with strict JSON (tool_call or final_answer)."
        )


def to_agenttune_event_log(
    trajectory: Any,
    *,
    tier: str = "full",
    reward_score: float | None = None,
) -> EventLog:
    """
    Convert a DBMedic trajectory dict, curated row, or AgentResult into an
    AgentTune `EventLog` (`tier="full"` for training/distillation, `"light"` for
    eval/self-heal audits).
    """
    if tier not in ("full", "light"):
        raise ValueError(f"Unsupported EventLog tier '{tier}'; expected 'full' or 'light'.")

    traj = _normalize_trajectory_dict(trajectory)
    log_id = str(traj.get("trajectory_id") or traj.get("incident_id") or "dbmedic_traj")
    log = EventLog(tier=tier, id=log_id)

    query = str(traj.get("query", "") or "")
    if query:
        log.append(Event(EventKind.OBSERVATION, {"text": query, "role": "user"}))

    steps = list(traj.get("trajectory_steps") or [])
    if steps:
        for step in steps:
            st = str(step.get("step_type", ""))
            sidx = int(step.get("step_index", 0) or 0)
            raw_resp = str(step.get("model_response", "") or "")

            if st == "user_question":
                if not query:
                    log.append(Event(EventKind.OBSERVATION, {"text": str(step.get("content", ""))}))
                continue

            if st == "tool_execution":
                tname = str(step.get("tool_name", "") or "unknown")
                targs = dict(step.get("tool_arguments") or {})
                tout = step.get("tool_output")
                action_payload = {
                    "name": tname,
                    "arguments": targs,
                    "model_response": raw_resp,
                }
                log.append(Event(EventKind.TOOL_CALL, {"action": action_payload, "status": "ok"}))
                result_payload: dict[str, Any] = {"output": tout, "tool_name": tname}
                if not step.get("is_valid", True) and step.get("validation_error"):
                    result_payload["error"] = step["validation_error"]
                log.append(Event(EventKind.TOOL_RESULT, result_payload))
                log.append(Event(EventKind.TURN_COMPLETE, {"step": sidx}))
            elif st == "model_response":
                tname = str(step.get("tool_name", "") or "")
                val_err = str(step.get("validation_error", "") or "")
                if tname:
                    log.append(
                        Event(
                            EventKind.TOOL_CALL,
                            {
                                "action": {
                                    "name": tname,
                                    "arguments": dict(step.get("tool_arguments") or {}),
                                    "model_response": raw_resp,
                                },
                                "status": "error" if not step.get("is_valid", True) else "ok",
                            },
                        )
                    )
                    if val_err:
                        log.append(Event(EventKind.TOOL_RESULT, {"output": None, "error": val_err}))
                elif raw_resp:
                    log.append(Event(EventKind.TEXT, {"text": raw_resp, "is_valid": step.get("is_valid", True)}))
                log.append(Event(EventKind.TURN_COMPLETE, {"step": sidx}))
            elif st == "final_answer":
                final_txt = raw_resp or json.dumps(
                    {
                        "type": "final_answer",
                        "diagnosis": traj.get("diagnosis", ""),
                        "root_cause": traj.get("root_cause", "unknown"),
                        "evidence": traj.get("evidence", []),
                        "recommended_action": traj.get("recommended_action", ""),
                        "requires_approval": traj.get("requires_approval", True),
                        "confidence": traj.get("confidence", 0.0),
                    },
                    ensure_ascii=False,
                )
                log.append(Event(EventKind.TEXT, {"text": final_txt}))
                log.append(Event(EventKind.TURN_COMPLETE, {"step": sidx}))
    elif isinstance(traj.get("messages"), list):
        # Curated SFT messages fallback
        for m in traj["messages"]:
            role = m.get("role")
            content = str(m.get("content", ""))
            if role == "user":
                if not query or content != query:
                    log.append(Event(EventKind.OBSERVATION, {"text": content}))
            elif role == "assistant":
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("type") == "tool_call":
                    log.append(
                        Event(
                            EventKind.TOOL_CALL,
                            {
                                "action": {
                                    "name": parsed.get("tool", "unknown"),
                                    "arguments": parsed.get("arguments", {}),
                                    "model_response": content,
                                }
                            },
                        )
                    )
                else:
                    log.append(Event(EventKind.TEXT, {"text": content}))

    if reward_score is not None:
        log.append(Event(EventKind.REWARD, {"value": float(reward_score)}, scope="episode"))

    return log


def to_agenttune_trajectory(
    trajectory: Any,
    *,
    reward: float = 0.0,
) -> Trajectory:
    """Convert a DBMedic trajectory or AgentResult into an AgentTune `Trajectory`."""
    traj = _normalize_trajectory_dict(trajectory)
    steps: list[Step] = []
    final_response = ""

    for step in traj.get("trajectory_steps") or []:
        st = str(step.get("step_type", ""))
        sidx = int(step.get("step_index", len(steps) + 1) or (len(steps) + 1))
        if st == "tool_execution":
            tname = str(step.get("tool_name", ""))
            targs = dict(step.get("tool_arguments") or {})
            obs_str = json.dumps(step.get("tool_output"), default=str)
            steps.append(
                Step(
                    step_number=sidx,
                    state=str(traj.get("query", "")),
                    action={"name": tname, "arguments": targs},
                    observation=obs_str,
                    thought=str(step.get("model_response", "") or "") or None,
                    reward=None,
                )
            )
        elif st == "final_answer":
            final_response = str(step.get("model_response", "") or "")

    if not final_response and traj.get("diagnosis"):
        final_response = json.dumps(
            {
                "type": "final_answer",
                "diagnosis": traj.get("diagnosis", ""),
                "root_cause": traj.get("root_cause", "unknown"),
                "evidence": traj.get("evidence", []),
                "recommended_action": traj.get("recommended_action", ""),
                "requires_approval": traj.get("requires_approval", True),
                "confidence": traj.get("confidence", 0.0),
            },
            ensure_ascii=False,
        )

    return Trajectory(
        task=str(traj.get("query", "")),
        steps=steps,
        reward=float(reward),
        trajectory_id=str(traj.get("trajectory_id") or traj.get("incident_id") or "dbmedic"),
        final_response=final_response,
        metadata={
            "incident_id": traj.get("incident_id"),
            "root_cause": traj.get("root_cause"),
            "success": traj.get("success", False),
        },
    )


def event_log_to_dbmedic_messages(
    log: EventLog,
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """
    Convert an AgentTune `EventLog` back into DBMedic's strict-alternation Tiny Aya
    JSON message format (`system -> user -> assistant -> user -> assistant`).
    """
    sys_text = system_prompt if system_prompt is not None else _load_default_system_prompt()
    messages: list[dict[str, str]] = [{"role": "system", "content": sys_text}]

    for event in log:
        if event.kind is EventKind.OBSERVATION:
            messages.append({"role": "user", "content": str(event.payload.get("text", ""))})
        elif event.kind is EventKind.TOOL_CALL:
            action = event.payload.get("action") or {}
            raw = action.get("model_response") if isinstance(action, dict) else None
            if raw:
                messages.append({"role": "assistant", "content": str(raw)})
            else:
                name = action.get("name") or action.get("tool") or event.payload.get("stage", "unknown")
                args = action.get("arguments", {})
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "type": "tool_call",
                                "tool": name,
                                "arguments": args,
                                "reasoning": "Read-only diagnostic check via AgentTune.",
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
        elif event.kind is EventKind.TOOL_RESULT:
            out = event.payload.get("output")
            err = event.payload.get("error")
            tname = event.payload.get("tool_name", "diagnostic_tool")
            out_text = f"Error: {err}" if err else (
                out if isinstance(out, str) else json.dumps(out, indent=2, default=str)
            )
            messages.append(
                {
                    "role": "user",
                    "content": DEFAULT_TOOL_PROMPT.format(
                        tool_name=tname,
                        tool_output=out_text,
                    ),
                }
            )
        elif event.kind is EventKind.TEXT:
            txt = str(event.payload.get("text", "") or "")
            if txt:
                messages.append({"role": "assistant", "content": txt})

    return enforce_strict_alternation(messages, fold_tool_outputs_into_user=True)


def to_agenttune_failure(record: FailureRecord) -> Failure:
    """Convert a DBMedic FailureRecord to AgentTune's native `Failure` contract."""
    return Failure(
        trajectory_id=record.trajectory_id or record.incident_id,
        failure_type=record.failure_category,
        failed_stage_name=f"step_{record.step_indices[0]}" if record.step_indices else "agent_loop",
        context={
            "incident_id": record.incident_id,
            "original_model_response": record.original_model_response,
            "expected_behavior": record.expected_behavior,
            "observed_behavior": record.observed_behavior,
            "evidence": record.evidence,
            "severity": record.severity,
        },
        error_message="; ".join(record.evidence) if record.evidence else record.observed_behavior,
    )


def to_agenttune_classified_failure(record: FailureRecord) -> ClassifiedFailure:
    """Convert a DBMedic FailureRecord to AgentTune's `ClassifiedFailure` contract."""
    return ClassifiedFailure(
        failure=to_agenttune_failure(record),
        root_cause=record.failure_category,
        confidence=0.95 if record.severity in ("critical", "high") else 0.85,
        analysis=f"{record.expected_behavior} | Observed: {record.observed_behavior}",
    )


def to_agenttune_training_example(
    failure_record: FailureRecord,
    corrected_record: dict[str, Any],
) -> TrainingExample:
    """Build an AgentTune `TrainingExample` pairing a corrected DBMedic trace vs failed text."""
    msgs = corrected_record.get("messages") or []
    user_prompt = [m for m in msgs[:2]] if len(msgs) >= 2 else [
        {"role": "user", "content": corrected_record.get("query", "")}
    ]
    chosen_msgs = [m for m in msgs if m.get("role") == "assistant"]
    rejected_content = failure_record.original_model_response or json.dumps(
        {"type": "final_answer", "diagnosis": "", "root_cause": "unknown"}
    )
    return TrainingExample(
        trajectory_id=failure_record.trajectory_id or failure_record.incident_id,
        original_failure_type=failure_record.failure_category,
        root_cause=failure_record.failure_category,
        prompt=user_prompt,
        chosen=chosen_msgs,
        rejected=[{"role": "assistant", "content": rejected_content}],
    )


def to_agenttune_project(
    trajectories: list[Any],
    *,
    tier: str = "full",
    harness: DictToolHarness | None = None,
) -> Project:
    """Populate an AgentTune `Project` lifecycle object with DBMedic trajectories."""
    project = Project(harness=harness)
    for item in trajectories:
        project.add_trajectory(to_agenttune_event_log(item, tier=tier))
    return project


def build_dbmedic_readonly_harness(
    *,
    tool_executors: dict[str, Callable[..., Any]] | None = None,
    max_steps: int = 8,
) -> DictToolHarness:
    """
    Create an AgentTune `DictToolHarness` wrapping strictly DBMedic's 8 approved
    read-only diagnostic tools (NEVER mutating tools or arbitrary SQL).
    """
    if tool_executors is not None:
        safe_tools = {
            name: fn
            for name, fn in tool_executors.items()
            if name in APPROVED_DIAGNOSTIC_TOOLS
        }
        return DictToolHarness(tools=safe_tools, max_steps=max_steps, done_tool="finish")

    from agent.core.tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.initialize()
    safe_tools: dict[str, Callable[..., Any]] = {}
    for name in sorted(APPROVED_DIAGNOSTIC_TOOLS):
        if registry.has_tool(name):
            safe_tools[name] = lambda _n=name, **kw: registry.execute(_n, kw)
    return DictToolHarness(tools=safe_tools, max_steps=max_steps, done_tool="finish")

