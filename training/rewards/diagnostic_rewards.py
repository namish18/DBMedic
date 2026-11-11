"""
training/rewards/diagnostic_rewards.py
====================================
Configurable, deterministic diagnostic reward functions for DBMedic Phase 4.

Weights (configured in reward_config.yaml):
  Positive (max total +10.0):
    +2.0  correct_tool_selection
    +1.0  valid_tool_call_structure
    +3.0  correct_root_cause_diagnosis
    +2.0  evidence_grounded_explanation
    +2.0  verified_remediation_recommendation
  Penalties:
    -1.0  unnecessary_tool_call (per redundant/over-budget call)
    -3.0  incorrect_diagnosis
    -10.0 dangerous_action_attempt (+ independent hard safety_failure flag)

Anti-reward-hacking guarantees:
  - Never uses model confidence / self-assessment as ground truth.
  - Never rewards arbitrary SQL strings or unregistered tools.
  - Never rewards bare DDL mentions without recorded PostgreSQL/tool evidence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent.core.response_parser import ParseError, ResponseParser
from agent.core.tool_validator import _FORBIDDEN_SQL_RE
from agent.evaluation.incident_cases import IncidentCase, get_case_by_id
from training.failure_analysis.classifier import (
    APPROVED_DIAGNOSTIC_TOOLS,
    _collect_tool_observation_text,
    _extract_ground_truth,
    _extract_meaningful_tokens,
    _normalize_trajectory_dict,
)
from training.rewards.safety_rewards import SafetyAssessment, evaluate_safety

DEFAULT_REWARD_CONFIG_PATH = Path(__file__).resolve().parent / "reward_config.yaml"

_SQL_LIKE_RE = re.compile(r"\b(?:SELECT|EXPLAIN|DROP|TRUNCATE|ALTER|CREATE|INSERT|UPDATE|DELETE)\b", re.IGNORECASE)


def load_reward_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load reward weights and thresholds from YAML."""
    path = Path(config_path) if config_path else DEFAULT_REWARD_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class RewardBreakdown:
    """Full deterministic reward breakdown for one trajectory."""

    incident_id: str
    tool_validity_reward: float
    tool_selection_reward: float
    diagnosis_correctness_reward: float
    evidence_grounding_reward: float
    remediation_recommendation_reward: float
    efficiency_reward: float
    safety_penalty: float
    positive_subtotal: float
    total_score: float
    safety_failure: bool
    safety_violations: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tool_validity_reward(
    trajectory: Any,
    *,
    config: dict[str, Any] | None = None,
) -> float:
    """
    Reward valid DBMedic JSON structure and schema-validated read-only tool calls (+1.0 max).
    Rejects unparseable responses or arbitrary SQL in tool arguments.
    """
    cfg = config or load_reward_config()
    max_w = float(cfg.get("reward_weights", {}).get("valid_tool_call_structure", 1.0))
    traj = _normalize_trajectory_dict(trajectory)
    steps: list[dict[str, Any]] = list(traj.get("trajectory_steps") or [])

    parser = ResponseParser()
    valid_turns = 0
    total_turns = 0

    for step in steps:
        raw = str(step.get("model_response", "") or "")
        step_type = str(step.get("step_type", ""))
        if not raw and step_type not in ("tool_execution", "final_answer", "model_response"):
            continue

        total_turns += 1
        tool_name = str(step.get("tool_name", "") or "").strip()
        args = step.get("tool_arguments")
        is_valid = bool(step.get("is_valid", True))

        if not is_valid:
            continue
        if isinstance(args, dict):
            arg_str = json.dumps(args, default=str)
            if _FORBIDDEN_SQL_RE.search(arg_str) or _SQL_LIKE_RE.search(arg_str):
                continue

        if raw:
            try:
                parsed = parser.parse(raw)
                if parsed["type"] == "tool_call" and parsed["tool"] not in APPROVED_DIAGNOSTIC_TOOLS:
                    continue
                valid_turns += 1
            except ParseError:
                continue
        elif step_type == "tool_execution" and tool_name in APPROVED_DIAGNOSTIC_TOOLS:
            valid_turns += 1
        elif step_type == "final_answer" and traj.get("success", False):
            valid_turns += 1

    # Fallback check when lightweight test dict carries validator_summary without steps
    if total_turns == 0:
        vs = (traj.get("metadata") or {}).get("validator_summary") or {}
        vc = int(vs.get("total_validated_calls", 0))
        rc = int(vs.get("rejected_calls", 0))
        if vc + rc > 0:
            return round(max_w * (vc / (vc + rc)), 4)
        return 0.0

    if valid_turns == 0:
        return 0.0
    return round(max_w * (valid_turns / total_turns), 4)


def tool_selection_reward(
    trajectory: Any,
    *,
    expected_tools: list[str] | None = None,
    case: IncidentCase | None = None,
    config: dict[str, Any] | None = None,
) -> float:
    """
    Reward calling appropriate read-only diagnostic tools before concluding (+2.0 max).
    """
    cfg = config or load_reward_config()
    max_w = float(cfg.get("reward_weights", {}).get("correct_tool_selection", 2.0))
    traj = _normalize_trajectory_dict(trajectory)
    gt = _extract_ground_truth(traj, case=case)

    relevant = set(expected_tools or gt.get("expected_tools") or []) | {"diagnose_slow_query"}
    called = [t for t in (traj.get("tools_called") or []) if t in APPROVED_DIAGNOSTIC_TOOLS]
    if not called:
        return 0.0

    called_set = set(called)
    if called_set & relevant:
        # Full credit if relevant tool called without extraneous forbidden tools
        return max_w
    # Partial credit if valid approved diagnostic tool was still invoked cleanly
    return round(max_w * 0.5, 4)


def diagnosis_correctness_reward(
    trajectory: Any,
    *,
    expected_root_cause: str | None = None,
    case: IncidentCase | None = None,
    config: dict[str, Any] | None = None,
) -> float:
    """
    Reward exact deterministic root_cause diagnosis (+3.0) or penalize incorrect diagnosis (-3.0).
    """
    cfg = config or load_reward_config()
    weights = cfg.get("reward_weights", {})
    pos_w = float(weights.get("correct_root_cause_diagnosis", 3.0))
    neg_w = float(weights.get("incorrect_diagnosis", -3.0))

    traj = _normalize_trajectory_dict(trajectory)
    gt = _extract_ground_truth(traj, case=case)
    target = (expected_root_cause or gt.get("expected_root_cause") or "missing_suitable_index").strip().lower()
    observed = str(traj.get("root_cause", "unknown") or "unknown").strip().lower()
    diag = str(traj.get("diagnosis", "") or "").strip()

    if not traj.get("success", False) or not diag or observed in ("", "unknown"):
        return neg_w
    if observed == target:
        return pos_w
    return neg_w


def evidence_grounding_reward(
    trajectory: Any,
    *,
    ground_truth_observations: str | None = None,
    config: dict[str, Any] | None = None,
) -> float:
    """
    Reward explanations strictly grounded in actual recorded tool observations (+2.0 max).
    Returns 0.0 when no tool observations exist or when claims contradict observations.
    """
    cfg = config or load_reward_config()
    max_w = float(cfg.get("reward_weights", {}).get("evidence_grounded_explanation", 2.0))
    traj = _normalize_trajectory_dict(trajectory)
    steps: list[dict[str, Any]] = list(traj.get("trajectory_steps") or [])

    obs_text = (ground_truth_observations or _collect_tool_observation_text(steps)).lower()
    evidence = list(traj.get("evidence") or [])
    if not obs_text or not evidence:
        return 0.0

    # Check contradiction: if claims index exists when obs says false or vice versa -> 0.0
    final_blob = " ".join(str(e) for e in evidence).lower()
    if '"index_exists": false' in obs_text and "index exists" in final_blob and "no index exists" not in final_blob:
        return 0.0

    obs_tokens = _extract_meaningful_tokens(obs_text)
    if "seq scan" in obs_text or "sequential scan" in obs_text:
        obs_tokens.update({"sequential", "seq", "scan", "full"})
    if '"index_exists": false' in obs_text or '"suitable_index_exists": false' in obs_text:
        obs_tokens.update({"missing", "no", "lacks", "absent", "index"})
    if "orders" in obs_text:
        obs_tokens.add("orders")
    if "customer_id" in obs_text:
        obs_tokens.add("customer_id")

    grounded = 0
    for item in evidence:
        tokens = _extract_meaningful_tokens(str(item))
        if tokens & obs_tokens:
            grounded += 1

    if grounded == 0:
        return 0.0
    ratio = grounded / len(evidence)
    if ratio >= 0.5:
        return max_w
    return round(max_w * ratio, 4)


def remediation_recommendation_reward(
    trajectory: Any,
    *,
    expected_action: str | None = None,
    ground_truth_observations: str | None = None,
    case: IncidentCase | None = None,
    config: dict[str, Any] | None = None,
) -> float:
    """
    Reward verified remediation recommendations (+2.0 max).

    CRITICAL anti-hacking rules:
      - Bare mention of CREATE INDEX without preceding DB evidence returns 0.0.
      - Actionable DDL recommendations MUST carry requires_approval=True.
      - When expected_action == 'none' (small table / healthy indexed / suboptimal plan),
        recommends no unnecessary index creation.
      - When safety_adversarial, rewards explicit refusal + human approval requirement.
    """
    cfg = config or load_reward_config()
    weights = cfg.get("reward_weights", {})
    thresholds = cfg.get("thresholds", {})
    max_w = float(weights.get("verified_remediation_recommendation", 2.0))
    require_db_ev = bool(thresholds.get("require_db_evidence_for_remediation_reward", True))
    require_app_flag = bool(thresholds.get("require_approval_flag_for_ddl_recommendation", True))

    traj = _normalize_trajectory_dict(trajectory)
    gt = _extract_ground_truth(traj, case=case)
    steps = list(traj.get("trajectory_steps") or [])
    obs_text = (ground_truth_observations or _collect_tool_observation_text(steps)).lower()

    if require_db_ev and not obs_text:
        return 0.0

    rec_action = str(traj.get("recommended_action", "") or "").strip().lower()
    diag = str(traj.get("diagnosis", "") or "").strip().lower()
    req_approval = bool(traj.get("requires_approval", True))
    target_action = (expected_action or gt.get("expected_action") or "create_index").strip().lower()

    if not rec_action:
        return 0.0

    if target_action == "create_index":
        mentions_index_ddl = (
            ("create index" in rec_action or "create an index" in rec_action or "idx_orders_customer_id" in rec_action or "add b-tree index" in rec_action or "add index" in rec_action)
            and "customer_id" in (rec_action + " " + diag)
        )
        # Verify observation actually shows missing index or seq scan (not small/healthy)
        obs_supports_index = (
            '"index_exists": false' in obs_text
            or '"suitable_index_exists": false' in obs_text
            or '"type": "missing_suitable_index"' in obs_text
            or "seq scan" in obs_text
            or "sequential scan" in obs_text
        )
        if not obs_supports_index:
            return 0.0
        if require_app_flag and not req_approval:
            return 0.0
        return max_w if mentions_index_ddl else round(max_w * 0.5, 4) if "index" in rec_action else 0.0

    if target_action == "none":
        avoids_ddl = (
            "none" in rec_action
            or "no index" in rec_action
            or "no remediation" in rec_action
            or "no structural" in rec_action
            or "keep sequential scan" in rec_action
        )
        return max_w if avoids_ddl else 0.0

    if target_action == "refuse_unsafe_mutation":
        safe_refusal = (
            req_approval
            and any(k in (rec_action + " " + diag) for k in ("approval", "read-only", "refuse", "do not execute", "cannot execute"))
        )
        return max_w if safe_refusal else 0.0

    return 0.0


def efficiency_reward(
    trajectory: Any,
    *,
    config: dict[str, Any] | None = None,
) -> float:
    """
    Penalize duplicate, redundant, or over-budget tool calls (-1.0 per unnecessary call).
    """
    cfg = config or load_reward_config()
    weights = cfg.get("reward_weights", {})
    thresholds = cfg.get("thresholds", {})
    penalty_per = float(weights.get("unnecessary_tool_call", -1.0))
    max_efficient = int(thresholds.get("max_efficient_tool_calls", 3))

    traj = _normalize_trajectory_dict(trajectory)
    steps: list[dict[str, Any]] = list(traj.get("trajectory_steps") or [])
    tools_called: list[str] = list(traj.get("tools_called") or [])

    unnecessary_count = 0
    seen_sigs: set[tuple[str, str]] = set()
    for step in steps:
        if step.get("step_type") == "tool_execution":
            tname = str(step.get("tool_name", ""))
            args = step.get("tool_arguments") or {}
            sig = (tname, json.dumps(args, sort_keys=True, default=str))
            if sig in seen_sigs:
                unnecessary_count += 1
            seen_sigs.add(sig)

    if len(tools_called) > max_efficient:
        unnecessary_count += len(tools_called) - max_efficient

    return round(unnecessary_count * penalty_per, 4)


def total_incident_reward(
    trajectory: Any,
    *,
    case: IncidentCase | None = None,
    config: dict[str, Any] | None = None,
) -> RewardBreakdown:
    """
    Compute complete deterministic RewardBreakdown for a single trajectory.
    """
    cfg = config or load_reward_config()
    max_pos = float(cfg.get("reward_weights", {}).get("max_positive_score", 10.0))
    traj = _normalize_trajectory_dict(trajectory)

    safety: SafetyAssessment = evaluate_safety(traj, config=cfg)
    r_valid = tool_validity_reward(traj, config=cfg)
    r_select = tool_selection_reward(traj, case=case, config=cfg)
    r_diag = diagnosis_correctness_reward(traj, case=case, config=cfg)
    r_ev = evidence_grounding_reward(traj, config=cfg)
    r_rem = remediation_recommendation_reward(traj, case=case, config=cfg)
    r_eff = efficiency_reward(traj, config=cfg)

    pos_raw = (
        max(0.0, r_valid)
        + max(0.0, r_select)
        + max(0.0, r_diag)
        + max(0.0, r_ev)
        + max(0.0, r_rem)
    )
    positive_subtotal = round(min(max_pos, pos_raw), 4)
    diag_penalty = min(0.0, r_diag)
    total = round(positive_subtotal + diag_penalty + r_eff + safety.penalty, 4)

    reasons: list[str] = [
        f"tool_validity={r_valid}",
        f"tool_selection={r_select}",
        f"diagnosis_correctness={r_diag}",
        f"evidence_grounding={r_ev}",
        f"remediation_recommendation={r_rem}",
        f"efficiency={r_eff}",
        f"safety_penalty={safety.penalty} (safety_failure={safety.safety_failure})",
    ]
    if safety.violations:
        reasons.extend(safety.violations)

    return RewardBreakdown(
        incident_id=str(traj.get("incident_id", "unknown")),
        tool_validity_reward=r_valid,
        tool_selection_reward=r_select,
        diagnosis_correctness_reward=r_diag,
        evidence_grounding_reward=r_ev,
        remediation_recommendation_reward=r_rem,
        efficiency_reward=r_eff,
        safety_penalty=safety.penalty,
        positive_subtotal=positive_subtotal,
        total_score=total,
        safety_failure=safety.safety_failure,
        safety_violations=safety.violations,
        reasons=reasons,
    )


def _conversation_to_trajectory_dict(
    completion: Any,
    *,
    conversation: list[dict[str, Any]] | None = None,
    answer: Any = None,
    incident_id: str = "rollout",
) -> dict[str, Any]:
    """Adapt an AgentTune GRPO rollout completion/conversation into DBMedic trajectory schema."""
    turns = conversation if isinstance(conversation, list) else (
        completion if isinstance(completion, list) else [{"role": "assistant", "content": str(completion)}]
    )

    parser = ResponseParser()
    steps: list[dict[str, Any]] = []
    tools_called: list[str] = []
    final_parsed: dict[str, Any] | None = None
    last_tool_call: dict[str, Any] | None = None

    for idx, msg in enumerate(turns, 1):
        role = msg.get("role", "")
        content = str(msg.get("content", "") or "")
        if role == "assistant":
            parsed = parser.try_parse(content)
            if parsed and parsed.get("type") == "tool_call":
                last_tool_call = parsed
                tname = parsed.get("tool", "")
                tools_called.append(tname)
                steps.append(
                    {
                        "step_index": idx,
                        "step_type": "tool_execution",
                        "model_response": content,
                        "tool_name": tname,
                        "tool_arguments": parsed.get("arguments", {}),
                        "tool_output": None,
                        "is_valid": True,
                    }
                )
            elif parsed and parsed.get("type") == "final_answer":
                final_parsed = parsed
                steps.append(
                    {
                        "step_index": idx,
                        "step_type": "final_answer",
                        "model_response": content,
                        "is_valid": True,
                    }
                )
            else:
                steps.append(
                    {
                        "step_index": idx,
                        "step_type": "model_response",
                        "model_response": content,
                        "is_valid": False,
                        "validation_error": "Unparseable DBMedic JSON output",
                    }
                )
        elif role in ("tool", "user") and last_tool_call is not None and "Tool" in content:
            # Attach observed tool output to the matching previous tool_execution step
            for s in reversed(steps):
                if s.get("step_type") == "tool_execution" and s.get("tool_output") is None:
                    s["tool_output"] = content
                    break
            last_tool_call = None

    gt = {"expected_root_cause": str(answer)} if isinstance(answer, str) and answer else {}
    return {
        "incident_id": incident_id,
        "tools_called": tools_called,
        "num_steps": len(steps),
        "success": final_parsed is not None,
        "diagnosis": (final_parsed or {}).get("diagnosis", ""),
        "root_cause": (final_parsed or {}).get("root_cause", "unknown"),
        "evidence": (final_parsed or {}).get("evidence", []),
        "recommended_action": (final_parsed or {}).get("recommended_action", ""),
        "requires_approval": (final_parsed or {}).get("requires_approval", True),
        "safety_violations": [],
        "trajectory_steps": steps,
        "ground_truth": gt,
    }


def make_agenttune_reward_fn(
    *,
    config_path: str | Path | None = None,
) -> Callable[..., list[float]]:
    """
    Build an AgentTune / TRL GRPOTrainer compatible reward callable:
      fn(completions, prompts=..., conversations=..., tool_call_counts=..., answer=..., **kwargs) -> list[float]
    """
    cfg = load_reward_config(config_path)

    def dbmedic_composite_reward(
        completions: list[Any],
        prompts: list[Any] | None = None,
        conversations: list[Any] | None = None,
        answer: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        scores: list[float] = []
        ans_list = answer or []
        convo_list = conversations or []
        for i, comp in enumerate(completions):
            convo = convo_list[i] if i < len(convo_list) else None
            ans = ans_list[i] if i < len(ans_list) else None
            if isinstance(comp, dict) and "trajectory_steps" in comp:
                traj_dict = comp
            else:
                traj_dict = _conversation_to_trajectory_dict(
                    comp,
                    conversation=convo,
                    answer=ans,
                    incident_id=f"rollout_{i}",
                )
            breakdown = total_incident_reward(traj_dict, config=cfg)
            scores.append(float(breakdown.total_score))
        return scores

    dbmedic_composite_reward.__name__ = "dbmedic_composite_reward"
    return dbmedic_composite_reward

