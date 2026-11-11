"""
agent/audit/auditkit_adapter.py
================================
Adapter that bridges DBMedic trajectories / AgentTune EventLog data
into the AuditKIT evaluation episode format.

AuditKIT (Lexsi Labs, LSAL-1.2) public API surface used here:
  - ak.evaluate(dataset, model, metrics, config) -> RunResult
  - RunResult.summary() -> dict
  - Metric families: agent.*, safety.*, performance.*
  - SHA-256 configuration fingerprinting (automatic, per-run)

If auditkit is not installed or is unavailable, AuditKITStub provides
a structurally identical interface backed by DBMedic's own deterministic
metrics so recorded-trajectory evaluation can proceed offline without any
live model inference or auditkit package dependency.

License note: AuditKIT is released under LSAL-1.2 (source-available,
non-commercial). Distribution of AuditKIT itself is subject to that
license. This adapter file contains only DBMedic-specific code and does
not redistribute AuditKIT source.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import real AuditKIT; fall back to stub transparently
# ---------------------------------------------------------------------------
try:
    import auditkit as ak  # type: ignore[import]

    _AUDITKIT_AVAILABLE = True
    _AUDITKIT_VERSION = getattr(ak, "__version__", "unknown")
    logger.info("AuditKIT %s loaded successfully.", _AUDITKIT_VERSION)
except ModuleNotFoundError:
    ak = None  # type: ignore[assignment]
    _AUDITKIT_AVAILABLE = False
    _AUDITKIT_VERSION = "not-installed"
    logger.warning(
        "auditkit package not found. Using AuditKITStub for recorded-trajectory "
        "evaluation. Install with: pip install auditkit"
    )


AUDITKIT_AVAILABLE: bool = _AUDITKIT_AVAILABLE
AUDITKIT_VERSION: str = _AUDITKIT_VERSION


# ---------------------------------------------------------------------------
# Episode schema (AuditKIT-compatible structure)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStep:
    """One step in an evaluation episode."""

    step_index: int
    step_type: str           # "model_response" | "tool_call" | "tool_result" | "final_answer"
    timestamp: str
    model_response: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_output: Any = None
    validation_error: str | None = None
    generation_time_ms: float | None = None
    is_valid: bool = True


@dataclass
class Episode:
    """
    One complete agent evaluation episode in AuditKIT-compatible format.

    Maps to the AuditKIT agent task episode structure used by ak.evaluate().
    Fields align with AuditKIT's internal episode schema to allow direct
    pass-through when the library is available.
    """

    episode_id: str
    incident_id: str
    user_request: str
    model_id: str
    model_revision: str | None

    # Trajectory
    steps: list[EpisodeStep] = field(default_factory=list)

    # Outcome
    final_diagnosis: str = ""
    final_root_cause: str = ""
    final_evidence: list[str] = field(default_factory=list)
    final_recommended_action: str = ""
    requires_approval: bool = True

    # Verification (independent of model's self-report)
    verified_success: bool | None = None
    expected_root_cause: str = ""
    outcome_verified_by: str = "dbmedic_deterministic_checker"

    # Safety
    safety_violations: list[str] = field(default_factory=list)
    safety_severity: str = "none"   # none | low | medium | high | critical

    # AgentTune compatibility
    agenttune_event_log: list[dict[str, Any]] = field(default_factory=list)

    # Performance
    total_time_ms: float | None = None
    model_inference_time_ms: float | None = None
    tool_execution_time_ms: float | None = None
    num_model_calls: int = 0
    num_tool_calls: int = 0

    # Metadata
    hardware_info: dict[str, Any] = field(default_factory=dict)
    generation_config: dict[str, Any] = field(default_factory=dict)
    evaluation_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    simulated: bool = False
    split: str = "baseline"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# AuditKITStub — identical interface when auditkit package absent
# ---------------------------------------------------------------------------

@dataclass
class StubRunResult:
    """Mirrors AuditKIT RunResult.summary() output format."""

    episodes: list[dict[str, Any]]
    metrics: dict[str, Any]
    config_fingerprint: str
    auditkit_version: str = "stub"
    evaluation_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> dict[str, Any]:
        return {
            "auditkit_version": self.auditkit_version,
            "evaluation_timestamp": self.evaluation_timestamp,
            "config_fingerprint": self.config_fingerprint,
            "num_episodes": len(self.episodes),
            "metrics": self.metrics,
        }


class AuditKITStub:
    """
    Offline stub implementing the AuditKIT evaluate() interface.

    Used when auditkit is not installed. Computes metrics deterministically
    from DBMedic's own metric functions so recorded episodes can be evaluated
    without live model inference or external package dependencies.
    """

    VERSION = "stub-1.0.0"
    IS_STUB = True

    def evaluate(
        self,
        dataset: list[dict[str, Any]],
        metrics: list[str],
        config: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> StubRunResult:
        """
        Evaluate a list of episode dicts using DBMedic deterministic metrics.

        Parameters
        ----------
        dataset : list[dict]
            Episode dicts (from Episode.to_dict()).
        metrics : list[str]
            AuditKIT metric names (e.g. "agent.TaskCompletion").
        config : dict | None
            Evaluation configuration (passed through; not used offline).
        model : str | None
            Model identifier string (informational only in stub mode).

        Returns
        -------
        StubRunResult
            Compatible with the real AuditKIT RunResult.summary() shape.
        """
        config = config or {}
        cfg_blob = json.dumps(
            {"metrics": sorted(metrics), "config": config, "model": model},
            sort_keys=True,
        ).encode()
        fingerprint = hashlib.sha256(cfg_blob).hexdigest()

        computed = self._compute_stub_metrics(dataset, metrics)

        return StubRunResult(
            episodes=dataset,
            metrics=computed,
            config_fingerprint=fingerprint,
        )

    # ------------------------------------------------------------------
    # Internal metric computation (backed by DBMedic deterministic logic)
    # ------------------------------------------------------------------

    def _compute_stub_metrics(
        self,
        episodes: list[dict[str, Any]],
        requested_metrics: list[str],
    ) -> dict[str, Any]:
        """Compute metric values from episode dicts."""
        from agent.audit.tool_metrics import compute_tool_metrics
        from agent.audit.safety_metrics import compute_safety_metrics
        from agent.audit.performance_metrics import compute_performance_metrics

        result: dict[str, Any] = {}

        metric_map = {
            "agent.TaskCompletion": "task_completion",
            "agent.ToolCallF1": "tool_call_f1",
            "agent.ToolCallValidity": "tool_call_validity",
            "agent.ToolSelectionJudge": "tool_selection_accuracy",
            "agent.RedundantToolCalls": "redundant_tool_calls",
            "agent.AgentLoopDetection": "agent_loop_detected",
            "agent.TrajectoryMatch": "trajectory_match",
            "safety.KeywordDetector": "safety_keyword_violations",
            "safety.ThreatCategory": "threat_category",
            "performance.LatencyStats": "latency_stats",
        }

        tool_metrics = compute_tool_metrics(episodes)
        safety_m = compute_safety_metrics(episodes)
        perf_m = compute_performance_metrics(episodes)

        for metric in requested_metrics:
            stub_key = metric_map.get(metric, metric)
            if stub_key in tool_metrics:
                result[metric] = tool_metrics[stub_key]
            elif stub_key in safety_m:
                result[metric] = safety_m[stub_key]
            elif stub_key in perf_m:
                result[metric] = perf_m[stub_key]
            else:
                result[metric] = {
                    "value": None,
                    "reason": f"metric '{metric}' not computable offline in stub mode",
                    "scored": False,
                }

        return result


# ---------------------------------------------------------------------------
# AuditKITAdapter — main public class
# ---------------------------------------------------------------------------

class AuditKITAdapter:
    """
    Converts DBMedic trajectories / AgentTune EventLog data into AuditKIT
    episode format and runs evaluation through either the real AuditKIT
    library or the offline stub.

    Usage
    -----
    adapter = AuditKITAdapter()
    episode = adapter.trajectory_to_episode(trajectory_dict)
    result  = adapter.evaluate_episodes([episode], mode="recorded")
    report  = result.summary()
    """

    DBMEDIC_STANDARD_METRICS = [
        "agent.TaskCompletion",
        "agent.ToolCallF1",
        "agent.ToolCallValidity",
        "agent.ToolSelectionJudge",
        "agent.RedundantToolCalls",
        "agent.AgentLoopDetection",
        "safety.KeywordDetector",
        "safety.ThreatCategory",
        "performance.LatencyStats",
    ]

    def __init__(self, use_stub: bool | None = None) -> None:
        """
        Parameters
        ----------
        use_stub : bool | None
            Force stub mode (True), force real AuditKIT (False), or auto-detect
            based on package availability (None, default).
        """
        if use_stub is None:
            self._use_stub = not AUDITKIT_AVAILABLE
        else:
            self._use_stub = use_stub

        if self._use_stub:
            self._client: Any = AuditKITStub()
            self.version = AuditKITStub.VERSION
            logger.info("AuditKITAdapter initialized in stub mode.")
        else:
            self._client = ak
            self.version = AUDITKIT_VERSION
            logger.info("AuditKITAdapter initialized with real AuditKIT %s.", self.version)

    @property
    def is_stub(self) -> bool:
        return self._use_stub

    # ------------------------------------------------------------------
    # Trajectory → Episode conversion
    # ------------------------------------------------------------------

    def trajectory_to_episode(
        self,
        trajectory: dict[str, Any],
        *,
        expected_root_cause: str = "",
        split: str = "baseline",
    ) -> Episode:
        """
        Convert a DBMedic trajectory dict (from TrajectoryLogger) into an Episode.

        Parameters
        ----------
        trajectory : dict
            A single trajectory record (JSONL line from trajectories.jsonl).
        expected_root_cause : str
            Ground-truth root cause for independent outcome verification.
        split : str
            Dataset split label.

        Returns
        -------
        Episode
            AuditKIT-compatible episode object.
        """
        incident_id = trajectory.get("incident_id", "unknown")
        episode_id = f"ep_{incident_id}_{int(time.time() * 1000)}"

        steps = self._convert_steps(trajectory.get("trajectory_steps", []))

        # Independent outcome verification — never trusts model's self-report alone
        from agent.audit.outcome_verifier import OutcomeVerifier
        verifier = OutcomeVerifier()
        verified = verifier.verify(
            trajectory=trajectory,
            expected_root_cause=expected_root_cause,
        )

        # Safety severity aggregation
        violations = trajectory.get("safety_violations", [])
        severity = self._severity_from_violations(violations)

        # AgentTune EventLog passthrough (if available in trajectory)
        at_log = trajectory.get("agenttune_event_log", [])

        return Episode(
            episode_id=episode_id,
            incident_id=incident_id,
            user_request=trajectory.get("query", ""),
            model_id=trajectory.get("model_id", "unknown"),
            model_revision=trajectory.get("model_revision"),
            steps=steps,
            final_diagnosis=trajectory.get("diagnosis", ""),
            final_root_cause=trajectory.get("root_cause", "unknown"),
            final_evidence=trajectory.get("evidence", []),
            final_recommended_action=trajectory.get("recommended_action", ""),
            requires_approval=trajectory.get("requires_approval", True),
            verified_success=verified["success"],
            expected_root_cause=expected_root_cause,
            outcome_verified_by=verified["verifier"],
            safety_violations=violations,
            safety_severity=severity,
            agenttune_event_log=at_log,
            total_time_ms=trajectory.get("total_time_ms"),
            num_model_calls=len([
                s for s in trajectory.get("trajectory_steps", [])
                if s.get("step_type") == "model_response"
            ]),
            num_tool_calls=len(trajectory.get("tools_called", [])),
            simulated=trajectory.get("simulated", False),
            split=split,
            hardware_info=trajectory.get("hardware_info", {}),
            generation_config=trajectory.get("generation_config", {}),
        )

    def agenttune_eventlog_to_episode(
        self,
        event_log_data: list[dict[str, Any]],
        *,
        incident_id: str = "unknown",
        user_request: str = "",
        model_id: str = "unknown",
        expected_root_cause: str = "",
        split: str = "baseline",
    ) -> Episode:
        """
        Build an Episode directly from AgentTune EventLog entries.

        Supports recorded-trajectory evaluation without live inference.
        """
        episode_id = f"ep_at_{incident_id}_{int(time.time() * 1000)}"

        steps: list[EpisodeStep] = []
        tool_calls = []
        for idx, event in enumerate(event_log_data):
            kind = event.get("kind", "")
            step_type = {
                "TOOL_CALL": "tool_call",
                "TOOL_RESULT": "tool_result",
                "TEXT": "model_response",
                "TURN_COMPLETE": "model_response",
                "OBSERVATION": "tool_result",
                "REWARD": "reward",
            }.get(kind, "unknown")

            ep_step = EpisodeStep(
                step_index=idx,
                step_type=step_type,
                timestamp=event.get("timestamp", datetime.now(timezone.utc).isoformat()),
                model_response=event.get("content") if step_type == "model_response" else None,
                tool_name=event.get("tool_name"),
                tool_arguments=event.get("tool_arguments"),
                tool_output=event.get("tool_result"),
                is_valid=not event.get("error"),
            )
            steps.append(ep_step)
            if step_type == "tool_call" and event.get("tool_name"):
                tool_calls.append(event["tool_name"])

        return Episode(
            episode_id=episode_id,
            incident_id=incident_id,
            user_request=user_request,
            model_id=model_id,
            model_revision=None,
            steps=steps,
            agenttune_event_log=event_log_data,
            expected_root_cause=expected_root_cause,
            num_tool_calls=len(tool_calls),
            split=split,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_episodes(
        self,
        episodes: list[Episode],
        *,
        mode: str = "recorded",
        metrics: list[str] | None = None,
        config: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Any:
        """
        Run AuditKIT evaluation over a list of episodes.

        Parameters
        ----------
        episodes : list[Episode]
        mode : str
            "recorded" (offline, no live inference) or "live" (live model).
        metrics : list[str] | None
            AuditKIT metric names. Defaults to DBMEDIC_STANDARD_METRICS.
        config : dict | None
            Passed to ak.evaluate() or the stub.
        model : str | None
            Model identifier (for live mode).

        Returns
        -------
        RunResult or StubRunResult with .summary() method.
        """
        if metrics is None:
            metrics = self.DBMEDIC_STANDARD_METRICS

        dataset = [ep.to_dict() for ep in episodes]
        cfg = config or {}
        cfg.setdefault("mode", mode)
        cfg.setdefault("max_turns", 10)

        if self._use_stub:
            return self._client.evaluate(dataset, metrics=metrics, config=cfg, model=model)
        else:
            # Real AuditKIT call — ak.evaluate() per documented API
            return ak.evaluate(
                dataset=dataset,
                model=model or "dbmedic/recorded",
                metrics=metrics,
                config=cfg,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_steps(self, raw_steps: list[dict[str, Any]]) -> list[EpisodeStep]:
        steps = []
        for s in raw_steps:
            steps.append(
                EpisodeStep(
                    step_index=s.get("step_index", 0),
                    step_type=s.get("step_type", "unknown"),
                    timestamp=s.get("timestamp", ""),
                    model_response=s.get("model_response"),
                    tool_name=s.get("tool_name"),
                    tool_arguments=s.get("tool_arguments"),
                    tool_output=s.get("tool_output"),
                    validation_error=s.get("validation_error"),
                    generation_time_ms=s.get("generation_time_ms"),
                    is_valid=s.get("is_valid", True),
                )
            )
        return steps

    @staticmethod
    def _severity_from_violations(violations: list[str]) -> str:
        if not violations:
            return "none"
        severe_keywords = {"drop", "truncate", "delete", "inject", "credential"}
        for v in violations:
            if any(k in v.lower() for k in severe_keywords):
                return "critical"
        return "medium"

