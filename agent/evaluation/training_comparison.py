"""
agent/evaluation/training_comparison.py
=========================================
Phase 4 Model Comparison, AgentTune Self-Healing Lifecycle, and Deployment Gate.

Provides:
  - `ModelComparisonEvaluator`: side-by-side comparison across all 8 Phase 3 metrics
    plus training telemetry (`training_time_sec`, `gpu_memory_usage_mb`, etc.).
  - `evaluate_deployment_gate`: combines AgentTune `DeploymentGate` with DBMedic
    safety/non-regression enforcement and mandatory `human_approved=True`.
  - `SelfHealingPipeline`: connects
    COLLECT -> CLASSIFY -> GENERATE CORRECTION -> VERIFY -> CURATE ->
    RETRAIN -> EVALUATE -> DEPLOYMENT GATE
    wrapping AgentTune's native `SelfHealLoop`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.evaluation.incident_cases import (
    BASELINE_EVAL_CASES,
    HIDDEN_EVAL_CASES,
    VALIDATION_CASES,
    get_case_by_id,
)
from agent.evaluation.metrics import compute_all_metrics
from agenttune.agentic.heal_loop import SelfHealLoop
from agenttune.decide.closed_loop.contracts import (
    ClassifiedFailure,
    Failure,
    TrainingExample,
)
from agenttune.decide.closed_loop.deployment_gate import DeploymentGate, GateDecision
from training.agenttune.trajectory_adapter import (
    to_agenttune_classified_failure,
    to_agenttune_failure,
    to_agenttune_training_example,
)
from training.data.curator import DatasetCurator
from training.failure_analysis.analyzer import TrajectoryFailureAnalyzer
from training.failure_analysis.classifier import FailureClassifier, FailureRecord
from training.failure_analysis.correction_generator import CorrectionGenerator
from training.rewards.safety_rewards import evaluate_safety

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

METRIC_KEYS: tuple[str, ...] = (
    "tool_call_validity_rate",
    "tool_selection_accuracy",
    "task_success_rate",
    "average_tool_calls_per_task",
    "average_steps_per_task",
    "safety_violation_count",
    "evidence_grounding_accuracy",
    "diagnosis_accuracy",
)


def _flatten_runs_from_results_file(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    raw = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    runs: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("runs"), list):
            runs.extend(item["runs"])
        elif isinstance(item, dict):
            runs.append(item)
    return runs


def _extract_metric_values(summary_or_metrics: dict[str, Any]) -> dict[str, float | int | None]:
    m = summary_or_metrics.get("metrics", summary_or_metrics)
    out: dict[str, float | int | None] = {}
    if not isinstance(m, dict) or m.get("num_runs", 0) == 0:
        for k in METRIC_KEYS:
            out[k] = None
        return out
    for k in METRIC_KEYS:
        entry = m.get(k)
        if isinstance(entry, dict) and "value" in entry:
            out[k] = entry["value"]
        elif isinstance(entry, (int, float)):
            out[k] = entry
        else:
            out[k] = None
    return out


def _normalize_run_for_metrics(run: dict[str, Any]) -> dict[str, Any]:
    copied = dict(run)
    if "_trajectory_steps" not in copied and "trajectory_steps" in copied:
        copied["_trajectory_steps"] = copied["trajectory_steps"]
    return copied


@dataclass
class DBMedicGateResult:
    """Comprehensive deployment decision combining AgentTune Gate + DBMedic Safety Policy."""

    approved: bool
    human_approved: bool
    agenttune_gate_approved: bool
    agenttune_decision: dict[str, Any]
    checks: dict[str, bool]
    blocking_reasons: list[str] = field(default_factory=list)
    recommendation: str = "hold_keep_base"
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_deployment_gate(
    *,
    base_runs: list[dict[str, Any]],
    candidate_runs: list[dict[str, Any]],
    hidden_base_runs: list[dict[str, Any]] | None = None,
    hidden_candidate_runs: list[dict[str, Any]] | None = None,
    human_approved: bool = False,
    task_regression_tol: float = 0.0,
) -> DBMedicGateResult:
    """
    Execute the Phase 4 deployment gate:
      1. AgentTune `DeploymentGate().evaluate_decision(...)`
      2. Zero critical safety failures (`safety_failure=False`)
      3. Zero safety violation increase vs base (`cand_violations <= base_violations`)
      4. Zero hidden test regression (when hidden runs supplied)
      5. Improved or non-regressed tool validity & task success
      6. Real non-empty evaluation runs exist
      7. Explicit `human_approved=True`
    """
    gate = DeploymentGate(seed=42)
    blocking: list[str] = []

    norm_base = [_normalize_run_for_metrics(r) for r in base_runs]
    norm_cand = [_normalize_run_for_metrics(r) for r in candidate_runs]

    has_real_eval = len(norm_base) > 0 and len(norm_cand) > 0

    base_by_id: dict[str, dict[str, Any]] = {}
    cand_by_id: dict[str, dict[str, Any]] = {}
    for r in norm_base:
        iid = str(r.get("incident_id", "")).split("_run")[0]
        base_by_id[iid] = r
    for r in norm_cand:
        iid = str(r.get("incident_id", "")).split("_run")[0]
        cand_by_id[iid] = r

    common_ids = sorted(set(base_by_id.keys()) | set(cand_by_id.keys()))
    test_set: list[dict[str, Any]] = []
    for iid in common_ids:
        case = get_case_by_id(iid)
        gt = (base_by_id.get(iid) or cand_by_id.get(iid) or {}).get("ground_truth") or {}
        expected = (
            gt.get("expected_root_cause")
            or (case.expected_root_cause if case is not None else None)
            or "missing_suitable_index"
        )
        test_set.append(
            {
                "pipeline_id": iid,
                "input_text": iid,
                "expected_verdict": expected,
            }
        )

    def _runner(mapping: dict[str, dict[str, Any]]) -> Callable[[Any], str]:
        def run_fn(iid: Any) -> str:
            rec = mapping.get(str(iid))
            if not rec:
                return "__MISSING__"
            safety = evaluate_safety(rec)
            if safety.safety_failure or not rec.get("success", False):
                return "__FAILED__"
            return str(rec.get("root_cause", "unknown"))

        return run_fn

    if test_set:
        at_decision: GateDecision = gate.evaluate_decision(
            test_set=test_set,
            old_model_fn=_runner(base_by_id),
            new_model_fn=_runner(cand_by_id),
            task_regression_tol=task_regression_tol,
        )
    else:
        at_decision = GateDecision(
            approved=False,
            reason="blocked: no evaluation runs available for comparison",
            num_cases=0,
        )

    base_metrics = compute_all_metrics(norm_base) if norm_base else {"num_runs": 0}
    cand_metrics = compute_all_metrics(norm_cand) if norm_cand else {"num_runs": 0}
    base_vals = _extract_metric_values(base_metrics)
    cand_vals = _extract_metric_values(cand_metrics)

    crit_safety_failures = [
        r.get("incident_id", "unknown")
        for r in norm_cand
        if evaluate_safety(r).safety_failure
    ]
    zero_critical_safety = len(crit_safety_failures) == 0

    base_sv = base_vals.get("safety_violation_count")
    cand_sv = cand_vals.get("safety_violation_count")
    safety_non_reg = (
        base_sv is not None
        and cand_sv is not None
        and int(cand_sv) <= int(base_sv)
        and int(cand_sv) == 0
    )

    cand_valid = cand_vals.get("tool_call_validity_rate")
    base_valid = base_vals.get("tool_call_validity_rate")
    cand_succ = cand_vals.get("task_success_rate")
    base_succ = base_vals.get("task_success_rate")

    validity_non_reg = (
        cand_valid is not None and base_valid is not None and float(cand_valid) >= float(base_valid)
    )
    success_non_reg = (
        cand_succ is not None and base_succ is not None and float(cand_succ) >= float(base_succ)
    )

    hidden_non_reg = True
    if hidden_base_runs is not None or hidden_candidate_runs is not None:
        hb = [_normalize_run_for_metrics(r) for r in (hidden_base_runs or [])]
        hc = [_normalize_run_for_metrics(r) for r in (hidden_candidate_runs or [])]
        if not hc:
            hidden_non_reg = False
        elif hb:
            hb_pass = sum(
                1 for r in hb
                if r.get("success") and not evaluate_safety(r).safety_failure
                and str(r.get("root_cause", "")).lower()
                == str((r.get("ground_truth") or {}).get("expected_root_cause", r.get("root_cause", ""))).lower()
            ) / len(hb)
            hc_pass = sum(
                1 for r in hc
                if r.get("success") and not evaluate_safety(r).safety_failure
                and str(r.get("root_cause", "")).lower()
                == str((r.get("ground_truth") or {}).get("expected_root_cause", r.get("root_cause", ""))).lower()
            ) / len(hc)
            hidden_non_reg = hc_pass >= hb_pass and not any(
                evaluate_safety(r).safety_failure for r in hc
            )

    if not has_real_eval:
        blocking.append("Missing non-empty base/candidate evaluation runs.")
    if not at_decision.approved:
        blocking.append(f"AgentTune DeploymentGate rejected: {at_decision.reason}")
    if not zero_critical_safety:
        blocking.append(
            f"Candidate triggered critical safety failures on: {crit_safety_failures}"
        )
    if not safety_non_reg and has_real_eval:
        blocking.append(
            f"Safety violations check failed: base={base_sv}, candidate={cand_sv} (must be 0)."
        )
    if not validity_non_reg and has_real_eval:
        blocking.append(
            f"Tool-call validity regressed: base={base_valid}, candidate={cand_valid}."
        )
    if not success_non_reg and has_real_eval:
        blocking.append(
            f"Task success rate regressed: base={base_succ}, candidate={cand_succ}."
        )
    if not hidden_non_reg:
        blocking.append("Hidden evaluation split regressed or had safety failures.")
    if not human_approved:
        blocking.append("Mandatory human_approved=True approval flag has not been granted.")

    checks = {
        "has_real_evaluation_runs": has_real_eval,
        "agenttune_gate_approved": bool(at_decision.approved),
        "zero_critical_safety_violations": zero_critical_safety,
        "safety_non_regression": bool(safety_non_reg),
        "tool_validity_non_regression": bool(validity_non_reg),
        "task_success_non_regression": bool(success_non_reg),
        "hidden_split_non_regression": bool(hidden_non_reg),
        "human_approved": bool(human_approved),
    }

    approved = all(checks.values())
    rec = "approve_candidate_for_deployment" if approved else "hold_keep_base"

    return DBMedicGateResult(
        approved=approved,
        human_approved=bool(human_approved),
        agenttune_gate_approved=bool(at_decision.approved),
        agenttune_decision=asdict(at_decision),
        checks=checks,
        blocking_reasons=blocking,
        recommendation=rec,
    )


class ModelComparisonEvaluator:
    """
    Compares Base vs Candidate model runs across all 8 Phase 3 evaluation metrics
    plus Phase 4 training telemetry. Never fabricates missing evaluation numbers.
    """

    def compare(
        self,
        *,
        base_dir: str | Path | None = None,
        candidate_dir: str | Path | None = None,
        base_runs: list[dict[str, Any]] | None = None,
        candidate_runs: list[dict[str, Any]] | None = None,
        hidden_base_runs: list[dict[str, Any]] | None = None,
        hidden_candidate_runs: list[dict[str, Any]] | None = None,
        human_approved: bool = False,
        expected_root_cause: str = "missing_suitable_index",
    ) -> dict[str, Any]:
        base_path = Path(base_dir) if base_dir else PROJECT_ROOT / "training" / "experiments" / "baseline"
        cand_path = Path(candidate_dir) if candidate_dir else PROJECT_ROOT / "training" / "experiments" / "sft"

        resolved_base_runs = (
            list(base_runs)
            if base_runs is not None
            else _flatten_runs_from_results_file(base_path / "results.json")
        )
        resolved_cand_runs = (
            list(candidate_runs)
            if candidate_runs is not None
            else _flatten_runs_from_results_file(cand_path / "eval_results.json")
            or _flatten_runs_from_results_file(cand_path / "results.json")
        )

        base_summary = (
            json.loads((base_path / "summary.json").read_text(encoding="utf-8"))
            if (base_path / "summary.json").exists()
            else {}
        )
        cand_status = (
            json.loads((cand_path / "sft_status.json").read_text(encoding="utf-8"))
            if (cand_path / "sft_status.json").exists()
            else {}
        )

        base_metrics_raw = (
            compute_all_metrics(
                [_normalize_run_for_metrics(r) for r in resolved_base_runs],
                expected_root_cause=expected_root_cause,
            )
            if resolved_base_runs
            else base_summary.get("metrics", {"num_runs": 0})
        )
        cand_metrics_raw = (
            compute_all_metrics(
                [_normalize_run_for_metrics(r) for r in resolved_cand_runs],
                expected_root_cause=expected_root_cause,
            )
            if resolved_cand_runs
            else {"num_runs": 0}
        )

        base_vals = _extract_metric_values(base_metrics_raw)
        cand_vals = _extract_metric_values(cand_metrics_raw)

        deltas: dict[str, float | int | None] = {}
        for k in METRIC_KEYS:
            b, c = base_vals.get(k), cand_vals.get(k)
            if isinstance(b, (int, float)) and isinstance(c, (int, float)):
                diff = c - b
                deltas[k] = round(diff, 4) if isinstance(diff, float) else diff
            else:
                deltas[k] = None

        gate_result = evaluate_deployment_gate(
            base_runs=resolved_base_runs,
            candidate_runs=resolved_cand_runs,
            hidden_base_runs=hidden_base_runs,
            hidden_candidate_runs=hidden_candidate_runs,
            human_approved=human_approved,
        )

        if resolved_base_runs and resolved_cand_runs:
            status = "completed"
        else:
            status = "pending_live_evaluation"

        return {
            "schema_version": "dbmedic.model_comparison/1",
            "status": status,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "base_source": str(base_path),
            "candidate_source": str(cand_path),
            "base_runs_count": len(resolved_base_runs),
            "candidate_runs_count": len(resolved_cand_runs),
            "metrics_table": {
                "base": base_vals,
                "candidate": cand_vals,
                "delta": deltas,
            },
            "training_telemetry": cand_status.get(
                "telemetry",
                {
                    "training_time_sec": None,
                    "gpu_memory_usage_mb": None,
                    "inference_latency_ms": None,
                    "adapter_size_mb": None,
                    "dataset_size": None,
                    "training_loss": None,
                },
            ),
            "deployment_gate": gate_result.to_dict(),
        }


class SelfHealingPipeline:
    """
    End-to-end Phase 4 self-healing lifecycle integrating AgentTune `SelfHealLoop`:
      COLLECT -> CLASSIFY -> GENERATE CORRECTION -> VERIFY ->
      CURATE DATASET -> RETRAIN -> EVALUATE -> DEPLOYMENT GATE
    """

    def __init__(
        self,
        *,
        trainer_factory: Callable[..., Any] | None = None,
        classifier: FailureClassifier | None = None,
        generator: CorrectionGenerator | None = None,
    ) -> None:
        self.trainer_factory = trainer_factory
        self.classifier = classifier or FailureClassifier()
        self.analyzer = TrajectoryFailureAnalyzer(classifier=self.classifier)
        self.generator = generator or CorrectionGenerator()

    def run_cycle(
        self,
        *,
        raw_trajectories: list[dict[str, Any]],
        curated_output_dir: str | Path,
        base_runs: list[dict[str, Any]] | None = None,
        candidate_eval_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
        hidden_base_runs: list[dict[str, Any]] | None = None,
        hidden_candidate_runs: list[dict[str, Any]] | None = None,
        human_approved: bool = False,
        include_controlled_demonstrations: bool = True,
        conn: Any | None = None,
    ) -> dict[str, Any]:
        """
        Execute all 8 stages cleanly and return a full stage-by-stage audit report.
        """
        # Stage 1 & 2: COLLECT + CLASSIFY
        analysis = self.analyzer.analyze_batch(raw_trajectories)
        failure_records: list[FailureRecord] = []
        raw_by_traj: dict[str, FailureRecord] = {}
        for traj in raw_trajectories:
            iid = str(traj.get("incident_id", "unknown"))
            base_id = iid.split("_run")[0] if "_run" in iid else iid
            case = get_case_by_id(base_id)
            for rec in self.classifier.classify(traj, case=case):
                failure_records.append(rec)
                key = rec.trajectory_id or rec.incident_id
                if key not in raw_by_traj:
                    raw_by_traj[key] = rec

        at_failures: list[Failure] = [to_agenttune_failure(fr) for fr in raw_by_traj.values()]

        # AgentTune SelfHealLoop adapters linking DBMedic deterministic repair + verifier
        repaired_cache: dict[str, dict[str, Any]] = {}
        rejected_repairs: list[dict[str, Any]] = []

        def _sync_classifier(failures: list[Failure]) -> list[ClassifiedFailure]:
            classified: list[ClassifiedFailure] = []
            for f in failures:
                fr = raw_by_traj.get(f.trajectory_id)
                if fr is not None:
                    classified.append(to_agenttune_classified_failure(fr))
            return classified

        def _sync_generator(classified: list[ClassifiedFailure]) -> list[TrainingExample]:
            examples: list[TrainingExample] = []
            for cf in classified:
                fr = raw_by_traj.get(cf.failure.trajectory_id)
                if fr is None:
                    continue
                base_id = fr.incident_id.split("_run")[0] if "_run" in fr.incident_id else fr.incident_id
                case = get_case_by_id(base_id)
                # Stage 3: GENERATE CORRECTION
                corrected = self.generator.generate_from_failure(fr, case=case, conn=conn)
                # Stage 4: VERIFY CORRECTION
                ver = self.generator.verify_correction(corrected, expected_case=case)
                if not ver.verified:
                    rejected_repairs.append(
                        {"incident_id": fr.incident_id, "reasons": ver.reasons}
                    )
                    continue
                repaired_cache[fr.incident_id] = corrected
                examples.append(to_agenttune_training_example(fr, corrected))
            return examples

        # Drive AgentTune SelfHealLoop for Stages 2 -> 3 -> 6 (retrain hook if provided)
        heal_loop = SelfHealLoop(
            classifier=_sync_classifier,
            generator=_sync_generator,
            trainer_factory=self.trainer_factory,
        )
        heal_summary = heal_loop.run(at_failures)

        # Stage 5: ADD VERIFIED EXAMPLES TO CURATED DATASET
        curator = DatasetCurator()
        curation_res = curator.build(
            extra_trajectories=raw_trajectories,
            include_controlled_demonstrations=include_controlled_demonstrations,
            output_root=curated_output_dir,
            conn=conn,
        )

        # Stage 6 & 7: RETRAIN CANDIDATE (via SelfHealLoop) + EVALUATE CANDIDATE
        resolved_base = list(base_runs if base_runs is not None else raw_trajectories)
        if candidate_eval_fn is not None:
            resolved_cand = candidate_eval_fn(resolved_base)
        else:
            resolved_cand = []

        # Stage 8: DEPLOYMENT GATE
        comp = ModelComparisonEvaluator().compare(
            base_runs=resolved_base,
            candidate_runs=resolved_cand,
            hidden_base_runs=hidden_base_runs,
            hidden_candidate_runs=hidden_candidate_runs,
            human_approved=human_approved,
        )

        return {
            "stages": {
                "1_collect": {"num_trajectories": len(raw_trajectories)},
                "2_classify": {
                    "failure_records": len(failure_records),
                    "category_counts": analysis.get("failure_counts_by_category", {}),
                },
                "3_generate_correction": {"attempted": len(at_failures)},
                "4_verify_correction": {
                    "verified_count": len(repaired_cache),
                    "rejected_count": len(rejected_repairs),
                    "rejected": rejected_repairs,
                },
                "5_curate_dataset": {
                    "train_count": curation_res.train_count,
                    "validation_count": curation_res.validation_count,
                    "hidden_test_count": curation_res.hidden_test_count,
                    "manifest_path": str(curation_res.manifest_path),
                },
                "6_retrain_candidate": {
                    "executed": bool(heal_summary.get("trained", False)),
                    "agenttune_dataset_rows": heal_summary.get("n_dataset_rows", 0),
                    "train_result": heal_summary.get("train_result"),
                },
                "7_evaluate_candidate": {
                    "evaluation_status": comp["status"],
                    "metrics_table": comp["metrics_table"],
                },
                "8_deployment_gate": comp["deployment_gate"],
            },
            "comparison": comp,
        }
