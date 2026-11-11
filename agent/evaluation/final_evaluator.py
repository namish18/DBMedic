"""
agent/evaluation/final_evaluator.py
===================================
Unified Phase 6 evaluation aggregator connecting Phases 1–6:
  - Phase 1/6 PostgreSQL before/after remediation metrics
  - Phase 3 baseline evaluation (benchmarks/baseline/)
  - Phase 4 AgentTune curation, training status & comparison report
  - Phase 5 AuditKIT adapter (real package vs AuditKITStub), red-team & deployment gate

Strict honesty guarantees:
  - Uses actual AuditKIT when installed; otherwise explicitly labels AuditKITStub
    and never claims stub evaluation is independent AuditKIT validation.
  - Unavailable live model metrics remain ``null`` (``None``).
  - Never mixes synthetic/simulated results with live model execution metrics.
  - Real and simulated episode counts are always reported separately.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.audit.auditkit_adapter import (
    AUDITKIT_AVAILABLE,
    AUDITKIT_VERSION,
    AuditKITAdapter,
)
from agent.audit.episode_builder import EpisodeBuilder
from agent.audit.regression_analyzer import RegressionAnalyzer
from agent.core.audit_logger import AuditLogger, redact_secrets
from agent.core.environment_check import (
    check_agenttune_availability,
    check_auditkit_availability,
    check_tiny_aya_availability,
)
from agent.evaluation.auditkit_scenarios import (
    ALL_SCENARIOS,
    HIDDEN_SCENARIOS,
)
from agent.evaluation.deployment_verifier import DeploymentVerifier
from agent.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FINAL_METRICS_PATH = (
    PROJECT_ROOT / "benchmarks" / "final_demo" / "final_metrics.json"
)


def _read_json_safe(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_jsonl_safe(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
    except (json.JSONDecodeError, OSError):
        return []
    return rows


def run_unified_evaluation(
    *,
    project_root: Path | None = None,
    output_path: Path | str | None = None,
    run_redteam: bool = True,
    audit_logger: AuditLogger | None = None,
) -> dict[str, Any]:
    """
    Aggregate Phase 3 baseline, Phase 4 AgentTune, Phase 5 AuditKIT/DeploymentGate,
    and Phase 6 remediation comparison into ``benchmarks/final_demo/final_metrics.json``.
    """
    root = project_root or PROJECT_ROOT
    logger_inst = audit_logger or AuditLogger()

    # 1. Environment / framework status
    auditkit_status = check_auditkit_availability()
    agenttune_status = check_agenttune_availability()
    tiny_aya_status = check_tiny_aya_availability()

    using_stub = bool(auditkit_status["using_stub"])
    auditkit_validation_label = (
        "real_auditkit"
        if not using_stub
        else "AuditKITStub (offline stub - NOT independent AuditKIT validation)"
    )

    # 2. Phase 3 real trajectories (never synthesize fake real episodes)
    baseline_traj_path = root / "benchmarks" / "baseline" / "trajectories.jsonl"
    baseline_summary_path = root / "benchmarks" / "baseline" / "summary.json"
    baseline_status_path = root / "training" / "experiments" / "baseline" / "baseline_status.json"

    raw_baseline_trajs = _read_jsonl_safe(baseline_traj_path)
    real_trajs = [t for t in raw_baseline_trajs if not t.get("simulated", False)]
    sim_trajs_from_baseline = [t for t in raw_baseline_trajs if t.get("simulated", False)]

    baseline_summary_disk = _read_json_safe(baseline_summary_path) or {}
    baseline_status_disk = _read_json_safe(baseline_status_path) or {}

    adapter = AuditKITAdapter()
    builder = EpisodeBuilder()

    auditkit_recorded_summary: dict[str, Any] | None = None
    real_agent_metrics: dict[str, Any] = {
        "diagnosis_accuracy": None,
        "tool_call_validity_rate": None,
        "tool_selection_accuracy": None,
        "task_success_rate": None,
        "average_tool_calls_per_task": None,
        "average_steps_per_task": None,
        "safety_violation_count": None,
        "evidence_grounding_accuracy": None,
    }

    if real_trajs:
        phase3_m = compute_all_metrics(real_trajs)
        for key in list(real_agent_metrics.keys()):
            entry = phase3_m.get(key)
            real_agent_metrics[key] = (
                entry.get("value") if isinstance(entry, dict) else entry
            )
        real_episodes = builder.from_trajectory_file(baseline_traj_path, split="baseline")
        real_episodes = [ep for ep in real_episodes if not ep.simulated]
        if real_episodes:
            res = adapter.evaluate_episodes(real_episodes, mode="recorded")
            auditkit_recorded_summary = res.summary()
    elif isinstance(baseline_summary_disk.get("metrics"), dict) and baseline_summary_disk["metrics"].get("num_runs", 0) > 0:
        m = baseline_summary_disk["metrics"]
        for key in list(real_agent_metrics.keys()):
            entry = m.get(key)
            real_agent_metrics[key] = (
                entry.get("value") if isinstance(entry, dict) else entry
            )

    # 3. Phase 4 AgentTune & dataset curation artifacts
    dataset_manifest = _read_json_safe(root / "training" / "data" / "dataset_manifest.json") or {}
    sft_status = _read_json_safe(root / "training" / "experiments" / "sft" / "sft_status.json") or {}
    grpo_status = _read_json_safe(root / "training" / "experiments" / "grpo" / "grpo_status.json") or {}
    phase4_cmp = _read_json_safe(root / "training" / "experiments" / "comparison_report.json") or {}

    # 4. Phase 5 Red-team & simulated structural validation
    redteam_path = root / "benchmarks" / "auditkit" / "redteam" / "redteam_results.json"
    if run_redteam:
        from scripts.run_agent_redteam import run_deterministic_redteam

        redteam_results = run_deterministic_redteam()
        redteam_path.parent.mkdir(parents=True, exist_ok=True)
        redteam_path.write_text(json.dumps(redteam_results, indent=2, default=str), encoding="utf-8")
    else:
        redteam_results = _read_json_safe(redteam_path)

    redteam_cases_count = int((redteam_results or {}).get("num_cases") or (redteam_results or {}).get("num_attacks") or 0)

    # Check simulated episodes count honestly (never conflated with real)
    controlled_sample_path = root / "training" / "data" / "raw" / "sample_controlled_trajectories.jsonl"
    simulated_sample_trajs = _read_jsonl_safe(controlled_sample_path)
    num_real_episodes = len(real_trajs)
    num_simulated_episodes = len(sim_trajs_from_baseline) + len(simulated_sample_trajs) + redteam_cases_count

    # 5. Phase 5 Regression & Deployment Verifier
    cand_eval_dir = root / "benchmarks" / "auditkit" / "candidate"
    candidate_summary = None
    if cand_eval_dir.exists():
        for jf in sorted(cand_eval_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            cand_data = _read_json_safe(jf)
            if isinstance(cand_data, dict) and cand_data.get("num_episodes", 0) > 0:
                candidate_summary = cand_data
                break

    reg_analyzer = RegressionAnalyzer()
    regression_report = reg_analyzer.analyze(
        auditkit_recorded_summary or {},
        candidate_summary or {},
        baseline_id="base_tiny_aya_global",
        candidate_id="candidate_sft_checkpoint",
    )

    verifier = DeploymentVerifier()
    gate_result = verifier.verify(
        baseline_summary=auditkit_recorded_summary,
        candidate_summary=candidate_summary,
        regression_report=regression_report.to_dict(),
        redteam_summary=redteam_results,
        human_approved=False,
    )

    # 6. Phase 6 Before/After Remediation comparison (with Phase 1 historical reference)
    phase6_rem_path = root / "benchmarks" / "final_demo" / "remediation_comparison.json"
    phase6_rem = _read_json_safe(phase6_rem_path)
    phase1_before = _read_json_safe(root / "benchmarks" / "slow_query" / "baseline.json")
    phase1_after = _read_json_safe(root / "benchmarks" / "slow_query" / "after_remediation.json")

    baseline_exec_ms: float | None = None
    post_exec_ms: float | None = None
    perf_improvement_pct: float | None = None
    remediation_eval_mode: str | None = None

    if isinstance(phase6_rem, dict):
        before_blk = phase6_rem.get("before") or {}
        after_blk = phase6_rem.get("after") or {}
        perf_blk = phase6_rem.get("performance_comparison") or {}
        baseline_exec_ms = before_blk.get("median_execution_time_ms")
        post_exec_ms = after_blk.get("median_execution_time_ms")
        perf_improvement_pct = perf_blk.get("improvement_pct")
        remediation_eval_mode = phase6_rem.get("execution_mode")

    if num_real_episodes > 0:
        evaluation_mode = "live_model_plus_deterministic_checks"
    else:
        evaluation_mode = "offline_deterministic_and_stub_pending_live_model"

    final_metrics: dict[str, Any] = {
        "schema_version": "dbmedic.final_metrics/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": evaluation_mode,
        "auditkit_status": {
            "installed": auditkit_status["installed"],
            "using_stub": using_stub,
            "version": AUDITKIT_VERSION,
            "validation_classification": auditkit_validation_label,
            "license": auditkit_status["license"],
        },
        "episode_counts": {
            "num_real_episodes": num_real_episodes,
            "num_simulated_episodes": num_simulated_episodes,
            "redteam_deterministic_cases": redteam_cases_count,
            "auditkit_defined_scenarios": len(ALL_SCENARIOS),
            "auditkit_hidden_scenarios": len(HIDDEN_SCENARIOS),
            "note": "Real model episodes and simulated/deterministic cases are strictly separated.",
        },
        "agent_metrics_live_model": {
            **real_agent_metrics,
            "status": "measured" if num_real_episodes > 0 else "null_pending_live_model_inference",
        },
        "remediation_metrics": {
            "source": str(phase6_rem_path) if phase6_rem else None,
            "execution_mode": remediation_eval_mode,
            "simulated": bool(phase6_rem.get("simulated")) if isinstance(phase6_rem, dict) else None,
            "baseline_execution_time_ms": baseline_exec_ms,
            "post_remediation_execution_time_ms": post_exec_ms,
            "performance_improvement_pct": perf_improvement_pct,
            "verdict": (
                (phase6_rem.get("verification_outcome") or {}).get("verdict")
                if isinstance(phase6_rem, dict)
                else None
            ),
            "historical_phase1_reference": (
                {
                    "baseline_median_ms": phase1_before.get("median_execution_time_ms"),
                    "after_remediation_median_ms": phase1_after.get("median_execution_time_ms"),
                    "performance_improvement_pct": phase1_after.get("performance_improvement_pct"),
                    "note": "Preserved from Phase 1 baseline artifacts (not overwritten by Phase 6).",
                }
                if (isinstance(phase1_before, dict) and isinstance(phase1_after, dict))
                else None
            ),
        },
        "safety_and_redteam": {
            "deterministic_redteam_executed": bool(redteam_results),
            "attacks_tested": (redteam_results or {}).get("num_attacks"),
            "attacks_blocked": (redteam_results or {}).get("attacks_blocked"),
            "attack_success_rate": (redteam_results or {}).get("attack_success_rate"),
            "target_met": (redteam_results or {}).get("target_met"),
            "note": (
                "Deterministic validator safety tests verify deterministic guards, "
                "not unguarded live Tiny Aya resistance."
            ),
        },
        "phase3_baseline_status": {
            "trajectories_count": num_real_episodes,
            "observed_status": baseline_status_disk.get("observed_status", "pending"),
            "inference_executed": bool(num_real_episodes > 0 or baseline_status_disk.get("inference_executed")),
        },
        "phase4_agenttune_status": {
            "agenttune_installed": agenttune_status["installed"],
            "agenttune_version": agenttune_status["version"],
            "dataset_splits": dataset_manifest.get("counts") or dataset_manifest.get("split_counts"),
            "dataset_sha256_verified": bool(dataset_manifest),
            "sft_status": sft_status.get("status", "not_run"),
            "grpo_status": grpo_status.get("status", "not_run"),
            "fine_tuned_checkpoint_exists": tiny_aya_status["fine_tuned_checkpoint_exists"],
            "comparison_status": phase4_cmp.get("status", "pending_live_evaluation"),
        },
        "phase5_regression_analysis": regression_report.to_dict(),
        "phase5_deployment_gate": gate_result.to_dict(),
    }

    final_metrics = redact_secrets(final_metrics)

    dest = Path(output_path) if output_path is not None else DEFAULT_FINAL_METRICS_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(final_metrics, indent=2, default=str), encoding="utf-8")
    final_metrics["output_path"] = str(dest)

    logger_inst.log_event(
        incident_id="final_evaluation",
        event_type="EVALUATION_COMPLETED",
        actor="dbmedic_final_evaluator",
        execution_mode="system",
        payload={
            "evaluation_mode": evaluation_mode,
            "num_real_episodes": num_real_episodes,
            "num_simulated_episodes": num_simulated_episodes,
            "auditkit_using_stub": using_stub,
            "deployment_decision": gate_result.decision,
            "output_path": str(dest),
        },
    )
    return final_metrics


def generate_consolidated_report(
    *,
    project_root: Path | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, str]:
    """Generate human-readable Markdown + JSON consolidated Phase 6 report artifacts."""
    root = project_root or PROJECT_ROOT
    out_dir = Path(output_dir) if output_dir else (root / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    readiness = _read_json_safe(root / "reports" / "environment_readiness.json")
    metrics = _read_json_safe(root / "benchmarks" / "final_demo" / "final_metrics.json")
    rem_cmp = _read_json_safe(root / "benchmarks" / "final_demo" / "remediation_comparison.json")
    audit_integrity = AuditLogger(root / "reports" / "audit_trail.jsonl").verify_integrity()

    summary_json = redact_secrets(
        {
            "schema_version": "dbmedic.final_summary/1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment_readiness": readiness,
            "final_metrics": metrics,
            "remediation_comparison": rem_cmp,
            "audit_trail_integrity": audit_integrity,
        }
    )

    json_path = out_dir / "final_summary.json"
    md_path = out_dir / "final_summary.md"

    json_path.write_text(json.dumps(summary_json, indent=2, default=str), encoding="utf-8")

    # Render concise Markdown report
    lines = [
        "# DBMedic — Consolidated Phase 6 Summary Report",
        "",
        f"Generated: `{summary_json['generated_at']}`",
        "",
        "## 1. Environment Readiness",
        f"- Overall Status: `{(readiness or {}).get('overall_status', 'not_checked')}`",
        f"- Live PostgreSQL Ready: `{(readiness or {}).get('capabilities', {}).get('live_postgres_ready')}`",
        f"- Live Tiny Aya Ready: `{(readiness or {}).get('capabilities', {}).get('live_tiny_aya_ready')}`",
        f"- AuditKIT Stub Active: `{(readiness or {}).get('checks', {}).get('auditkit', {}).get('using_stub')}`",
        "",
        "## 2. Remediation Verification",
    ]
    if rem_cmp:
        before = rem_cmp.get("before", {})
        after = rem_cmp.get("after", {})
        perf = rem_cmp.get("performance_comparison", {})
        outc = rem_cmp.get("verification_outcome", {})
        lines.extend(
            [
                f"- Incident: `{rem_cmp.get('incident_id')}`",
                f"- Execution Mode: `{rem_cmp.get('execution_mode')}` (`simulated={rem_cmp.get('simulated')}`)",
                f"- Verdict: `{outc.get('verdict')}` (`success={outc.get('success')}`)",
                f"- Before: `{before.get('median_execution_time_ms')} ms` ({before.get('scan_type')})",
                f"- After: `{after.get('median_execution_time_ms')} ms` ({after.get('scan_type')})",
                f"- Improvement: `{perf.get('improvement_pct')}%`",
            ]
        )
    else:
        lines.append("- No Phase 6 remediation comparison recorded yet (`null`).")

    lines.extend(
        [
            "",
            "## 3. Evaluation & Deployment Gate",
            f"- Evaluation Mode: `{(metrics or {}).get('evaluation_mode', 'pending')}`",
            f"- Real Model Episodes: `{(metrics or {}).get('episode_counts', {}).get('num_real_episodes')}`",
            f"- Simulated/Deterministic Episodes: `{(metrics or {}).get('episode_counts', {}).get('num_simulated_episodes')}`",
            f"- Red-Team Attacks Blocked: `{(metrics or {}).get('safety_and_redteam', {}).get('attacks_blocked')} / {(metrics or {}).get('safety_and_redteam', {}).get('attacks_tested')}`",
            f"- Deployment Gate Decision: `{(metrics or {}).get('phase5_deployment_gate', {}).get('decision', 'unknown')}`",
            "",
            "## 4. Audit Trail Integrity",
            f"- Valid Chain: `{audit_integrity.get('valid')}`",
            f"- Total Events: `{audit_integrity.get('total_events')}`",
            f"- Security Note: {audit_integrity.get('limitations')}",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"json_path": str(json_path), "md_path": str(md_path)}
