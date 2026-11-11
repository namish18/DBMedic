"""
scripts/compare_audit_results.py
===================================
Compare AuditKIT evaluation results between base and candidate models.

Integrates with Phase 4 ModelComparisonEvaluator and AuditKIT regression analysis.

Usage:
  python scripts/compare_audit_results.py
  python scripts/compare_audit_results.py --base-dir benchmarks/auditkit/baseline
                                           --candidate-dir benchmarks/auditkit/candidate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("compare_audit_results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare AuditKIT evaluation results.")
    p.add_argument("--base-dir", default="benchmarks/auditkit/baseline")
    p.add_argument("--candidate-dir", default="benchmarks/auditkit/candidate")
    p.add_argument("--output", default="benchmarks/auditkit/comparisons/comparison_report.json")
    return p.parse_args()


def _load_latest_result(directory: Path) -> dict | None:
    """Load the most recent evaluation result JSON from a directory."""
    if not directory.exists():
        return None
    jsons = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for jf in jsons:
        try:
            return json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def main() -> None:
    args = parse_args()

    base_dir = PROJECT_ROOT / args.base_dir
    cand_dir = PROJECT_ROOT / args.candidate_dir
    output_path = PROJECT_ROOT / args.output

    logger.info("Loading baseline results from %s", base_dir)
    baseline = _load_latest_result(base_dir)

    logger.info("Loading candidate results from %s", cand_dir)
    candidate = _load_latest_result(cand_dir)

    # Run regression analysis
    from agent.audit.regression_analyzer import RegressionAnalyzer
    from agent.audit.report_generator import ReportGenerator
    from agent.audit.auditkit_adapter import AUDITKIT_VERSION

    analyzer = RegressionAnalyzer()
    regression_report = analyzer.analyze(
        baseline or {},
        candidate or {},
        baseline_id="base_tiny_aya_global",
        candidate_id="candidate_sft_checkpoint",
    )

    # Print comparison table
    print("\n" + "=" * 70)
    print("DBMedic AuditKIT — Base vs. Candidate Comparison")
    print("=" * 70)

    metric_keys = [
        "tool_call_validity_rate",
        "tool_selection_accuracy",
        "task_success_rate",
        "average_tool_calls_per_task",
        "average_steps_per_task",
        "safety_violation_count",
        "evidence_grounding_accuracy",
        "diagnosis_accuracy",
        # Phase 5 extensions
        "redteam_attack_success_rate",
        "inference_latency_ms",
        "end_to_end_latency_ms",
        "tool_execution_failures",
        "evaluation_sample_count",
    ]

    def _get_val(result: dict | None, key: str) -> str:
        if result is None:
            return "null (pending)"
        v = result.get("metrics", result).get(key)
        if v is None:
            return "null (pending)"
        if isinstance(v, dict):
            val = v.get("value")
            return "null (pending)" if val is None else str(val)
        return str(v)

    print(f"{'Metric':<40} {'Base':>18} {'Candidate':>18}")
    print("-" * 78)
    for k in metric_keys:
        bv = _get_val(baseline, k)
        cv = _get_val(candidate, k)
        print(f"  {k:<38} {bv:>18} {cv:>18}")

    print("\nRegression Analysis:")
    print(f"  Overall verdict  : {regression_report.overall_verdict}")
    print(f"  Regressions      : {len(regression_report.regressions)}")
    print(f"  Improvements     : {len(regression_report.improvements)}")
    print(f"  Uncomputable     : {len(regression_report.uncomputable)}")

    if regression_report.regressions:
        print("\n  Detected regressions:")
        for r in regression_report.regressions:
            print(f"    ⚠ {r['metric']}: {r['baseline']} → {r['candidate']} (Δ={r['delta']:+.4f})")

    # Save comparison
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = {
        "base_model": "CohereLabs/tiny-aya-global",
        "candidate_model": "pending_sft_checkpoint",
        "baseline_summary": baseline,
        "candidate_summary": candidate,
        "regression_report": regression_report.to_dict(),
    }
    output_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    logger.info("Comparison saved to %s", output_path)

    # Generate full report
    gen = ReportGenerator()
    report_paths = gen.generate(
        auditkit_version=AUDITKIT_VERSION,
        base_model_id="CohereLabs/tiny-aya-global",
        base_model_revision=None,
        candidate_model_id=None,
        candidate_model_revision=None,
        baseline_results=baseline,
        candidate_results=candidate,
        redteam_results=None,
        performance_results=None,
        regression_report=regression_report.to_dict(),
        deployment_gate=None,
    )
    print(f"\nFull report  : {report_paths['json_path']}")
    print(f"Markdown     : {report_paths['md_path']}")


if __name__ == "__main__":
    main()

