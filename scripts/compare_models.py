"""
scripts/compare_models.py
=========================
Compare Base Tiny Aya vs Fine-Tuned Candidate across all 8 Phase 3 evaluation
metrics + training telemetry, and run AgentTune + DBMedic Deployment Gate.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --base-dir benchmarks/baseline --candidate-dir training/experiments/sft --dry-run
    python scripts/compare_models.py --human-approve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation.training_comparison import ModelComparisonEvaluator


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Base vs Trained DBMedic models")
    parser.add_argument(
        "--base-dir",
        default="training/experiments/baseline",
        help="Directory holding base model baseline results/summary",
    )
    parser.add_argument(
        "--candidate-dir",
        default="training/experiments/sft",
        help="Directory holding candidate model eval results & sft_status.json",
    )
    parser.add_argument(
        "--output",
        default="training/experiments/comparison_report.json",
        help="Output JSON comparison report path",
    )
    parser.add_argument(
        "--human-approve",
        action="store_true",
        help="Grant mandatory human_approved=True sign-off for deployment gate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate framework comparison report safely without failing on pending live evals",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir
    cand_dir = Path(args.candidate_dir)
    if not cand_dir.is_absolute():
        cand_dir = PROJECT_ROOT / cand_dir
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    report = ModelComparisonEvaluator().compare(
        base_dir=base_dir,
        candidate_dir=cand_dir,
        human_approved=args.human_approve,
    )
    report["dry_run"] = bool(args.dry_run)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 70)
    print("DBMedic Base vs Candidate Model Comparison")
    print("=" * 70)
    print(f"Status          : {report['status']}")
    print(f"Base runs       : {report['base_runs_count']}")
    print(f"Candidate runs  : {report['candidate_runs_count']}")
    print(f"Gate approved   : {report['deployment_gate']['approved']} ({report['deployment_gate']['recommendation']})")
    print("\nMetrics Table (Base -> Candidate | Delta):")
    base_t = report["metrics_table"]["base"]
    cand_t = report["metrics_table"]["candidate"]
    delta_t = report["metrics_table"]["delta"]
    for k in base_t:
        print(f"  {k:<32} base={base_t[k]}  cand={cand_t[k]}  delta={delta_t[k]}")
    print("=" * 70)
    print(f"Comparison report written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

