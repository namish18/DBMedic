"""
scripts/run_auditkit_evaluation.py
=====================================
CLI for AuditKIT evaluation of DBMedic.

Modes:
  --mode recorded   Offline evaluation using recorded trajectories (no model needed)
  --mode live       Live model inference (requires HF_TOKEN + CUDA + PostgreSQL)
    --model base       Evaluate CohereLabs/tiny-aya-global
    --model candidate  Evaluate fine-tuned checkpoint

Usage:
  python scripts/run_auditkit_evaluation.py --mode recorded
  python scripts/run_auditkit_evaluation.py --mode live --model base
  python scripts/run_auditkit_evaluation.py --mode live --model candidate
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
logger = logging.getLogger("run_auditkit_evaluation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run AuditKIT evaluation for DBMedic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["recorded", "live"],
        required=True,
        help="Evaluation mode: 'recorded' (offline) or 'live' (requires GPU + HF_TOKEN).",
    )
    p.add_argument(
        "--model",
        choices=["base", "candidate"],
        default="base",
        help="Model to evaluate in live mode (ignored for recorded mode).",
    )
    p.add_argument(
        "--trajectory-path",
        default=None,
        help="Path to trajectories.jsonl (recorded mode only).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for evaluation results.",
    )
    p.add_argument(
        "--split",
        default="baseline",
        help="Dataset split label (default: baseline).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from agent.audit.auditkit_adapter import AUDITKIT_AVAILABLE, AUDITKIT_VERSION
    from agent.evaluation.auditkit_evaluator import AuditKITEvaluator
    from agent.audit.report_generator import ReportGenerator

    logger.info("=" * 60)
    logger.info("DBMedic Phase 5 — AuditKIT Evaluation")
    logger.info("AuditKIT version  : %s", AUDITKIT_VERSION)
    logger.info("AuditKIT available: %s", AUDITKIT_AVAILABLE)
    logger.info("Mode              : %s", args.mode)
    if args.mode == "live":
        logger.info("Model             : %s", args.model)
    logger.info("=" * 60)

    evaluator = AuditKITEvaluator()
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "benchmarks" / "auditkit"
    )

    if args.mode == "recorded":
        trajectory_path = args.trajectory_path
        result = evaluator.evaluate_recorded(
            trajectory_path=trajectory_path,
            split=args.split,
        )
        split_dir = output_dir / "baseline"
    else:
        result = evaluator.evaluate_live(model=args.model)
        split_dir = output_dir / ("baseline" if args.model == "base" else "candidate")

    split_dir.mkdir(parents=True, exist_ok=True)
    result_path = split_dir / f"auditkit_evaluation_{args.mode}_{args.model if args.mode == 'live' else 'recorded'}.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    logger.info("Evaluation result written to %s", result_path)

    # Print summary
    print("\n--- Evaluation Summary ---")
    print(f"Mode        : {args.mode}")
    print(f"Episodes    : {result.get('num_episodes', 'N/A')}")
    print(f"Status      : {result.get('status', 'completed')}")
    if result.get("blockers"):
        print("BLOCKERS:")
        for b in result["blockers"]:
            print(f"  ✗ {b}")
    if result.get("metrics"):
        print("Metrics:")
        for k, v in result["metrics"].items():
            if isinstance(v, dict):
                val = v.get("value", "N/A")
                print(f"  {k}: {val}")
            else:
                print(f"  {k}: {v}")
    print(f"\nFull result: {result_path}")

    # Exit with error code if blocked
    if result.get("status") == "blocked":
        logger.error("Evaluation BLOCKED — see blockers above.")
        sys.exit(2)


if __name__ == "__main__":
    main()

