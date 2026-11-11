"""
scripts/train_agent.py
======================
CLI entry point for DBMedic Phase 4 AgentTune training orchestration.

Modes:
  preflight : Audit dataset integrity, HF authentication, GPU/RAM hardware, and DB readiness.
  sft       : Primary AgentTune SFT distillation on Tiny Aya Global.
  grpo      : Secondary/experimental agentic GRPO reinforcement learning (strictly gated).

Exit codes:
  0 : preflight passed / dry-run validated / live training completed
  1 : configuration or validation failure
  2 : live training honestly blocked by unmet environment/gating prerequisites

Usage:
    python scripts/train_agent.py --mode preflight
    python scripts/train_agent.py --mode sft --dry-run
    python scripts/train_agent.py --mode sft
    python scripts/train_agent.py --mode grpo --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.agenttune.trainer import DBMedicTrainingPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DBMedic Phase 4 AgentTune training stages")
    parser.add_argument(
        "--mode",
        choices=["preflight", "sft", "grpo"],
        required=True,
        help="Pipeline stage to validate or run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, AgentTune Project projection, and LoRA setup without training",
    )
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="Allow CPU execution ONLY for tiny local smoke-test models",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override base model ID or local model directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override experiment output directory",
    )
    parser.add_argument(
        "--no-db-check",
        action="store_true",
        help="Skip PostgreSQL reachability check during offline preflight",
    )
    args = parser.parse_args()

    pipeline = DBMedicTrainingPipeline()

    if args.mode == "preflight":
        report = pipeline.preflight(check_db=not args.no_db_check)
        preflight_path = PROJECT_ROOT / "training" / "experiments" / "preflight_status.json"
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["dataset_validation"]["passed"] else 1

    if args.mode == "sft":
        res = pipeline.run_sft(
            output_dir=args.output_dir,
            model_id=args.model_id,
            dry_run=args.dry_run,
            allow_cpu_smoke=args.allow_cpu_smoke,
            check_db=not args.no_db_check,
        )
        print(json.dumps(res, indent=2))
        st = res.get("status")
        if st in ("completed", "dry_run_validated"):
            return 0
        if st == "blocked":
            return 2
        return 1

    if args.mode == "grpo":
        res = pipeline.run_grpo(
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            check_db=not args.no_db_check,
        )
        print(json.dumps(res, indent=2))
        st = res.get("status")
        if st in ("completed", "dry_run_validated"):
            return 0
        if st == "gated_blocked":
            return 2
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())

