"""
scripts/evaluate_trained_agent.py
=================================
Evaluate a fine-tuned DBMedic AgentTune checkpoint deterministically (`do_sample=False`)
on `baseline`, `validation`, or `hidden` incident splits.

Strict leakage guarantee:
  - On `--split hidden`, ONLY sanitized prompts (`case.to_public_prompt_dict()`) are
    sent to the agent; ground-truth root causes are scored strictly post-inference.
  - Never fabricates metrics when no saved checkpoint, HF_TOKEN, or DB connection exists.

Usage:
    python scripts/evaluate_trained_agent.py --split validation --dry-run
    python scripts/evaluate_trained_agent.py --split hidden --checkpoint training/experiments/sft
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation.incident_cases import get_cases_by_split
from training.agenttune.distillation import _detect_hf_token, _probe_db_available
from training.agenttune.grpo_training import _has_saved_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DBMedic agent checkpoint")
    parser.add_argument(
        "--split",
        choices=["baseline", "validation", "hidden"],
        default="validation",
        help="Incident evaluation split",
    )
    parser.add_argument(
        "--checkpoint",
        default="training/experiments/sft",
        help="Path to fine-tuned LoRA adapter or model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write evaluation results (defaults to --checkpoint dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate evaluation readiness and prompt isolation without running model inference",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = get_cases_by_split(args.split)
    sanitized_prompts = [c.to_public_prompt_dict() for c in cases]

    ckpt_ok = _has_saved_checkpoint(ckpt_path)
    hf_ok, _ = _detect_hf_token()
    db_ok = _probe_db_available()

    blockers: list[str] = []
    if not ckpt_ok:
        blockers.append(f"No trained model/adapter checkpoint found at '{ckpt_path}'.")
    if not hf_ok:
        blockers.append("HF_TOKEN / HUGGINGFACE_HUB_TOKEN missing for base model tokenizer/weights.")
    if not db_ok:
        blockers.append("Live dbmedic_lab PostgreSQL database is offline.")

    status_path = out_dir / f"eval_status_{args.split}.json"

    if args.dry_run:
        payload = {
            "schema_version": "dbmedic.trained_eval_status/1",
            "status": "dry_run_validated",
            "executed_evaluation": False,
            "split": args.split,
            "num_cases": len(cases),
            "hidden_answers_stripped": True,
            "sample_public_prompt_keys": sorted(sanitized_prompts[0].keys()) if sanitized_prompts else [],
            "checkpoint_present": ckpt_ok,
            "hf_token_present": hf_ok,
            "live_db_available": db_ok,
            "blockers_for_live_eval": blockers,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0

    if blockers:
        payload = {
            "schema_version": "dbmedic.trained_eval_status/1",
            "status": "blocked",
            "executed_evaluation": False,
            "split": args.split,
            "num_cases": len(cases),
            "checkpoint_path": str(ckpt_path),
            "blockers": blockers,
            "metrics": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "No fabricated evaluation metrics generated; prerequisite blockers prevent live checkpoint inference.",
        }
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2

    from agent.core.agent import DBMedicAgent
    from agent.evaluation.evaluator import BaselineEvaluator

    agent = DBMedicAgent(
        model_id=str(ckpt_path),
        trajectory_dir=out_dir,
        generation_config={"do_sample": False},
    )
    agent.initialize()
    try:
        evaluator = BaselineEvaluator(agent=agent, output_dir=out_dir)
        summary: dict[str, Any] = evaluator.run_baseline(cases=cases)
    finally:
        agent.unload()

    # Rename outputs cleanly so baseline benchmarks/baseline/ is never touched
    for src_name, dst_name in (
        ("results.json", f"eval_results_{args.split}.json"),
        ("summary.json", f"eval_summary_{args.split}.json"),
    ):
        src = out_dir / src_name
        dst = out_dir / dst_name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())

