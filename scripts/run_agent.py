"""
scripts/run_agent.py
=====================
CLI entry point for the DBMedic autonomous diagnostic agent.

Usage:
    python scripts/run_agent.py --query "Why is my orders query slow?"
    python scripts/run_agent.py --incident slow_query_001
    python scripts/run_agent.py --evaluate --max-cases 3
    python scripts/run_agent.py --list-incidents

Options:
    --query TEXT         Run agent on a free-text database question
    --incident ID        Run agent on a known incident ID
    --evaluate           Run baseline evaluation on all incident cases
    --max-cases N        Maximum cases to evaluate (for quick test)
    --model-id ID        Hugging Face model ID (default: CohereLabs/tiny-aya-global)
    --max-steps N        Maximum agent steps (default: 10)
    --max-tools N        Maximum tool calls per run (default: 5)
    --no-log             Do not log trajectory to JSONL
    --list-incidents     List all known incident IDs
    --verbose            Enable debug logging

Security:
    HF_TOKEN must be set in .env or environment for gated model access.
    Never pass credentials on the command line.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy transformers logs unless verbose
    if not verbose:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("accelerate").setLevel(logging.WARNING)


def print_separator(title: str = "") -> None:
    width = 65
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("=" * width)


def print_result(result: Any, verbose: bool = False) -> None:
    """Print a formatted agent result to stdout."""
    print_separator("DBMedic Agent Result")

    status = "✓ SUCCESS" if result.success else "✗ FAILED"
    print(f"Status        : {status}")
    print(f"Incident ID   : {result.incident_id}")
    print(f"Model         : {result.model_id}")
    print(f"Steps         : {result.num_steps}")
    print(f"Time          : {result.total_time_ms:.0f} ms")
    print()

    print_separator("Selected Tools")
    if result.tools_called:
        for i, tool in enumerate(result.tools_called, 1):
            print(f"  {i}. {tool}")
    else:
        print("  (no tools called)")
    print()

    print_separator("Diagnosis")
    print(f"  {result.diagnosis}")
    print()
    print(f"  Root Cause   : {result.root_cause}")
    print(f"  Confidence   : {result.confidence:.2f}")
    print()

    print_separator("Evidence")
    if result.evidence:
        for i, ev in enumerate(result.evidence, 1):
            print(f"  {i}. {ev}")
    else:
        print("  (no evidence recorded)")
    print()

    print_separator("Recommended Action")
    print(f"  {result.recommended_action or 'None'}")
    if result.requires_approval:
        print("  ⚠  Human approval required before execution")
    print()

    print_separator("Safety Status")
    if result.safety_violations:
        print(f"  ⚠  {len(result.safety_violations)} safety violation(s):")
        for v in result.safety_violations:
            print(f"     - {v}")
    else:
        print("  ✓  No safety violations")
    print()

    if result.error:
        print_separator("Error")
        print(f"  {result.error}")
        print()

    print_separator()


def cmd_run_query(args: argparse.Namespace) -> int:
    """Run agent on a free-text query."""
    from agent.core.agent import DBMedicAgent

    agent = DBMedicAgent(
        model_id=args.model_id,
        max_steps=args.max_steps,
        max_tool_calls=args.max_tools,
    )

    print(f"\nInitializing DBMedic agent (model: {args.model_id})...")
    print("Note: First run requires downloading model weights (~2–4 GB).")
    print()

    result = agent.run(
        query=args.query,
        log_trajectory=not args.no_log,
    )

    print_result(result, verbose=args.verbose)
    agent.unload()

    return 0 if result.success else 1


def cmd_run_incident(args: argparse.Namespace) -> int:
    """Run agent on a known incident ID."""
    from agent.core.agent import DBMedicAgent, KNOWN_INCIDENTS

    if args.incident not in KNOWN_INCIDENTS:
        print(f"Error: Unknown incident '{args.incident}'")
        print(f"Available incidents: {list(KNOWN_INCIDENTS.keys())}")
        return 1

    agent = DBMedicAgent(
        model_id=args.model_id,
        max_steps=args.max_steps,
        max_tool_calls=args.max_tools,
    )

    print(f"\nInitializing DBMedic agent (model: {args.model_id})...")
    incident = KNOWN_INCIDENTS[args.incident]
    print(f"Incident: {args.incident} — {incident['description']}")
    print()

    result = agent.run_incident(
        incident_id=args.incident,
        log_trajectory=not args.no_log,
    )

    print_result(result, verbose=args.verbose)
    agent.unload()

    return 0 if result.success else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run baseline evaluation."""
    from agent.core.agent import DBMedicAgent
    from agent.evaluation.evaluator import BaselineEvaluator
    from agent.evaluation.incident_cases import BASELINE_EVAL_CASES

    agent = DBMedicAgent(
        model_id=args.model_id,
        max_steps=args.max_steps,
        max_tool_calls=args.max_tools,
    )

    print(f"\nInitializing DBMedic agent (model: {args.model_id})...")
    agent.initialize()

    max_cases = args.max_cases if hasattr(args, "max_cases") else None
    cases = BASELINE_EVAL_CASES
    if max_cases:
        cases = cases[:max_cases]

    print(f"\nRunning baseline evaluation on {len(cases)} cases...")
    print("Results will be saved to: benchmarks/baseline/")
    print()

    evaluator = BaselineEvaluator(agent=agent)
    summary = evaluator.run_baseline(cases=cases, max_cases=max_cases)

    agent.unload()
    return 0


def cmd_list_incidents() -> int:
    """List all known incident IDs."""
    from agent.core.agent import KNOWN_INCIDENTS
    from agent.evaluation.incident_cases import ALL_CASES

    print_separator("Known Incident IDs (agent.py)")
    for iid, info in KNOWN_INCIDENTS.items():
        print(f"  {iid}: {info['description']}")
    print()

    print_separator("Evaluation Dataset (incident_cases.py)")
    from agent.evaluation.incident_cases import BASELINE_EVAL_CASES, TRAINING_CANDIDATES, HIDDEN_EVAL_CASES
    print(f"  Baseline eval cases  : {len(BASELINE_EVAL_CASES)}")
    print(f"  Training candidates  : {len(TRAINING_CANDIDATES)}")
    print(f"  Hidden eval cases    : {len(HIDDEN_EVAL_CASES)} (answers not shown)")
    print()
    for case in BASELINE_EVAL_CASES:
        print(f"  [{case.split}] {case.incident_id}: {case.query[:60]}...")
    return 0


# ---------------------------------------------------------------------------
# Type annotation import (avoid circular in print_result)
# ---------------------------------------------------------------------------
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DBMedic — AI-powered PostgreSQL diagnostic agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", metavar="TEXT", help="Run agent on a free-text question")
    mode.add_argument("--incident", metavar="ID", help="Run agent on a known incident ID")
    mode.add_argument("--evaluate", action="store_true", help="Run baseline evaluation")
    mode.add_argument("--list-incidents", action="store_true", help="List available incident IDs")

    parser.add_argument(
        "--model-id", default="CohereLabs/tiny-aya-global",
        help="Hugging Face model ID (default: CohereLabs/tiny-aya-global)"
    )
    parser.add_argument("--max-steps", type=int, default=10, help="Max agent steps (default: 10)")
    parser.add_argument("--max-tools", type=int, default=5, help="Max tool calls per run (default: 5)")
    parser.add_argument("--max-cases", type=int, default=None, help="Max evaluation cases to run")
    parser.add_argument("--no-log", action="store_true", help="Disable trajectory logging")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    if args.list_incidents:
        return cmd_list_incidents()
    elif args.query:
        return cmd_run_query(args)
    elif args.incident:
        return cmd_run_incident(args)
    elif args.evaluate:
        return cmd_evaluate(args)
    else:
        print("No command specified. Use --help for usage.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

