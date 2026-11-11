"""
scripts/verify_deployment.py
==============================
Deployment verification for DBMedic Phase 5.

Integrates AuditKIT regression analysis with the Phase 4 deployment gate.

Usage:
  python scripts/verify_deployment.py
  python scripts/verify_deployment.py --human-approve

The deployment decision will remain 'hold_keep_base' until all real
evaluation data is available AND human approval is explicitly granted.
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
logger = logging.getLogger("verify_deployment")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify deployment readiness for DBMedic.")
    p.add_argument(
        "--human-approve",
        action="store_true",
        default=False,
        help="Grant human approval (only use after reviewing all results).",
    )
    p.add_argument(
        "--base-dir",
        default="benchmarks/auditkit/baseline",
    )
    p.add_argument(
        "--candidate-dir",
        default="benchmarks/auditkit/candidate",
    )
    p.add_argument(
        "--redteam-results",
        default="benchmarks/auditkit/redteam/redteam_results.json",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


def _load_latest(directory: Path) -> dict | None:
    if not directory.exists():
        return None
    jsons = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for jf in jsons:
        d = _load_json(jf)
        if d is not None:
            return d
    return None


def main() -> None:
    args = parse_args()

    base_dir = PROJECT_ROOT / args.base_dir
    cand_dir = PROJECT_ROOT / args.candidate_dir
    redteam_path = PROJECT_ROOT / args.redteam_results

    baseline = _load_latest(base_dir)
    candidate = _load_latest(cand_dir)
    redteam = _load_json(redteam_path)

    # Run regression analysis
    from agent.audit.regression_analyzer import RegressionAnalyzer
    from agent.evaluation.deployment_verifier import DeploymentVerifier

    analyzer = RegressionAnalyzer()
    regression_report = analyzer.analyze(
        baseline or {},
        candidate or {},
        baseline_id="base_tiny_aya_global",
        candidate_id="candidate_sft_checkpoint",
    )

    # Run deployment verification
    verifier = DeploymentVerifier()
    result = verifier.verify(
        baseline_summary=baseline,
        candidate_summary=candidate,
        regression_report=regression_report.to_dict(),
        redteam_summary=redteam,
        human_approved=args.human_approve,
    )

    # Save result
    saved_path = verifier.save_result(result)

    # Print summary
    print("\n" + "=" * 60)
    print("DBMedic Deployment Verification")
    print("=" * 60)
    print(f"Decision        : {result.decision}")
    print(f"Approved        : {result.approved}")
    print(f"Human approved  : {result.human_approved}")
    print()
    print("Gate checks:")
    for k, v in result.checks.items():
        status = "✓" if v is True else ("✗" if v is False else "?")
        print(f"  {status} {k}: {v}")

    if result.regressions:
        print("\nRegressions:")
        for r in result.regressions:
            print(f"  ⚠ {r}")

    if result.blockers:
        print("\nBlockers:")
        for b in result.blockers:
            print(f"  ✗ {b}")

    print(f"\nNote: {result.note}")
    print(f"Result saved to: {saved_path}")

    if not result.approved:
        logger.warning("Deployment NOT approved — decision: %s", result.decision)
        sys.exit(2 if result.decision == "hold_keep_base" else 1)
    else:
        logger.info("Deployment approved.")


if __name__ == "__main__":
    main()

