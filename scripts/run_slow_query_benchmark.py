"""
scripts/run_slow_query_benchmark.py
=====================================
Run the slow-query benchmark for the orders.customer_id incident, optionally
apply the approved remediation, and compare before/after results.

Usage — baseline only:
    python scripts/run_slow_query_benchmark.py

Usage — with remediation (requires explicit --approve flag):
    python scripts/run_slow_query_benchmark.py --remediate --approve

Usage — compare saved results:
    python scripts/run_slow_query_benchmark.py --compare

Flags:
  --remediate    After the baseline, also apply the index remediation.
  --approve      Provides explicit approval for the remediation DDL.
                 Required when --remediate is passed; refused otherwise.
  --compare      Load baseline.json and after_remediation.json and print a
                 side-by-side comparison (no database connection needed).
  --runs N       Number of benchmark iterations (default 7).
  --customer N   customer_id to query (default 4242).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from database.scenarios.slow_query import (
    BASELINE_FILE,
    REMEDIATION_FILE,
    TARGET_CUSTOMER_ID,
    compare_results,
    load_result,
    run_benchmark,
    save_result,
)
from agent.tools.query_analyzer import apply_remediation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_runner")


def _get_lab_url() -> str:
    url = os.getenv("LAB_DATABASE_URL")
    if not url:
        raise EnvironmentError("LAB_DATABASE_URL is not set. Check your .env file.")
    return url


def _print_result_summary(label: str, data: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Incident ID  : {data['incident_id']}")
    print(f"  Scan type    : {data['scan_type']}")
    print(f"  Median time  : {data['median_execution_time_ms']:.3f} ms")
    print(f"  Shared hits  : {data['shared_hit_blocks']}")
    print(f"  Shared reads : {data['shared_read_blocks']}")
    print(f"  Remediation  : {data['remediation_status']}")


def _print_comparison(comparison: dict) -> None:
    improved = comparison["performance_improved"]
    symbol = "✓" if improved else "✗"
    print(f"\n{'='*60}")
    print("  Before vs After Comparison")
    print(f"{'='*60}")
    print(f"  Baseline scan    : {comparison['baseline_scan']}")
    print(f"  After scan       : {comparison['after_scan']}")
    print(f"  Baseline median  : {comparison['baseline_median_ms']:.3f} ms")
    print(f"  After median     : {comparison['after_remediation_median_ms']:.3f} ms")
    print(f"  Improvement      : {comparison['improvement_ms']:.3f} ms  "
          f"({comparison['improvement_pct']:.1f}%)")
    print(f"  Performance OK?  : {symbol} {'YES' if improved else 'NO'}")


def run_baseline(lab_url: str, customer_id: int, n_runs: int) -> None:
    """Run baseline benchmark and save results."""
    logger.info("Running BASELINE benchmark …")
    with psycopg.connect(lab_url) as conn:
        result = run_benchmark(conn, customer_id=customer_id, n_runs=n_runs)

    save_result(result, BASELINE_FILE)
    _print_result_summary("BASELINE", result.to_dict())
    logger.info("Baseline saved → %s", BASELINE_FILE)


def run_remediation_and_benchmark(
    lab_url: str,
    customer_id: int,
    n_runs: int,
    approved: bool,
) -> None:
    """Apply the approved index remediation then rerun the benchmark."""
    logger.info("Applying remediation (approved=%s) …", approved)
    with psycopg.connect(lab_url) as conn:
        outcome = apply_remediation(
            conn,
            "create_idx_orders_customer_id",
            approved=approved,
        )
        logger.info("Remediation outcome: %s", outcome)

        logger.info("Running POST-REMEDIATION benchmark …")
        result = run_benchmark(
            conn,
            customer_id=customer_id,
            n_runs=n_runs,
            remediation_status="applied",
        )

    save_result(result, REMEDIATION_FILE)
    _print_result_summary("AFTER REMEDIATION", result.to_dict())
    logger.info("Post-remediation result saved → %s", REMEDIATION_FILE)


def compare_saved() -> None:
    """Load and compare previously saved benchmark files."""
    if not BASELINE_FILE.exists():
        logger.error("Baseline file not found: %s", BASELINE_FILE)
        sys.exit(1)
    if not REMEDIATION_FILE.exists():
        logger.error("Post-remediation file not found: %s", REMEDIATION_FILE)
        sys.exit(1)

    before = load_result(BASELINE_FILE)
    after = load_result(REMEDIATION_FILE)

    _print_result_summary("BASELINE", before)
    _print_result_summary("AFTER REMEDIATION", after)

    comparison = compare_results(before, after)
    _print_comparison(comparison)

    # Persist comparison inline into after_remediation.json for record-keeping
    after["performance_improvement_pct"] = comparison["improvement_pct"]
    with open(REMEDIATION_FILE, "w", encoding="utf-8") as fh:
        json.dump(after, fh, indent=2)
    logger.info("Comparison complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DBMedic slow-query benchmark"
    )
    parser.add_argument(
        "--remediate",
        action="store_true",
        help="Apply the index remediation after the baseline benchmark.",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Provide explicit approval for the remediation DDL (required with --remediate).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare saved baseline and post-remediation results (no DB needed).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=7,
        metavar="N",
        help="Number of benchmark iterations (default 7).",
    )
    parser.add_argument(
        "--customer",
        type=int,
        default=TARGET_CUSTOMER_ID,
        metavar="N",
        help=f"customer_id to query (default {TARGET_CUSTOMER_ID}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.compare:
        compare_saved()
        return

    if args.remediate and not args.approve:
        logger.error(
            "ERROR: --remediate requires --approve. "
            "Re-run with both flags to confirm you have reviewed the DDL."
        )
        sys.exit(1)

    lab_url = _get_lab_url()

    # Always run baseline first (unless only comparing)
    run_baseline(lab_url, customer_id=args.customer, n_runs=args.runs)

    if args.remediate:
        run_remediation_and_benchmark(
            lab_url,
            customer_id=args.customer,
            n_runs=args.runs,
            approved=args.approve,
        )
        compare_saved()


if __name__ == "__main__":
    main()

