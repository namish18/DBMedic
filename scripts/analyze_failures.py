"""
scripts/analyze_failures.py
===========================
Run DBMedic's deterministic 10-category failure classifier on JSONL trajectories
and emit structured analysis + failure records under `training/data/raw/`.

Usage:
    python scripts/analyze_failures.py
    python scripts/analyze_failures.py --input training/data/raw/sample_controlled_trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.failure_analysis.analyzer import TrajectoryFailureAnalyzer


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze DBMedic agent trajectory failures")
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSONL trajectory path (defaults to existing real or controlled sample file)",
    )
    parser.add_argument(
        "--output",
        default="training/data/raw/failure_analysis_summary.json",
        help="Output JSON summary path",
    )
    args = parser.parse_args()

    if args.input:
        in_path = Path(args.input)
        if not in_path.is_absolute():
            in_path = PROJECT_ROOT / in_path
    else:
        candidates = [
            PROJECT_ROOT / "benchmarks" / "baseline" / "trajectories.jsonl",
            PROJECT_ROOT / "training" / "data" / "raw" / "live_agent_trajectories.jsonl",
            PROJECT_ROOT / "training" / "data" / "raw" / "sample_controlled_trajectories.jsonl",
        ]
        in_path = next((p for p in candidates if p.exists() and p.stat().st_size > 0), candidates[-1])

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    analyzer = TrajectoryFailureAnalyzer()
    report = analyzer.analyze_file(in_path, output_path=out_path)

    print("=" * 65)
    print("DBMedic Failure Analysis Summary")
    print("=" * 65)
    print(f"Source               : {report.get('source_path')}")
    print(f"Total trajectories   : {report.get('total_trajectories', 0)}")
    print(f"Passed trajectories  : {report.get('passed_trajectories', 0)}")
    print(f"Failed trajectories  : {report.get('failed_trajectories', 0)}")
    print(f"Safety failure runs  : {report.get('safety_failure_trajectories', 0)}")
    print(f"Total failure records: {report.get('total_failure_records', 0)}")
    print("\nFailure Counts by Category:")
    for cat, cnt in (report.get("failure_counts_by_category") or {}).items():
        print(f"  {cat:<30} {cnt}")
    print("=" * 65)
    print(f"Summary saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

