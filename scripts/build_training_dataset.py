"""
scripts/build_training_dataset.py
=================================
Build the verified, deduplicated, leakage-free Phase 4 AgentTune training splits
(`train.jsonl`, `validation.jsonl`, `hidden_test.jsonl`) and SHA-256 manifest.

Usage:
    python scripts/build_training_dataset.py
    python scripts/build_training_dataset.py --trajectory-files path1.jsonl path2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.data.curator import build_curated_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build curated DBMedic Phase 4 training dataset")
    parser.add_argument(
        "--trajectory-files",
        nargs="*",
        default=None,
        help="Optional trajectory JSONL files to classify, repair, and curate",
    )
    parser.add_argument(
        "--no-controlled-demos",
        action="store_true",
        help="Exclude controlled Phase 2 human-reviewed demonstrations",
    )
    parser.add_argument(
        "--no-sample-trajectories",
        action="store_true",
        help="Ignore sample_controlled_trajectories.jsonl if present",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override output root directory (defaults to training/data/)",
    )
    args = parser.parse_args()

    tfiles: list[Path] = []
    if args.trajectory_files:
        for f in args.trajectory_files:
            p = Path(f)
            tfiles.append(p if p.is_absolute() else PROJECT_ROOT / p)
    else:
        for default_p in (
            PROJECT_ROOT / "benchmarks" / "baseline" / "trajectories.jsonl",
            PROJECT_ROOT / "training" / "data" / "raw" / "live_agent_trajectories.jsonl",
        ):
            if default_p.exists() and default_p.stat().st_size > 0:
                tfiles.append(default_p)
        sample_p = PROJECT_ROOT / "training" / "data" / "raw" / "sample_controlled_trajectories.jsonl"
        if not args.no_sample_trajectories and sample_p.exists() and sample_p.stat().st_size > 0:
            tfiles.append(sample_p)

    res = build_curated_dataset(
        trajectory_files=tfiles,
        include_controlled_demonstrations=not args.no_controlled_demos,
        output_root=args.output_root,
    )

    print("=" * 65)
    print("DBMedic Phase 4 Dataset Curation Complete")
    print("=" * 65)
    print(f"Train examples       : {res.train_count}")
    print(f"Validation examples  : {res.validation_count}")
    print(f"Hidden test prompts  : {res.hidden_test_count} (answers stripped)")
    print(f"Rejected candidates  : {res.rejected_count}")
    print(f"Duplicates removed   : {res.duplicates_removed}")
    print(f"Provenance breakdown : {json.dumps(res.provenance_counts)}")
    print(f"Manifest path        : {res.manifest_path}")
    print("SHA-256 Hashes:")
    for k, v in (res.manifest.get("sha256") or {}).items():
        print(f"  {k:<20} {v}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())

