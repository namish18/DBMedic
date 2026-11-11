"""
scripts/validate_training_dataset.py
====================================
Audit curated DBMedic Phase 4 dataset splits for:
  - SHA-256 integrity against `dataset_manifest.json`
  - Zero scenario_group / ID / query leakage across train/val/hidden/baseline
  - Stripped hidden answers in `hidden_test.jsonl`
  - Tiny Aya strict alternating turns & completion-only masking
  - CorrectionGenerator verification & zero safety violations

Usage:
    python scripts/validate_training_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.data.validator import validate_dataset_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate curated DBMedic Phase 4 dataset splits")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset directory root (default: training/data)",
    )
    args = parser.parse_args()

    report = validate_dataset_artifacts(args.data_dir)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

