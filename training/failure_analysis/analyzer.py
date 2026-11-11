"""
training/failure_analysis/analyzer.py
=====================================
Batch analyzer over collected DBMedic JSONL trajectories.

Aggregates failure category distributions, pairs trajectories with incident
metadata, optionally invokes AgentTune's FailureDetector for structural loop
audit signals, and exports machine-readable analysis summaries.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from agent.evaluation.incident_cases import IncidentCase, get_case_by_id
from agent.evaluation.trajectory_logger import _redact_secrets
from training.failure_analysis.classifier import (
    FAILURE_CATEGORIES,
    FailureClassifier,
    FailureRecord,
)

logger = logging.getLogger(__name__)


def load_trajectories_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Safely load trajectory entries from a JSONL file."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %d in %s: %s", line_num, file_path, exc)
    return entries


class TrajectoryFailureAnalyzer:
    """Runs deterministic failure analysis across trajectory batches."""

    def __init__(
        self,
        classifier: FailureClassifier | None = None,
    ) -> None:
        self.classifier = classifier or FailureClassifier()

    def analyze_batch(
        self,
        trajectories: list[dict[str, Any]],
        cases_by_id: dict[str, IncidentCase] | None = None,
    ) -> dict[str, Any]:
        """Analyze a list of trajectory dicts and produce structured summary + failure list."""
        all_failures: list[FailureRecord] = []
        per_trajectory: list[dict[str, Any]] = []
        category_counts: Counter[str] = Counter({cat: 0 for cat in FAILURE_CATEGORIES})
        passed_count = 0
        failed_count = 0
        safety_failure_count = 0

        for traj in trajectories:
            incident_id = str(traj.get("incident_id", "unknown"))
            base_id = incident_id.split("_run")[0] if "_run" in incident_id else incident_id
            case = (cases_by_id or {}).get(incident_id) or (cases_by_id or {}).get(base_id) or get_case_by_id(base_id)

            records = self.classifier.classify(traj, case=case)
            cats = [r.failure_category for r in records]
            has_safety = "unsafe_action_attempt" in cats

            if records:
                failed_count += 1
            else:
                passed_count += 1
            if has_safety:
                safety_failure_count += 1

            for r in records:
                category_counts[r.failure_category] += 1
                all_failures.append(r)

            per_trajectory.append(
                {
                    "incident_id": incident_id,
                    "query": traj.get("query", ""),
                    "passed": len(records) == 0,
                    "safety_failure": has_safety,
                    "failure_categories": cats,
                    "failure_count": len(records),
                    "tools_called": traj.get("tools_called", []),
                    "root_cause": traj.get("root_cause", "unknown"),
                }
            )

        total = len(trajectories)
        return {
            "total_trajectories": total,
            "passed_trajectories": passed_count,
            "failed_trajectories": failed_count,
            "pass_rate": round(passed_count / total, 4) if total else 0.0,
            "safety_failure_trajectories": safety_failure_count,
            "total_failure_records": len(all_failures),
            "failure_counts_by_category": dict(category_counts),
            "per_trajectory": per_trajectory,
            "failures": [r.to_dict() for r in all_failures],
        }

    def analyze_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Load JSONL trajectories, analyze them, and optionally write JSON report."""
        trajectories = load_trajectories_jsonl(input_path)
        report = self.analyze_batch(trajectories)
        report["source_path"] = str(input_path)
        sanitized = _redact_secrets(report)

        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(
                json.dumps(sanitized, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Failure analysis report written to %s", out_file)
        return sanitized

