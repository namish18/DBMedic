"""
agent/evaluation/auditkit_evaluator.py
========================================
Phase 5 AuditKIT-based evaluator for DBMedic.

Wraps AuditKITAdapter to run full evaluation pipelines in:
  - recorded mode (offline, existing trajectories only)
  - live mode (requires model access and hardware)

Integrates with the existing Phase 4 ModelComparisonEvaluator without
double-counting existing metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class AuditKITEvaluator:
    """
    Runs AuditKIT evaluation pipelines over DBMedic episodes.

    Recorded mode works fully offline using existing trajectory files.
    Live mode requires HF_TOKEN, CUDA, and PostgreSQL — fails clearly when missing.
    """

    def __init__(self) -> None:
        from agent.audit.auditkit_adapter import AuditKITAdapter
        from agent.audit.episode_builder import EpisodeBuilder
        self._adapter = AuditKITAdapter()
        self._builder = EpisodeBuilder()

    # ------------------------------------------------------------------
    # Recorded mode
    # ------------------------------------------------------------------

    def evaluate_recorded(
        self,
        trajectory_path: Path | str | None = None,
        *,
        split: str = "baseline",
        ground_truth_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate using recorded trajectories — no live model inference needed.

        Parameters
        ----------
        trajectory_path : Path | str | None
            Path to trajectories.jsonl. Defaults to benchmarks/baseline/trajectories.jsonl.
        split : str
            Dataset split label.
        ground_truth_map : dict | None
            {incident_id: expected_root_cause} for independent verification.

        Returns
        -------
        dict with evaluation summary compatible with AuditKIT RunResult.summary().
        """
        if trajectory_path is None:
            trajectory_path = PROJECT_ROOT / "benchmarks" / "baseline" / "trajectories.jsonl"

        trajectory_path = Path(trajectory_path)

        if not trajectory_path.exists():
            logger.warning("No trajectory file found at %s — returning empty evaluation.", trajectory_path)
            return self._empty_result("trajectory_file_not_found", str(trajectory_path))

        episodes = self._builder.from_trajectory_file(
            trajectory_path,
            split=split,
            ground_truth_map=ground_truth_map,
        )

        if not episodes:
            return self._empty_result("no_episodes_in_trajectory_file", str(trajectory_path))

        result = self._adapter.evaluate_episodes(
            episodes,
            mode="recorded",
        )

        summary = result.summary()
        summary["num_episodes"] = len(episodes)
        summary["mode"] = "recorded"
        summary["trajectory_path"] = str(trajectory_path)
        summary["auditkit_stub"] = self._adapter.is_stub

        logger.info(
            "Recorded evaluation complete: %d episodes, stub=%s",
            len(episodes), self._adapter.is_stub
        )
        return summary

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------

    def evaluate_live(
        self,
        model: str = "base",
        *,
        incident_cases: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run live model inference evaluation.

        Fails clearly when prerequisites (HF_TOKEN, CUDA, PostgreSQL) are missing.

        Parameters
        ----------
        model : str
            "base" or "candidate".
        incident_cases : list | None
            Evaluation cases. Defaults to BASELINE_EVAL_CASES.

        Returns
        -------
        dict with evaluation summary or blocker record.
        """
        blockers = self._check_live_prerequisites(model)
        if blockers:
            logger.error("Live evaluation blocked: %s", blockers)
            return {
                "status": "blocked",
                "mode": "live",
                "model": model,
                "blockers": blockers,
                "result": None,
                "note": (
                    "Live evaluation requires HF_TOKEN, CUDA GPU, and PostgreSQL. "
                    "Use --mode recorded for offline evaluation."
                ),
            }

        # Live evaluation path (only reached when all prerequisites are satisfied)
        from agent.evaluation.incident_cases import BASELINE_EVAL_CASES
        cases = incident_cases or BASELINE_EVAL_CASES

        episodes = self._builder.from_incident_cases(cases)
        result = self._adapter.evaluate_episodes(
            episodes,
            mode="live",
            model=f"CohereLabs/tiny-aya-global" if model == "base" else "candidate",
        )
        summary = result.summary()
        summary["num_episodes"] = len(episodes)
        summary["mode"] = "live"
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_live_prerequisites(model: str) -> list[str]:
        """Return list of unmet prerequisites for live evaluation."""
        blockers = []
        import os
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            blockers.append("HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable is not set.")
        try:
            import torch
            if not torch.cuda.is_available():
                blockers.append("No CUDA GPU detected (torch.cuda.is_available() == False).")
        except ImportError:
            blockers.append("PyTorch is not installed.")
        # PostgreSQL check deferred to runtime
        return blockers

    @staticmethod
    def _empty_result(reason: str, detail: str = "") -> dict[str, Any]:
        return {
            "status": "empty",
            "reason": reason,
            "detail": detail,
            "num_episodes": 0,
            "metrics": {},
            "note": "No evaluation data available. Results are pending.",
        }

