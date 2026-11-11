"""
agent/audit/episode_builder.py
================================
Builds AuditKIT-compatible episodes from multiple DBMedic data sources:
  - Recorded trajectory JSONL files (offline, no model needed)
  - AgentTune EventLog structures from Phase 4
  - Synthetic evaluation dataset cases (IncidentCase → Episode)

All episodes preserve:
  - User request (never modified)
  - Model responses and tool calls
  - Tool arguments and observations
  - Final diagnosis and verified incident outcome
  - Safety violations and execution metadata
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EpisodeBuilder:
    """
    Builds lists of AuditKIT Episode objects from various DBMedic sources.

    Usage
    -----
    builder = EpisodeBuilder()
    episodes = builder.from_trajectory_file(Path("benchmarks/baseline/trajectories.jsonl"))
    episodes = builder.from_incident_cases(BASELINE_EVAL_CASES)
    """

    def __init__(self) -> None:
        from agent.audit.auditkit_adapter import AuditKITAdapter
        self._adapter = AuditKITAdapter()

    # ------------------------------------------------------------------
    # From recorded trajectory JSONL
    # ------------------------------------------------------------------

    def from_trajectory_file(
        self,
        path: Path | str,
        *,
        split: str = "baseline",
        ground_truth_map: dict[str, str] | None = None,
    ) -> list[Any]:
        """
        Load a trajectory JSONL file and convert each line to an Episode.

        Parameters
        ----------
        path : Path | str
            Path to a .jsonl file of agent trajectories.
        split : str
            Dataset split label for all episodes.
        ground_truth_map : dict | None
            {incident_id: expected_root_cause} for independent verification.
            If omitted, verification will be attempted without ground truth.

        Returns
        -------
        list[Episode]
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Trajectory file not found: %s", path)
            return []

        ground_truth_map = ground_truth_map or {}
        episodes = []

        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    traj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed trajectory line %d: %s", line_num, exc)
                    continue

                incident_id = traj.get("incident_id", f"unknown_{line_num}")
                expected_rc = ground_truth_map.get(incident_id, "")
                ep = self._adapter.trajectory_to_episode(
                    traj,
                    expected_root_cause=expected_rc,
                    split=split,
                )
                episodes.append(ep)

        logger.info("Built %d episodes from %s", len(episodes), path)
        return episodes

    # ------------------------------------------------------------------
    # From IncidentCase definitions
    # ------------------------------------------------------------------

    def from_incident_cases(
        self,
        cases: list[Any],
        *,
        trajectory_jsonl: Path | str | None = None,
    ) -> list[Any]:
        """
        Build episodes from IncidentCase definitions.

        If a trajectory_jsonl path is given, each case's trajectory is looked
        up by incident_id. If not found, a synthetic episode stub is built
        (useful for describing the evaluation dataset without live results).

        Parameters
        ----------
        cases : list[IncidentCase]
            Evaluation cases from incident_cases.py.
        trajectory_jsonl : Path | str | None
            Optional path to recorded trajectory JSONL for matching.

        Returns
        -------
        list[Episode]
        """
        from agent.audit.auditkit_adapter import Episode, EpisodeStep

        # Load recorded trajectories for matching
        recorded: dict[str, dict[str, Any]] = {}
        if trajectory_jsonl is not None:
            tpath = Path(trajectory_jsonl)
            if tpath.exists():
                with open(tpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            traj = json.loads(line)
                            iid = traj.get("incident_id", "")
                            if iid:
                                recorded[iid] = traj
                        except json.JSONDecodeError:
                            pass

        episodes = []
        for case in cases:
            iid = case.incident_id
            if iid in recorded:
                ep = self._adapter.trajectory_to_episode(
                    recorded[iid],
                    expected_root_cause=case.expected_root_cause,
                    split=case.split,
                )
            else:
                # Synthetic stub episode — no live trajectory available
                ep = Episode(
                    episode_id=f"ep_stub_{iid}",
                    incident_id=iid,
                    user_request=case.query,
                    model_id="pending",
                    model_revision=None,
                    steps=[],
                    expected_root_cause=case.expected_root_cause,
                    verified_success=None,
                    outcome_verified_by="no_trajectory_available",
                    split=case.split,
                    simulated=True,
                )
            episodes.append(ep)

        logger.info("Built %d episodes from %d incident cases", len(episodes), len(cases))
        return episodes

    # ------------------------------------------------------------------
    # From AgentTune EventLog structures
    # ------------------------------------------------------------------

    def from_agenttune_eventlog(
        self,
        event_log_data: list[dict[str, Any]],
        *,
        incident_id: str = "unknown",
        user_request: str = "",
        model_id: str = "unknown",
        expected_root_cause: str = "",
        split: str = "baseline",
    ) -> Any:
        """Build a single episode from AgentTune EventLog data."""
        return self._adapter.agenttune_eventlog_to_episode(
            event_log_data,
            incident_id=incident_id,
            user_request=user_request,
            model_id=model_id,
            expected_root_cause=expected_root_cause,
            split=split,
        )

    # ------------------------------------------------------------------
    # From evaluation benchmark dataset JSONL
    # ------------------------------------------------------------------

    def from_benchmark_dataset(
        self,
        dataset_path: Path | str,
        *,
        split: str = "baseline",
    ) -> list[Any]:
        """
        Load a prepared evaluation benchmark JSONL dataset.

        Each line must have:
          {"incident_id": ..., "query": ..., "expected_root_cause": ..., ...}

        Ground-truth answers are NEVER passed to the evaluated model —
        they are used only for independent outcome verification.
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.warning("Benchmark dataset not found: %s", dataset_path)
            return []

        from agent.audit.auditkit_adapter import Episode

        episodes = []
        with open(dataset_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line %d: %s", line_num, exc)
                    continue

                iid = record.get("incident_id", f"unknown_{line_num}")
                ep = Episode(
                    episode_id=f"ep_bench_{iid}",
                    incident_id=iid,
                    user_request=record.get("query", ""),
                    model_id=record.get("model_id", "pending"),
                    model_revision=record.get("model_revision"),
                    expected_root_cause=record.get("expected_root_cause", ""),
                    split=split,
                    simulated=record.get("simulated", False),
                )
                episodes.append(ep)

        logger.info("Loaded %d episodes from benchmark dataset %s", len(episodes), dataset_path)
        return episodes

