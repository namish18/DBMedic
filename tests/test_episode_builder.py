"""
tests/test_episode_builder.py
================================
Unit tests for EpisodeBuilder.

Tests:
  - From trajectory JSONL file
  - From IncidentCase definitions
  - From AgentTune EventLog
  - Stub episodes for missing trajectories
  - Serialization
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


SAMPLE_TRAJ_LINE = json.dumps({
    "incident_id": "baseline_001",
    "query": "Slow orders query",
    "model_id": "tiny-aya",
    "model_revision": None,
    "success": True,
    "num_steps": 2,
    "tools_called": ["get_query_execution_plan"],
    "diagnosis": "Missing index",
    "root_cause": "missing_suitable_index",
    "evidence": ["Seq Scan detected"],
    "recommended_action": "CREATE INDEX",
    "requires_approval": True,
    "safety_violations": [],
    "trajectory_steps": [
        {
            "step_index": 0,
            "step_type": "tool_call",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "tool_name": "get_query_execution_plan",
            "tool_arguments": {},
            "tool_output": {"plan": "Seq Scan"},
            "is_valid": True,
        }
    ],
    "total_time_ms": 500,
})


@pytest.mark.unit
class TestEpisodeBuilder:

    def test_from_trajectory_file_missing_file(self):
        from agent.audit.episode_builder import EpisodeBuilder
        builder = EpisodeBuilder()
        episodes = builder.from_trajectory_file(Path("/nonexistent/path.jsonl"))
        assert episodes == []

    def test_from_trajectory_file_valid(self):
        from agent.audit.episode_builder import EpisodeBuilder
        builder = EpisodeBuilder()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(SAMPLE_TRAJ_LINE + "\n")
            tmppath = Path(f.name)

        try:
            episodes = builder.from_trajectory_file(
                tmppath,
                ground_truth_map={"baseline_001": "missing_suitable_index"},
            )
            assert len(episodes) == 1
            assert episodes[0].incident_id == "baseline_001"
            assert episodes[0].verified_success is True
        finally:
            tmppath.unlink(missing_ok=True)

    def test_from_trajectory_file_skips_malformed(self):
        from agent.audit.episode_builder import EpisodeBuilder
        builder = EpisodeBuilder()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write("{malformed json\n")
            f.write(SAMPLE_TRAJ_LINE + "\n")
            tmppath = Path(f.name)

        try:
            episodes = builder.from_trajectory_file(tmppath)
            # Only the valid line should produce an episode
            assert len(episodes) == 1
        finally:
            tmppath.unlink(missing_ok=True)

    def test_from_incident_cases_no_trajectory(self):
        """Cases without matching trajectories produce stub episodes."""
        from agent.audit.episode_builder import EpisodeBuilder
        from agent.evaluation.incident_cases import BASELINE_EVAL_CASES

        builder = EpisodeBuilder()
        episodes = builder.from_incident_cases(BASELINE_EVAL_CASES[:3])

        assert len(episodes) == 3
        for ep in episodes:
            # Stub episodes have simulated=True
            assert ep.simulated is True
            assert ep.verified_success is None

    def test_from_incident_cases_with_matching_trajectory(self):
        from agent.audit.episode_builder import EpisodeBuilder
        from agent.evaluation.incident_cases import BASELINE_EVAL_CASES

        builder = EpisodeBuilder()

        # Write a trajectory file matching the first baseline case
        first_case = BASELINE_EVAL_CASES[0]
        traj = {
            "incident_id": first_case.incident_id,
            "query": first_case.query,
            "model_id": "tiny-aya",
            "model_revision": None,
            "success": True,
            "num_steps": 2,
            "tools_called": ["get_query_execution_plan"],
            "diagnosis": "Missing index on orders.customer_id",
            "root_cause": first_case.expected_root_cause,
            "evidence": ["Seq Scan on orders"],
            "recommended_action": "CREATE INDEX",
            "requires_approval": True,
            "safety_violations": [],
            "trajectory_steps": [],
            "total_time_ms": 400,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps(traj) + "\n")
            tmppath = Path(f.name)

        try:
            episodes = builder.from_incident_cases(
                [first_case],
                trajectory_jsonl=tmppath,
            )
            assert len(episodes) == 1
            # Should NOT be simulated — a real trajectory matched
            assert episodes[0].simulated is False
            assert episodes[0].incident_id == first_case.incident_id
        finally:
            tmppath.unlink(missing_ok=True)

    def test_from_agenttune_eventlog(self):
        from agent.audit.episode_builder import EpisodeBuilder

        builder = EpisodeBuilder()
        event_log = [
            {"kind": "TOOL_CALL", "tool_name": "get_query_execution_plan",
             "tool_arguments": {}, "timestamp": "2026-01-01T00:00:01+00:00"},
            {"kind": "TOOL_RESULT", "tool_result": {"plan": "Seq Scan"}, "timestamp": "2026-01-01T00:00:02+00:00"},
        ]
        ep = builder.from_agenttune_eventlog(
            event_log,
            incident_id="at_ep_001",
            user_request="Diagnose query",
            model_id="tiny-aya",
            expected_root_cause="missing_suitable_index",
        )
        assert ep.incident_id == "at_ep_001"
        assert len(ep.steps) == 2
        assert ep.agenttune_event_log == event_log

