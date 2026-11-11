"""
tests/test_trajectory_adapter.py
================================
Unit tests for AgentTune bridge (`training/agenttune/trajectory_adapter.py`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation.incident_cases import get_case_by_id
from agenttune.agentic.events import EventKind
from agenttune.agentic.heal_loop import SelfHealLoop
from training.agenttune.trajectory_adapter import (
    build_dbmedic_readonly_harness,
    event_log_to_dbmedic_messages,
    to_agenttune_classified_failure,
    to_agenttune_event_log,
    to_agenttune_failure,
    to_agenttune_project,
    to_agenttune_training_example,
    to_agenttune_trajectory,
)
from training.data.formatter import validate_alternation
from training.failure_analysis.classifier import APPROVED_DIAGNOSTIC_TOOLS, FailureRecord
from training.failure_analysis.correction_generator import CorrectionGenerator


@pytest.mark.unit
class TestTrajectoryAdapter:
    def test_event_log_full_and_light_and_project_distill_preview(self):
        gen = CorrectionGenerator()
        rec = gen.generate_for_case(get_case_by_id("train_001"))

        full_log = to_agenttune_event_log(rec, tier="full", reward_score=10.0)
        assert full_log.tier == "full"
        assert full_log.rewards("episode") == [10.0]

        light_log = to_agenttune_event_log(rec, tier="light")
        assert light_log.tier == "light"
        assert len(light_log.to_audit_records()) >= 1

        proj = to_agenttune_project([rec], tier="full")
        sft_rows = proj.sft_dataset("sft")
        assert len(sft_rows) == 1
        assert len(sft_rows[0]["messages"]) >= 3

        traj = to_agenttune_trajectory(rec, reward=9.5)
        assert traj.reward == 9.5
        assert len(traj.steps) >= 1

    def test_roundtrip_event_log_to_tiny_aya_strict_alternation(self):
        gen = CorrectionGenerator()
        rec = gen.generate_for_case(get_case_by_id("train_002"))
        log = to_agenttune_event_log(rec, tier="full")
        msgs = event_log_to_dbmedic_messages(log)
        ok, err = validate_alternation(msgs)
        assert ok, err
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[-1]["role"] == "assistant"
        parsed_final = json.loads(msgs[-1]["content"])
        assert parsed_final["type"] == "final_answer"

    def test_readonly_harness_excludes_mutating_tools(self):
        mock_execs = {name: (lambda _n=name, **kw: {"tool": _n, "ok": True}) for name in APPROVED_DIAGNOSTIC_TOOLS}
        mock_execs["apply_remediation"] = lambda **kw: {"mutated": True}
        mock_execs["execute_sql"] = lambda **kw: {"mutated": True}
        harness = build_dbmedic_readonly_harness(tool_executors=mock_execs)
        assert set(harness.tools.keys()) == set(APPROVED_DIAGNOSTIC_TOOLS)
        assert "apply_remediation" not in harness.tools
        assert "execute_sql" not in harness.tools

    def test_self_heal_loop_integration_with_failure_contracts(self):
        gen = CorrectionGenerator()
        case = get_case_by_id("train_001")
        fr = FailureRecord(
            incident_id="train_001",
            failure_category="malformed_tool_call",
            original_model_response="bad output",
            expected_behavior="Valid JSON",
            observed_behavior="Plain text",
            evidence=["Parse error"],
            trajectory_id="t1",
        )
        at_fail = to_agenttune_failure(fr)
        repaired = gen.generate_from_failure(fr, case=case)

        loop = SelfHealLoop(
            classifier=lambda fails: [to_agenttune_classified_failure(fr) for _ in fails],
            generator=lambda cfs: [to_agenttune_training_example(fr, repaired) for _ in cfs],
        )
        res = loop.run([at_fail])
        assert res["n_failures"] == 1
        assert res["n_classified"] == 1
        assert res["n_generated"] == 1
        assert res["n_dataset_rows"] == 1

