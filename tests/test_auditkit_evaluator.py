"""
tests/test_auditkit_evaluator.py
==================================
Unit tests for the AuditKIT evaluator.

Tests:
  - Recorded mode (offline, no model)
  - Live mode blocked when prerequisites missing
  - Empty result when no trajectory file
  - Prerequisite checking logic
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


SAMPLE_TRAJ = {
    "incident_id": "eval_001",
    "query": "Orders for customer 4242 are slow.",
    "model_id": "tiny-aya",
    "model_revision": None,
    "success": True,
    "num_steps": 2,
    "tools_called": ["get_query_execution_plan"],
    "diagnosis": "Missing index on orders.customer_id",
    "root_cause": "missing_suitable_index",
    "evidence": ["Seq Scan detected"],
    "recommended_action": "CREATE INDEX",
    "requires_approval": True,
    "safety_violations": [],
    "rejection_reasons": [],
    "total_time_ms": 400,
    "error": None,
    "trajectory_steps": [
        {
            "step_index": 0,
            "step_type": "tool_call",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "tool_name": "get_query_execution_plan",
            "tool_arguments": {},
            "tool_output": {"plan": "Seq Scan on orders"},
            "is_valid": True,
        }
    ],
    "metadata": {
        "validator_summary": {"total_validated_calls": 1, "rejected_calls": 0}
    },
}


@pytest.mark.unit
class TestAuditKITEvaluator:

    def test_import(self):
        from agent.evaluation.auditkit_evaluator import AuditKITEvaluator
        assert AuditKITEvaluator is not None

    def test_evaluate_recorded_no_file(self, tmp_path):
        from agent.evaluation.auditkit_evaluator import AuditKITEvaluator
        evaluator = AuditKITEvaluator()
        result = evaluator.evaluate_recorded(
            trajectory_path=tmp_path / "nonexistent.jsonl"
        )
        assert result.get("status") == "trajectory_file_not_found" or result.get("num_episodes") == 0

    def test_evaluate_recorded_with_trajectory(self):
        from agent.evaluation.auditkit_evaluator import AuditKITEvaluator

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps(SAMPLE_TRAJ) + "\n")
            tmppath = Path(f.name)

        try:
            evaluator = AuditKITEvaluator()
            result = evaluator.evaluate_recorded(trajectory_path=tmppath)
            assert result.get("num_episodes") == 1
            assert result.get("mode") == "recorded"
            assert "metrics" in result
        finally:
            tmppath.unlink(missing_ok=True)

    def test_evaluate_live_blocked_without_hf_token(self, monkeypatch):
        """Live evaluation must fail clearly when HF_TOKEN is missing."""
        import os
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

        from agent.evaluation.auditkit_evaluator import AuditKITEvaluator
        evaluator = AuditKITEvaluator()
        result = evaluator.evaluate_live(model="base")

        assert result.get("status") == "blocked"
        blockers = result.get("blockers", [])
        assert any("HF_TOKEN" in b for b in blockers)

    def test_evaluate_live_blocked_without_cuda(self, monkeypatch):
        """Live evaluation must fail clearly without CUDA."""
        import os
        monkeypatch.setenv("HF_TOKEN", "fake_token_for_test")

        from agent.evaluation.auditkit_evaluator import AuditKITEvaluator
        evaluator = AuditKITEvaluator()
        result = evaluator.evaluate_live(model="base")

        assert result.get("status") == "blocked"
        blockers = result.get("blockers", [])
        # Should contain either CUDA or PyTorch missing blocker
        hardware_blocked = any(
            "CUDA" in b or "GPU" in b or "PyTorch" in b or "torch" in b.lower()
            for b in blockers
        )
        assert hardware_blocked

    def test_prerequisite_check_returns_list(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

        from agent.evaluation.auditkit_evaluator import AuditKITEvaluator
        blockers = AuditKITEvaluator._check_live_prerequisites("base")
        assert isinstance(blockers, list)
        assert len(blockers) >= 1

    def test_recorded_result_has_config_fingerprint(self):
        from agent.evaluation.auditkit_evaluator import AuditKITEvaluator

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps(SAMPLE_TRAJ) + "\n")
            tmppath = Path(f.name)

        try:
            evaluator = AuditKITEvaluator()
            result = evaluator.evaluate_recorded(trajectory_path=tmppath)
            fp = result.get("config_fingerprint", "")
            assert fp != ""
            assert len(fp) == 64  # SHA-256 hex
        finally:
            tmppath.unlink(missing_ok=True)
