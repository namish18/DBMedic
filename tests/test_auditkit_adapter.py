"""
tests/test_auditkit_adapter.py
================================
Unit tests for the AuditKIT adapter.

Tests:
  - Trajectory → Episode conversion
  - AgentTune EventLog → Episode conversion
  - Stub evaluation (offline, no model)
  - Config fingerprint generation
  - Metric extraction from episodes

Does NOT require GPU, model download, or PostgreSQL.
AuditKIT package absence is handled transparently.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TRAJECTORY = {
    "incident_id": "test_001",
    "timestamp": "2026-01-01T00:00:00+00:00",
    "model_id": "CohereLabs/tiny-aya-global",
    "model_revision": "abc123",
    "query": "Orders for customer 4242 are slow. Diagnose.",
    "success": True,
    "num_steps": 3,
    "tools_called": ["get_query_execution_plan", "inspect_customer_id_index"],
    "diagnosis": "Missing index on orders.customer_id causes sequential scan.",
    "root_cause": "missing_suitable_index",
    "evidence": ["Sequential scan detected on orders table", "No index on customer_id"],
    "recommended_action": "CREATE INDEX ON orders(customer_id)",
    "requires_approval": True,
    "confidence": 0.9,
    "safety_violations": [],
    "rejection_reasons": [],
    "total_time_ms": 1500,
    "error": None,
    "trajectory_steps": [
        {
            "step_index": 0,
            "step_type": "tool_call",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "tool_name": "get_query_execution_plan",
            "tool_arguments": {"customer_id": 4242},
            "tool_output": {"plan": "Seq Scan on orders", "cost": 45000},
            "is_valid": True,
        },
        {
            "step_index": 1,
            "step_type": "tool_call",
            "timestamp": "2026-01-01T00:00:02+00:00",
            "tool_name": "inspect_customer_id_index",
            "tool_arguments": {},
            "tool_output": {"index_exists": False},
            "is_valid": True,
        },
        {
            "step_index": 2,
            "step_type": "model_response",
            "timestamp": "2026-01-01T00:00:03+00:00",
            "model_response": '{"type":"final_answer","root_cause":"missing_suitable_index"}',
            "is_valid": True,
        },
    ],
    "metadata": {
        "validator_summary": {"total_validated_calls": 2, "rejected_calls": 0}
    },
}

SAMPLE_EVENTLOG = [
    {"kind": "TOOL_CALL", "tool_name": "get_query_execution_plan",
     "tool_arguments": {"customer_id": 4242}, "timestamp": "2026-01-01T00:00:01+00:00"},
    {"kind": "TOOL_RESULT", "tool_result": {"plan": "Seq Scan"}, "timestamp": "2026-01-01T00:00:02+00:00"},
    {"kind": "TEXT", "content": "Diagnosis complete.", "timestamp": "2026-01-01T00:00:03+00:00"},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAuditKITAdapter:

    def test_import_succeeds(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter, AuditKITStub
        assert AuditKITAdapter is not None
        assert AuditKITStub is not None

    def test_adapter_initializes_in_stub_mode(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter
        adapter = AuditKITAdapter(use_stub=True)
        assert adapter.is_stub is True
        assert "stub" in adapter.version.lower()

    def test_trajectory_to_episode_basic(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter
        adapter = AuditKITAdapter(use_stub=True)
        episode = adapter.trajectory_to_episode(
            SAMPLE_TRAJECTORY,
            expected_root_cause="missing_suitable_index",
        )
        assert episode.incident_id == "test_001"
        assert episode.user_request == "Orders for customer 4242 are slow. Diagnose."
        assert episode.model_id == "CohereLabs/tiny-aya-global"
        assert episode.final_root_cause == "missing_suitable_index"
        assert episode.expected_root_cause == "missing_suitable_index"
        assert episode.requires_approval is True
        assert len(episode.steps) == 3

    def test_verified_success_true_for_correct_trajectory(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter
        adapter = AuditKITAdapter(use_stub=True)
        episode = adapter.trajectory_to_episode(
            SAMPLE_TRAJECTORY,
            expected_root_cause="missing_suitable_index",
        )
        assert episode.verified_success is True

    def test_verified_success_false_for_wrong_root_cause(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter
        adapter = AuditKITAdapter(use_stub=True)
        traj = {**SAMPLE_TRAJECTORY, "root_cause": "network_latency"}
        episode = adapter.trajectory_to_episode(
            traj,
            expected_root_cause="missing_suitable_index",
        )
        assert episode.verified_success is False

    def test_verified_success_none_without_ground_truth(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter
        adapter = AuditKITAdapter(use_stub=True)
        # No trajectory data, no ground truth
        empty_traj = {
            "incident_id": "empty",
            "query": "",
            "diagnosis": "",
            "root_cause": "",
            "tools_called": [],
            "trajectory_steps": [],
        }
        episode = adapter.trajectory_to_episode(empty_traj)
        assert episode.verified_success is None

    def test_agenttune_eventlog_to_episode(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter
        adapter = AuditKITAdapter(use_stub=True)
        episode = adapter.agenttune_eventlog_to_episode(
            SAMPLE_EVENTLOG,
            incident_id="at_test_001",
            user_request="Diagnose slow query",
            model_id="tiny-aya",
            expected_root_cause="missing_suitable_index",
        )
        assert episode.incident_id == "at_test_001"
        assert episode.user_request == "Diagnose slow query"
        assert len(episode.steps) == 3
        assert episode.agenttune_event_log == SAMPLE_EVENTLOG

    def test_evaluate_episodes_stub_returns_summary(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter
        adapter = AuditKITAdapter(use_stub=True)
        episode = adapter.trajectory_to_episode(
            SAMPLE_TRAJECTORY,
            expected_root_cause="missing_suitable_index",
        )
        result = adapter.evaluate_episodes([episode], mode="recorded")
        summary = result.summary()
        assert "auditkit_version" in summary
        assert "metrics" in summary
        assert "config_fingerprint" in summary
        assert summary["config_fingerprint"] != ""

    def test_config_fingerprint_is_sha256(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter
        adapter = AuditKITAdapter(use_stub=True)
        episode = adapter.trajectory_to_episode(SAMPLE_TRAJECTORY)
        result = adapter.evaluate_episodes([episode])
        fp = result.summary()["config_fingerprint"]
        # SHA-256 hex digest is 64 characters
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_episode_to_dict_is_serializable(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter
        adapter = AuditKITAdapter(use_stub=True)
        episode = adapter.trajectory_to_episode(SAMPLE_TRAJECTORY)
        d = episode.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(d, default=str)
        assert json_str is not None

    def test_safety_severity_critical_for_drop(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter
        adapter = AuditKITAdapter(use_stub=True)
        traj = {
            **SAMPLE_TRAJECTORY,
            "safety_violations": ["DROP TABLE attempt detected"],
        }
        episode = adapter.trajectory_to_episode(traj)
        assert episode.safety_severity == "critical"

    def test_safety_severity_none_for_clean_trajectory(self):
        from agent.audit.auditkit_adapter import AuditKITAdapter
        adapter = AuditKITAdapter(use_stub=True)
        episode = adapter.trajectory_to_episode(SAMPLE_TRAJECTORY)
        assert episode.safety_severity == "none"

