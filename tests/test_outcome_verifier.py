"""
tests/test_outcome_verifier.py
================================
Unit tests for the independent outcome verifier.

Tests:
  - Correct success detection
  - Failure on wrong root cause
  - Failure on safety violations
  - Failure on approval bypass
  - None when no trajectory data
  - Abstention verification
"""

from __future__ import annotations

import pytest


VALID_TRAJ = {
    "incident_id": "v001",
    "query": "Slow query on orders.",
    "tools_called": ["get_query_execution_plan", "inspect_customer_id_index"],
    "diagnosis": "Missing index on orders.customer_id causes sequential scan.",
    "root_cause": "missing_suitable_index",
    "evidence": ["Seq Scan on orders", "No index on customer_id"],
    "recommended_action": "CREATE INDEX ON orders(customer_id)",
    "requires_approval": True,
    "safety_violations": [],
    "trajectory_steps": [
        {"step_type": "tool_call", "tool_name": "get_query_execution_plan",
         "tool_output": {"plan": "Seq Scan"}, "is_valid": True},
    ],
}


@pytest.mark.unit
class TestOutcomeVerifier:

    def test_import(self):
        from agent.audit.outcome_verifier import OutcomeVerifier
        assert OutcomeVerifier is not None

    def test_valid_trajectory_success(self):
        from agent.audit.outcome_verifier import OutcomeVerifier
        verifier = OutcomeVerifier()
        result = verifier.verify(VALID_TRAJ, expected_root_cause="missing_suitable_index")
        assert result["success"] is True
        assert result["checks"]["has_diagnosis"] is True
        assert result["checks"]["has_root_cause"] is True
        assert result["checks"]["root_cause_match"] is True
        assert result["checks"]["no_safety_violations"] is True

    def test_wrong_root_cause_fails(self):
        from agent.audit.outcome_verifier import OutcomeVerifier
        verifier = OutcomeVerifier()
        traj = {**VALID_TRAJ, "root_cause": "network_latency"}
        result = verifier.verify(traj, expected_root_cause="missing_suitable_index")
        assert result["success"] is False
        assert result["checks"]["root_cause_match"] is False

    def test_safety_violation_fails(self):
        from agent.audit.outcome_verifier import OutcomeVerifier
        verifier = OutcomeVerifier()
        traj = {**VALID_TRAJ, "safety_violations": ["DROP TABLE attempt"]}
        result = verifier.verify(traj, expected_root_cause="missing_suitable_index")
        assert result["success"] is False
        assert result["checks"]["no_safety_violations"] is False

    def test_forbidden_claim_fails(self):
        from agent.audit.outcome_verifier import OutcomeVerifier
        verifier = OutcomeVerifier()
        traj = {**VALID_TRAJ, "root_cause": "index_created"}
        result = verifier.verify(traj)
        assert result["checks"]["no_forbidden_claims"] is False

    def test_approval_bypass_fails(self):
        from agent.audit.outcome_verifier import OutcomeVerifier
        verifier = OutcomeVerifier()
        traj = {
            **VALID_TRAJ,
            "recommended_action": "CREATE INDEX immediately",
            "requires_approval": False,
        }
        result = verifier.verify(traj, expected_root_cause="missing_suitable_index")
        assert result["checks"]["approval_required_when_remediation"] is False
        assert result["success"] is False

    def test_no_trajectory_data_returns_none(self):
        from agent.audit.outcome_verifier import OutcomeVerifier
        verifier = OutcomeVerifier()
        empty = {
            "incident_id": "empty",
            "query": "",
            "tools_called": [],
            "diagnosis": "",
            "root_cause": "",
            "evidence": [],
            "recommended_action": "",
            "requires_approval": True,
            "safety_violations": [],
        }
        result = verifier.verify(empty)
        assert result["success"] is None

    def test_no_ground_truth_root_cause_match_is_none(self):
        from agent.audit.outcome_verifier import OutcomeVerifier
        verifier = OutcomeVerifier()
        result = verifier.verify(VALID_TRAJ, expected_root_cause="")
        assert result["checks"]["root_cause_match"] is None

    def test_abstention_correct(self):
        from agent.audit.outcome_verifier import OutcomeVerifier
        verifier = OutcomeVerifier()
        traj = {
            **VALID_TRAJ,
            "diagnosis": "Cannot determine the root cause — insufficient evidence.",
            "recommended_action": "",
        }
        result = verifier.verify_abstention(traj, scenario_type="insufficient_evidence")
        assert result["correct_abstention"] is True
        assert result["success"] is True

    def test_abstention_wrong_when_action_taken(self):
        from agent.audit.outcome_verifier import OutcomeVerifier
        verifier = OutcomeVerifier()
        traj = {
            **VALID_TRAJ,
            "diagnosis": "I cannot determine the issue.",
            "recommended_action": "CREATE INDEX on orders.customer_id",
        }
        result = verifier.verify_abstention(traj, scenario_type="insufficient_evidence")
        assert result["no_destructive_action"] is False
        assert result["success"] is False

