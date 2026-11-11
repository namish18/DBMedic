"""
tests/test_deployment_verifier.py
====================================
Unit tests for the deployment verifier.

Tests:
  - Gate rejects when no evaluation data
  - Gate holds when candidate data missing
  - Gate rejects with safety regression
  - Gate rejects without human approval
  - Gate rejects when unauthorized actions succeeded
  - Null values are never treated as passing
  - Approval bypass check
"""

from __future__ import annotations

import pytest


BASELINE_SUMMARY = {
    "num_episodes": 10,
    "config_fingerprint": "a" * 64,
    "metrics": {
        "tool_call_validity_rate": 0.90,
        "task_success_rate": 0.80,
        "safety_violation_count": 0,
        "diagnosis_accuracy": 0.80,
        "evidence_grounding_accuracy": 0.75,
    },
}

CANDIDATE_SUMMARY = {
    "num_episodes": 10,
    "config_fingerprint": "b" * 64,
    "metrics": {
        "tool_call_validity_rate": 0.92,
        "task_success_rate": 0.85,
        "safety_violation_count": 0,
        "diagnosis_accuracy": 0.85,
        "evidence_grounding_accuracy": 0.80,
    },
}

REGRESSION_PASS = {
    "baseline_id": "base",
    "candidate_id": "candidate",
    "regressions": [],
    "improvements": [],
    "unchanged": [],
    "uncomputable": [],
    "has_critical_regression": False,
    "has_safety_regression": False,
    "overall_verdict": "pass",
}

REDTEAM_CLEAN = {
    "num_cases": 10,
    "attacks_succeeded": 0,
    "attacks_blocked": 10,
    "metrics": {
        "threat_category": {"attacks_succeeded": 0, "attacks_blocked": 10}
    },
}


@pytest.mark.unit
class TestDeploymentVerifier:

    def test_import(self):
        from agent.evaluation.deployment_verifier import DeploymentVerifier
        assert DeploymentVerifier is not None

    def test_hold_when_no_data(self):
        from agent.evaluation.deployment_verifier import DeploymentVerifier
        verifier = DeploymentVerifier()
        result = verifier.verify(
            baseline_summary=None,
            candidate_summary=None,
            regression_report=None,
            redteam_summary=None,
            human_approved=False,
        )
        assert result.decision == "hold_keep_base"
        assert result.approved is False

    def test_hold_when_candidate_missing(self):
        from agent.evaluation.deployment_verifier import DeploymentVerifier
        verifier = DeploymentVerifier()
        result = verifier.verify(
            baseline_summary=BASELINE_SUMMARY,
            candidate_summary=None,
            regression_report=REGRESSION_PASS,
            redteam_summary=REDTEAM_CLEAN,
            human_approved=True,
        )
        assert result.decision == "hold_keep_base"
        assert result.approved is False

    def test_rejected_without_human_approval(self):
        from agent.evaluation.deployment_verifier import DeploymentVerifier
        verifier = DeploymentVerifier()
        result = verifier.verify(
            baseline_summary=BASELINE_SUMMARY,
            candidate_summary=CANDIDATE_SUMMARY,
            regression_report=REGRESSION_PASS,
            redteam_summary=REDTEAM_CLEAN,
            human_approved=False,  # <-- missing
        )
        assert result.approved is False
        assert result.checks["human_approved"] is False
        assert any("human" in b.lower() for b in result.blockers)

    def test_rejected_with_safety_regression(self):
        from agent.evaluation.deployment_verifier import DeploymentVerifier
        verifier = DeploymentVerifier()
        safety_regression = {
            **REGRESSION_PASS,
            "has_safety_regression": True,
            "has_critical_regression": True,
            "regressions": [{"metric": "safety_violation_count", "severity": "critical"}],
            "overall_verdict": "fail",
        }
        result = verifier.verify(
            baseline_summary=BASELINE_SUMMARY,
            candidate_summary=CANDIDATE_SUMMARY,
            regression_report=safety_regression,
            redteam_summary=REDTEAM_CLEAN,
            human_approved=True,
        )
        assert result.decision in {"rejected", "hold_keep_base"}
        assert result.approved is False

    def test_rejected_when_unauthorized_actions_succeeded(self):
        from agent.evaluation.deployment_verifier import DeploymentVerifier
        verifier = DeploymentVerifier()
        redteam_failed = {
            "num_cases": 10,
            "attacks_succeeded": 2,
            "metrics": {
                "threat_category": {"attacks_succeeded": 2, "attacks_blocked": 8}
            },
        }
        result = verifier.verify(
            baseline_summary=BASELINE_SUMMARY,
            candidate_summary=CANDIDATE_SUMMARY,
            regression_report=REGRESSION_PASS,
            redteam_summary=redteam_failed,
            human_approved=True,
        )
        assert result.checks["zero_unauthorized_actions"] is False
        assert result.approved is False

    def test_null_metrics_never_treated_as_passing(self):
        """
        CRITICAL: If eval data is missing (num_episodes=0), gate must hold.
        """
        from agent.evaluation.deployment_verifier import DeploymentVerifier
        verifier = DeploymentVerifier()
        empty_base = {"num_episodes": 0, "config_fingerprint": "a" * 64, "metrics": {}}
        result = verifier.verify(
            baseline_summary=empty_base,
            candidate_summary=CANDIDATE_SUMMARY,
            regression_report=REGRESSION_PASS,
            redteam_summary=REDTEAM_CLEAN,
            human_approved=True,
        )
        assert result.checks["real_baseline_evaluation"] is False
        assert result.approved is False

    def test_hold_when_provenance_missing(self):
        from agent.evaluation.deployment_verifier import DeploymentVerifier
        verifier = DeploymentVerifier()
        # Missing config_fingerprint
        base_no_fp = {**BASELINE_SUMMARY}
        base_no_fp.pop("config_fingerprint", None)
        result = verifier.verify(
            baseline_summary=base_no_fp,
            candidate_summary=CANDIDATE_SUMMARY,
            regression_report=REGRESSION_PASS,
            redteam_summary=REDTEAM_CLEAN,
            human_approved=True,
        )
        assert result.checks["evaluation_provenance_verified"] is False

    def test_result_to_dict_serializable(self):
        import json
        from agent.evaluation.deployment_verifier import DeploymentVerifier
        verifier = DeploymentVerifier()
        result = verifier.verify(
            baseline_summary=None,
            candidate_summary=None,
            regression_report=None,
            redteam_summary=None,
            human_approved=False,
        )
        d = result.to_dict()
        json_str = json.dumps(d, default=str)
        assert json_str is not None
        assert "decision" in d
        assert d["decision"] == "hold_keep_base"

