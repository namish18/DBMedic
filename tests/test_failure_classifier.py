"""
tests/test_failure_classifier.py
================================
Unit tests for DBMedic Phase 4 10-category FailureClassifier and TrajectoryFailureAnalyzer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation.incident_cases import TRAINING_CANDIDATES, get_case_by_id
from training.failure_analysis.analyzer import TrajectoryFailureAnalyzer
from training.failure_analysis.classifier import FAILURE_CATEGORIES, FailureClassifier


def _valid_trajectory(incident_id: str = "train_001") -> dict:
    case = get_case_by_id(incident_id)
    assert case is not None
    tool_call_msg = json.dumps(
        {
            "type": "tool_call",
            "tool": "analyze_query_performance",
            "arguments": {"customer_id": case.customer_id},
            "reasoning": "Inspect real execution timing and sequential scan.",
        }
    )
    tool_call_idx = json.dumps(
        {
            "type": "tool_call",
            "tool": "inspect_customer_id_index",
            "arguments": {},
            "reasoning": "Verify whether orders.customer_id has an index.",
        }
    )
    final_msg = json.dumps(
        {
            "type": "final_answer",
            "diagnosis": "The orders table is experiencing a Sequential Scan (~18.4 ms) due to no suitable index on customer_id.",
            "root_cause": "missing_suitable_index",
            "evidence": [
                "Execution plan ran Sequential Scan on 200,000 rows in orders.",
                "inspect_customer_id_index returned index_exists=false.",
            ],
            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id) (requires human approval)",
            "requires_approval": True,
            "confidence": 0.95,
        }
    )
    return {
        "trajectory_id": f"traj_{incident_id}",
        "incident_id": incident_id,
        "query": case.query,
        "tools_called": ["analyze_query_performance", "inspect_customer_id_index"],
        "num_steps": 3,
        "success": True,
        "diagnosis": "Sequential Scan (~18.4 ms) due to missing index on customer_id in orders.",
        "root_cause": "missing_suitable_index",
        "evidence": [
            "Sequential Scan on 200,000 rows in orders (18.4 ms).",
            "orders.customer_id has index_exists=false.",
        ],
        "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
        "requires_approval": True,
        "safety_violations": [],
        "trajectory_steps": [
            {
                "step_index": 1,
                "step_type": "tool_execution",
                "model_response": tool_call_msg,
                "tool_name": "analyze_query_performance",
                "tool_arguments": {"customer_id": case.customer_id},
                "tool_output": {"scan_type": "Seq Scan", "execution_time_ms": 18.42, "table_name": "orders"},
                "is_valid": True,
            },
            {
                "step_index": 2,
                "step_type": "tool_execution",
                "model_response": tool_call_idx,
                "tool_name": "inspect_customer_id_index",
                "tool_arguments": {},
                "tool_output": {"table_name": "orders", "target_column": "customer_id", "index_exists": False},
                "is_valid": True,
            },
            {
                "step_index": 3,
                "step_type": "final_answer",
                "model_response": final_msg,
                "is_valid": True,
            },
        ],
    }


@pytest.mark.unit
class TestFailureClassifier:
    def test_all_ten_categories_constant_present(self):
        assert len(FAILURE_CATEGORIES) == 10
        assert set(FAILURE_CATEGORIES) == {
            "malformed_tool_call",
            "invalid_tool_arguments",
            "incorrect_tool_selection",
            "unnecessary_tool_call",
            "incomplete_diagnosis",
            "incorrect_root_cause",
            "unsupported_claim",
            "excessive_agent_steps",
            "unsafe_action_attempt",
            "model_termination_failure",
        }

    def test_valid_paraphrased_trajectory_passes_without_failures(self):
        clf = FailureClassifier()
        traj = _valid_trajectory("train_001")
        # Paraphrase diagnosis phrasing cleanly — must stay tolerated!
        traj["diagnosis"] = "Customer lookup performs full seq scan across orders since no index covers customer_id."
        failures = clf.classify(traj, case=TRAINING_CANDIDATES[0])
        assert failures == []

    def test_classifies_malformed_tool_call(self):
        clf = FailureClassifier()
        traj = _valid_trajectory()
        traj["trajectory_steps"][0] = {
            "step_index": 1,
            "step_type": "model_response",
            "model_response": "Not JSON at all",
            "is_valid": False,
            "validation_error": "Parse failure",
        }
        cats = {r.failure_category for r in clf.classify(traj)}
        assert "malformed_tool_call" in cats

    def test_classifies_invalid_tool_arguments(self):
        clf = FailureClassifier()
        traj = _valid_trajectory()
        traj["trajectory_steps"][0]["tool_arguments"] = {"table_name": "secret_table"}
        traj["trajectory_steps"][0]["is_valid"] = False
        traj["trajectory_steps"][0]["validation_error"] = "Invalid table_name"
        cats = {r.failure_category for r in clf.classify(traj)}
        assert "invalid_tool_arguments" in cats

    def test_classifies_incorrect_tool_selection(self):
        clf = FailureClassifier()
        traj = _valid_trajectory("train_004")
        traj["tools_called"] = ["get_table_size"]
        cats = {r.failure_category for r in clf.classify(traj, case=get_case_by_id("train_004"))}
        assert "incorrect_tool_selection" in cats

    def test_classifies_unnecessary_tool_call(self):
        clf = FailureClassifier()
        traj = _valid_trajectory()
        # Repeat identical call
        step1 = dict(traj["trajectory_steps"][0])
        traj["trajectory_steps"].insert(1, step1)
        traj["tools_called"] = ["analyze_query_performance", "analyze_query_performance", "inspect_customer_id_index"]
        cats = {r.failure_category for r in clf.classify(traj)}
        assert "unnecessary_tool_call" in cats

    def test_classifies_incomplete_diagnosis(self):
        clf = FailureClassifier()
        traj = _valid_trajectory()
        traj["diagnosis"] = ""
        traj["root_cause"] = "unknown"
        traj["evidence"] = []
        cats = {r.failure_category for r in clf.classify(traj)}
        assert "incomplete_diagnosis" in cats

    def test_classifies_incorrect_root_cause(self):
        clf = FailureClassifier()
        traj = _valid_trajectory()
        traj["root_cause"] = "disk_corruption"
        cats = {r.failure_category for r in clf.classify(traj)}
        assert "incorrect_root_cause" in cats

    def test_classifies_unsupported_claim_on_contradiction(self):
        clf = FailureClassifier()
        traj = _valid_trajectory()
        traj["evidence"] = ["An active index exists on customer_id partitioning"]
        traj["diagnosis"] = "index exists on customer_id already."
        cats = {r.failure_category for r in clf.classify(traj)}
        assert "unsupported_claim" in cats

    def test_classifies_excessive_agent_steps(self):
        clf = FailureClassifier(max_steps_threshold=8)
        traj = _valid_trajectory()
        traj["num_steps"] = 10
        cats = {r.failure_category for r in clf.classify(traj)}
        assert "excessive_agent_steps" in cats

    def test_classifies_unsafe_action_attempt_and_forbidden_tool(self):
        clf = FailureClassifier()
        traj = _valid_trajectory()
        traj["trajectory_steps"][0]["tool_name"] = "apply_remediation"
        traj["requires_approval"] = False
        cats = {r.failure_category for r in clf.classify(traj)}
        assert "unsafe_action_attempt" in cats

    def test_classifies_model_termination_failure(self):
        clf = FailureClassifier()
        traj = _valid_trajectory()
        traj["success"] = False
        traj["trajectory_steps"] = traj["trajectory_steps"][:1]
        cats = {r.failure_category for r in clf.classify(traj)}
        assert "model_termination_failure" in cats

    def test_batch_analyzer_aggregates_counts(self):
        analyzer = TrajectoryFailureAnalyzer()
        good = _valid_trajectory("train_001")
        bad = _valid_trajectory("train_002")
        bad["root_cause"] = "network_jitter"
        report = analyzer.analyze_batch([good, bad])
        assert report["total_trajectories"] == 2
        assert report["passed_trajectories"] == 1
        assert report["failed_trajectories"] == 1
        assert report["failure_counts_by_category"]["incorrect_root_cause"] == 1

