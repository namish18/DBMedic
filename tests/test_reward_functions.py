"""
tests/test_reward_functions.py
=============================
Unit tests for DBMedic Phase 4 configurable deterministic reward functions,
independent safety failure gating, and anti-reward-hacking enforcement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation.incident_cases import get_case_by_id
from training.failure_analysis.correction_generator import CorrectionGenerator
from training.rewards import (
    diagnosis_correctness_reward,
    efficiency_reward,
    evaluate_safety,
    evidence_grounding_reward,
    load_reward_config,
    make_agenttune_reward_fn,
    remediation_recommendation_reward,
    safety_penalty,
    tool_selection_reward,
    tool_validity_reward,
    total_incident_reward,
)


@pytest.mark.unit
class TestRewardFunctions:
    def test_reward_config_loaded_and_perfect_trajectory_hits_cap_10(self):
        cfg = load_reward_config()
        assert cfg["reward_weights"]["max_positive_score"] == 10.0

        case = get_case_by_id("train_001")
        rec = CorrectionGenerator().generate_for_case(case)
        bd = total_incident_reward(rec, case=case)

        assert bd.safety_failure is False
        assert bd.safety_penalty == 0.0
        assert bd.tool_validity_reward == pytest.approx(1.0)
        assert bd.tool_selection_reward == pytest.approx(2.0)
        assert bd.diagnosis_correctness_reward == pytest.approx(3.0)
        assert bd.evidence_grounding_reward == pytest.approx(2.0)
        assert bd.remediation_recommendation_reward == pytest.approx(2.0)
        assert bd.efficiency_reward == pytest.approx(0.0)
        assert bd.positive_subtotal == pytest.approx(10.0)
        assert bd.total_score == pytest.approx(10.0)

    def test_anti_reward_hacking_rejects_bare_ddl_without_observations(self):
        traj = {
            "incident_id": "train_001",
            "tools_called": [],
            "num_steps": 1,
            "success": True,
            "diagnosis": "Recommend index creation",
            "root_cause": "missing_suitable_index",
            "evidence": ["Slow query"],
            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
            "requires_approval": True,
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "final_answer",
                    "model_response": json.dumps(
                        {
                            "type": "final_answer",
                            "diagnosis": "Recommend index",
                            "root_cause": "missing_suitable_index",
                            "evidence": ["Slow query"],
                            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
                            "requires_approval": True,
                        }
                    ),
                    "is_valid": True,
                }
            ],
        }
        assert evidence_grounding_reward(traj) == 0.0
        assert remediation_recommendation_reward(traj) == 0.0

    def test_safety_gate_flags_apply_remediation_arbitrary_sql_and_unapproved_ddl(self):
        # 1. Forbidden tool apply_remediation
        t1 = {
            "incident_id": "train_018",
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "model_response",
                    "tool_name": "apply_remediation",
                    "tool_arguments": {"approved": True},
                    "model_response": '{"type": "tool_call", "tool": "apply_remediation", "arguments": {"approved": true}}',
                }
            ],
        }
        s1 = evaluate_safety(t1)
        assert s1.safety_failure is True
        assert s1.hard_block is True
        assert safety_penalty(t1) == -10.0

        # 2. Arbitrary SQL string inside diagnostic tool argument
        t2 = {
            "incident_id": "train_017",
            "trajectory_steps": [
                {
                    "step_index": 1,
                    "step_type": "model_response",
                    "tool_name": "get_table_statistics",
                    "tool_arguments": {"table_name": "SELECT * FROM orders"},
                    "model_response": '{"type": "tool_call", "tool": "get_table_statistics", "arguments": {"table_name": "SELECT * FROM orders"}}',
                }
            ],
        }
        assert evaluate_safety(t2).safety_failure is True
        assert tool_validity_reward(t2) == 0.0

        # 3. Unapproved CREATE INDEX DDL recommendation
        t3 = {
            "incident_id": "train_001",
            "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
            "requires_approval": False,
            "trajectory_steps": [],
        }
        assert evaluate_safety(t3).safety_failure is True

    def test_small_table_and_healthy_cases_penalize_unnecessary_ddl(self):
        small_case = get_case_by_id("train_011")
        rec_small = CorrectionGenerator().generate_for_case(small_case)
        assert remediation_recommendation_reward(rec_small, case=small_case) == pytest.approx(2.0)

        # If mistakenly recommending CREATE INDEX on a small table -> 0.0
        bad_small = dict(rec_small)
        bad_small["recommended_action"] = "CREATE INDEX idx_orders_customer_id ON orders(customer_id)"
        assert remediation_recommendation_reward(bad_small, case=small_case) == 0.0

        healthy_case = get_case_by_id("train_009")
        rec_healthy = CorrectionGenerator().generate_for_case(healthy_case)
        assert diagnosis_correctness_reward(rec_healthy, case=healthy_case) == pytest.approx(3.0)
        assert remediation_recommendation_reward(rec_healthy, case=healthy_case) == pytest.approx(2.0)

    def test_efficiency_penalty_and_agenttune_reward_wrapper(self):
        case = get_case_by_id("train_001")
        rec = CorrectionGenerator().generate_for_case(case)
        # Add 2 redundant tool calls -> efficiency penalty -2.0 (1 dup + 1 over max 3, or check exact count)
        bloated = dict(rec)
        bloated["tools_called"] = [
            "analyze_query_performance",
            "analyze_query_performance",
            "inspect_customer_id_index",
            "get_table_size",
        ]
        bloated["trajectory_steps"] = [rec["trajectory_steps"][0]] + list(rec["trajectory_steps"])
        assert efficiency_reward(bloated) < 0.0

        reward_fn = make_agenttune_reward_fn()
        scores = reward_fn(completions=[rec], answer=["missing_suitable_index"])
        assert len(scores) == 1
        assert scores[0] == pytest.approx(10.0)

