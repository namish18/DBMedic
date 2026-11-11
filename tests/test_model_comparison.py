"""
tests/test_model_comparison.py
=============================
Unit tests for Phase 4 Base vs Candidate comparison across all 8 Phase 3 metrics,
AgentTune DeploymentGate + DBMedic safety/human-approval gating, and full 8-stage
SelfHealingPipeline execution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evaluation.incident_cases import HIDDEN_EVAL_CASES, TRAINING_CANDIDATES, get_case_by_id
from agent.evaluation.training_comparison import (
    METRIC_KEYS,
    ModelComparisonEvaluator,
    SelfHealingPipeline,
    evaluate_deployment_gate,
)
from training.failure_analysis.correction_generator import CorrectionGenerator


def _make_eval_run(
    incident_id: str,
    *,
    success: bool = True,
    root_cause: str = "missing_suitable_index",
    valid_calls: int = 1,
    rejected_calls: int = 0,
    safety_violations: list[str] | None = None,
    requires_approval: bool = True,
) -> dict:
    case = get_case_by_id(incident_id)
    obs = {
        "scan_type": "Sequential Scan",
        "table_name": "orders",
        "target_column": "customer_id",
        "index_exists": False,
    }
    return {
        "incident_id": f"{incident_id}_run1",
        "query": case.query if case else "Slow orders query",
        "success": success,
        "diagnosis": "Sequential scan on orders due to missing index on customer_id" if success else "",
        "root_cause": root_cause,
        "evidence": ["Sequential scan on orders", "customer_id index_exists false"] if success else [],
        "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id) (requires human approval)",
        "requires_approval": requires_approval,
        "tools_called": ["analyze_query_performance", "inspect_customer_id_index"] if valid_calls else [],
        "num_steps": 3,
        "safety_violations": safety_violations or [],
        "rejection_reasons": [],
        "metadata": {
            "validator_summary": {
                "total_validated_calls": valid_calls,
                "rejected_calls": rejected_calls,
            }
        },
        "ground_truth": {"expected_root_cause": case.expected_root_cause if case else "missing_suitable_index"},
        "trajectory_steps": [
            {
                "step_index": 1,
                "step_type": "tool_execution",
                "tool_name": "analyze_query_performance",
                "tool_arguments": {"customer_id": 4242},
                "tool_output": obs,
                "model_response": json.dumps({"type": "tool_call", "tool": "analyze_query_performance", "arguments": {"customer_id": 4242}}),
                "is_valid": True,
            },
            {
                "step_index": 2,
                "step_type": "final_answer",
                "model_response": json.dumps(
                    {
                        "type": "final_answer",
                        "diagnosis": "Sequential scan on orders",
                        "root_cause": root_cause,
                        "evidence": ["Sequential scan on orders", "customer_id index_exists false"],
                        "recommended_action": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
                        "requires_approval": requires_approval,
                    }
                ),
                "is_valid": True,
            },
        ],
    }


@pytest.mark.unit
class TestModelComparisonAndSelfHealing:
    def test_pending_comparison_reports_none_without_fabricating(self, tmp_path: Path):
        comp = ModelComparisonEvaluator().compare(
            base_dir=PROJECT_ROOT / "training" / "experiments" / "baseline",
            candidate_dir=tmp_path,
        )
        assert comp["status"] == "pending_live_evaluation"
        assert comp["base_runs_count"] == 0
        assert comp["candidate_runs_count"] == 0
        for k in METRIC_KEYS:
            assert comp["metrics_table"]["base"][k] is None
            assert comp["metrics_table"]["candidate"][k] is None
            assert comp["metrics_table"]["delta"][k] is None
        assert comp["deployment_gate"]["approved"] is False

    def test_deployment_gate_blocks_without_human_approval_and_on_safety_or_regression(self):
        base = [
            _make_eval_run("slow_query_001", success=False, root_cause="unknown", valid_calls=0, rejected_calls=1),
            _make_eval_run("slow_query_002", success=True, root_cause="missing_suitable_index", valid_calls=2),
        ]
        better_cand = [
            _make_eval_run("slow_query_001", success=True, root_cause="missing_suitable_index", valid_calls=2),
            _make_eval_run("slow_query_002", success=True, root_cause="missing_suitable_index", valid_calls=2),
        ]

        # 1. Even with clear improvement, human_approved=False MUST block deployment!
        g_no_human = evaluate_deployment_gate(base_runs=base, candidate_runs=better_cand, human_approved=False)
        assert g_no_human.agenttune_gate_approved is True
        assert g_no_human.approved is False
        assert g_no_human.checks["human_approved"] is False

        # 2. Candidate with safety violation -> blocked even if human_approved=True!
        unsafe_cand = [
            better_cand[0],
            _make_eval_run("slow_query_002", success=True, safety_violations=["Attempted apply_remediation"]),
        ]
        g_unsafe = evaluate_deployment_gate(base_runs=base, candidate_runs=unsafe_cand, human_approved=True)
        assert g_unsafe.approved is False
        assert g_unsafe.checks["zero_critical_safety_violations"] is False

        # 3. Regressed hidden test -> blocked!
        hb = [_make_eval_run(HIDDEN_EVAL_CASES[0].incident_id, success=True, root_cause=HIDDEN_EVAL_CASES[0].expected_root_cause)]
        hc = [_make_eval_run(HIDDEN_EVAL_CASES[0].incident_id, success=True, root_cause="wrong_cause")]
        g_hidden_reg = evaluate_deployment_gate(
            base_runs=base,
            candidate_runs=better_cand,
            hidden_base_runs=hb,
            hidden_candidate_runs=hc,
            human_approved=True,
        )
        assert g_hidden_reg.approved is False
        assert g_hidden_reg.checks["hidden_split_non_regression"] is False

        # 4. All checks pass + human_approved=True -> approved!
        g_ok = evaluate_deployment_gate(
            base_runs=base,
            candidate_runs=better_cand,
            hidden_base_runs=hb,
            hidden_candidate_runs=hb,
            human_approved=True,
        )
        assert g_ok.approved is True
        assert g_ok.recommendation == "approve_candidate_for_deployment"

    def test_self_healing_pipeline_end_to_end_eight_stages(self, tmp_path: Path):
        # Raw failing trajectory with wrong root cause
        bad_run = _make_eval_run("train_001", success=True, root_cause="network_timeout", valid_calls=1)

        class _FakeTrainer:
            def train(self):
                return {"training_loss": 0.12, "status": "ok"}

        pipeline = SelfHealingPipeline(
            trainer_factory=lambda **kw: _FakeTrainer(),
        )
        report = pipeline.run_cycle(
            raw_trajectories=[bad_run],
            curated_output_dir=tmp_path / "curated",
            base_runs=[bad_run],
            candidate_eval_fn=lambda runs: [
                _make_eval_run("train_001", success=True, root_cause="missing_suitable_index", valid_calls=2)
            ],
            human_approved=True,
        )
        stages = report["stages"]
        assert stages["1_collect"]["num_trajectories"] == 1
        assert stages["2_classify"]["failure_records"] >= 1
        assert stages["3_generate_correction"]["attempted"] == 1
        assert stages["4_verify_correction"]["verified_count"] == 1
        assert stages["5_curate_dataset"]["train_count"] >= 18
        assert stages["6_retrain_candidate"]["executed"] is True
        assert stages["7_evaluate_candidate"]["evaluation_status"] == "completed"
        assert stages["8_deployment_gate"]["approved"] is True

