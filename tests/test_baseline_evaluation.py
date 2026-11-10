"""
tests/test_baseline_evaluation.py
===================================
Tests for the baseline evaluation framework.

Verifies:
  - Metric computations on synthetic result dicts
  - Trajectory logging and loading
  - Incident case dataset structure
  - Evaluator integration with mocked agent

No model download, no live database required.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers — synthetic result construction
# ---------------------------------------------------------------------------

def _make_result(
    success: bool = True,
    root_cause: str = "missing_suitable_index",
    tools_called: list[str] | None = None,
    num_steps: int = 3,
    safety_violations: list[str] | None = None,
    num_valid_calls: int = 1,
    num_rejected: int = 0,
    evidence: list[str] | None = None,
) -> dict:
    return {
        "incident_id": "slow_query_001",
        "query": "Why is my query slow?",
        "success": success,
        "diagnosis": "Sequential scan detected" if success else "",
        "root_cause": root_cause,
        "evidence": evidence if evidence is not None else ["Sequential scan", "No index on customer_id"],
        "recommended_action": "Create index",
        "requires_approval": True,
        "tools_called": tools_called if tools_called is not None else ["inspect_customer_id_index"],
        "num_steps": num_steps,
        "safety_violations": safety_violations if safety_violations is not None else [],
        "rejection_reasons": [],
        "total_time_ms": 1500.0,
        "confidence": 0.9 if success else 0.0,
        "error": "",
        "metadata": {
            "validator_summary": {
                "total_validated_calls": num_valid_calls,
                "rejected_calls": num_rejected,
            }
        },
        "_trajectory_steps": [],
    }


# ---------------------------------------------------------------------------
# Metric unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMetrics:
    def test_task_success_rate_all_success(self):
        from agent.evaluation.metrics import compute_task_success_rate
        results = [_make_result(success=True) for _ in range(5)]
        rate = compute_task_success_rate(results)
        assert rate == 1.0

    def test_task_success_rate_all_failure(self):
        from agent.evaluation.metrics import compute_task_success_rate
        results = [_make_result(success=False) for _ in range(5)]
        rate = compute_task_success_rate(results)
        assert rate == 0.0

    def test_task_success_rate_mixed(self):
        from agent.evaluation.metrics import compute_task_success_rate
        results = [
            _make_result(success=True),
            _make_result(success=True),
            _make_result(success=False),
            _make_result(success=False),
        ]
        rate = compute_task_success_rate(results)
        assert rate == pytest.approx(0.5, abs=0.01)

    def test_task_success_requires_non_empty_diagnosis(self):
        from agent.evaluation.metrics import compute_task_success_rate
        results = [
            {**_make_result(success=True), "diagnosis": ""},  # Empty diagnosis = not really success
        ]
        rate = compute_task_success_rate(results)
        assert rate == 0.0

    def test_diagnosis_accuracy_correct(self):
        from agent.evaluation.metrics import compute_diagnosis_accuracy
        results = [_make_result(root_cause="missing_suitable_index") for _ in range(3)]
        accuracy = compute_diagnosis_accuracy(results, "missing_suitable_index")
        assert accuracy == 1.0

    def test_diagnosis_accuracy_wrong_root_cause(self):
        from agent.evaluation.metrics import compute_diagnosis_accuracy
        results = [_make_result(root_cause="unknown")]
        accuracy = compute_diagnosis_accuracy(results, "missing_suitable_index")
        assert accuracy == 0.0

    def test_tool_call_validity_rate_all_valid(self):
        from agent.evaluation.metrics import compute_tool_call_validity_rate
        results = [_make_result(num_valid_calls=3, num_rejected=0) for _ in range(3)]
        rate = compute_tool_call_validity_rate(results)
        assert rate == 1.0

    def test_tool_call_validity_rate_some_rejected(self):
        from agent.evaluation.metrics import compute_tool_call_validity_rate
        results = [_make_result(num_valid_calls=3, num_rejected=1)]
        rate = compute_tool_call_validity_rate(results)
        assert rate == pytest.approx(0.75, abs=0.01)

    def test_average_tool_calls(self):
        from agent.evaluation.metrics import compute_average_tool_calls
        results = [
            _make_result(tools_called=["a", "b", "c"]),
            _make_result(tools_called=["c"]),
            _make_result(tools_called=[]),
        ]
        avg = compute_average_tool_calls(results)
        # (3 + 1 + 0) / 3 = 1.333...
        assert avg == pytest.approx(4 / 3, abs=0.01)

    def test_safety_violation_count_zero(self):
        from agent.evaluation.metrics import compute_safety_violation_count
        results = [_make_result(safety_violations=[]) for _ in range(5)]
        count = compute_safety_violation_count(results)
        assert count == 0

    def test_safety_violation_count_nonzero(self):
        from agent.evaluation.metrics import compute_safety_violation_count
        results = [
            _make_result(safety_violations=["SQL injection attempt"]),
            _make_result(safety_violations=["DROP TABLE rejected", "another"]),
        ]
        count = compute_safety_violation_count(results)
        assert count == 3

    def test_tool_selection_accuracy_relevant_tool_used(self):
        from agent.evaluation.metrics import compute_tool_selection_accuracy
        results = [_make_result(tools_called=["inspect_customer_id_index"])]
        accuracy = compute_tool_selection_accuracy(results)
        assert accuracy == 1.0

    def test_tool_selection_accuracy_no_relevant_tool(self):
        from agent.evaluation.metrics import compute_tool_selection_accuracy
        results = [_make_result(tools_called=["some_irrelevant_tool"])]
        accuracy = compute_tool_selection_accuracy(
            results,
            expected_relevant_tools=["inspect_customer_id_index", "diagnose_slow_query"],
        )
        assert accuracy == 0.0

    def test_empty_results_handled(self):
        from agent.evaluation.metrics import compute_all_metrics
        metrics = compute_all_metrics([])
        assert metrics.get("num_runs") == 0
        assert "error" in metrics

    def test_compute_all_metrics_structure(self):
        from agent.evaluation.metrics import compute_all_metrics
        results = [_make_result() for _ in range(3)]
        metrics = compute_all_metrics(results)
        required_keys = {
            "num_runs",
            "tool_call_validity_rate",
            "tool_selection_accuracy",
            "task_success_rate",
            "average_tool_calls_per_task",
            "average_steps_per_task",
            "safety_violation_count",
            "evidence_grounding_accuracy",
            "diagnosis_accuracy",
        }
        assert required_keys.issubset(set(metrics.keys()))

    def test_format_metrics_report_returns_string(self):
        from agent.evaluation.metrics import compute_all_metrics, format_metrics_report
        results = [_make_result()]
        metrics = compute_all_metrics(results)
        report = format_metrics_report(metrics)
        assert isinstance(report, str)
        assert "task_success_rate" in report


# ---------------------------------------------------------------------------
# Trajectory logger tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTrajectoryLogger:
    def _make_mock_result(self):
        result = MagicMock()
        result.incident_id = "slow_query_001"
        result.model_id = "CohereLabs/tiny-aya-global"
        result.model_revision = "abc123"
        result.query = "Why is my query slow?"
        result.success = True
        result.num_steps = 3
        result.tools_called = ["inspect_customer_id_index"]
        result.diagnosis = "Missing index"
        result.root_cause = "missing_suitable_index"
        result.evidence = ["No index exists"]
        result.recommended_action = "Create index"
        result.requires_approval = True
        result.confidence = 0.9
        result.safety_violations = []
        result.rejection_reasons = []
        result.total_time_ms = 1500.0
        result.error = ""
        result.metadata = {}

        step = MagicMock()
        step.step_index = 1
        step.step_type = "tool_execution"
        step.timestamp = "2026-01-01T00:00:00Z"
        step.is_valid = True
        step.model_response_text = ""
        step.tool_name = "inspect_customer_id_index"
        step.tool_arguments = {}
        step.tool_output = {"index_exists": False}
        step.validation_error = ""
        step.generation_time_ms = 100.0
        result.trajectory = [step]

        return result

    def test_log_writes_to_jsonl(self):
        from agent.evaluation.trajectory_logger import TrajectoryLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrajectoryLogger(tmpdir)
            mock_result = self._make_mock_result()
            logger.log(mock_result)

            assert (Path(tmpdir) / "trajectories.jsonl").exists()
            lines = (Path(tmpdir) / "trajectories.jsonl").read_text().strip().split("\n")
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["incident_id"] == "slow_query_001"

    def test_log_multiple_trajectories(self):
        from agent.evaluation.trajectory_logger import TrajectoryLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrajectoryLogger(tmpdir)
            for _ in range(3):
                logger.log(self._make_mock_result())

            entries = logger.load_all()
            assert len(entries) == 3

    def test_secrets_not_logged(self):
        from agent.evaluation.trajectory_logger import TrajectoryLogger, _redact_secrets
        sensitive = {
            "HF_TOKEN": "hf_secret_token_1234",
            "data": "normal_data",
            "nested": {"POSTGRES_PASSWORD": "super_secret"},
        }
        redacted = _redact_secrets(sensitive)
        assert redacted["HF_TOKEN"] == "[REDACTED]"
        assert redacted["data"] == "normal_data"
        assert redacted["nested"]["POSTGRES_PASSWORD"] == "[REDACTED]"

    def test_load_all_returns_empty_on_missing_file(self):
        from agent.evaluation.trajectory_logger import TrajectoryLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrajectoryLogger(tmpdir)
            entries = logger.load_all()
            assert entries == []


# ---------------------------------------------------------------------------
# Incident cases dataset tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIncidentCases:
    def test_at_least_10_baseline_cases(self):
        from agent.evaluation.incident_cases import BASELINE_EVAL_CASES
        assert len(BASELINE_EVAL_CASES) >= 10

    def test_all_baseline_cases_have_required_fields(self):
        from agent.evaluation.incident_cases import BASELINE_EVAL_CASES
        for case in BASELINE_EVAL_CASES:
            assert case.incident_id, f"Missing incident_id"
            assert case.query, f"Missing query in {case.incident_id}"
            assert case.expected_root_cause, f"Missing expected_root_cause in {case.incident_id}"
            assert case.expected_tools_to_call, f"Missing expected_tools in {case.incident_id}"

    def test_all_incident_ids_unique(self):
        from agent.evaluation.incident_cases import ALL_CASES
        ids = [c.incident_id for c in ALL_CASES]
        assert len(ids) == len(set(ids)), "Duplicate incident IDs found"

    def test_baseline_cases_marked_correctly(self):
        from agent.evaluation.incident_cases import BASELINE_EVAL_CASES
        for case in BASELINE_EVAL_CASES:
            assert case.split == "baseline"

    def test_hidden_cases_not_in_baseline(self):
        from agent.evaluation.incident_cases import BASELINE_EVAL_CASES, HIDDEN_EVAL_CASES
        baseline_ids = {c.incident_id for c in BASELINE_EVAL_CASES}
        hidden_ids = {c.incident_id for c in HIDDEN_EVAL_CASES}
        assert not baseline_ids & hidden_ids, "Hidden cases must not appear in baseline"

    def test_get_case_by_id_returns_correct_case(self):
        from agent.evaluation.incident_cases import get_case_by_id
        case = get_case_by_id("slow_query_001")
        assert case is not None
        assert case.incident_id == "slow_query_001"

    def test_get_case_by_id_returns_none_for_unknown(self):
        from agent.evaluation.incident_cases import get_case_by_id
        assert get_case_by_id("nonexistent_case_xyz") is None

    def test_expected_root_cause_matches_standard(self):
        from agent.evaluation.incident_cases import BASELINE_EVAL_CASES
        # All slow-query baseline cases should have the same root cause
        for case in BASELINE_EVAL_CASES:
            assert case.expected_root_cause == "missing_suitable_index"
