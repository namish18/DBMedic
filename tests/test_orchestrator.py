"""
tests/test_orchestrator.py
===========================
Tests for the AgentOrchestrator — multi-step loop, parse retries,
max step enforcement, and safe handling of model failures.

All tests use mocked model client — no model download required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_model_client(responses: list[str]):
    """Mock model client that returns responses in sequence."""
    client = MagicMock()
    client.model_id = "CohereLabs/tiny-aya-global"
    client.model_revision = "mock_revision"
    call_count = [0]

    def generate(messages, **kwargs):
        idx = min(call_count[0], len(responses) - 1)
        text = responses[idx]
        call_count[0] += 1
        return {
            "text": text,
            "generation_time_ms": 100.0,
            "model_id": "CohereLabs/tiny-aya-global",
            "model_revision": "mock_revision",
        }

    client.generate.side_effect = generate
    return client


def _make_registry_with_tool(tool_output=None):
    """Mock registry that executes a single mock tool."""
    registry = MagicMock()
    registry.has_tool.return_value = True
    registry.tool_names = ["inspect_customer_id_index", "diagnose_slow_query"]
    registry.get_schema.return_value = {
        "name": "inspect_customer_id_index",
        "description": "Inspect index",
        "parameters": {},
        "required": [],
    }
    registry.execute.return_value = tool_output or {
        "index_exists": False,
        "table_name": "orders",
        "target_column": "customer_id",
    }
    return registry


def _make_validator(valid: bool = True):
    """Mock validator that always passes or always fails."""
    from agent.core.tool_validator import ValidationResult

    validator = MagicMock()
    validator.session_summary.return_value = {
        "total_validated_calls": 1 if valid else 0,
        "calls_per_tool": {},
        "rejected_calls": 0 if valid else 1,
        "rejections": [],
        "calls_remaining": 4,
    }

    result = ValidationResult(
        is_valid=valid,
        tool_name="inspect_customer_id_index",
        arguments={},
        rejection_reason="" if valid else "Test rejection",
        sanitized_arguments={},
    )
    validator.validate.return_value = result
    return validator


def _make_orchestrator(model_client, registry=None, validator=None, max_steps=5):
    from agent.core.orchestrator import AgentOrchestrator
    from agent.core.response_parser import ResponseParser

    registry = registry or _make_registry_with_tool()
    validator = validator or _make_validator(valid=True)
    parser = ResponseParser()

    return AgentOrchestrator(
        model_client=model_client,
        tool_registry=registry,
        tool_validator=validator,
        response_parser=parser,
        max_steps=max_steps,
        max_tool_calls=3,
    )


@pytest.mark.unit
class TestOrchestratorSingleStep:
    def test_immediate_final_answer(self):
        """Model provides a final answer on first response."""
        response = '{"type": "final_answer", "diagnosis": "Missing index on customer_id", "root_cause": "missing_suitable_index", "evidence": ["Sequential scan detected"], "recommended_action": "Create index", "requires_approval": true, "confidence": 0.9}'

        client = _make_model_client([response])
        orchestrator = _make_orchestrator(client)

        result = orchestrator.run("Why is my query slow?")

        assert result.success is True
        assert result.root_cause == "missing_suitable_index"
        assert "Missing index" in result.diagnosis
        assert result.confidence == 0.9

    def test_tool_call_then_final_answer(self):
        """Model calls a tool then provides final answer."""
        tool_call = '{"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}, "reasoning": "Check if index exists"}'
        final = '{"type": "final_answer", "diagnosis": "No index on customer_id", "root_cause": "missing_suitable_index", "evidence": ["Index does not exist"], "recommended_action": "Create index", "requires_approval": true, "confidence": 0.85}'

        client = _make_model_client([tool_call, final])
        orchestrator = _make_orchestrator(client)

        result = orchestrator.run("What is wrong with orders?")

        assert result.success is True
        assert len(result.tools_called) == 1
        assert result.tools_called[0] == "inspect_customer_id_index"

    def test_trajectory_has_correct_steps(self):
        """Trajectory records all steps correctly."""
        tool_call = '{"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}}'
        final = '{"type": "final_answer", "diagnosis": "Missing index", "root_cause": "missing_suitable_index", "evidence": [], "recommended_action": "Create index", "requires_approval": true, "confidence": 0.8}'

        client = _make_model_client([tool_call, final])
        orchestrator = _make_orchestrator(client)
        result = orchestrator.run("Diagnose query")

        step_types = [s.step_type for s in result.trajectory]
        assert "user_question" in step_types
        assert "tool_execution" in step_types
        assert "final_answer" in step_types


@pytest.mark.unit
class TestOrchestratorMaxSteps:
    def test_max_steps_enforced(self):
        """Agent stops after max_steps even without final answer."""
        # Always return a tool call (never a final answer)
        tool_call = '{"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}}'

        client = _make_model_client([tool_call] * 20)
        orchestrator = _make_orchestrator(client, max_steps=3)

        result = orchestrator.run("Infinite loop test")

        # Should not hang; steps bounded
        assert result.num_steps <= 5  # Allow some buffer for retry steps

    def test_failed_run_has_success_false(self):
        """When agent never produces final answer, success=False."""
        # Always return a JSON parse failure
        client = _make_model_client(["NOT JSON AT ALL"] * 10)
        orchestrator = _make_orchestrator(client, max_steps=3)

        result = orchestrator.run("Test parse failure handling")

        assert result.success is False


@pytest.mark.unit
class TestOrchestratorValidationRejection:
    def test_rejected_tool_call_does_not_execute(self):
        """When validator rejects, tool is never executed."""
        tool_call = '{"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}}'
        final = '{"type": "final_answer", "diagnosis": "Could not diagnose", "root_cause": "unknown", "evidence": [], "recommended_action": "none", "requires_approval": false, "confidence": 0.1}'

        client = _make_model_client([tool_call, final])
        validator = _make_validator(valid=False)
        registry = _make_registry_with_tool()

        orchestrator = _make_orchestrator(client, registry=registry, validator=validator)
        result = orchestrator.run("Test rejection")

        # Registry execute should NOT have been called
        registry.execute.assert_not_called()

    def test_rejection_recorded_in_result(self):
        """Rejected tool calls appear in rejection_reasons."""
        tool_call = '{"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}}'
        final = '{"type": "final_answer", "diagnosis": "Rejected", "root_cause": "unknown", "evidence": [], "recommended_action": "none", "requires_approval": false, "confidence": 0.1}'

        client = _make_model_client([tool_call, final])
        validator = _make_validator(valid=False)

        orchestrator = _make_orchestrator(client, validator=validator)
        result = orchestrator.run("Test rejection tracking")

        assert len(result.rejection_reasons) >= 1


@pytest.mark.unit
class TestOrchestratorModelFailure:
    def test_inference_error_handled_gracefully(self):
        """InferenceError from model client is caught and handled."""
        from agent.core.model_client import InferenceError

        client = MagicMock()
        client.model_id = "CohereLabs/tiny-aya-global"
        client.model_revision = None
        client.generate.side_effect = InferenceError("GPU OOM")

        orchestrator = _make_orchestrator(client, max_steps=2)
        result = orchestrator.run("Test model failure")

        assert result.success is False
        assert result.error  # Has some error message

    def test_tool_execution_failure_handled(self):
        """Tool runtime errors don't crash the agent."""
        tool_call = '{"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}}'
        final = '{"type": "final_answer", "diagnosis": "Tool failed but continuing", "root_cause": "unknown", "evidence": [], "recommended_action": "retry", "requires_approval": false, "confidence": 0.2}'

        client = _make_model_client([tool_call, final])
        registry = _make_registry_with_tool()
        registry.execute.side_effect = RuntimeError("DB connection refused")

        orchestrator = _make_orchestrator(client, registry=registry)
        result = orchestrator.run("Test tool failure")

        # Agent should still produce a result (possibly with error in diagnosis)
        assert result is not None
        # Tool was attempted
        assert "inspect_customer_id_index" in result.tools_called

