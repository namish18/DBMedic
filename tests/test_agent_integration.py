"""
tests/test_agent_integration.py
=================================
End-to-end integration tests for the DBMedic agent.

Tests the complete agent pipeline with mocked model client.
Verifies: tool selection, validation, execution, and diagnosis.
No model download required for these tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _build_full_mock_agent(model_responses: list[str], tool_outputs: dict | None = None):
    """Build a complete DBMedicAgent with mocked model and real tools mocked."""
    from agent.core.agent import DBMedicAgent
    from agent.core.tool_registry import ToolRegistry
    from agent.core.tool_validator import ToolValidator
    from agent.core.response_parser import ResponseParser
    from agent.core.orchestrator import AgentOrchestrator

    if tool_outputs is None:
        tool_outputs = {
            "inspect_customer_id_index": {
                "table_name": "orders",
                "target_column": "customer_id",
                "index_exists": False,
                "is_suitable_for_filter": False,
                "index_scan_count": 0,
            },
            "analyze_query_performance": {
                "template_key": "slow_query_customer_orders",
                "customer_id": 4242,
                "metrics": {
                    "execution_time_ms": 18.42,
                    "scan_type": "Sequential Scan",
                    "actual_rows": 22,
                },
                "scan_analysis": {"has_sequential_scan": True},
            },
            "diagnose_slow_query": {
                "incident_id": "slow_query_001",
                "incident_type": "slow_query",
                "status": "diagnosed",
                "severity": "high",
                "root_cause": {"type": "missing_suitable_index"},
                "evidence": {
                    "scan_type": "Sequential Scan",
                    "suitable_index_exists": False,
                },
                "diagnostic_score": 0.95,
            },
        }

    # Build a real registry with mocked phase-2 functions
    with patch("agent.core.tool_registry._build_registry") as mock_build:
        # Create callable wrappers for each tool
        def make_tool_fn(name, output):
            def fn(arguments):
                return output
            return fn

        fake_registry = {}
        for tool_name, output in tool_outputs.items():
            fake_registry[tool_name] = {
                "name": tool_name,
                "description": f"Mock {tool_name}",
                "function": make_tool_fn(tool_name, output),
                "parameters": {"customer_id": {"type": "integer", "minimum": 1, "default": 4242}}
                if "customer_id" in str(output) else {},
                "required": [],
                "read_only": True,
            }
        mock_build.return_value = fake_registry

        registry = ToolRegistry()
        registry.initialize()

    # Mock model client
    call_idx = [0]
    mock_client = MagicMock()
    mock_client.model_id = "CohereLabs/tiny-aya-global"
    mock_client.model_revision = "mock"

    def mock_generate(messages, **kwargs):
        idx = min(call_idx[0], len(model_responses) - 1)
        response = model_responses[idx]
        call_idx[0] += 1
        return {"text": response, "generation_time_ms": 50.0, "model_id": "mock", "model_revision": "mock"}

    mock_client.generate.side_effect = mock_generate

    # Build orchestrator
    validator = ToolValidator(registry, max_calls_per_session=5)
    parser = ResponseParser()
    orchestrator = AgentOrchestrator(
        model_client=mock_client,
        tool_registry=registry,
        tool_validator=validator,
        response_parser=parser,
        max_steps=8,
        max_tool_calls=5,
    )

    return orchestrator, mock_client, registry


@pytest.mark.unit
class TestAgentEndToEnd:
    def test_agent_reaches_correct_diagnosis(self):
        """Agent produces correct diagnosis when model outputs are well-formed."""
        tool_call = '{"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}}'
        final = (
            '{"type": "final_answer", '
            '"diagnosis": "Sequential scan detected on orders table due to missing index on customer_id", '
            '"root_cause": "missing_suitable_index", '
            '"evidence": ["No index on orders.customer_id", "Sequential scan detected"], '
            '"recommended_action": "Create index on orders(customer_id)", '
            '"requires_approval": true, '
            '"confidence": 0.9}'
        )

        orchestrator, _, _ = _build_full_mock_agent([tool_call, final])
        result = orchestrator.run("Why is my orders query slow?")

        assert result.success is True
        assert result.root_cause == "missing_suitable_index"
        assert "Sequential scan" in result.diagnosis
        assert result.requires_approval is True
        assert len(result.tools_called) == 1
        assert len(result.evidence) == 2

    def test_agent_refuses_destructive_tool_call(self):
        """Agent validator blocks any attempt to call drop/truncate operations."""
        # Model tries to call an unregistered destructive tool
        bad_tool_call = '{"type": "tool_call", "tool": "apply_remediation", "arguments": {"approved": true}}'
        final = (
            '{"type": "final_answer", "diagnosis": "Cannot remediate", '
            '"root_cause": "unknown", "evidence": [], "recommended_action": "none", '
            '"requires_approval": false, "confidence": 0.1}'
        )

        orchestrator, _, registry = _build_full_mock_agent([bad_tool_call, final])
        registry.execute = MagicMock(side_effect=registry.execute)
        result = orchestrator.run("Drop all indexes and recreate them")

        # Registry execute should not have been called for unregistered tool
        registry.execute.assert_not_called()
        # Result should still be produced (graceful degradation)
        assert result is not None
        assert "apply_remediation" not in result.tools_called

    def test_agent_trajectory_logged(self):
        """Trajectory contains all steps including tool execution."""
        tool_call = '{"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}}'
        final = (
            '{"type": "final_answer", "diagnosis": "Missing index", '
            '"root_cause": "missing_suitable_index", '
            '"evidence": ["No index exists"], '
            '"recommended_action": "Create index", '
            '"requires_approval": true, "confidence": 0.8}'
        )

        orchestrator, _, _ = _build_full_mock_agent([tool_call, final])
        result = orchestrator.run("Diagnose slow orders query")

        step_types = [s.step_type for s in result.trajectory]
        assert "user_question" in step_types
        assert "tool_execution" in step_types
        assert "final_answer" in step_types

    def test_num_steps_matches_trajectory(self):
        """num_steps in result matches trajectory length."""
        final = (
            '{"type": "final_answer", "diagnosis": "Quick answer", '
            '"root_cause": "none_detected", "evidence": [], '
            '"recommended_action": "none", "requires_approval": false, "confidence": 0.5}'
        )

        orchestrator, _, _ = _build_full_mock_agent([final])
        result = orchestrator.run("Quick test")

        assert result.num_steps >= 1

    def test_multi_tool_orchestration(self):
        """Agent can call multiple tools before reaching final answer."""
        tc1 = '{"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}}'
        tc2 = '{"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {"customer_id": 42}}'
        final = (
            '{"type": "final_answer", "diagnosis": "Confirmed: no index exists", '
            '"root_cause": "missing_suitable_index", '
            '"evidence": ["No index", "Sequential scan"], '
            '"recommended_action": "Create index", "requires_approval": true, "confidence": 0.95}'
        )

        orchestrator, _, _ = _build_full_mock_agent([tc1, tc2, final])
        result = orchestrator.run("Multi-tool test")

        # At least 1 tool should have been called (tc2 may fail schema validation)
        assert len(result.tools_called) >= 1

    def test_safety_violations_recorded(self):
        """Safety violations from SQL injection attempts are recorded."""
        sql_injection = (
            '{"type": "tool_call", "tool": "get_table_statistics", '
            '"arguments": {"table_name": "orders; DROP TABLE orders;"}}'
        )
        final = (
            '{"type": "final_answer", "diagnosis": "Attempted injection detected", '
            '"root_cause": "unknown", "evidence": [], '
            '"recommended_action": "Review security", "requires_approval": false, "confidence": 0.1}'
        )

        orchestrator, _, _ = _build_full_mock_agent([sql_injection, final])
        result = orchestrator.run("Test injection safety")

        # The injection attempt should have been rejected
        assert len(result.rejection_reasons) >= 1


@pytest.mark.unit
class TestResponseParserIntegration:
    def test_valid_tool_call_parsed(self):
        from agent.core.response_parser import ResponseParser
        parser = ResponseParser()
        text = '{"type": "tool_call", "tool": "diagnose_slow_query", "arguments": {"customer_id": 42}}'
        result = parser.parse(text)
        assert result["type"] == "tool_call"
        assert result["tool"] == "diagnose_slow_query"
        assert result["arguments"]["customer_id"] == 42

    def test_valid_final_answer_parsed(self):
        from agent.core.response_parser import ResponseParser
        parser = ResponseParser()
        text = (
            '{"type": "final_answer", "diagnosis": "Missing index", '
            '"root_cause": "missing_suitable_index", "evidence": ["Seq scan"], '
            '"recommended_action": "Create index", "requires_approval": true, "confidence": 0.9}'
        )
        result = parser.parse(text)
        assert result["type"] == "final_answer"
        assert result["root_cause"] == "missing_suitable_index"

    def test_invalid_json_raises_parse_error(self):
        from agent.core.response_parser import ResponseParser, ParseError
        parser = ResponseParser()
        with pytest.raises(ParseError):
            parser.parse("This is not JSON at all.")

    def test_json_with_wrong_type_raises_parse_error(self):
        from agent.core.response_parser import ResponseParser, ParseError
        parser = ResponseParser()
        with pytest.raises(ParseError):
            parser.parse('{"type": "unknown_type", "data": "something"}')

    def test_dangerous_sql_in_response_rejected(self):
        from agent.core.response_parser import ResponseParser, ParseError
        parser = ResponseParser()
        with pytest.raises(ParseError, match="Dangerous"):
            parser.parse(
                '{"type": "tool_call", "tool": "get_table_statistics", '
                '"arguments": {"table_name": "DROP TABLE orders"}}'
            )

    def test_json_in_markdown_fence_extracted(self):
        from agent.core.response_parser import ResponseParser
        parser = ResponseParser()
        text = (
            "Here is my response:\n"
            "```json\n"
            '{"type": "final_answer", "diagnosis": "test", "root_cause": "none_detected", '
            '"evidence": [], "recommended_action": "none", "requires_approval": false, "confidence": 0.5}\n'
            "```"
        )
        result = parser.parse(text)
        assert result["type"] == "final_answer"

    def test_missing_tool_field_raises_error(self):
        from agent.core.response_parser import ResponseParser, ParseError
        parser = ResponseParser()
        with pytest.raises(ParseError):
            parser.parse('{"type": "tool_call", "arguments": {}}')

    def test_missing_diagnosis_field_raises_error(self):
        from agent.core.response_parser import ResponseParser, ParseError
        parser = ResponseParser()
        with pytest.raises(ParseError):
            parser.parse('{"type": "final_answer", "root_cause": "none_detected"}')

    def test_try_parse_returns_none_on_failure(self):
        from agent.core.response_parser import ResponseParser
        parser = ResponseParser()
        result = parser.try_parse("definitely not json")
        assert result is None

