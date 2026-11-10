"""
tests/test_tool_validator.py
=============================
Tests for ToolValidator — argument validation, SQL rejection, rate limiting.

All deterministic — no live database or model required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.core.tool_validator import ToolValidator


def _make_registry(tools: dict | None = None):
    """Return a MagicMock registry with configurable tool list."""
    registry = MagicMock()
    if tools is None:
        tools = {
            "get_query_execution_plan": {
                "name": "get_query_execution_plan",
                "description": "Get execution plan",
                "parameters": {
                    "template_key": {"type": "string", "enum": ["slow_query_customer_orders"]},
                    "customer_id": {"type": "integer", "minimum": 1, "default": 4242},
                },
                "required": [],
            },
            "analyze_query_performance": {
                "name": "analyze_query_performance",
                "description": "Analyze performance",
                "parameters": {
                    "customer_id": {"type": "integer", "minimum": 1, "default": 4242},
                },
                "required": [],
            },
            "get_table_statistics": {
                "name": "get_table_statistics",
                "description": "Get table stats",
                "parameters": {
                    "table_name": {
                        "type": "string",
                        "enum": ["orders", "customers", "products", "order_items"],
                        "default": "orders",
                    },
                },
                "required": [],
            },
            "inspect_customer_id_index": {
                "name": "inspect_customer_id_index",
                "description": "Inspect index",
                "parameters": {},
                "required": [],
            },
            "diagnose_slow_query": {
                "name": "diagnose_slow_query",
                "description": "Full diagnosis",
                "parameters": {
                    "customer_id": {"type": "integer", "minimum": 1, "default": 4242},
                },
                "required": [],
            },
        }
    registry.tool_names = list(tools.keys())
    registry.has_tool.side_effect = lambda name: name in tools
    registry.get_schema.side_effect = lambda name: tools[name]
    return registry


@pytest.mark.unit
class TestToolValidatorBasic:
    def test_valid_tool_call_accepted(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        result = validator.validate({
            "type": "tool_call",
            "tool": "analyze_query_performance",
            "arguments": {"customer_id": 4242},
        })
        assert result.is_valid is True
        assert result.tool_name == "analyze_query_performance"

    def test_valid_tool_with_no_arguments(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        result = validator.validate({
            "type": "tool_call",
            "tool": "inspect_customer_id_index",
            "arguments": {},
        })
        assert result.is_valid is True

    def test_empty_tool_name_rejected(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        result = validator.validate({"type": "tool_call", "tool": "", "arguments": {}})
        assert result.is_valid is False
        assert "non-empty" in result.rejection_reason.lower()

    def test_unregistered_tool_rejected(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        result = validator.validate({
            "type": "tool_call",
            "tool": "drop_all_tables",
            "arguments": {},
        })
        assert result.is_valid is False
        assert "not registered" in result.rejection_reason

    def test_apply_remediation_rejected(self):
        """apply_remediation is not registered — must be rejected."""
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        result = validator.validate({
            "type": "tool_call",
            "tool": "apply_remediation",
            "arguments": {"remediation_key": "create_idx_orders_customer_id", "approved": True},
        })
        assert result.is_valid is False

    def test_unknown_argument_rejected(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        result = validator.validate({
            "type": "tool_call",
            "tool": "analyze_query_performance",
            "arguments": {"customer_id": 42, "sql": "SELECT * FROM pg_shadow"},
        })
        assert result.is_valid is False
        assert "Unknown argument" in result.rejection_reason or "Unknown arguments" in result.rejection_reason

    def test_wrong_type_argument_rejected(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        result = validator.validate({
            "type": "tool_call",
            "tool": "analyze_query_performance",
            "arguments": {"customer_id": "not_a_number"},
        })
        assert result.is_valid is False

    def test_invalid_enum_value_rejected(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        result = validator.validate({
            "type": "tool_call",
            "tool": "get_table_statistics",
            "arguments": {"table_name": "pg_shadow"},
        })
        assert result.is_valid is False

    def test_negative_customer_id_rejected(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        result = validator.validate({
            "type": "tool_call",
            "tool": "analyze_query_performance",
            "arguments": {"customer_id": -1},
        })
        assert result.is_valid is False

    def test_zero_customer_id_rejected(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        result = validator.validate({
            "type": "tool_call",
            "tool": "analyze_query_performance",
            "arguments": {"customer_id": 0},
        })
        assert result.is_valid is False


@pytest.mark.unit
class TestSQLInjectionRejection:
    """SQL injection patterns in arguments must always be rejected."""

    def _validator(self):
        return ToolValidator(_make_registry(), max_calls_per_session=10)

    def test_drop_table_in_argument_rejected(self):
        validator = self._validator()
        result = validator.validate({
            "type": "tool_call",
            "tool": "get_table_statistics",
            "arguments": {"table_name": "orders; DROP TABLE orders;"},
        })
        assert result.is_valid is False

    def test_union_select_in_argument_rejected(self):
        validator = self._validator()
        result = validator.validate({
            "type": "tool_call",
            "tool": "get_query_execution_plan",
            "arguments": {"template_key": "slow_query_customer_orders UNION SELECT * FROM pg_shadow"},
        })
        assert result.is_valid is False

    def test_delete_from_rejected(self):
        validator = self._validator()
        result = validator.validate({
            "type": "tool_call",
            "tool": "get_table_statistics",
            "arguments": {"table_name": "DELETE FROM orders"},
        })
        assert result.is_valid is False

    def test_truncate_rejected(self):
        validator = self._validator()
        result = validator.validate({
            "type": "tool_call",
            "tool": "get_table_statistics",
            "arguments": {"table_name": "TRUNCATE orders"},
        })
        assert result.is_valid is False

    def test_insert_into_rejected(self):
        validator = self._validator()
        result = validator.validate({
            "type": "tool_call",
            "tool": "get_table_statistics",
            "arguments": {"table_name": "INSERT INTO orders VALUES(1)"},
        })
        assert result.is_valid is False

    def test_information_schema_rejected(self):
        validator = self._validator()
        result = validator.validate({
            "type": "tool_call",
            "tool": "get_table_statistics",
            "arguments": {"table_name": "INFORMATION_SCHEMA.tables"},
        })
        assert result.is_valid is False


@pytest.mark.unit
class TestRateLimiting:
    def test_session_limit_enforced(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=2)

        tc = {"type": "tool_call", "tool": "inspect_customer_id_index", "arguments": {}}
        r1 = validator.validate(tc)
        r2 = validator.validate(tc)
        r3 = validator.validate(tc)  # Over limit

        assert r1.is_valid is True
        assert r2.is_valid is True
        assert r3.is_valid is False
        assert "limit" in r3.rejection_reason.lower()

    def test_per_tool_limit_enforced(self):
        validator = ToolValidator(
            _make_registry(),
            max_calls_per_session=10,
            max_calls_per_tool=2,
        )
        tc = {"type": "tool_call", "tool": "diagnose_slow_query", "arguments": {}}

        r1 = validator.validate(tc)
        r2 = validator.validate(tc)
        r3 = validator.validate(tc)  # Over per-tool limit

        assert r1.is_valid is True
        assert r2.is_valid is True
        assert r3.is_valid is False

    def test_session_summary_tracks_rejections(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        validator.validate({"type": "tool_call", "tool": "nonexistent", "arguments": {}})
        summary = validator.session_summary()
        assert summary["rejected_calls"] == 1
        assert len(summary["rejections"]) == 1

    def test_total_calls_increments_on_valid(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        validator.validate({
            "type": "tool_call",
            "tool": "inspect_customer_id_index",
            "arguments": {},
        })
        assert validator.total_calls == 1

    def test_total_calls_does_not_increment_on_rejection(self):
        validator = ToolValidator(_make_registry(), max_calls_per_session=5)
        validator.validate({"type": "tool_call", "tool": "bad_tool", "arguments": {}})
        assert validator.total_calls == 0

