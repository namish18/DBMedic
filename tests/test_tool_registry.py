"""
tests/test_tool_registry.py
============================
Tests for the ToolRegistry — registration, lookup, and execution dispatch.

All tests use mocked DB connections; no live PostgreSQL required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Unit Tests — Registry structure and tool lookup
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestToolRegistryStructure:
    @pytest.fixture(autouse=True)
    def mock_tools(self):
        """Mock all Phase 2 tool imports so registry can init without DB."""
        with patch("agent.core.tool_registry._build_registry") as mock_build:
            # Build a minimal fake registry
            def fake_callable(arguments):
                return {"result": "ok", "tool_called_with": arguments}

            mock_build.return_value = {
                "get_query_execution_plan": {
                    "name": "get_query_execution_plan",
                    "description": "Get execution plan",
                    "function": fake_callable,
                    "parameters": {
                        "template_key": {"type": "string", "enum": ["slow_query_customer_orders"]},
                        "customer_id": {"type": "integer", "minimum": 1, "default": 4242},
                    },
                    "required": [],
                    "read_only": True,
                },
                "analyze_query_performance": {
                    "name": "analyze_query_performance",
                    "description": "Analyze query performance",
                    "function": fake_callable,
                    "parameters": {
                        "customer_id": {"type": "integer", "minimum": 1, "default": 4242},
                    },
                    "required": [],
                    "read_only": True,
                },
                "get_table_statistics": {
                    "name": "get_table_statistics",
                    "description": "Get table statistics",
                    "function": fake_callable,
                    "parameters": {
                        "table_name": {"type": "string", "enum": ["orders"], "default": "orders"},
                    },
                    "required": [],
                    "read_only": True,
                },
                "get_table_size": {
                    "name": "get_table_size",
                    "description": "Get table size",
                    "function": fake_callable,
                    "parameters": {
                        "table_name": {"type": "string", "enum": ["orders"], "default": "orders"},
                    },
                    "required": [],
                    "read_only": True,
                },
                "get_table_indexes": {
                    "name": "get_table_indexes",
                    "description": "Get table indexes",
                    "function": fake_callable,
                    "parameters": {
                        "table_name": {"type": "string", "enum": ["orders"], "default": "orders"},
                    },
                    "required": [],
                    "read_only": True,
                },
                "get_index_usage": {
                    "name": "get_index_usage",
                    "description": "Get index usage",
                    "function": fake_callable,
                    "parameters": {
                        "table_name": {"type": "string", "enum": ["orders"], "default": "orders"},
                    },
                    "required": [],
                    "read_only": True,
                },
                "inspect_customer_id_index": {
                    "name": "inspect_customer_id_index",
                    "description": "Inspect customer_id index",
                    "function": fake_callable,
                    "parameters": {},
                    "required": [],
                    "read_only": True,
                },
                "diagnose_slow_query": {
                    "name": "diagnose_slow_query",
                    "description": "Full diagnostic engine",
                    "function": fake_callable,
                    "parameters": {
                        "customer_id": {"type": "integer", "minimum": 1, "default": 4242},
                    },
                    "required": [],
                    "read_only": True,
                },
            }
            yield

    def _make_registry(self):
        from agent.core.tool_registry import ToolRegistry
        r = ToolRegistry()
        r.initialize()
        return r

    def test_registry_initializes_with_eight_tools(self):
        registry = self._make_registry()
        assert len(registry.tool_names) == 8

    def test_all_expected_tools_registered(self):
        registry = self._make_registry()
        expected = {
            "get_query_execution_plan",
            "analyze_query_performance",
            "get_table_statistics",
            "get_table_size",
            "get_table_indexes",
            "get_index_usage",
            "inspect_customer_id_index",
            "diagnose_slow_query",
        }
        assert expected == set(registry.tool_names)

    def test_has_tool_returns_true_for_registered(self):
        registry = self._make_registry()
        assert registry.has_tool("diagnose_slow_query") is True

    def test_has_tool_returns_false_for_unregistered(self):
        registry = self._make_registry()
        assert registry.has_tool("drop_tables") is False
        assert registry.has_tool("apply_remediation") is False

    def test_get_tool_raises_key_error_for_unknown(self):
        registry = self._make_registry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get_tool("nonexistent_tool")

    def test_get_schema_returns_correct_structure(self):
        registry = self._make_registry()
        schema = registry.get_schema("get_query_execution_plan")
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert "required" in schema

    def test_execute_calls_registered_function(self):
        registry = self._make_registry()
        result = registry.execute("analyze_query_performance", {"customer_id": 42})
        assert result["result"] == "ok"
        assert result["tool_called_with"]["customer_id"] == 42

    def test_execute_raises_key_error_for_unregistered(self):
        registry = self._make_registry()
        with pytest.raises(KeyError):
            registry.execute("arbitrary_sql_runner", {})

    def test_remediation_tool_not_registered(self):
        """apply_remediation must NOT be in the registry — safety constraint."""
        registry = self._make_registry()
        assert not registry.has_tool("apply_remediation")
        assert not registry.has_tool("create_index")
        assert not registry.has_tool("drop_table")

    def test_double_initialize_is_idempotent(self):
        registry = self._make_registry()
        count_before = len(registry.tool_names)
        registry.initialize()  # Second call
        assert len(registry.tool_names) == count_before

    def test_describe_all_returns_string(self):
        registry = self._make_registry()
        desc = registry.describe_all()
        assert isinstance(desc, str)
        assert "get_query_execution_plan" in desc

