"""
agent/core/tool_registry.py
============================
Registry of approved diagnostic tools for the DBMedic agent.

Registers the Phase 2 diagnostic tools with their schemas and
callable references. The agent may only execute tools registered here.

Design constraints:
  - Only read-only diagnostic tools are registered.
  - Remediation tools (apply_remediation) are explicitly excluded.
  - Each tool has a strict argument schema for validation.
  - Tool functions are wrapped to enforce read-only connections.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool Schema Definitions
# ---------------------------------------------------------------------------
# Each tool entry contains:
#   - name: str                (unique tool identifier)
#   - description: str         (what it does)
#   - function: Callable       (the actual Python function to call)
#   - parameters: dict         (JSON Schema-style argument spec)
#   - required: list[str]      (required argument names)
#   - read_only: bool          (must be True for all registered tools)

_TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def _build_registry() -> dict[str, dict[str, Any]]:
    """Build the tool registry by importing and inspecting Phase 2 tools."""
    from agent.tools.query_analyzer import (
        get_query_execution_plan,
        analyze_query_performance,
    )
    from agent.tools.table_monitor import (
        get_table_statistics,
        get_table_size,
    )
    from agent.tools.index_monitor import (
        get_table_indexes,
        get_index_usage,
        inspect_customer_id_index,
    )
    from agent.core.diagnosis_engine import diagnose_slow_query

    registry: dict[str, dict[str, Any]] = {
        # ------------------------------------------------------------------
        # Query Analyzer Tools
        # ------------------------------------------------------------------
        "get_query_execution_plan": {
            "name": "get_query_execution_plan",
            "description": (
                "Retrieve PostgreSQL EXPLAIN (FORMAT JSON) execution plan for the "
                "registered slow query template. Use to identify scan types (Seq Scan "
                "vs Index Scan) and query cost estimates."
            ),
            "function": _wrap_query_execution_plan(get_query_execution_plan),
            "parameters": {
                "template_key": {
                    "type": "string",
                    "description": "Registered query template key",
                    "enum": ["slow_query_customer_orders"],
                    "default": "slow_query_customer_orders",
                },
                "customer_id": {
                    "type": "integer",
                    "description": "Customer ID filter value",
                    "minimum": 1,
                    "default": 4242,
                },
            },
            "required": [],
            "read_only": True,
        },

        "analyze_query_performance": {
            "name": "analyze_query_performance",
            "description": (
                "Run EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) to get real execution "
                "metrics: actual execution time, buffer hits/reads, actual rows. "
                "Use to measure true query cost."
            ),
            "function": _wrap_analyze_query_performance(analyze_query_performance),
            "parameters": {
                "customer_id": {
                    "type": "integer",
                    "description": "Customer ID filter value",
                    "minimum": 1,
                    "default": 4242,
                },
            },
            "required": [],
            "read_only": True,
        },

        # ------------------------------------------------------------------
        # Table Monitor Tools
        # ------------------------------------------------------------------
        "get_table_statistics": {
            "name": "get_table_statistics",
            "description": (
                "Retrieve comprehensive table statistics from pg_stat_user_tables: "
                "row count estimates, sequential scan counts, index scan counts, "
                "table size, dead tuples."
            ),
            "function": _wrap_table_tool(get_table_statistics, default_table="orders"),
            "parameters": {
                "table_name": {
                    "type": "string",
                    "description": "Table name (must be in allowed list)",
                    "enum": ["orders", "customers", "products", "order_items"],
                    "default": "orders",
                },
            },
            "required": [],
            "read_only": True,
        },

        "get_table_size": {
            "name": "get_table_size",
            "description": (
                "Retrieve table, index, and total size in bytes and human-readable form. "
                "Use to understand storage footprint."
            ),
            "function": _wrap_table_tool(get_table_size, default_table="orders"),
            "parameters": {
                "table_name": {
                    "type": "string",
                    "description": "Table name (must be in allowed list)",
                    "enum": ["orders", "customers", "products", "order_items"],
                    "default": "orders",
                },
            },
            "required": [],
            "read_only": True,
        },

        # ------------------------------------------------------------------
        # Index Monitor Tools
        # ------------------------------------------------------------------
        "get_table_indexes": {
            "name": "get_table_indexes",
            "description": (
                "Retrieve all index definitions for a table: index names, covered "
                "columns, index type, uniqueness, and usage statistics."
            ),
            "function": _wrap_table_tool(get_table_indexes, default_table="orders"),
            "parameters": {
                "table_name": {
                    "type": "string",
                    "description": "Table name (must be in allowed list)",
                    "enum": ["orders", "customers", "products", "order_items"],
                    "default": "orders",
                },
            },
            "required": [],
            "read_only": True,
        },

        "get_index_usage": {
            "name": "get_index_usage",
            "description": (
                "Retrieve operational index usage statistics from pg_stat_user_indexes: "
                "scan counts, tuples read, tuples fetched. Shows how often each index is used."
            ),
            "function": _wrap_table_tool(get_index_usage, default_table="orders"),
            "parameters": {
                "table_name": {
                    "type": "string",
                    "description": "Table name (must be in allowed list)",
                    "enum": ["orders", "customers", "products", "order_items"],
                    "default": "orders",
                },
            },
            "required": [],
            "read_only": True,
        },

        "inspect_customer_id_index": {
            "name": "inspect_customer_id_index",
            "description": (
                "Specifically inspect whether a suitable index covering orders.customer_id "
                "exists. Returns index existence, definition, and usage statistics."
            ),
            "function": _wrap_no_args_tool(inspect_customer_id_index),
            "parameters": {},
            "required": [],
            "read_only": True,
        },

        # ------------------------------------------------------------------
        # Diagnosis Engine
        # ------------------------------------------------------------------
        "diagnose_slow_query": {
            "name": "diagnose_slow_query",
            "description": (
                "Run the full deterministic diagnostic engine: combines execution plan, "
                "table statistics, and index inspection to produce a structured JSON "
                "diagnostic report with root cause, evidence, and recommendation."
            ),
            "function": _wrap_diagnose_slow_query(diagnose_slow_query),
            "parameters": {
                "customer_id": {
                    "type": "integer",
                    "description": "Customer ID filter value",
                    "minimum": 1,
                    "default": 4242,
                },
            },
            "required": [],
            "read_only": True,
        },
    }

    return registry


# ---------------------------------------------------------------------------
# Wrapper functions to adapt Phase 2 signatures for the agent
# (Phase 2 tools accept optional conn parameter; agent calls without conn)
# ---------------------------------------------------------------------------

def _wrap_query_execution_plan(fn: Callable) -> Callable:
    """Wrap get_query_execution_plan to accept only agent-safe arguments."""
    def wrapper(arguments: dict[str, Any]) -> Any:
        kwargs: dict[str, Any] = {}
        if "template_key" in arguments:
            kwargs["template_key"] = str(arguments["template_key"])
        if "customer_id" in arguments:
            customer_id = int(arguments["customer_id"])
            kwargs["params"] = {"customer_id": customer_id}
        return fn(conn=None, **kwargs)
    return wrapper


def _wrap_analyze_query_performance(fn: Callable) -> Callable:
    """Wrap analyze_query_performance to accept only agent-safe arguments."""
    def wrapper(arguments: dict[str, Any]) -> Any:
        kwargs: dict[str, Any] = {}
        if "customer_id" in arguments:
            kwargs["customer_id"] = int(arguments["customer_id"])
        return fn(conn=None, **kwargs)
    return wrapper


def _wrap_table_tool(fn: Callable, default_table: str) -> Callable:
    """Wrap table tools that take table_name as first arg."""
    def wrapper(arguments: dict[str, Any]) -> Any:
        table_name = str(arguments.get("table_name", default_table))
        return fn(table_name, conn=None)
    return wrapper


def _wrap_no_args_tool(fn: Callable) -> Callable:
    """Wrap tools that take no arguments."""
    def wrapper(arguments: dict[str, Any]) -> Any:
        return fn(conn=None)
    return wrapper


def _wrap_diagnose_slow_query(fn: Callable) -> Callable:
    """Wrap diagnose_slow_query for agent use."""
    def wrapper(arguments: dict[str, Any]) -> Any:
        kwargs: dict[str, Any] = {}
        if "customer_id" in arguments:
            kwargs["customer_id"] = int(arguments["customer_id"])
        return fn(conn=None, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Public Registry API
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Registry of approved diagnostic tools available to the DBMedic agent.
    Thread-safe read-only access after initialization.
    """

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, Any]] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Load and register all Phase 2 tools."""
        if self._initialized:
            return
        self._registry = _build_registry()
        self._initialized = True
        logger.info(
            "Tool registry initialized with %d tools: %s",
            len(self._registry),
            list(self._registry.keys()),
        )

    @property
    def tool_names(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._registry.keys())

    def has_tool(self, name: str) -> bool:
        """Return True if tool_name is registered."""
        return name in self._registry

    def get_tool(self, name: str) -> dict[str, Any]:
        """Return tool entry dict. Raises KeyError if not found."""
        if name not in self._registry:
            raise KeyError(f"Tool '{name}' is not registered. Available: {self.tool_names}")
        return self._registry[name]

    def get_schema(self, name: str) -> dict[str, Any]:
        """Return the parameter schema for a tool."""
        entry = self.get_tool(name)
        return {
            "name": entry["name"],
            "description": entry["description"],
            "parameters": entry["parameters"],
            "required": entry["required"],
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a registered tool with the given arguments.

        Parameters
        ----------
        name: str
            Registered tool name.
        arguments: dict
            Validated argument dictionary.

        Returns
        -------
        Tool output (any JSON-serializable type).

        Raises
        ------
        KeyError: If tool is not registered.
        RuntimeError: If tool execution fails.
        """
        entry = self.get_tool(name)
        fn = entry["function"]

        logger.info("Executing tool: %s with arguments: %s", name, arguments)

        try:
            result = fn(arguments)
            logger.debug("Tool '%s' returned: %s", name, type(result).__name__)
            return result
        except Exception as exc:
            logger.error("Tool '%s' execution failed: %s", name, exc)
            raise RuntimeError(f"Tool '{name}' failed: {exc}") from exc

    def describe_all(self) -> str:
        """Return a human-readable description of all registered tools."""
        lines: list[str] = ["Available diagnostic tools:"]
        for name, entry in self._registry.items():
            lines.append(f"\n  {name}")
            lines.append(f"    Description: {entry['description'][:80]}...")
            if entry["parameters"]:
                lines.append(f"    Parameters: {list(entry['parameters'].keys())}")
        return "\n".join(lines)

