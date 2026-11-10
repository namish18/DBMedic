"""
agent/core/tool_validator.py
=============================
Strict validation of model-generated tool calls before execution.

Enforces:
  - Valid JSON structure (handled by ResponseParser before this)
  - Tool must exist in the registry
  - Arguments must match the tool schema
  - No unknown arguments
  - No arbitrary SQL values
  - No destructive operations
  - No unregistered function names
  - Per-session tool call limits
  - Records all rejected calls with reasons

Security model:
  - Model output is treated as untrusted input.
  - Arguments are validated independently of model intent.
  - Validation is fail-secure (reject on doubt).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SQL injection / destructive patterns forbidden anywhere in arguments
FORBIDDEN_SQL_PATTERNS = [
    r"\bDROP\b",
    r"\bTRUNCATE\b",
    r"\bDELETE\s+FROM\b",
    r"\bINSERT\s+INTO\b",
    r"\bUPDATE\s+\w+\s+SET\b",
    r"\bALTER\s+TABLE\b",
    r"\bALTER\s+SYSTEM\b",
    r"\bCREATE\s+(INDEX|TABLE|DATABASE|USER|ROLE)\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bEXEC(?:UTE)?\b",
    r"\bXP_\w+",                    # SQL Server extended procs (defensive)
    r";\s*--",                      # Inline comment injection
    r"'\s*OR\s+'?\d+'\s*=\s*'?\d+", # Classic injection
    r"\bUNION\s+(?:ALL\s+)?SELECT\b", # UNION injection
    r"\bINFORMATION_SCHEMA\b",      # Schema enumeration (read-only but unsafe)
    r"\bPG_READ_FILE\b",            # Filesystem access
    r"\bLO_IMPORT\b",               # Large-object injection
    r"\bCOPY\s+\w+\s+FROM\b",       # COPY FROM injection
]

_FORBIDDEN_SQL_RE = re.compile(
    "|".join(FORBIDDEN_SQL_PATTERNS),
    re.IGNORECASE,
)

# Allowed table names (must match tool allowlist)
ALLOWED_TABLE_NAMES = frozenset({"orders", "customers", "products", "order_items"})

# Allowed query template keys
ALLOWED_TEMPLATE_KEYS = frozenset({"slow_query_customer_orders"})

# Type validators
_TYPE_VALIDATORS = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a tool call validation."""
    is_valid: bool
    tool_name: str
    arguments: dict[str, Any]
    rejection_reason: str = ""
    sanitized_arguments: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ToolValidator:
    """
    Validates model-generated tool calls before execution.

    Instantiate once per session. Tracks call count per tool and total.
    """

    def __init__(
        self,
        registry: Any,  # ToolRegistry (avoid circular import)
        max_calls_per_session: int = 5,
        max_calls_per_tool: int = 3,
    ) -> None:
        self.registry = registry
        self.max_calls_per_session = max_calls_per_session
        self.max_calls_per_tool = max_calls_per_tool
        self._total_calls: int = 0
        self._calls_per_tool: dict[str, int] = {}
        self._rejected_calls: list[dict[str, Any]] = []

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def rejected_calls(self) -> list[dict[str, Any]]:
        return list(self._rejected_calls)

    def validate(self, tool_call: dict[str, Any]) -> ValidationResult:
        """
        Validate a parsed tool_call dict.

        Parameters
        ----------
        tool_call: dict
            Parsed tool call: {"type": "tool_call", "tool": str, "arguments": dict}

        Returns
        -------
        ValidationResult with is_valid flag and rejection_reason if rejected.
        """
        tool_name = tool_call.get("tool", "")
        arguments = tool_call.get("arguments", {})

        logger.debug("Validating tool call: tool=%s, args=%s", tool_name, arguments)

        # 1. Tool name must be a non-empty string
        if not isinstance(tool_name, str) or not tool_name.strip():
            return self._reject(tool_name, arguments, "Tool name must be a non-empty string")

        tool_name = tool_name.strip()

        # 2. Tool must exist in registry
        if not self.registry.has_tool(tool_name):
            registered = self.registry.tool_names
            return self._reject(
                tool_name, arguments,
                f"Tool '{tool_name}' is not registered. "
                f"Registered tools: {registered}"
            )

        # 3. Arguments must be a dict
        if not isinstance(arguments, dict):
            return self._reject(tool_name, arguments, "Arguments must be a JSON object")

        # 4. Session-level rate limit
        if self._total_calls >= self.max_calls_per_session:
            return self._reject(
                tool_name, arguments,
                f"Session tool call limit reached ({self.max_calls_per_session}). "
                f"Agent must provide a final answer."
            )

        # 5. Per-tool rate limit
        tool_calls_so_far = self._calls_per_tool.get(tool_name, 0)
        if tool_calls_so_far >= self.max_calls_per_tool:
            return self._reject(
                tool_name, arguments,
                f"Tool '{tool_name}' has been called {tool_calls_so_far} times. "
                f"Maximum per-tool calls: {self.max_calls_per_tool}"
            )

        # 6. Validate arguments against schema
        schema_result = self._validate_arguments(tool_name, arguments)
        if schema_result is not None:
            return self._reject(tool_name, arguments, schema_result)

        # 7. Check for dangerous content in argument values
        dangerous = self._check_dangerous_content(arguments)
        if dangerous:
            return self._reject(
                tool_name, arguments,
                f"Dangerous content detected in arguments: {dangerous}. "
                f"SQL injection or destructive operations are forbidden."
            )

        # 8. Apply semantic-level argument validation
        semantic_result = self._validate_argument_semantics(tool_name, arguments)
        if semantic_result is not None:
            return self._reject(tool_name, arguments, semantic_result)

        # All checks passed
        self._total_calls += 1
        self._calls_per_tool[tool_name] = tool_calls_so_far + 1

        # Build sanitized arguments (apply defaults, type coercion)
        sanitized = self._sanitize_arguments(tool_name, arguments)

        logger.info(
            "Tool call validated: %s (session call #%d)",
            tool_name,
            self._total_calls,
        )

        return ValidationResult(
            is_valid=True,
            tool_name=tool_name,
            arguments=arguments,
            sanitized_arguments=sanitized,
        )

    # ------------------------------------------------------------------
    # Argument schema validation
    # ------------------------------------------------------------------

    def _validate_arguments(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str | None:
        """
        Validate arguments against the tool's registered schema.
        Returns error message string if invalid, None if valid.
        """
        schema = self.registry.get_schema(tool_name)
        parameters = schema.get("parameters", {})
        required = schema.get("required", [])

        # Check required arguments are present
        for req_key in required:
            if req_key not in arguments:
                return f"Missing required argument: '{req_key}'"

        # Check no unknown arguments
        unknown = set(arguments.keys()) - set(parameters.keys())
        if unknown:
            return f"Unknown arguments: {unknown}. Allowed: {set(parameters.keys())}"

        # Type check each provided argument
        for arg_name, arg_value in arguments.items():
            if arg_name not in parameters:
                return f"Unknown argument: '{arg_name}'"

            param_spec = parameters[arg_name]
            param_type = param_spec.get("type")
            expected_types = _TYPE_VALIDATORS.get(param_type, None)

            if expected_types and not isinstance(arg_value, expected_types):
                # Allow int where float expected
                if param_type == "integer" and isinstance(arg_value, float) and arg_value == int(arg_value):
                    pass  # Accept 4242.0 as integer
                else:
                    return (
                        f"Argument '{arg_name}' has wrong type. "
                        f"Expected {param_type}, got {type(arg_value).__name__}"
                    )

            # Enum check
            if "enum" in param_spec:
                allowed_values = param_spec["enum"]
                if arg_value not in allowed_values:
                    return (
                        f"Argument '{arg_name}' has invalid value '{arg_value}'. "
                        f"Allowed values: {allowed_values}"
                    )

            # Minimum check for integers
            if param_type == "integer" and "minimum" in param_spec:
                if int(arg_value) < param_spec["minimum"]:
                    return (
                        f"Argument '{arg_name}' value {arg_value} is below "
                        f"minimum {param_spec['minimum']}"
                    )

        return None

    def _validate_argument_semantics(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str | None:
        """
        Apply semantic validation rules specific to DBMedic tools.
        Returns error message string if invalid, None if valid.
        """
        # customer_id must be a positive integer
        if "customer_id" in arguments:
            cid = arguments["customer_id"]
            try:
                cid_int = int(cid)
            except (ValueError, TypeError):
                return f"customer_id must be an integer, got {cid!r}"
            if cid_int <= 0:
                return f"customer_id must be positive, got {cid_int}"

        # table_name must be in allowed list
        if "table_name" in arguments:
            tbl = arguments.get("table_name", "")
            if isinstance(tbl, str) and tbl.lower() not in ALLOWED_TABLE_NAMES:
                return (
                    f"table_name '{tbl}' is not permitted. "
                    f"Allowed: {sorted(ALLOWED_TABLE_NAMES)}"
                )

        # template_key must be in allowed list
        if "template_key" in arguments:
            key = arguments.get("template_key", "")
            if key not in ALLOWED_TEMPLATE_KEYS:
                return (
                    f"template_key '{key}' is not permitted. "
                    f"Allowed: {sorted(ALLOWED_TEMPLATE_KEYS)}"
                )

        return None

    # ------------------------------------------------------------------
    # Dangerous content detection
    # ------------------------------------------------------------------

    def _check_dangerous_content(
        self, arguments: dict[str, Any]
    ) -> list[str]:
        """
        Check all argument values for dangerous SQL patterns.
        Returns list of field names with dangerous content.
        """
        dangerous: list[str] = []
        for key, value in arguments.items():
            value_str = str(value) if not isinstance(value, str) else value
            if _FORBIDDEN_SQL_RE.search(value_str):
                dangerous.append(key)
        return dangerous

    # ------------------------------------------------------------------
    # Argument sanitization
    # ------------------------------------------------------------------

    def _sanitize_arguments(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Apply defaults and type coercion to arguments.
        Returns a clean argument dict ready for tool execution.
        """
        schema = self.registry.get_schema(tool_name)
        parameters = schema.get("parameters", {})
        sanitized: dict[str, Any] = {}

        # Apply defaults for missing optional args
        for param_name, param_spec in parameters.items():
            if param_name in arguments:
                value = arguments[param_name]
                # Type coercion
                if param_spec.get("type") == "integer":
                    value = int(value)
                elif param_spec.get("type") == "string":
                    value = str(value)
                sanitized[param_name] = value
            elif "default" in param_spec:
                sanitized[param_name] = param_spec["default"]

        return sanitized

    # ------------------------------------------------------------------
    # Rejection tracking
    # ------------------------------------------------------------------

    def _reject(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str,
    ) -> ValidationResult:
        """Record and return a rejection."""
        logger.warning("Tool call rejected: tool=%s, reason=%s", tool_name, reason)
        self._rejected_calls.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "reason": reason,
        })
        return ValidationResult(
            is_valid=False,
            tool_name=tool_name,
            arguments=arguments,
            rejection_reason=reason,
        )

    # ------------------------------------------------------------------
    # Session info
    # ------------------------------------------------------------------

    def session_summary(self) -> dict[str, Any]:
        """Return a summary of this validation session."""
        return {
            "total_validated_calls": self._total_calls,
            "calls_per_tool": dict(self._calls_per_tool),
            "rejected_calls": len(self._rejected_calls),
            "rejections": self._rejected_calls,
            "calls_remaining": max(0, self.max_calls_per_session - self._total_calls),
        }

