"""
agent/core/response_parser.py
==============================
Parses model-generated text into structured tool calls or final answers.

Handles:
  - Clean JSON responses
  - JSON embedded in markdown code fences
  - Partial / truncated JSON (with limited repair)
  - Multiple JSON objects in one response (uses first valid one)

Never executes or trusts parsed content — returns structured dicts only.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_RESPONSE_TYPES = {"tool_call", "final_answer"}

# SQL injection patterns to detect in any parsed field
DANGEROUS_SQL_PATTERNS = [
    r"\bDROP\s+TABLE\b",
    r"\bDROP\s+DATABASE\b",
    r"\bTRUNCATE\s+TABLE\b",
    r"\bDELETE\s+FROM\b",
    r"\bINSERT\s+INTO\b",
    r"\bUPDATE\s+\w+\s+SET\b",
    r"\bALTER\s+TABLE\b",
    r"\bCREATE\s+TABLE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r";\s*--",                      # SQL comment injection
    r"'\s*OR\s+'?\d+'\s*=\s*'?\d+", # Classic injection
]

_DANGEROUS_SQL_RE = re.compile(
    "|".join(DANGEROUS_SQL_PATTERNS),
    re.IGNORECASE,
)


class ParseError(ValueError):
    """Raised when model output cannot be parsed into a valid response."""


class ResponseParser:
    """
    Parses raw model text output into structured DBMedic response dicts.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, text: str) -> dict[str, Any]:
        """
        Parse model output text into a structured response dict.

        Returns one of:
            {"type": "tool_call",    "tool": str, "arguments": dict, ...}
            {"type": "final_answer", "diagnosis": str, ...}

        Raises ParseError if the text cannot be parsed.
        """
        if not text or not text.strip():
            raise ParseError("Model returned empty response")

        # Attempt extraction strategies in order
        candidates = self._extract_json_candidates(text)

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                validated = self._validate_structure(parsed)
                self._check_for_dangerous_content(validated)
                return validated
            except (json.JSONDecodeError, ParseError, ValueError) as exc:
                last_error = exc
                continue

        raise ParseError(
            f"Could not parse model output as valid DBMedic response. "
            f"Last error: {last_error}. "
            f"Raw output (first 500 chars): {text[:500]}"
        )

    def try_parse(self, text: str) -> dict[str, Any] | None:
        """
        Parse model output, returning None instead of raising on failure.
        """
        try:
            return self.parse(text)
        except ParseError as exc:
            logger.debug("Parse attempt failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # JSON extraction strategies
    # ------------------------------------------------------------------

    def _extract_json_candidates(self, text: str) -> list[str]:
        """
        Extract potential JSON strings from model output using multiple strategies.
        Returns a list of candidate strings to try parsing (best-first).
        """
        candidates: list[str] = []
        stripped = text.strip()

        # Strategy 1: Entire output is already JSON
        if stripped.startswith("{"):
            candidates.append(stripped)

        # Strategy 2: Extract from markdown code fence ```json ... ```
        fence_matches = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        for match in fence_matches:
            match = match.strip()
            if match.startswith("{"):
                candidates.append(match)

        # Strategy 3: Find the first {...} block in the text
        brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group(0))

        # Strategy 4: Find all {...} blocks (multi-level)
        # Simple greedy extraction of outermost braces
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    block = text[start:i + 1]
                    if block not in candidates:
                        candidates.append(block)
                    start = -1

        return candidates

    # ------------------------------------------------------------------
    # Structure validation
    # ------------------------------------------------------------------

    def _validate_structure(self, data: Any) -> dict[str, Any]:
        """
        Validate that parsed JSON has correct DBMedic response structure.
        """
        if not isinstance(data, dict):
            raise ParseError(f"Response must be a JSON object, got {type(data).__name__}")

        response_type = data.get("type")
        if response_type not in VALID_RESPONSE_TYPES:
            raise ParseError(
                f"Response 'type' must be one of {VALID_RESPONSE_TYPES}, "
                f"got '{response_type}'"
            )

        if response_type == "tool_call":
            return self._validate_tool_call(data)
        elif response_type == "final_answer":
            return self._validate_final_answer(data)
        else:
            raise ParseError(f"Unknown response type: {response_type}")

    def _validate_tool_call(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate tool_call response structure."""
        if "tool" not in data:
            raise ParseError("tool_call response missing 'tool' field")

        tool_name = data.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ParseError("'tool' must be a non-empty string")

        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ParseError("'arguments' must be a JSON object")

        return {
            "type": "tool_call",
            "tool": tool_name.strip(),
            "arguments": arguments,
            "reasoning": str(data.get("reasoning", "")),
        }

    def _validate_final_answer(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate final_answer response structure."""
        if "diagnosis" not in data:
            raise ParseError("final_answer response missing 'diagnosis' field")

        diagnosis = data.get("diagnosis", "")
        if not isinstance(diagnosis, str) or not diagnosis.strip():
            raise ParseError("'diagnosis' must be a non-empty string")

        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            # Try to convert string evidence to list
            if isinstance(evidence, str):
                evidence = [evidence] if evidence else []
            else:
                evidence = []

        return {
            "type": "final_answer",
            "diagnosis": str(diagnosis).strip(),
            "root_cause": str(data.get("root_cause", "unknown")),
            "evidence": [str(e) for e in evidence],
            "recommended_action": str(data.get("recommended_action", "")),
            "requires_approval": bool(data.get("requires_approval", True)),
            "confidence": float(data.get("confidence", 0.0)),
        }

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def _check_for_dangerous_content(self, data: dict[str, Any]) -> None:
        """
        Scan all string values in the parsed response for dangerous SQL patterns.
        Raises ParseError if any are found.
        """
        dangerous_fields = self._find_dangerous_strings(data)
        if dangerous_fields:
            raise ParseError(
                f"Dangerous SQL pattern detected in model response fields: "
                f"{dangerous_fields}. Response rejected."
            )

    def _find_dangerous_strings(self, obj: Any, path: str = "") -> list[str]:
        """Recursively search for dangerous SQL in string values."""
        dangerous: list[str] = []

        if isinstance(obj, str):
            if _DANGEROUS_SQL_RE.search(obj):
                dangerous.append(f"{path}={obj[:100]!r}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                child_path = f"{path}.{k}" if path else k
                dangerous.extend(self._find_dangerous_strings(v, child_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                dangerous.extend(self._find_dangerous_strings(item, f"{path}[{i}]"))

        return dangerous

