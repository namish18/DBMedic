"""
agent/evaluation/trajectory_logger.py
======================================
Records agent execution trajectories to JSONL for AgentTune training.

Each trajectory entry records:
  - Incident ID, model ID, model revision
  - User query
  - Generated model responses
  - Tool calls, arguments, and outputs
  - Validation errors
  - Number of steps, final diagnosis
  - Ground-truth comparison (where available)
  - Success/failure status, safety violations
  - Timestamp

Security:
  - Credentials and tokens are never logged
  - Sensitive environment variable values are redacted
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fields that must NEVER appear in trajectory logs
_REDACTED_ENV_KEYS = {
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "POSTGRES_PASSWORD",
    "DATABASE_URL",
    "LAB_DATABASE_URL",
    "SECRET_KEY",
    "API_KEY",
}


def _redact_secrets(obj: Any, depth: int = 0) -> Any:
    """Recursively redact sensitive values from dicts/lists."""
    if depth > 10:
        return obj  # Prevent infinite recursion
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]" if k.upper() in _REDACTED_ENV_KEYS
            else _redact_secrets(v, depth + 1)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_redact_secrets(item, depth + 1) for item in obj]
    return obj


class TrajectoryLogger:
    """
    Logs agent trajectories to JSONL file for AgentTune training.
    """

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories_path = self.output_dir / "trajectories.jsonl"

    def log(self, result: Any) -> None:
        """
        Append one agent run trajectory to the JSONL file.

        Parameters
        ----------
        result: AgentResult
            Completed agent run result.
        """
        entry = self._build_entry(result)
        entry = _redact_secrets(entry)

        try:
            with open(self.trajectories_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
            logger.debug("Trajectory logged to %s", self.trajectories_path)
        except OSError as exc:
            logger.error("Failed to write trajectory: %s", exc)

    def _build_entry(self, result: Any) -> dict[str, Any]:
        """Build a trajectory entry dict from an AgentResult."""
        # Serialize trajectory steps
        steps = []
        for step in result.trajectory:
            step_dict: dict[str, Any] = {
                "step_index": step.step_index,
                "step_type": step.step_type,
                "timestamp": step.timestamp,
                "is_valid": step.is_valid,
            }
            if step.model_response_text:
                step_dict["model_response"] = step.model_response_text
            if step.tool_name:
                step_dict["tool_name"] = step.tool_name
            if step.tool_arguments:
                step_dict["tool_arguments"] = step.tool_arguments
            if step.tool_output is not None:
                # Truncate very large tool outputs
                try:
                    output_str = json.dumps(step.tool_output, default=str)
                    if len(output_str) > 4000:
                        step_dict["tool_output"] = step.tool_output  # Let JSON handle truncation below
                    else:
                        step_dict["tool_output"] = step.tool_output
                except (TypeError, ValueError):
                    step_dict["tool_output"] = str(step.tool_output)[:2000]
            if step.validation_error:
                step_dict["validation_error"] = step.validation_error
            if step.generation_time_ms:
                step_dict["generation_time_ms"] = step.generation_time_ms

            steps.append(step_dict)

        return {
            "incident_id": result.incident_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": result.model_id,
            "model_revision": result.model_revision,
            "query": result.query,
            "success": result.success,
            "num_steps": result.num_steps,
            "tools_called": result.tools_called,
            "diagnosis": result.diagnosis,
            "root_cause": result.root_cause,
            "evidence": result.evidence,
            "recommended_action": result.recommended_action,
            "requires_approval": result.requires_approval,
            "confidence": result.confidence,
            "safety_violations": result.safety_violations,
            "rejection_reasons": result.rejection_reasons,
            "total_time_ms": result.total_time_ms,
            "error": result.error,
            "trajectory_steps": steps,
            "metadata": result.metadata,
        }

    def load_all(self) -> list[dict[str, Any]]:
        """Load all logged trajectories from the JSONL file."""
        if not self.trajectories_path.exists():
            return []
        entries = []
        with open(self.trajectories_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed trajectory line %d: %s", line_num, exc)
        return entries

