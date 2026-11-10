"""
agent/core/orchestrator.py
==========================
Agent orchestration loop for DBMedic.

Implements the complete ReAct-style agent loop:
  1. Receive user question
  2. Construct model messages with system prompt
  3. Request model response
  4. Parse response (tool call or final answer)
  5. Validate tool calls
  6. Execute approved diagnostic tools
  7. Append tool output to conversation history
  8. Ask model to continue
  9. Stop on final answer or max steps

Safety constraints:
  - Never retries an unsafe tool call
  - Limited JSON parse retries for malformed output
  - Hard stop at max_steps regardless of model state
  - All tool calls go through validator before execution
  - Trajectory recorded for AgentTune training
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    """One step in the agent trajectory."""
    step_index: int
    step_type: str          # "user_question" | "model_response" | "tool_execution" | "final_answer"
    content: Any            # Raw content of this step
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_response_text: str = ""
    tool_name: str = ""
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None
    validation_error: str = ""
    generation_time_ms: float = 0.0
    is_valid: bool = True


@dataclass
class AgentResult:
    """Final result of an agent run."""
    incident_id: str
    query: str
    success: bool
    diagnosis: str
    root_cause: str
    evidence: list[str]
    recommended_action: str
    requires_approval: bool
    tools_called: list[str]
    num_steps: int
    safety_violations: list[str]
    rejection_reasons: list[str]
    trajectory: list[TrajectoryStep]
    model_id: str
    model_revision: str | None
    total_time_ms: float
    error: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "incident_id": self.incident_id,
            "query": self.query,
            "success": self.success,
            "diagnosis": self.diagnosis,
            "root_cause": self.root_cause,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "requires_approval": self.requires_approval,
            "tools_called": self.tools_called,
            "num_steps": self.num_steps,
            "safety_violations": self.safety_violations,
            "rejection_reasons": self.rejection_reasons,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "total_time_ms": self.total_time_ms,
            "error": self.error,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Orchestrates the DBMedic agent loop.

    Coordinates between model client, tool registry, validator, and parser.
    """

    def __init__(
        self,
        model_client: Any,
        tool_registry: Any,
        tool_validator: Any,
        response_parser: Any,
        max_steps: int = 10,
        max_tool_calls: int = 5,
        max_parse_retries: int = 2,
        incident_id_prefix: str = "agent_run",
    ) -> None:
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.tool_validator = tool_validator
        self.response_parser = response_parser
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_parse_retries = max_parse_retries
        self.incident_id_prefix = incident_id_prefix

        # Load prompt templates
        self._system_prompt = self._load_system_prompt()
        self._tool_call_prompt_template = self._load_tool_call_prompt()

    def _load_system_prompt(self) -> str:
        prompt_path = PROJECT_ROOT / "agent" / "prompts" / "system_prompt.txt"
        try:
            return prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("system_prompt.txt not found, using minimal fallback")
            return (
                "You are DBMedic, a PostgreSQL diagnostic assistant. "
                "Respond only in JSON format as either a tool_call or final_answer."
            )

    def _load_tool_call_prompt(self) -> str:
        prompt_path = PROJECT_ROOT / "agent" / "prompts" / "tool_call_prompt.txt"
        try:
            return prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "Tool result for '{tool_name}':\n{tool_output}\n\nRespond with JSON only."

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        incident_id: str | None = None,
    ) -> AgentResult:
        """
        Execute the agent loop for a user query.

        Parameters
        ----------
        query: str
            The user's database question.
        incident_id: str | None
            Optional incident identifier. Auto-generated if None.

        Returns
        -------
        AgentResult containing diagnosis, evidence, trajectory, and metadata.
        """
        if incident_id is None:
            incident_id = f"{self.incident_id_prefix}_{uuid.uuid4().hex[:8]}"

        start_time = time.perf_counter()
        trajectory: list[TrajectoryStep] = []
        tools_called: list[str] = []
        safety_violations: list[str] = []
        rejection_reasons: list[str] = []

        logger.info("Starting agent run: incident_id=%s, query=%s", incident_id, query)

        # Initialize conversation with system prompt + user question
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]

        # Record initial step
        trajectory.append(TrajectoryStep(
            step_index=0,
            step_type="user_question",
            content=query,
        ))

        step_index = 1
        final_answer: dict[str, Any] | None = None
        last_error = ""

        try:
            while step_index <= self.max_steps:
                logger.info("Agent step %d / %d", step_index, self.max_steps)

                # --------------------------------------------------------
                # 1. Generate model response
                # --------------------------------------------------------
                gen_result = self._generate_with_retry(messages)
                if gen_result is None:
                    last_error = "Model inference failed after retries"
                    break

                model_text = gen_result["text"]
                generation_time_ms = gen_result.get("generation_time_ms", 0.0)

                logger.debug("Model response (step %d): %s", step_index, model_text[:300])

                # --------------------------------------------------------
                # 2. Parse response
                # --------------------------------------------------------
                from agent.core.response_parser import ParseError
                try:
                    parsed = self.response_parser.parse(model_text)
                except ParseError as exc:
                    err_msg = str(exc)
                    if "Dangerous" in err_msg or "SQL" in err_msg or "injection" in err_msg:
                        # Safety violation at parse stage
                        rejection_reasons.append(err_msg)
                        safety_violations.append(err_msg)
                        trajectory.append(TrajectoryStep(
                            step_index=step_index,
                            step_type="model_response",
                            content=model_text,
                            model_response_text=model_text,
                            generation_time_ms=generation_time_ms,
                            is_valid=False,
                            validation_error=err_msg,
                        ))
                        messages.append({"role": "assistant", "content": model_text})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Your response was rejected for safety reasons: {err_msg}\n"
                                "Please provide a safe response."
                            ),
                        })
                        step_index += 1
                        continue

                    # Record failed parse
                    trajectory.append(TrajectoryStep(
                        step_index=step_index,
                        step_type="model_response",
                        content=model_text,
                        model_response_text=model_text,
                        generation_time_ms=generation_time_ms,
                        is_valid=False,
                        validation_error="Could not parse model output as valid JSON response",
                    ))
                    last_error = f"Parse failure at step {step_index}"
                    # Add a correction message and retry
                    messages.append({"role": "assistant", "content": model_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Please respond with ONLY a valid JSON object in one of these formats:\n"
                            '{"type": "tool_call", "tool": "<name>", "arguments": {}}\n'
                            'OR\n'
                            '{"type": "final_answer", "diagnosis": "<text>", "root_cause": "<type>", '
                            '"evidence": [], "recommended_action": "<text>", "requires_approval": true, "confidence": 0.5}'
                        ),
                    })
                    step_index += 1
                    continue

                # --------------------------------------------------------
                # 3. Handle parsed response
                # --------------------------------------------------------
                if parsed["type"] == "final_answer":
                    trajectory.append(TrajectoryStep(
                        step_index=step_index,
                        step_type="final_answer",
                        content=parsed,
                        model_response_text=model_text,
                        generation_time_ms=generation_time_ms,
                    ))
                    final_answer = parsed
                    logger.info("Agent reached final answer at step %d", step_index)
                    break

                elif parsed["type"] == "tool_call":
                    # --------------------------------------------------------
                    # 4. Validate tool call
                    # --------------------------------------------------------
                    validation = self.tool_validator.validate(parsed)

                    if not validation.is_valid:
                        reason = validation.rejection_reason
                        rejection_reasons.append(reason)

                        # Safety violation if it was a dangerous content rejection
                        if "Dangerous" in reason or "SQL" in reason or "injection" in reason:
                            safety_violations.append(reason)
                            logger.warning("SAFETY VIOLATION: %s", reason)

                        trajectory.append(TrajectoryStep(
                            step_index=step_index,
                            step_type="model_response",
                            content=parsed,
                            model_response_text=model_text,
                            tool_name=parsed.get("tool", ""),
                            tool_arguments=parsed.get("arguments", {}),
                            generation_time_ms=generation_time_ms,
                            is_valid=False,
                            validation_error=reason,
                        ))

                        # Tell model about the rejection (but never retry unsafe calls)
                        messages.append({"role": "assistant", "content": model_text})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Your tool call was rejected: {reason}\n"
                                "Please either call a different valid tool or provide a final answer."
                            ),
                        })
                        step_index += 1
                        continue

                    # --------------------------------------------------------
                    # 5. Execute validated tool
                    # --------------------------------------------------------
                    tool_name = validation.tool_name
                    sanitized_args = validation.sanitized_arguments

                    tool_output, tool_error = self._execute_tool(tool_name, sanitized_args)
                    tools_called.append(tool_name)

                    # Serialize tool output for model
                    if tool_error:
                        tool_output_str = f"Error: {tool_error}"
                        logger.warning("Tool '%s' returned error: %s", tool_name, tool_error)
                    else:
                        try:
                            tool_output_str = json.dumps(tool_output, indent=2, default=str)
                        except (TypeError, ValueError):
                            tool_output_str = str(tool_output)

                    trajectory.append(TrajectoryStep(
                        step_index=step_index,
                        step_type="tool_execution",
                        content=tool_output,
                        model_response_text=model_text,
                        tool_name=tool_name,
                        tool_arguments=sanitized_args,
                        tool_output=tool_output,
                        generation_time_ms=generation_time_ms,
                    ))

                    # --------------------------------------------------------
                    # 6. Append tool result to conversation
                    # --------------------------------------------------------
                    messages.append({"role": "assistant", "content": model_text})

                    tool_result_msg = self._tool_call_prompt_template.format(
                        tool_name=tool_name,
                        tool_output=tool_output_str,
                    )
                    messages.append({"role": "user", "content": tool_result_msg})

                    logger.info(
                        "Tool '%s' executed. Tools called so far: %d",
                        tool_name,
                        len(tools_called),
                    )

                step_index += 1

                # Hard stop if max tool calls reached
                if len(tools_called) >= self.max_tool_calls and final_answer is None:
                    logger.warning(
                        "Max tool calls (%d) reached. Requesting final answer.",
                        self.max_tool_calls,
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have used {len(tools_called)} tools. "
                            "Based on the evidence collected so far, provide your final diagnosis now. "
                            'Respond with: {"type": "final_answer", ...}'
                        ),
                    })

        except Exception as exc:
            last_error = str(exc)
            logger.error("Agent loop encountered unexpected error: %s", exc, exc_info=True)

        # ----------------------------------------------------------------
        # Build result
        # ----------------------------------------------------------------
        total_time_ms = (time.perf_counter() - start_time) * 1000
        executed_steps = len([s for s in trajectory if s.step_type != "user_question"])

        if final_answer:
            result = AgentResult(
                incident_id=incident_id,
                query=query,
                success=True,
                diagnosis=final_answer.get("diagnosis", ""),
                root_cause=final_answer.get("root_cause", "unknown"),
                evidence=final_answer.get("evidence", []),
                recommended_action=final_answer.get("recommended_action", ""),
                requires_approval=final_answer.get("requires_approval", True),
                confidence=final_answer.get("confidence", 0.0),
                tools_called=tools_called,
                num_steps=executed_steps,
                safety_violations=safety_violations,
                rejection_reasons=rejection_reasons,
                trajectory=trajectory,
                model_id=self.model_client.model_id,
                model_revision=self.model_client.model_revision,
                total_time_ms=round(total_time_ms, 1),
                metadata={
                    "max_steps": self.max_steps,
                    "max_tool_calls": self.max_tool_calls,
                    "validator_summary": self.tool_validator.session_summary(),
                },
            )
        else:
            result = AgentResult(
                incident_id=incident_id,
                query=query,
                success=False,
                diagnosis="Agent did not produce a final answer",
                root_cause="unknown",
                evidence=[],
                recommended_action="",
                requires_approval=True,
                confidence=0.0,
                tools_called=tools_called,
                num_steps=executed_steps,
                safety_violations=safety_violations,
                rejection_reasons=rejection_reasons,
                trajectory=trajectory,
                model_id=self.model_client.model_id,
                model_revision=getattr(self.model_client, 'model_revision', None),
                total_time_ms=round(total_time_ms, 1),
                error=last_error,
                metadata={
                    "max_steps": self.max_steps,
                    "max_tool_calls": self.max_tool_calls,
                    "validator_summary": self.tool_validator.session_summary(),
                },
            )

        logger.info(
            "Agent run complete: success=%s, steps=%d, tools=%d, time=%.1fms",
            result.success,
            result.num_steps,
            len(result.tools_called),
            result.total_time_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_with_retry(
        self, messages: list[dict[str, str]], max_retries: int = 2
    ) -> dict[str, Any] | None:
        """Generate a model response with retry on inference failures."""
        from agent.core.model_client import InferenceError

        for attempt in range(max_retries + 1):
            try:
                return self.model_client.generate(messages)
            except InferenceError as exc:
                logger.warning(
                    "Inference error (attempt %d/%d): %s",
                    attempt + 1, max_retries + 1, exc,
                )
                if attempt == max_retries:
                    return None
                time.sleep(1.0)  # Brief pause before retry

        return None

    def _parse_with_retry(
        self, text: str
    ) -> dict[str, Any] | None:
        """Parse model output with limited retries on parse failures."""
        from agent.core.response_parser import ParseError

        try:
            return self.response_parser.parse(text)
        except ParseError as exc:
            logger.debug("Initial parse failed: %s", exc)
            return None

    def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[Any, str]:
        """
        Execute a tool. Returns (output, error_message).
        Error message is empty string on success.
        """
        try:
            output = self.tool_registry.execute(tool_name, arguments)
            return output, ""
        except Exception as exc:
            return None, str(exc)

