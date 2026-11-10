"""
agent/core/agent.py
====================
Public interface for the DBMedic autonomous diagnostic agent.

Usage:
    agent = DBMedicAgent()
    result = agent.run("Why is my orders query taking so long?")

    # Or using an incident ID:
    result = agent.run_incident("slow_query_001")

The agent integrates:
  - TinyAyaClient: Loads and runs CohereLabs/tiny-aya-global
  - ToolRegistry: Manages Phase 2 diagnostic tools
  - ToolValidator: Validates all model-generated tool calls
  - ResponseParser: Parses model output into structured responses
  - AgentOrchestrator: Runs the full ReAct agent loop
  - TrajectoryLogger: Records all runs to JSONL for AgentTune

Returns a structured AgentResult with diagnosis, evidence, trajectory, and metadata.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_TOOL_CALLS = 5
DEFAULT_MODEL_ID = "CohereLabs/tiny-aya-global"
DEFAULT_TRAJECTORY_DIR = PROJECT_ROOT / "benchmarks" / "baseline"

# Incident ID → query mapping for known incidents
KNOWN_INCIDENTS: dict[str, dict[str, Any]] = {
    "slow_query_001": {
        "query": "Why is my orders query taking so long?",
        "description": "Slow query incident on orders table with customer_id filter",
        "customer_id": 4242,
    },
    "slow_query_002": {
        "query": "The orders table query for customer 1234 is very slow. What is wrong?",
        "description": "Slow query variant with different customer ID",
        "customer_id": 1234,
    },
    "slow_query_003": {
        "query": "Why does querying orders by customer_id perform a full table scan?",
        "description": "Direct sequential scan question",
        "customer_id": 4242,
    },
}


class DBMedicAgent:
    """
    DBMedic autonomous diagnostic agent.

    Integrates Tiny Aya Global with Phase 2 diagnostic tools.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        trajectory_dir: str | Path | None = None,
        generation_config: dict[str, Any] | None = None,
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.trajectory_dir = Path(trajectory_dir or DEFAULT_TRAJECTORY_DIR)
        self.generation_config = generation_config
        self.device_map = device_map

        # Components initialized lazily
        self._model_client = None
        self._tool_registry = None
        self._response_parser = None
        self._trajectory_logger = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Initialize all agent components.

        Loads the model, registers tools, and sets up logging.
        Call this explicitly before run() if you want to control load timing.
        """
        if self._initialized:
            return

        logger.info("Initializing DBMedic agent...")

        # Model client
        from agent.core.model_client import TinyAyaClient
        self._model_client = TinyAyaClient(
            model_id=self.model_id,
            generation_config=self.generation_config,
            device_map=self.device_map,
        )
        self._model_client.load()

        # Tool registry
        from agent.core.tool_registry import ToolRegistry
        self._tool_registry = ToolRegistry()
        self._tool_registry.initialize()

        # Response parser
        from agent.core.response_parser import ResponseParser
        self._response_parser = ResponseParser()

        # Trajectory logger
        from agent.evaluation.trajectory_logger import TrajectoryLogger
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._trajectory_logger = TrajectoryLogger(self.trajectory_dir)

        self._initialized = True
        logger.info("DBMedic agent initialized successfully")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        incident_id: str | None = None,
        log_trajectory: bool = True,
    ) -> Any:  # AgentResult
        """
        Run the agent on a user query.

        Parameters
        ----------
        query: str
            The database question to investigate.
        incident_id: str | None
            Optional incident identifier for tracking.
        log_trajectory: bool
            Whether to log the trajectory to JSONL file.

        Returns
        -------
        AgentResult containing diagnosis, evidence, and trajectory.
        """
        if not self._initialized:
            self.initialize()

        # Create fresh validator for this run
        from agent.core.tool_validator import ToolValidator
        from agent.core.orchestrator import AgentOrchestrator

        validator = ToolValidator(
            registry=self._tool_registry,
            max_calls_per_session=self.max_tool_calls,
        )

        orchestrator = AgentOrchestrator(
            model_client=self._model_client,
            tool_registry=self._tool_registry,
            tool_validator=validator,
            response_parser=self._response_parser,
            max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls,
            incident_id_prefix="agent",
        )

        result = orchestrator.run(query, incident_id=incident_id)

        if log_trajectory and self._trajectory_logger:
            self._trajectory_logger.log(result)

        return result

    def run_incident(
        self,
        incident_id: str,
        log_trajectory: bool = True,
    ) -> Any:  # AgentResult
        """
        Run the agent on a known incident by ID.

        Parameters
        ----------
        incident_id: str
            Known incident identifier (e.g., "slow_query_001").

        Returns
        -------
        AgentResult
        """
        if incident_id not in KNOWN_INCIDENTS:
            available = list(KNOWN_INCIDENTS.keys())
            raise ValueError(
                f"Unknown incident ID '{incident_id}'. "
                f"Available incidents: {available}"
            )

        incident = KNOWN_INCIDENTS[incident_id]
        query = incident["query"]

        logger.info("Running incident: %s — %s", incident_id, incident["description"])

        return self.run(query, incident_id=incident_id, log_trajectory=log_trajectory)

    def unload(self) -> None:
        """Release model from memory."""
        if self._model_client:
            self._model_client.unload()
        self._initialized = False

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"DBMedicAgent(model_id='{self.model_id}', status={status})"

