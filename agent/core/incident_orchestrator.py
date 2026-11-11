"""
agent/core/incident_orchestrator.py
===================================
End-to-end DBMedic Phase 6 incident lifecycle orchestrator.

Coordinates Phase 1–5 components across one auditable, fail-safe state machine:

    CREATED -> INVESTIGATING -> DIAGNOSED -> AWAITING_APPROVAL -> APPROVED
    -> REMEDIATING -> VERIFYING -> RESOLVED

Also enforces:
  - REJECTED, FAILED, BLOCKED, NEEDS_INVESTIGATION
  - Strict rejection of invalid state transitions
  - Explicit execution modes: ``live_tiny_aya``, ``deterministic``, ``simulated``
  - Zero silent fallback from ``live_tiny_aya`` to ``deterministic``
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent.core.approval_manager import ApprovalManager
from agent.core.audit_logger import AuditLogger, redact_secrets
from agent.core.diagnosis_engine import (
    DEFAULT_LATENCY_THRESHOLD_MS,
    diagnose_slow_query,
)
from agent.core.environment_check import (
    DEFAULT_MODEL_ID,
    check_database_connection,
    check_tiny_aya_availability,
    parse_database_identity,
)
from agent.core.remediation_manager import (
    ALLOWED_REMEDIATION_SPECS,
    RemediationManager,
)
from agent.core.remediation_verifier import (
    DEFAULT_COMPARISON_OUTPUT_PATH,
    capture_post_remediation_metrics,
    capture_pre_remediation_metrics,
    verify_and_save_comparison,
    verify_index_created,
)
from agent.tools import get_lab_connection
from agent.tools.index_monitor import inspect_customer_id_index
from agent.tools.query_analyzer import (
    DEFAULT_CUSTOMER_ID,
    DEFAULT_QUERY_KEY,
    analyze_query_performance,
)
from agent.tools.table_monitor import get_table_statistics
from database.scenarios.slow_query import BENCHMARK_RUNS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INCIDENTS_DIR = PROJECT_ROOT / "reports" / "incidents"

# Lifecycle states
STATE_CREATED = "CREATED"
STATE_INVESTIGATING = "INVESTIGATING"
STATE_DIAGNOSED = "DIAGNOSED"
STATE_AWAITING_APPROVAL = "AWAITING_APPROVAL"
STATE_APPROVED = "APPROVED"
STATE_REMEDIATING = "REMEDIATING"
STATE_VERIFYING = "VERIFYING"
STATE_RESOLVED = "RESOLVED"
STATE_REJECTED = "REJECTED"
STATE_FAILED = "FAILED"
STATE_BLOCKED = "BLOCKED"
STATE_NEEDS_INVESTIGATION = "NEEDS_INVESTIGATION"

ALL_INCIDENT_STATES: frozenset[str] = frozenset(
    {
        STATE_CREATED,
        STATE_INVESTIGATING,
        STATE_DIAGNOSED,
        STATE_AWAITING_APPROVAL,
        STATE_APPROVED,
        STATE_REMEDIATING,
        STATE_VERIFYING,
        STATE_RESOLVED,
        STATE_REJECTED,
        STATE_FAILED,
        STATE_BLOCKED,
        STATE_NEEDS_INVESTIGATION,
    }
)

VALID_STATE_TRANSITIONS: dict[str, frozenset[str]] = {
    STATE_CREATED: frozenset({STATE_INVESTIGATING, STATE_BLOCKED, STATE_FAILED, STATE_NEEDS_INVESTIGATION}),
    STATE_INVESTIGATING: frozenset({STATE_DIAGNOSED, STATE_NEEDS_INVESTIGATION, STATE_BLOCKED, STATE_FAILED}),
    STATE_NEEDS_INVESTIGATION: frozenset({STATE_INVESTIGATING, STATE_DIAGNOSED, STATE_BLOCKED, STATE_FAILED}),
    STATE_DIAGNOSED: frozenset({STATE_AWAITING_APPROVAL, STATE_RESOLVED, STATE_NEEDS_INVESTIGATION, STATE_BLOCKED, STATE_FAILED}),
    STATE_AWAITING_APPROVAL: frozenset({STATE_APPROVED, STATE_REJECTED, STATE_BLOCKED, STATE_FAILED}),
    STATE_APPROVED: frozenset({STATE_REMEDIATING, STATE_BLOCKED, STATE_FAILED}),
    STATE_REMEDIATING: frozenset({STATE_VERIFYING, STATE_FAILED, STATE_BLOCKED}),
    STATE_VERIFYING: frozenset({STATE_RESOLVED, STATE_NEEDS_INVESTIGATION, STATE_FAILED}),
    STATE_REJECTED: frozenset({STATE_RESOLVED, STATE_NEEDS_INVESTIGATION}),
    STATE_BLOCKED: frozenset({STATE_INVESTIGATING, STATE_NEEDS_INVESTIGATION, STATE_FAILED}),
    STATE_FAILED: frozenset({STATE_NEEDS_INVESTIGATION}),
    STATE_RESOLVED: frozenset(),
}

EXECUTION_MODE_LIVE_TINY_AYA = "live_tiny_aya"
EXECUTION_MODE_DETERMINISTIC = "deterministic"
EXECUTION_MODE_SIMULATED = "simulated"
VALID_EXECUTION_MODES: frozenset[str] = frozenset(
    {EXECUTION_MODE_LIVE_TINY_AYA, EXECUTION_MODE_DETERMINISTIC, EXECUTION_MODE_SIMULATED}
)


class InvalidStateTransitionError(ValueError):
    """Raised when an incident lifecycle transition violates state-machine safety rules."""


class OrchestrationBlockedError(RuntimeError):
    """Raised when a required runtime dependency for the chosen execution mode is unavailable."""


class IncidentNotFoundError(KeyError):
    """Raised when an incident ID is not found in state storage."""


def normalize_execution_mode(mode: str) -> str:
    """Normalize CLI/internal execution mode names strictly without fallback."""
    cleaned = str(mode or "").strip().lower()
    mapping = {
        "tiny-aya": EXECUTION_MODE_LIVE_TINY_AYA,
        "tiny_aya": EXECUTION_MODE_LIVE_TINY_AYA,
        "live_tiny_aya": EXECUTION_MODE_LIVE_TINY_AYA,
        "live": EXECUTION_MODE_LIVE_TINY_AYA,
        "deterministic": EXECUTION_MODE_DETERMINISTIC,
        "simulated": EXECUTION_MODE_SIMULATED,
    }
    if cleaned not in mapping:
        raise ValueError(
            f"Unsupported execution mode '{mode}'. "
            "Valid explicit modes: tiny-aya, live_tiny_aya, deterministic, simulated."
        )
    return mapping[cleaned]


class IncidentOrchestrator:
    """Orchestrates complete DBMedic incidents with persisted state and safety gating."""

    def __init__(
        self,
        *,
        incidents_dir: Path | str | None = None,
        audit_logger: AuditLogger | None = None,
        approval_manager: ApprovalManager | None = None,
        remediation_manager: RemediationManager | None = None,
        agent_factory: Callable[..., Any] | None = None,
        comparison_output_path: Path | str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.incidents_dir = Path(incidents_dir) if incidents_dir else DEFAULT_INCIDENTS_DIR
        self.audit_logger = audit_logger or AuditLogger()
        self.approval_manager = approval_manager or ApprovalManager(
            audit_logger=self.audit_logger
        )
        self.remediation_manager = remediation_manager or RemediationManager(
            approval_manager=self.approval_manager,
            audit_logger=self.audit_logger,
        )
        self.agent_factory = agent_factory
        self.comparison_output_path = (
            Path(comparison_output_path)
            if comparison_output_path
            else DEFAULT_COMPARISON_OUTPUT_PATH
        )
        self.env = env

    # ------------------------------------------------------------------
    # State persistence & transition enforcement
    # ------------------------------------------------------------------

    def _incident_path(self, incident_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in incident_id)
        return self.incidents_dir / f"{safe_id}.json"

    def save_incident(self, incident: dict[str, Any]) -> dict[str, Any]:
        self.incidents_dir.mkdir(parents=True, exist_ok=True)
        incident["updated_at"] = datetime.now(timezone.utc).isoformat()
        sanitized = redact_secrets(incident)
        self._incident_path(incident["incident_id"]).write_text(
            json.dumps(sanitized, indent=2, default=str),
            encoding="utf-8",
        )
        return sanitized

    def load_incident(self, incident_id: str) -> dict[str, Any]:
        path = self._incident_path(incident_id)
        if not path.exists():
            raise IncidentNotFoundError(f"Incident '{incident_id}' does not exist in {self.incidents_dir}.")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_incidents(self) -> list[dict[str, Any]]:
        if not self.incidents_dir.exists():
            return []
        items: list[dict[str, Any]] = []
        for p in sorted(self.incidents_dir.glob("*.json")):
            try:
                items.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return items

    def transition_state(
        self,
        incident: dict[str, Any],
        target_state: str,
        *,
        reason: str = "",
        actor: str = "dbmedic_orchestrator",
    ) -> dict[str, Any]:
        """Enforce the incident lifecycle state machine; reject invalid transitions."""
        current = str(incident.get("status", ""))
        if target_state not in ALL_INCIDENT_STATES:
            raise InvalidStateTransitionError(f"Unknown lifecycle state '{target_state}'.")

        allowed = VALID_STATE_TRANSITIONS.get(current, frozenset())
        if target_state not in allowed:
            self.audit_logger.log_event(
                incident_id=incident["incident_id"],
                event_type="STATE_TRANSITION_REJECTED",
                actor=actor,
                execution_mode=incident.get("execution_mode", "system"),
                payload={
                    "from_state": current,
                    "attempted_state": target_state,
                    "reason": reason or "Transition violates lifecycle state machine.",
                },
            )
            raise InvalidStateTransitionError(
                f"Invalid incident transition '{current}' -> '{target_state}' "
                f"for '{incident['incident_id']}'. Allowed next states: {sorted(allowed)}."
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        incident["status"] = target_state
        incident.setdefault("state_history", []).append(
            {
                "from": current,
                "to": target_state,
                "timestamp": now_iso,
                "reason": reason,
                "actor": actor,
            }
        )
        self.save_incident(incident)

        self.audit_logger.log_event(
            incident_id=incident["incident_id"],
            event_type="INCIDENT_STATUS_CHANGED",
            actor=actor,
            execution_mode=incident.get("execution_mode", "system"),
            payload={"from_state": current, "to_state": target_state, "reason": reason},
        )
        return incident

    # ------------------------------------------------------------------
    # Lifecycle methods required by Phase 6 specification
    # ------------------------------------------------------------------

    def create_incident(
        self,
        *,
        incident_id: str = "slow_query_001",
        query_description: str = "Why is my orders query for customer_id taking so long?",
        customer_id: int = DEFAULT_CUSTOMER_ID,
        execution_mode: str = EXECUTION_MODE_DETERMINISTIC,
        model_id: str | None = None,
        simulated_index_exists: bool = False,
    ) -> dict[str, Any]:
        """Initialize and persist a new incident in ``CREATED`` state."""
        mode = normalize_execution_mode(execution_mode)
        resolved_model = (
            model_id
            if model_id is not None
            else (DEFAULT_MODEL_ID if mode == EXECUTION_MODE_LIVE_TINY_AYA else "deterministic_phase2_engine")
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        incident: dict[str, Any] = {
            "schema_version": "dbmedic.incident/1",
            "run_id": str(uuid.uuid4()),
            "incident_id": incident_id.strip(),
            "status": STATE_CREATED,
            "query": query_description,
            "customer_id": int(customer_id),
            "execution_mode": mode,
            "simulated": mode == EXECUTION_MODE_SIMULATED,
            "simulated_precondition": {
                "idx_orders_customer_id_exists": bool(simulated_index_exists),
            },
            "model_id": resolved_model,
            "created_at": now_iso,
            "updated_at": now_iso,
            "state_history": [
                {
                    "from": None,
                    "to": STATE_CREATED,
                    "timestamp": now_iso,
                    "reason": "Incident created",
                    "actor": "dbmedic_orchestrator",
                }
            ],
            "evidence": None,
            "pre_remediation_metrics": None,
            "diagnosis": None,
            "diagnosis_validation": None,
            "remediation_plan": None,
            "approval_request": None,
            "remediation_result": None,
            "post_remediation_metrics": None,
            "verification_report": None,
            "blocker_reason": None,
            "closure_summary": None,
        }
        self.save_incident(incident)

        self.audit_logger.log_event(
            incident_id=incident["incident_id"],
            event_type="INCIDENT_CREATED",
            actor="dbmedic_orchestrator",
            execution_mode=mode,
            payload={
                "run_id": incident["run_id"],
                "customer_id": incident["customer_id"],
                "query": incident["query"],
                "model_id": resolved_model,
            },
        )
        return incident

    def collect_incident_evidence(
        self,
        incident: dict[str, Any],
        *,
        conn: Any = None,
        n_benchmark_runs: int = BENCHMARK_RUNS,
    ) -> dict[str, Any]:
        """Collect PostgreSQL plan, table stats, index metadata, and baseline benchmark."""
        mode = incident["execution_mode"]
        self.transition_state(
            incident,
            STATE_INVESTIGATING,
            reason=f"Starting evidence collection in '{mode}' mode",
        )
        self.audit_logger.log_event(
            incident_id=incident["incident_id"],
            event_type="DIAGNOSTIC_STARTED",
            actor="dbmedic_orchestrator",
            execution_mode=mode,
            payload={"customer_id": incident["customer_id"]},
        )

        if mode == EXECUTION_MODE_SIMULATED:
            idx_sim = bool(
                (incident.get("simulated_precondition") or {}).get(
                    "idx_orders_customer_id_exists", False
                )
            )
            pre_metrics = capture_pre_remediation_metrics(
                customer_id=incident["customer_id"],
                n_runs=n_benchmark_runs,
                simulated=True,
                simulated_index_present=idx_sim,
            )
            evidence = {
                "simulated": True,
                "execution_mode": EXECUTION_MODE_SIMULATED,
                "target_database": parse_database_identity(
                    (self.env or {}).get("LAB_DATABASE_URL", "")
                    or "postgresql://dbmedic@localhost:5432/dbmedic_lab"
                ),
                "query_performance": {
                    "template_key": DEFAULT_QUERY_KEY,
                    "customer_id": incident["customer_id"],
                    "metrics": {
                        "scan_type": pre_metrics["scan_type"],
                        "execution_time_ms": pre_metrics["execution_time_ms"],
                        "planning_time_ms": pre_metrics["planning_time_ms"],
                        "actual_rows": pre_metrics["actual_rows"],
                        "estimated_rows": pre_metrics["estimated_rows"],
                        "shared_hit_blocks": pre_metrics["shared_hit_blocks"],
                        "shared_read_blocks": pre_metrics["shared_read_blocks"],
                    },
                    "scan_analysis": {
                        "has_sequential_scan": not idx_sim,
                        "primary_scan_type": pre_metrics["scan_type"],
                        "raw_node_type": pre_metrics["raw_scan_type"],
                        "scanned_relation": "orders",
                    },
                },
                "table_statistics": {
                    "table_name": "orders",
                    "estimated_row_count": 200_000,
                    "table_size_pretty": "34 MB",
                    "sequential_scan_count": 0 if idx_sim else 14,
                },
                "index_inspection": {
                    "table_name": "orders",
                    "target_column": "customer_id",
                    "index_exists": idx_sim,
                    "is_suitable_for_filter": idx_sim,
                    "index_name": "idx_orders_customer_id" if idx_sim else None,
                },
            }
        else:
            db_check = (
                {"connected": True, "target_database": parse_database_identity("postgresql://localhost:5432/dbmedic_lab")}
                if conn is not None
                else check_database_connection(env=self.env)
            )
            if not db_check.get("connected"):
                blocker = db_check.get("blocker") or "PostgreSQL dbmedic_lab is unreachable."
                incident["blocker_reason"] = blocker
                self.transition_state(incident, STATE_BLOCKED, reason=blocker)
                raise OrchestrationBlockedError(blocker)

            try:
                with get_lab_connection(conn, read_only=True) as db_conn:
                    perf = analyze_query_performance(
                        db_conn,
                        template_key=DEFAULT_QUERY_KEY,
                        customer_id=incident["customer_id"],
                    )
                    t_stats = get_table_statistics("orders", conn=db_conn)
                    idx_info = inspect_customer_id_index(conn=db_conn)
                    pre_metrics = capture_pre_remediation_metrics(
                        db_conn,
                        customer_id=incident["customer_id"],
                        n_runs=n_benchmark_runs,
                        simulated=False,
                    )
                    perf.pop("raw_plan", None)
                    pre_metrics.pop("raw_plan", None)
                evidence = {
                    "simulated": False,
                    "execution_mode": mode,
                    "target_database": db_check["target_database"],
                    "query_performance": perf,
                    "table_statistics": t_stats,
                    "index_inspection": idx_info,
                }
            except Exception as exc:
                incident["blocker_reason"] = redact_secrets(str(exc))
                self.transition_state(
                    incident,
                    STATE_FAILED,
                    reason=f"Evidence collection failed: {type(exc).__name__}",
                )
                raise

        incident["evidence"] = evidence
        incident["pre_remediation_metrics"] = pre_metrics
        self.save_incident(incident)

        self.audit_logger.log_event(
            incident_id=incident["incident_id"],
            event_type="DIAGNOSTIC_EVIDENCE",
            actor="dbmedic_orchestrator",
            execution_mode=mode,
            payload={
                "scan_type": pre_metrics["scan_type"],
                "median_execution_time_ms": pre_metrics["median_execution_time_ms"],
                "index_exists": evidence["index_inspection"].get("is_suitable_for_filter"),
                "table_rows": evidence["table_statistics"].get("estimated_row_count"),
            },
        )
        return incident

    def run_agent_diagnosis(
        self,
        incident: dict[str, Any],
        *,
        conn: Any = None,
    ) -> dict[str, Any]:
        """
        Execute diagnosis using the explicitly requested mode.

        Never silently substitutes deterministic execution if ``live_tiny_aya`` fails.
        """
        mode = incident["execution_mode"]

        if mode == EXECUTION_MODE_LIVE_TINY_AYA:
            self.audit_logger.log_event(
                incident_id=incident["incident_id"],
                event_type="MODEL_IDENTITY_RECORDED",
                actor="dbmedic_agent",
                execution_mode=mode,
                payload={"model_id": incident["model_id"], "requested_mode": mode},
            )
            if self.agent_factory is None:
                tiny_check = check_tiny_aya_availability(env=self.env)
                if not tiny_check["live_inference_ready"]:
                    blocker = "; ".join(tiny_check["blockers"])
                    incident["blocker_reason"] = blocker
                    self.transition_state(incident, STATE_BLOCKED, reason=blocker)
                    raise OrchestrationBlockedError(
                        f"Tiny Aya live mode blocked (no silent fallback allowed): {blocker}"
                    )

            try:
                if self.agent_factory is not None:
                    agent = self.agent_factory(model_id=incident["model_id"])
                else:
                    from agent.core.agent import DBMedicAgent

                    agent = DBMedicAgent(model_id=incident["model_id"])

                agent_res = agent.run(
                    query=incident["query"],
                    incident_id=incident["incident_id"],
                    log_trajectory=True,
                )
                if not getattr(agent_res, "success", False):
                    err = getattr(agent_res, "error", None) or "Tiny Aya inference failed"
                    incident["blocker_reason"] = str(err)
                    self.transition_state(incident, STATE_FAILED, reason=str(err))
                    raise OrchestrationBlockedError(f"Tiny Aya execution failed: {err}")

                tools_called = list(getattr(agent_res, "tools_called", []) or [])
                for tname in tools_called:
                    self.audit_logger.log_event(
                        incident_id=incident["incident_id"],
                        event_type="TOOL_SELECTED",
                        actor=incident["model_id"],
                        execution_mode=mode,
                        payload={"tool_name": tname},
                    )
                    self.audit_logger.log_event(
                        incident_id=incident["incident_id"],
                        event_type="TOOL_EXECUTED",
                        actor="tool_registry",
                        execution_mode=mode,
                        payload={"tool_name": tname},
                    )

                diagnosis = {
                    "incident_id": incident["incident_id"],
                    "incident_type": "slow_query",
                    "execution_mode": EXECUTION_MODE_LIVE_TINY_AYA,
                    "simulated": False,
                    "model_id": getattr(agent_res, "model_id", incident["model_id"]),
                    "model_revision": getattr(agent_res, "model_revision", None),
                    "tools_called": tools_called,
                    "num_steps": getattr(agent_res, "num_steps", 0),
                    "summary": getattr(agent_res, "diagnosis", ""),
                    "root_cause": {
                        "type": getattr(agent_res, "root_cause", "unknown"),
                        "description": getattr(agent_res, "diagnosis", ""),
                    },
                    "evidence_claims": list(getattr(agent_res, "evidence", []) or []),
                    "recommended_action_text": getattr(agent_res, "recommended_action", ""),
                    "recommendation": {
                        "action": (
                            "create_index"
                            if "missing" in str(getattr(agent_res, "root_cause", "")).lower()
                            or "index" in str(getattr(agent_res, "recommended_action", "")).lower()
                            else "none"
                        ),
                        "sql": (
                            ALLOWED_REMEDIATION_SPECS["create_idx_orders_customer_id"]["proposed_sql"]
                            if "index" in str(getattr(agent_res, "recommended_action", "")).lower()
                            else None
                        ),
                        "requires_approval": True,
                    },
                    "diagnostic_score": getattr(agent_res, "confidence", None),
                }
            except OrchestrationBlockedError:
                raise
            except Exception as exc:
                incident["blocker_reason"] = redact_secrets(str(exc))
                self.transition_state(
                    incident,
                    STATE_FAILED,
                    reason=f"Tiny Aya diagnosis error: {type(exc).__name__}",
                )
                raise OrchestrationBlockedError(
                    f"Tiny Aya execution failed (no silent fallback allowed): {exc}"
                ) from exc

        elif mode == EXECUTION_MODE_DETERMINISTIC:
            self.audit_logger.log_event(
                incident_id=incident["incident_id"],
                event_type="MODEL_IDENTITY_RECORDED",
                actor="deterministic_phase2_engine",
                execution_mode=mode,
                payload={"model_id": "deterministic_phase2_engine"},
            )
            tools = [
                "get_query_execution_plan",
                "analyze_query_performance",
                "get_table_statistics",
                "inspect_customer_id_index",
            ]
            for tname in tools:
                self.audit_logger.log_event(
                    incident_id=incident["incident_id"],
                    event_type="TOOL_SELECTED",
                    actor="deterministic_phase2_engine",
                    execution_mode=mode,
                    payload={"tool_name": tname},
                )
            with get_lab_connection(conn, read_only=True) as db_conn:
                report = diagnose_slow_query(
                    db_conn,
                    customer_id=incident["customer_id"],
                    latency_threshold_ms=DEFAULT_LATENCY_THRESHOLD_MS,
                )
            diagnosis = {
                **report,
                "incident_id": incident["incident_id"],
                "execution_mode": EXECUTION_MODE_DETERMINISTIC,
                "simulated": False,
                "model_id": "deterministic_phase2_engine",
                "tools_called": tools,
            }

        else:  # EXECUTION_MODE_SIMULATED
            from agent.core.diagnosis_engine import (
                calculate_diagnostic_score,
                create_diagnostic_report,
                generate_recommendation,
            )

            ev = incident.get("evidence") or {}
            pre = incident.get("pre_remediation_metrics") or {}
            t_stats = ev.get("table_statistics", {"estimated_row_count": 200_000})
            idx_info = ev.get("index_inspection", {"is_suitable_for_filter": False})
            suitable_exists = bool(idx_info.get("is_suitable_for_filter", False))

            ev_summary = {
                "scan_type": pre.get("scan_type", "Sequential Scan"),
                "execution_time_ms": pre.get("execution_time_ms", 24.3),
                "table_name": "orders",
                "filter_column": "customer_id",
                "suitable_index_exists": suitable_exists,
            }
            score = calculate_diagnostic_score(
                ev_summary,
                t_stats,
                row_selectivity=20 / max(int(t_stats.get("estimated_row_count", 200_000)), 1),
            )
            root_type = "none_detected" if suitable_exists else "missing_suitable_index"
            rec = generate_recommendation(root_type, "orders", "customer_id")
            report = create_diagnostic_report(
                incident_id=incident["incident_id"],
                incident_type="slow_query",
                status="optimized" if suitable_exists else "diagnosed",
                severity="low" if suitable_exists else "high",
                summary=(
                    "Query is using an index efficiently"
                    if suitable_exists
                    else "Simulated slow-query incident: missing index on orders(customer_id)"
                ),
                root_cause={
                    "type": root_type,
                    "description": (
                        "Index already exists"
                        if suitable_exists
                        else "Sequential scan detected on orders (200,000 rows) without customer_id index"
                    ),
                },
                evidence=ev_summary,
                recommendation=rec,
                diagnostic_score=score,
            )
            diagnosis = {
                **report,
                "execution_mode": EXECUTION_MODE_SIMULATED,
                "simulated": True,
                "model_id": "simulated_deterministic_engine",
                "tools_called": [
                    "get_query_execution_plan",
                    "get_table_statistics",
                    "inspect_customer_id_index",
                ],
            }

        incident["diagnosis"] = diagnosis
        self.transition_state(
            incident,
            STATE_DIAGNOSED,
            reason=f"Diagnosis complete: {diagnosis['root_cause'].get('type')}",
        )
        return incident

    def validate_diagnosis(self, incident: dict[str, Any]) -> dict[str, Any]:
        """Validate diagnosis against collected evidence and remediation policy."""
        diag = incident.get("diagnosis") or {}
        ev = incident.get("evidence") or {}
        idx_exists = bool((ev.get("index_inspection") or {}).get("is_suitable_for_filter", False))
        root_type = str((diag.get("root_cause") or {}).get("type", ""))
        rec = diag.get("recommendation") or {}
        rec_action = str(rec.get("action", "none"))

        checks: dict[str, bool] = {
            "has_evidence": bool(ev),
            "has_diagnosis": bool(diag),
            "root_cause_present": bool(root_type),
            "approval_enforced_when_mutating": (
                True if rec_action == "none" else bool(rec.get("requires_approval", False))
            ),
            "consistent_with_index_state": not (
                root_type == "missing_suitable_index" and idx_exists
            ),
        }
        passed = all(checks.values())
        validation = {
            "valid": passed,
            "checks": checks,
            "requires_remediation": rec_action == "create_index" and not idx_exists,
        }
        incident["diagnosis_validation"] = validation
        self.save_incident(incident)

        if not passed:
            self.transition_state(
                incident,
                STATE_NEEDS_INVESTIGATION,
                reason=f"Diagnosis validation failed checks: {checks}",
            )
        return incident

    def prepare_remediation(self, incident: dict[str, Any]) -> dict[str, Any]:
        """Prepare allowlisted remediation action specification (no DB mutation)."""
        validation = incident.get("diagnosis_validation") or {}
        if not validation.get("valid", False):
            raise InvalidStateTransitionError(
                "Cannot prepare remediation before diagnosis passes validation."
            )

        spec = ALLOWED_REMEDIATION_SPECS["create_idx_orders_customer_id"]
        action_id = f"rem_{incident['incident_id']}_create_idx_orders_customer_id"
        plan = {
            "action_id": action_id,
            "remediation_key": spec["remediation_key"],
            "proposed_sql": spec["proposed_sql"],
            "description": spec["description"],
            "table_name": spec["table_name"],
            "column_name": spec["column_name"],
            "index_name": spec["index_name"],
            "target_database": (incident.get("evidence") or {}).get("target_database")
            or parse_database_identity("postgresql://localhost:5432/dbmedic_lab"),
        }
        incident["remediation_plan"] = plan
        self.save_incident(incident)

        self.audit_logger.log_event(
            incident_id=incident["incident_id"],
            event_type="RECOMMENDATION_PROPOSED",
            actor="dbmedic_orchestrator",
            execution_mode=incident["execution_mode"],
            payload=plan,
        )
        return incident

    def request_human_approval(
        self,
        incident: dict[str, Any],
        *,
        ttl_seconds: int = 1800,
    ) -> dict[str, Any]:
        """Create human approval request and move incident to ``AWAITING_APPROVAL``."""
        plan = incident.get("remediation_plan")
        if not plan:
            raise InvalidStateTransitionError("prepare_remediation must run before requesting approval.")

        req = self.approval_manager.create_approval_request(
            incident_id=incident["incident_id"],
            action_id=plan["action_id"],
            target_database=plan["target_database"],
            proposed_sql=plan["proposed_sql"],
            action_description=plan["description"],
            evidence={
                "scan_type": (incident.get("pre_remediation_metrics") or {}).get("scan_type"),
                "median_execution_time_ms": (incident.get("pre_remediation_metrics") or {}).get(
                    "median_execution_time_ms"
                ),
                "root_cause": (incident.get("diagnosis") or {}).get("root_cause"),
                "index_inspection": (incident.get("evidence") or {}).get("index_inspection"),
            },
            remediation_key=plan["remediation_key"],
            ttl_seconds=ttl_seconds,
            execution_mode=incident["execution_mode"],
        )
        incident["approval_request"] = req
        self.transition_state(
            incident,
            STATE_AWAITING_APPROVAL,
            reason=f"Awaiting explicit human approval for action '{plan['action_id']}'",
        )
        return incident

    def submit_human_decision(
        self,
        incident: dict[str, Any],
        *,
        approved: bool,
        operator: str,
        confirmation_method: str = "interactive_cli",
        reason: str = "",
    ) -> dict[str, Any]:
        """Record human operator decision and advance state to APPROVED or REJECTED."""
        if incident["status"] != STATE_AWAITING_APPROVAL:
            raise InvalidStateTransitionError(
                f"Cannot record approval decision while incident status is '{incident['status']}'."
            )
        plan = incident.get("remediation_plan") or {}
        if approved:
            req = self.approval_manager.record_human_approval(
                incident_id=incident["incident_id"],
                action_id=plan["action_id"],
                approved_by=operator,
                confirmation_method=confirmation_method,
                expected_sql=plan["proposed_sql"],
            )
            incident["approval_request"] = req
            return self.transition_state(
                incident,
                STATE_APPROVED,
                reason=f"Explicitly approved by human operator '{req['approved_by']}'",
                actor=req["approved_by"],
            )

        req = self.approval_manager.record_human_rejection(
            incident_id=incident["incident_id"],
            action_id=plan["action_id"],
            rejected_by=operator,
            reason=reason or "Rejected by human operator",
        )
        incident["approval_request"] = req
        return self.transition_state(
            incident,
            STATE_REJECTED,
            reason=req.get("rejection_reason") or "Rejected by human operator",
            actor=operator,
        )

    def execute_approved_remediation(
        self,
        incident: dict[str, Any],
        *,
        conn: Any = None,
        approval_payload: dict[str, Any] | None = None,
        current_database_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Transition APPROVED -> REMEDIATING -> VERIFYING while running verified DDL."""
        if incident["status"] != STATE_APPROVED:
            # Log attempted bypass & reject via state machine
            return self.transition_state(
                incident,
                STATE_REMEDIATING,
                reason="Attempted remediation without APPROVED status",
            )

        plan = incident.get("remediation_plan") or {}
        self.transition_state(
            incident,
            STATE_REMEDIATING,
            reason=f"Executing approved remediation '{plan.get('action_id')}'",
        )

        try:
            rem_result = self.remediation_manager.execute_remediation(
                incident_id=incident["incident_id"],
                action_id=plan["action_id"],
                remediation_key=plan["remediation_key"],
                proposed_sql=plan["proposed_sql"],
                conn=conn,
                approval_payload=approval_payload,
                execution_mode=incident["execution_mode"],
                current_database_identity=current_database_identity,
                simulated=incident["execution_mode"] == EXECUTION_MODE_SIMULATED,
            )
        except Exception as exc:
            incident["blocker_reason"] = redact_secrets(str(exc))
            self.transition_state(
                incident,
                STATE_FAILED,
                reason=f"Remediation execution failed: {redact_secrets(str(exc))[:200]}",
            )
            raise

        incident["remediation_result"] = rem_result
        self.transition_state(
            incident,
            STATE_VERIFYING,
            reason="Remediation DDL complete; transitioning to verification",
        )
        return incident

    def verify_remediation(
        self,
        incident: dict[str, Any],
        *,
        conn: Any = None,
        n_benchmark_runs: int = BENCHMARK_RUNS,
        output_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run before/after benchmark + plan comparison and record outcome."""
        if incident["status"] != STATE_VERIFYING:
            raise InvalidStateTransitionError(
                f"Cannot verify remediation when incident status is '{incident['status']}' "
                f"(expected '{STATE_VERIFYING}')."
            )

        mode = incident["execution_mode"]
        is_sim = mode == EXECUTION_MODE_SIMULATED
        before_metrics = incident.get("pre_remediation_metrics")
        if not before_metrics:
            raise RuntimeError("Pre-remediation baseline metrics missing from incident record.")

        if is_sim:
            after_metrics = capture_post_remediation_metrics(
                customer_id=incident["customer_id"],
                n_runs=n_benchmark_runs,
                simulated=True,
            )
            idx_verify = verify_index_created(simulated=True)
        else:
            with get_lab_connection(conn, read_only=True) as db_conn:
                after_metrics = capture_post_remediation_metrics(
                    db_conn,
                    customer_id=incident["customer_id"],
                    n_runs=n_benchmark_runs,
                    simulated=False,
                )
                idx_verify = verify_index_created(db_conn, simulated=False)
                after_metrics.pop("raw_plan", None)

        comparison = verify_and_save_comparison(
            incident_id=incident["incident_id"],
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            index_verification=idx_verify,
            execution_mode=mode,
            output_path=output_path or self.comparison_output_path,
        )

        incident["post_remediation_metrics"] = after_metrics
        incident["verification_report"] = comparison
        self.save_incident(incident)

        outcome = comparison["verification_outcome"]
        self.audit_logger.log_event(
            incident_id=incident["incident_id"],
            event_type="VERIFICATION_COMPLETED",
            actor="dbmedic_remediation_verifier",
            execution_mode=mode,
            payload={
                "success": outcome["success"],
                "verdict": outcome["verdict"],
                "before_median_ms": before_metrics["median_execution_time_ms"],
                "after_median_ms": after_metrics["median_execution_time_ms"],
                "improvement_pct": comparison["performance_comparison"]["improvement_pct"],
                "before_scan": before_metrics["scan_type"],
                "after_scan": after_metrics["scan_type"],
                "output_path": comparison.get("output_path"),
            },
        )
        return incident

    def close_incident(self, incident: dict[str, Any]) -> dict[str, Any]:
        """Close incident cleanly into RESOLVED or NEEDS_INVESTIGATION."""
        outcome = (incident.get("verification_report") or {}).get("verification_outcome") or {}
        if incident["status"] == STATE_DIAGNOSED:
            val = incident.get("diagnosis_validation") or {}
            if val.get("valid") and not val.get("requires_remediation"):
                summary = {
                    "closed_at": datetime.now(timezone.utc).isoformat(),
                    "final_status": STATE_RESOLVED,
                    "reason": "No database modification required.",
                }
                incident["closure_summary"] = summary
                self.transition_state(incident, STATE_RESOLVED, reason=summary["reason"])
                self.audit_logger.log_event(
                    incident_id=incident["incident_id"],
                    event_type="INCIDENT_CLOSED",
                    actor="dbmedic_orchestrator",
                    execution_mode=incident["execution_mode"],
                    payload=summary,
                )
                return incident

        if incident["status"] == STATE_REJECTED:
            summary = {
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "final_status": STATE_RESOLVED,
                "reason": "Closed safely without remediation after operator rejection.",
            }
            incident["closure_summary"] = summary
            self.transition_state(incident, STATE_RESOLVED, reason=summary["reason"])
            self.audit_logger.log_event(
                incident_id=incident["incident_id"],
                event_type="INCIDENT_CLOSED",
                actor="dbmedic_orchestrator",
                execution_mode=incident["execution_mode"],
                payload=summary,
            )
            return incident

        if outcome.get("success") or outcome.get("planner_choice_legitimate"):
            summary = {
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "final_status": STATE_RESOLVED,
                "verdict": outcome.get("verdict"),
                "reason": outcome.get("reason"),
            }
            incident["closure_summary"] = summary
            self.transition_state(incident, STATE_RESOLVED, reason=str(summary["reason"]))
            self.audit_logger.log_event(
                incident_id=incident["incident_id"],
                event_type="INCIDENT_CLOSED",
                actor="dbmedic_orchestrator",
                execution_mode=incident["execution_mode"],
                payload=summary,
            )
            return incident

        summary = {
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "final_status": STATE_NEEDS_INVESTIGATION,
            "verdict": outcome.get("verdict"),
            "reason": outcome.get("reason") or "Remediation verification did not confirm improvement.",
        }
        incident["closure_summary"] = summary
        self.transition_state(incident, STATE_NEEDS_INVESTIGATION, reason=str(summary["reason"]))
        self.audit_logger.log_event(
            incident_id=incident["incident_id"],
            event_type="INCIDENT_CLOSED",
            actor="dbmedic_orchestrator",
            execution_mode=incident["execution_mode"],
            payload=summary,
        )
        return incident
