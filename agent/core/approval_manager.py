"""
agent/core/approval_manager.py
==============================
Human approval workflow and cryptographic binding for DBMedic remediations.

Security guarantees enforced:
  - The AI model can NEVER approve its own action.
  - A model-supplied ``human_approved: true`` boolean is never treated as authorization.
  - Every approval is bound to an exact ``incident_id``, ``action_id``,
    ``remediation_key``, canonical SQL SHA-256 digest, and target database identity.
  - Any modification to the proposed SQL after request/approval invalidates authorization.
  - Expired approval requests cannot be approved or executed.
  - Rejected requests cannot be executed.
  - Duplicate executions or duplicate approvals for the same action are blocked.
  - Target database identity is revalidated immediately prior to execution.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agent.core.audit_logger import AuditLogger, redact_secrets
from agent.core.environment_check import parse_database_identity

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_APPROVAL_STORE_PATH = PROJECT_ROOT / "reports" / "approvals.json"
DEFAULT_APPROVAL_TTL_SECONDS = 1800  # 30 minutes

APPROVAL_STATUS_PENDING = "PENDING"
APPROVAL_STATUS_APPROVED = "APPROVED"
APPROVAL_STATUS_REJECTED = "REJECTED"
APPROVAL_STATUS_EXPIRED = "EXPIRED"
APPROVAL_STATUS_EXECUTED = "EXECUTED"
APPROVAL_STATUS_INVALIDATED = "INVALIDATED"

_FORBIDDEN_MODEL_ACTORS = frozenset(
    {
        "model",
        "agent",
        "llm",
        "ai",
        "tiny_aya",
        "tiny-aya",
        "tiny_aya_global",
        "coherelabs/tiny-aya-global",
        "autonomous",
        "model_output",
        "auto",
        "self",
    }
)

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "127.0.0.1/32", "::1/128"})


class ApprovalError(PermissionError):
    """Base security error raised when remediation authorization fails."""


class ApprovalValidationError(ApprovalError):
    """Raised when an approval request has missing or malformed required fields."""


class ModelSelfApprovalError(ApprovalError):
    """Raised when an AI model or automated non-human source attempts approval."""


class ApprovalExpiredError(ApprovalError):
    """Raised when an approval request has passed its expiration timestamp."""


class SQLTamperingError(ApprovalError):
    """Raised when SQL differs from the SHA-256 bound to the approval request."""


class DuplicateApprovalError(ApprovalError):
    """Raised on duplicate approval or repeated execution attempts."""


class DatabaseIdentityMismatchError(ApprovalError):
    """Raised when target database identity does not match the approved database."""


def canonicalize_sql(sql: str) -> str:
    """Normalize SQL whitespace and trailing semicolon consistently for hashing."""
    if not isinstance(sql, str):
        raise ApprovalValidationError("proposed_sql must be a string")
    collapsed = re.sub(r"\s+", " ", sql.strip())
    return collapsed.rstrip(";").strip()


def compute_sql_sha256(sql: str) -> str:
    """Compute SHA-256 digest of canonicalized SQL statement."""
    return hashlib.sha256(canonicalize_sql(sql).encode("utf-8")).hexdigest()


def _normalize_db_identity(target_database: dict[str, Any] | str) -> dict[str, Any]:
    """Normalize target database metadata cleanly with zero credentials."""
    if isinstance(target_database, str):
        if "://" in target_database:
            parsed = parse_database_identity(target_database)
            return {
                "database": parsed.get("database"),
                "host": parsed.get("host"),
                "port": parsed.get("port") or 5432,
                "redacted_url": parsed.get("redacted_url"),
            }
        return {
            "database": target_database.strip(),
            "host": "localhost",
            "port": 5432,
            "redacted_url": None,
        }

    if not isinstance(target_database, dict):
        raise ApprovalValidationError("target_database must be a dict or database identifier")

    db_name = (
        target_database.get("live_current_database")
        or target_database.get("database")
        or target_database.get("dbname")
    )
    host = target_database.get("host") or target_database.get("server_addr")
    port = target_database.get("live_server_port") or target_database.get("port") or 5432
    return {
        "database": str(db_name).strip() if db_name else None,
        "host": str(host).strip() if host else None,
        "port": int(port) if port else 5432,
        "redacted_url": target_database.get("redacted_url"),
    }


def _hosts_compatible(expected_host: str | None, actual_host: str | None) -> bool:
    if not expected_host or not actual_host:
        return True
    eh = expected_host.strip().lower().split("/")[0]
    ah = actual_host.strip().lower().split("/")[0]
    if eh == ah:
        return True
    return eh in _LOOPBACK_HOSTS and ah in _LOOPBACK_HOSTS


def database_identities_match(
    expected_db: dict[str, Any] | str,
    actual_db: dict[str, Any] | str,
) -> tuple[bool, str]:
    """Compare expected approved target DB against live current DB identity."""
    exp = _normalize_db_identity(expected_db)
    act = _normalize_db_identity(actual_db)

    if not exp.get("database") or not act.get("database"):
        return False, f"Incomplete database identity: expected={exp.get('database')}, actual={act.get('database')}"

    if exp["database"] != act["database"]:
        return False, (
            f"Database name mismatch: approved for '{exp['database']}', "
            f"connected to '{act['database']}'."
        )

    if exp.get("port") and act.get("port") and int(exp["port"]) != int(act["port"]):
        return False, (
            f"Database port mismatch: approved for port {exp['port']}, "
            f"connected to port {act['port']}."
        )

    if not _hosts_compatible(exp.get("host"), act.get("host")):
        return False, (
            f"Database host mismatch: approved for '{exp.get('host')}', "
            f"connected to '{act.get('host')}'."
        )

    return True, "Database identity matches."


def _compute_record_binding_digest(record: dict[str, Any]) -> str:
    """Compute integrity binding digest tying approval to incident, action, SQL, DB & human."""
    target_db = _normalize_db_identity(record.get("target_database", {}))
    canonical = json.dumps(
        {
            "request_id": record.get("request_id"),
            "incident_id": record.get("incident_id"),
            "action_id": record.get("action_id"),
            "remediation_key": record.get("remediation_key"),
            "sql_sha256": record.get("sql_sha256"),
            "database": target_db.get("database"),
            "port": target_db.get("port"),
            "approved_by": record.get("approved_by"),
            "approved_at": record.get("approved_at"),
            "confirmation_method": record.get("confirmation_method"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _assert_human_actor(actor: str, confirmation_method: str | None) -> str:
    if not isinstance(actor, str) or not actor.strip():
        raise ModelSelfApprovalError("Human operator identifier is required for approval.")
    cleaned = actor.strip()
    lowered = cleaned.lower()
    if (
        lowered in _FORBIDDEN_MODEL_ACTORS
        or "tiny_aya" in lowered
        or "tiny-aya" in lowered
        or "model_output" in lowered
        or lowered.startswith("agent_")
    ):
        raise ModelSelfApprovalError(
            f"Automated/model actor '{actor}' is forbidden from approving database remediations."
        )

    valid_methods = {"interactive_cli", "explicit_human_operator", "dashboard_human_confirm"}
    if not confirmation_method or confirmation_method not in valid_methods:
        raise ModelSelfApprovalError(
            f"Invalid confirmation_method '{confirmation_method}'. Explicit human operator confirmation is required."
        )
    return cleaned


class ApprovalManager:
    """Persistent manager for DBMedic human remediation approvals."""

    def __init__(
        self,
        store_path: Path | str | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.store_path = (
            Path(store_path) if store_path is not None else DEFAULT_APPROVAL_STORE_PATH
        )
        self.audit_logger = audit_logger or AuditLogger()

    def _load_store(self) -> dict[str, dict[str, Any]]:
        if not self.store_path.exists():
            return {}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("requests"), dict):
                return data["requests"]
        except (json.JSONDecodeError, OSError):
            return {}
        return {}

    def _save_store(self, requests: dict[str, dict[str, Any]]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "dbmedic.approvals/1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "requests": redact_secrets(requests),
        }
        self.store_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _key(incident_id: str, action_id: str) -> str:
        return f"{incident_id}::{action_id}"

    def create_approval_request(
        self,
        *,
        incident_id: str,
        action_id: str,
        target_database: dict[str, Any] | str,
        proposed_sql: str,
        action_description: str,
        evidence: dict[str, Any] | list[str],
        remediation_key: str = "create_idx_orders_customer_id",
        ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
        now: datetime | None = None,
        execution_mode: str = "system",
    ) -> dict[str, Any]:
        """Create and persist a pending human approval request."""
        if not incident_id or not isinstance(incident_id, str):
            raise ApprovalValidationError("incident_id must be a non-empty string")
        if not action_id or not isinstance(action_id, str):
            raise ApprovalValidationError("action_id must be a non-empty string")
        if not proposed_sql or not isinstance(proposed_sql, str):
            raise ApprovalValidationError("proposed_sql must be a non-empty string")
        if not action_description or not isinstance(action_description, str):
            raise ApprovalValidationError("action_description must be a non-empty string")
        if not evidence:
            raise ApprovalValidationError("Supporting diagnostic evidence is required")
        if ttl_seconds <= 0:
            raise ApprovalValidationError("ttl_seconds must be positive")

        db_identity = _normalize_db_identity(target_database)
        if not db_identity.get("database"):
            raise ApprovalValidationError("target_database must include database name")

        current_time = now or datetime.now(timezone.utc)
        expires_at = current_time + timedelta(seconds=ttl_seconds)

        request: dict[str, Any] = {
            "request_id": str(uuid.uuid4()),
            "incident_id": incident_id.strip(),
            "action_id": action_id.strip(),
            "remediation_key": remediation_key.strip(),
            "target_database": db_identity,
            "proposed_sql": proposed_sql.strip(),
            "canonical_sql": canonicalize_sql(proposed_sql),
            "sql_sha256": compute_sql_sha256(proposed_sql),
            "action_description": action_description.strip(),
            "evidence": redact_secrets(evidence),
            "timestamp": current_time.isoformat(),
            "created_at": current_time.isoformat(),
            "expiration": expires_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "approval_status": APPROVAL_STATUS_PENDING,
            "approved_by": None,
            "approved_at": None,
            "confirmation_method": None,
            "rejected_by": None,
            "rejected_at": None,
            "rejection_reason": None,
            "executed_at": None,
            "approval_digest": None,
            "execution_mode": execution_mode,
        }

        requests = self._load_store()
        key = self._key(request["incident_id"], request["action_id"])
        requests[key] = request
        self._save_store(requests)

        self.audit_logger.log_event(
            incident_id=request["incident_id"],
            event_type="APPROVAL_REQUESTED",
            actor="dbmedic_approval_manager",
            execution_mode=execution_mode,
            payload={
                "request_id": request["request_id"],
                "action_id": request["action_id"],
                "remediation_key": request["remediation_key"],
                "target_database": request["target_database"],
                "proposed_sql": request["proposed_sql"],
                "sql_sha256": request["sql_sha256"],
                "expires_at": request["expires_at"],
            },
        )
        return request

    def get_request(
        self,
        incident_id: str,
        action_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve stored approval request for incident (latest if action_id omitted)."""
        requests = self._load_store()
        if action_id is not None:
            return requests.get(self._key(incident_id, action_id))

        matching = [
            req
            for req in requests.values()
            if req.get("incident_id") == incident_id
        ]
        if not matching:
            return None
        matching.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
        return matching[0]

    def validate_approval_request(
        self,
        request: dict[str, Any],
        *,
        expected_sql: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Validate structure, SQL integrity hash, and expiration of an approval request."""
        if not isinstance(request, dict):
            raise ApprovalValidationError("Approval request must be a dictionary")

        required_fields = (
            "request_id",
            "incident_id",
            "action_id",
            "target_database",
            "proposed_sql",
            "sql_sha256",
            "action_description",
            "evidence",
            "timestamp",
            "expiration",
            "approval_status",
        )
        missing = [f for f in required_fields if not request.get(f)]
        if missing:
            raise ApprovalValidationError(f"Approval request missing required fields: {missing}")

        actual_sql_hash = compute_sql_sha256(str(request["proposed_sql"]))
        if actual_sql_hash != request["sql_sha256"]:
            self._update_status(
                request["incident_id"],
                request["action_id"],
                APPROVAL_STATUS_INVALIDATED,
            )
            raise SQLTamperingError(
                "Approval request SQL digest mismatch — proposed_sql was altered."
            )

        if expected_sql is not None:
            expected_hash = compute_sql_sha256(expected_sql)
            if expected_hash != request["sql_sha256"]:
                self._update_status(
                    request["incident_id"],
                    request["action_id"],
                    APPROVAL_STATUS_INVALIDATED,
                )
                raise SQLTamperingError(
                    "Proposed SQL does not match the SQL bound to the approval request."
                )

        current_time = now or datetime.now(timezone.utc)
        exp_dt = _parse_iso(str(request.get("expires_at") or request["expiration"]))
        if current_time > exp_dt:
            self._update_status(
                request["incident_id"],
                request["action_id"],
                APPROVAL_STATUS_EXPIRED,
            )
            raise ApprovalExpiredError(
                f"Approval request '{request['request_id']}' expired at {exp_dt.isoformat()}."
            )

        return {"valid": True, "request_id": request["request_id"], "status": request["approval_status"]}

    def _update_status(self, incident_id: str, action_id: str, new_status: str) -> None:
        requests = self._load_store()
        key = self._key(incident_id, action_id)
        if key in requests:
            requests[key]["approval_status"] = new_status
            self._save_store(requests)

    def record_human_approval(
        self,
        *,
        incident_id: str,
        action_id: str,
        approved_by: str,
        confirmation_method: str = "interactive_cli",
        expected_sql: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Record an explicit human operator approval for a pending request."""
        try:
            operator = _assert_human_actor(approved_by, confirmation_method)
        except ModelSelfApprovalError as exc:
            self.audit_logger.log_event(
                incident_id=incident_id,
                event_type="APPROVAL_VALIDATION_FAILED",
                actor=str(approved_by),
                payload={"action_id": action_id, "reason": str(exc)},
            )
            raise

        requests = self._load_store()
        key = self._key(incident_id, action_id)
        request = requests.get(key)
        if request is None:
            raise ApprovalValidationError(
                f"No pending approval request found for incident='{incident_id}', action='{action_id}'."
            )

        status = request.get("approval_status")
        if status == APPROVAL_STATUS_EXECUTED:
            raise DuplicateApprovalError(
                f"Action '{action_id}' for incident '{incident_id}' has already been executed."
            )
        if status == APPROVAL_STATUS_APPROVED:
            raise DuplicateApprovalError(
                f"Action '{action_id}' for incident '{incident_id}' was already approved."
            )
        if status == APPROVAL_STATUS_REJECTED:
            raise ApprovalError(
                f"Action '{action_id}' for incident '{incident_id}' was rejected and cannot be approved."
            )
        if status == APPROVAL_STATUS_INVALIDATED:
            raise SQLTamperingError(
                f"Approval request '{request['request_id']}' was invalidated due to SQL tampering."
            )

        self.validate_approval_request(request, expected_sql=expected_sql, now=now)

        approved_at = (now or datetime.now(timezone.utc)).isoformat()
        request["approval_status"] = APPROVAL_STATUS_APPROVED
        request["approved_by"] = operator
        request["approved_at"] = approved_at
        request["confirmation_method"] = confirmation_method
        request["approval_digest"] = _compute_record_binding_digest(request)

        requests[key] = request
        self._save_store(requests)

        self.audit_logger.log_event(
            incident_id=incident_id,
            event_type="APPROVAL_GRANTED",
            actor=operator,
            execution_mode=str(request.get("execution_mode", "system")),
            payload={
                "request_id": request["request_id"],
                "action_id": action_id,
                "approved_by": operator,
                "confirmation_method": confirmation_method,
                "sql_sha256": request["sql_sha256"],
                "approval_digest": request["approval_digest"],
                "target_database": request["target_database"],
            },
        )
        return request

    def record_human_rejection(
        self,
        *,
        incident_id: str,
        action_id: str,
        rejected_by: str,
        reason: str = "Rejected by human operator",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Record explicit human operator rejection for an approval request."""
        if not rejected_by or not isinstance(rejected_by, str) or not rejected_by.strip():
            raise ApprovalValidationError("rejected_by operator identifier is required")

        requests = self._load_store()
        key = self._key(incident_id, action_id)
        request = requests.get(key)
        if request is None:
            raise ApprovalValidationError(
                f"No approval request found for incident='{incident_id}', action='{action_id}'."
            )
        if request.get("approval_status") == APPROVAL_STATUS_EXECUTED:
            raise DuplicateApprovalError(
                f"Action '{action_id}' has already been executed and cannot be rejected."
            )

        rejected_at = (now or datetime.now(timezone.utc)).isoformat()
        request["approval_status"] = APPROVAL_STATUS_REJECTED
        request["rejected_by"] = rejected_by.strip()
        request["rejected_at"] = rejected_at
        request["rejection_reason"] = reason.strip()

        requests[key] = request
        self._save_store(requests)

        self.audit_logger.log_event(
            incident_id=incident_id,
            event_type="APPROVAL_REJECTED",
            actor=rejected_by.strip(),
            execution_mode=str(request.get("execution_mode", "system")),
            payload={
                "request_id": request["request_id"],
                "action_id": action_id,
                "rejected_by": rejected_by.strip(),
                "reason": reason.strip(),
            },
        )
        return request

    def verify_approval_authorization(
        self,
        *,
        incident_id: str,
        action_id: str,
        proposed_sql: str,
        current_database_identity: dict[str, Any] | str,
        approval_payload: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Verify complete authorization prior to executing remediation DDL.

        Enforces:
          - Stored approval exists (model booleans alone are rejected)
          - Human operator identity is valid and non-AI
          - Status is APPROVED (never PENDING, REJECTED, EXPIRED, EXECUTED, INVALIDATED)
          - Not expired
          - SQL SHA-256 and binding digest match intact
          - Target database identity matches current live DB connection
        """
        if approval_payload is not None and isinstance(approval_payload, dict):
            # Explicitly reject raw boolean authorization from model outputs
            if "request_id" not in approval_payload or "approval_digest" not in approval_payload:
                raise ModelSelfApprovalError(
                    "Unbound boolean or forged approval payload cannot authorize remediation."
                )

        requests = self._load_store()
        key = self._key(incident_id, action_id)
        stored = requests.get(key)
        if stored is None:
            raise ApprovalValidationError(
                f"No stored approval authorization exists for incident='{incident_id}', action='{action_id}'."
            )

        status = stored.get("approval_status")
        if status == APPROVAL_STATUS_REJECTED:
            raise ApprovalError(
                f"Remediation blocked: action '{action_id}' was explicitly rejected by operator."
            )
        if status == APPROVAL_STATUS_EXECUTED:
            raise DuplicateApprovalError(
                f"Remediation blocked: action '{action_id}' has already been executed."
            )
        if status == APPROVAL_STATUS_INVALIDATED:
            raise SQLTamperingError(
                f"Remediation blocked: approval '{stored['request_id']}' was invalidated."
            )
        if status == APPROVAL_STATUS_EXPIRED:
            raise ApprovalExpiredError(
                f"Remediation blocked: approval '{stored['request_id']}' is expired."
            )
        if status != APPROVAL_STATUS_APPROVED:
            raise ApprovalError(
                f"Remediation blocked: approval status is '{status}', expected 'APPROVED'."
            )

        operator = _assert_human_actor(
            str(stored.get("approved_by") or ""),
            stored.get("confirmation_method"),
        )

        # Validate structure, expiration, and SQL tampering against current proposed_sql
        self.validate_approval_request(stored, expected_sql=proposed_sql, now=now)

        # Verify integrity binding digest (protects against manual edits to store fields)
        expected_digest = _compute_record_binding_digest(stored)
        if stored.get("approval_digest") != expected_digest:
            self._update_status(incident_id, action_id, APPROVAL_STATUS_INVALIDATED)
            raise SQLTamperingError(
                "Approval binding digest mismatch — approval record was modified after signing."
            )

        # Immediately revalidate target database identity
        db_ok, db_reason = database_identities_match(
            stored["target_database"],
            current_database_identity,
        )
        if not db_ok:
            raise DatabaseIdentityMismatchError(db_reason)

        return {
            "authorized": True,
            "request_id": stored["request_id"],
            "incident_id": incident_id,
            "action_id": action_id,
            "remediation_key": stored.get("remediation_key"),
            "approved_by": operator,
            "approved_at": stored.get("approved_at"),
            "confirmation_method": stored.get("confirmation_method"),
            "sql_sha256": stored.get("sql_sha256"),
            "target_database": stored.get("target_database"),
        }

    def mark_executed(
        self,
        *,
        incident_id: str,
        action_id: str,
        execution_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Transition an APPROVED request to EXECUTED to prevent duplicate runs."""
        requests = self._load_store()
        key = self._key(incident_id, action_id)
        stored = requests.get(key)
        if stored is None:
            raise ApprovalValidationError(f"Approval record not found for {incident_id}::{action_id}")
        if stored.get("approval_status") == APPROVAL_STATUS_EXECUTED:
            raise DuplicateApprovalError(f"Action '{action_id}' has already been executed.")
        if stored.get("approval_status") != APPROVAL_STATUS_APPROVED:
            raise ApprovalError(
                f"Cannot mark non-approved action (status={stored.get('approval_status')}) as executed."
            )
        stored["approval_status"] = APPROVAL_STATUS_EXECUTED
        stored["executed_at"] = datetime.now(timezone.utc).isoformat()
        if execution_result is not None:
            stored["execution_summary"] = redact_secrets(execution_result)
        requests[key] = stored
        self._save_store(requests)
        return stored


# ---------------------------------------------------------------------------
# Module-level public wrappers matching spec function signatures
# ---------------------------------------------------------------------------

def create_approval_request(
    *,
    incident_id: str,
    action_id: str,
    target_database: dict[str, Any] | str,
    proposed_sql: str,
    action_description: str,
    evidence: dict[str, Any] | list[str],
    remediation_key: str = "create_idx_orders_customer_id",
    ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
    now: datetime | None = None,
    store_path: Path | str | None = None,
    audit_logger: AuditLogger | None = None,
) -> dict[str, Any]:
    return ApprovalManager(store_path=store_path, audit_logger=audit_logger).create_approval_request(
        incident_id=incident_id,
        action_id=action_id,
        target_database=target_database,
        proposed_sql=proposed_sql,
        action_description=action_description,
        evidence=evidence,
        remediation_key=remediation_key,
        ttl_seconds=ttl_seconds,
        now=now,
    )


def validate_approval_request(
    request: dict[str, Any],
    *,
    expected_sql: str | None = None,
    now: datetime | None = None,
    store_path: Path | str | None = None,
    audit_logger: AuditLogger | None = None,
) -> dict[str, Any]:
    return ApprovalManager(store_path=store_path, audit_logger=audit_logger).validate_approval_request(
        request,
        expected_sql=expected_sql,
        now=now,
    )


def record_human_approval(
    *,
    incident_id: str,
    action_id: str,
    approved_by: str,
    confirmation_method: str = "interactive_cli",
    expected_sql: str | None = None,
    now: datetime | None = None,
    store_path: Path | str | None = None,
    audit_logger: AuditLogger | None = None,
) -> dict[str, Any]:
    return ApprovalManager(store_path=store_path, audit_logger=audit_logger).record_human_approval(
        incident_id=incident_id,
        action_id=action_id,
        approved_by=approved_by,
        confirmation_method=confirmation_method,
        expected_sql=expected_sql,
        now=now,
    )


def record_human_rejection(
    *,
    incident_id: str,
    action_id: str,
    rejected_by: str,
    reason: str = "Rejected by human operator",
    now: datetime | None = None,
    store_path: Path | str | None = None,
    audit_logger: AuditLogger | None = None,
) -> dict[str, Any]:
    return ApprovalManager(store_path=store_path, audit_logger=audit_logger).record_human_rejection(
        incident_id=incident_id,
        action_id=action_id,
        rejected_by=rejected_by,
        reason=reason,
        now=now,
    )


def verify_approval_authorization(
    *,
    incident_id: str,
    action_id: str,
    proposed_sql: str,
    current_database_identity: dict[str, Any] | str,
    approval_payload: dict[str, Any] | None = None,
    now: datetime | None = None,
    store_path: Path | str | None = None,
    audit_logger: AuditLogger | None = None,
) -> dict[str, Any]:
    return ApprovalManager(store_path=store_path, audit_logger=audit_logger).verify_approval_authorization(
        incident_id=incident_id,
        action_id=action_id,
        proposed_sql=proposed_sql,
        current_database_identity=current_database_identity,
        approval_payload=approval_payload,
        now=now,
    )

