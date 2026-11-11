"""
agent/core/remediation_manager.py
=================================
Gated PostgreSQL remediation wrapper over the Phase 1 ``apply_remediation`` executor.

Enforces defense-in-depth before any database modification occurs:
  1. Strict remediation allowlist (initially ONLY ``create_idx_orders_customer_id``)
  2. Identifier & SQL sanity validation (no DROP/TRUNCATE/arbitrary SQL)
  3. Cryptographically bound human approval verification via :class:`ApprovalManager`
  4. Live target database identity verification immediately prior to execution
  5. Idempotent pre-existence check for ``idx_orders_customer_id``
  6. Single execution attempt (no automatic retries on mutation failure)
  7. Post-creation index definition lookup & complete audit logging
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

from agent.core.approval_manager import (
    ApprovalManager,
    canonicalize_sql,
)
from agent.core.audit_logger import AuditLogger, redact_secrets
from agent.core.environment_check import parse_database_identity
from agent.tools.index_monitor import SAFE_IDENTIFIER_REGEX, get_table_indexes
from agent.tools.query_analyzer import (
    APPROVED_REMEDIATIONS,
    apply_remediation,
    index_exists,
)

logger = logging.getLogger(__name__)

# Predefined allowlisted remediation actions (no arbitrary SQL permitted)
ALLOWED_REMEDIATION_SPECS: dict[str, dict[str, Any]] = {
    "create_idx_orders_customer_id": {
        "remediation_key": "create_idx_orders_customer_id",
        "table_name": "orders",
        "column_name": "customer_id",
        "index_name": "idx_orders_customer_id",
        "proposed_sql": "CREATE INDEX idx_orders_customer_id ON orders(customer_id)",
        "phase1_ddl": APPROVED_REMEDIATIONS["create_idx_orders_customer_id"],
        "description": "Create B-tree index idx_orders_customer_id on orders(customer_id)",
    }
}

_FORBIDDEN_DDL_RE = re.compile(
    r"\b(DROP|TRUNCATE|DELETE|ALTER|UPDATE|INSERT|GRANT|REVOKE|COPY|EXECUTE|VACUUM\s+FULL)\b",
    re.IGNORECASE,
)


class DisallowedRemediationError(PermissionError):
    """Raised when a remediation action or SQL violates the Phase 6 allowlist."""


class RemediationExecutionError(RuntimeError):
    """Raised when an approved DDL execution fails (never auto-retried)."""


def get_allowed_remediation_spec(remediation_key: str) -> dict[str, Any]:
    """Validate remediation_key and identifiers against the immutable allowlist."""
    if not isinstance(remediation_key, str) or remediation_key not in ALLOWED_REMEDIATION_SPECS:
        raise DisallowedRemediationError(
            f"Remediation key '{remediation_key}' is not permitted. "
            f"Allowed keys: {list(ALLOWED_REMEDIATION_SPECS.keys())}"
        )
    spec = dict(ALLOWED_REMEDIATION_SPECS[remediation_key])
    for ident_field in ("table_name", "column_name", "index_name"):
        ident = spec[ident_field]
        if not SAFE_IDENTIFIER_REGEX.match(ident):
            raise DisallowedRemediationError(f"Unsafe SQL identifier in spec: {ident}")
    return spec


def validate_remediation_sql_allowlist(
    *,
    remediation_key: str,
    proposed_sql: str,
) -> dict[str, Any]:
    """Ensure proposed_sql strictly matches the allowlisted DDL for remediation_key."""
    if not isinstance(proposed_sql, str) or not proposed_sql.strip():
        raise DisallowedRemediationError("proposed_sql must be a non-empty SQL string")

    if _FORBIDDEN_DDL_RE.search(proposed_sql):
        raise DisallowedRemediationError(
            "Forbidden SQL keyword detected (DROP/TRUNCATE/ALTER/DML is strictly blocked)."
        )

    # Reject chained multi-statement payloads
    stripped_no_trailing = proposed_sql.strip().rstrip(";").strip()
    if ";" in stripped_no_trailing:
        raise DisallowedRemediationError(
            "Multiple SQL statements separated by ';' are strictly forbidden."
        )

    spec = get_allowed_remediation_spec(remediation_key)
    allowed_canonical = {
        canonicalize_sql(spec["proposed_sql"]).lower(),
        canonicalize_sql(spec["phase1_ddl"]).lower(),
    }
    if canonicalize_sql(proposed_sql).lower() not in allowed_canonical:
        raise DisallowedRemediationError(
            f"SQL statement '{proposed_sql}' does not match allowlisted DDL "
            f"'{spec['proposed_sql']}'."
        )
    return spec


def inspect_live_database_identity(
    conn: Any,
    fallback_url: str | None = None,
) -> dict[str, Any]:
    """Inspect current database identity from an active psycopg connection."""
    fallback = parse_database_identity(fallback_url or os.getenv("LAB_DATABASE_URL", ""))
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT current_database(), inet_server_addr()::text, inet_server_port();"
            )
            row = cur.fetchone()
        if row:
            return {
                "database": str(row[0]) if row[0] else fallback.get("database"),
                "host": str(row[1]) if row[1] else fallback.get("host"),
                "port": int(row[2]) if row[2] else (fallback.get("port") or 5432),
                "redacted_url": fallback.get("redacted_url"),
            }
    except Exception:
        pass
    return fallback


def _fetch_index_definition(conn: Any, table_name: str, index_name: str) -> str | None:
    """Look up the CREATE INDEX definition via Phase 2 ``get_table_indexes``."""
    try:
        indexes = get_table_indexes(table_name, conn=conn)
        for idx in indexes:
            if idx.get("index_name") == index_name:
                return str(idx.get("index_definition") or "")
    except Exception as exc:
        logger.debug("Index definition lookup error: %s", exc)
    return None


class RemediationManager:
    """Safe, allowlisted remediation executor backed by Phase 1 DDL & human approvals."""

    def __init__(
        self,
        *,
        approval_manager: ApprovalManager | None = None,
        audit_logger: AuditLogger | None = None,
        lab_database_url: str | None = None,
    ) -> None:
        self.audit_logger = audit_logger or AuditLogger()
        self.approval_manager = approval_manager or ApprovalManager(
            audit_logger=self.audit_logger
        )
        self.lab_database_url = lab_database_url

    def execute_remediation(
        self,
        *,
        incident_id: str,
        action_id: str,
        remediation_key: str,
        proposed_sql: str,
        conn: psycopg.Connection | None = None,
        approval_payload: dict[str, Any] | None = None,
        execution_mode: str = "deterministic",
        now: datetime | None = None,
        current_database_identity: dict[str, Any] | None = None,
        simulated: bool = False,
    ) -> dict[str, Any]:
        """
        Execute an allowlisted PostgreSQL remediation after strict approval & DB checks.

        Does NOT automatically retry on DDL errors.
        """
        # 1. Enforce strict allowlist & forbid arbitrary SQL / DROP / TRUNCATE
        spec = validate_remediation_sql_allowlist(
            remediation_key=remediation_key,
            proposed_sql=proposed_sql,
        )
        index_name = spec["index_name"]
        table_name = spec["table_name"]

        # 2. If simulated mode (explicitly requested offline without DB), enforce FULL
        #    approval verification first, then return clearly tagged simulation result.
        if simulated:
            target_id = current_database_identity or parse_database_identity(
                self.lab_database_url or os.getenv("LAB_DATABASE_URL", "")
            )
            auth = self.approval_manager.verify_approval_authorization(
                incident_id=incident_id,
                action_id=action_id,
                proposed_sql=proposed_sql,
                current_database_identity=target_id,
                approval_payload=approval_payload,
                now=now,
            )
            result = {
                "incident_id": incident_id,
                "action_id": action_id,
                "remediation_key": remediation_key,
                "ddl_applied": spec["phase1_ddl"],
                "index_name": index_name,
                "index_exists": True,
                "index_already_existed": False,
                "index_definition": f"CREATE INDEX {index_name} ON public.{table_name} USING btree ({spec['column_name']})",
                "elapsed_ms": 0.0,
                "success": True,
                "simulated": True,
                "execution_mode": "simulated",
                "approved_by": auth["approved_by"],
                "target_database": auth["target_database"],
                "executed_at": datetime.now(timezone.utc).isoformat(),
            }
            self.approval_manager.mark_executed(
                incident_id=incident_id,
                action_id=action_id,
                execution_result=result,
            )
            self.audit_logger.log_event(
                incident_id=incident_id,
                event_type="REMEDIATION_EXECUTED",
                actor=auth["approved_by"],
                execution_mode="simulated",
                payload=result,
            )
            return result

        # 3. Real PostgreSQL connection path (writable transaction required for DDL)
        owns_conn = conn is None
        lab_url = self.lab_database_url or os.getenv("LAB_DATABASE_URL", "")
        if owns_conn:
            if not lab_url:
                raise RemediationExecutionError(
                    "LAB_DATABASE_URL is not configured for live remediation."
                )
            conn = psycopg.connect(lab_url, connect_timeout=5)

        assert conn is not None
        try:
            live_db_id = current_database_identity or inspect_live_database_identity(
                conn, fallback_url=lab_url
            )

            # Revalidate authorization & database identity immediately before DDL
            try:
                auth = self.approval_manager.verify_approval_authorization(
                    incident_id=incident_id,
                    action_id=action_id,
                    proposed_sql=proposed_sql,
                    current_database_identity=live_db_id,
                    approval_payload=approval_payload,
                    now=now,
                )
            except Exception as auth_exc:
                self.audit_logger.log_event(
                    incident_id=incident_id,
                    event_type="REMEDIATION_BLOCKED",
                    actor="dbmedic_remediation_manager",
                    execution_mode=execution_mode,
                    payload={
                        "action_id": action_id,
                        "remediation_key": remediation_key,
                        "reason": redact_secrets(str(auth_exc)),
                        "target_database": live_db_id,
                    },
                )
                raise

            # Check whether index already exists (idempotent safety check)
            already_present = bool(index_exists(conn, index_name))
            if already_present:
                existing_def = _fetch_index_definition(conn, table_name, index_name)
                result = {
                    "incident_id": incident_id,
                    "action_id": action_id,
                    "remediation_key": remediation_key,
                    "ddl_applied": None,
                    "index_name": index_name,
                    "index_exists": True,
                    "index_already_existed": True,
                    "index_definition": existing_def,
                    "elapsed_ms": 0.0,
                    "success": True,
                    "simulated": False,
                    "execution_mode": execution_mode,
                    "approved_by": auth["approved_by"],
                    "target_database": live_db_id,
                    "executed_at": datetime.now(timezone.utc).isoformat(),
                    "note": (
                        f"Index '{index_name}' already existed on '{table_name}'; "
                        "no duplicate DDL executed."
                    ),
                }
                self.approval_manager.mark_executed(
                    incident_id=incident_id,
                    action_id=action_id,
                    execution_result=result,
                )
                self.audit_logger.log_event(
                    incident_id=incident_id,
                    event_type="REMEDIATION_EXECUTED",
                    actor=auth["approved_by"],
                    execution_mode=execution_mode,
                    payload=result,
                )
                return result

            # Execute Phase 1 apply_remediation ONCE — zero automatic retries
            try:
                phase1_result = apply_remediation(conn, remediation_key, approved=True)
            except Exception as ddl_exc:
                try:
                    conn.rollback()
                except Exception:
                    pass
                self.audit_logger.log_event(
                    incident_id=incident_id,
                    event_type="REMEDIATION_FAILED",
                    actor="dbmedic_remediation_manager",
                    execution_mode=execution_mode,
                    payload={
                        "action_id": action_id,
                        "remediation_key": remediation_key,
                        "error": redact_secrets(str(ddl_exc)),
                    },
                )
                raise RemediationExecutionError(
                    f"Remediation DDL '{remediation_key}' failed (no auto-retry): {redact_secrets(str(ddl_exc))}"
                ) from ddl_exc

            # Ensure planner stats reflect the newly indexed column
            try:
                with conn.cursor() as cur:
                    cur.execute(f"ANALYZE {table_name};")
                conn.commit()
            except Exception as analyze_exc:
                logger.warning("Post-remediation ANALYZE %s failed: %s", table_name, analyze_exc)

            verified_def = _fetch_index_definition(conn, table_name, index_name)
            verified_exists = bool(phase1_result.get("index_exists", False)) or bool(
                index_exists(conn, index_name)
            )
            result = {
                "incident_id": incident_id,
                "action_id": action_id,
                "remediation_key": remediation_key,
                "ddl_applied": phase1_result.get("ddl_applied"),
                "index_name": index_name,
                "index_exists": verified_exists,
                "index_already_existed": False,
                "index_definition": verified_def,
                "elapsed_ms": phase1_result.get("elapsed_ms", 0.0),
                "success": bool(phase1_result.get("success", False) and verified_exists),
                "simulated": False,
                "execution_mode": execution_mode,
                "approved_by": auth["approved_by"],
                "target_database": live_db_id,
                "executed_at": datetime.now(timezone.utc).isoformat(),
            }

            if result["success"]:
                self.approval_manager.mark_executed(
                    incident_id=incident_id,
                    action_id=action_id,
                    execution_result=result,
                )
                self.audit_logger.log_event(
                    incident_id=incident_id,
                    event_type="REMEDIATION_EXECUTED",
                    actor=auth["approved_by"],
                    execution_mode=execution_mode,
                    payload=result,
                )
            else:
                self.audit_logger.log_event(
                    incident_id=incident_id,
                    event_type="REMEDIATION_FAILED",
                    actor="dbmedic_remediation_manager",
                    execution_mode=execution_mode,
                    payload=result,
                )
            return result
        finally:
            if owns_conn:
                try:
                    conn.close()
                except Exception:
                    pass

