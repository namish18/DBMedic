"""
agent/core/audit_logger.py
==========================
Append-only audit trail with SHA-256 hash chaining and strict credential redaction.

Records the complete DBMedic incident lifecycle:
  - Incident creation & state transitions (including rejected transitions)
  - Diagnostic execution, model identity, and execution mode
  - Tool selection and tool execution summaries
  - Diagnostic evidence & recommendations
  - Approval requests, approvals, and rejections
  - Remediation execution and verification results
  - Final incident closure

Honest security note on hash chaining:
  Each event computes a SHA-256 digest over its canonical JSON body (including
  ``prev_event_hash``) to detect accidental corruption or naive single-line
  modification. This local file hash chain does NOT provide cryptographic
  tamper-proofing against an adversary with filesystem write permissions, who
  could rewrite and re-hash the entire file. Production deployments require
  forwarding events to external append-only WORM storage or signing with an
  external KMS key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_LOG_PATH = PROJECT_ROOT / "reports" / "audit_trail.jsonl"

GENESIS_HASH = "0" * 64

HASH_CHAIN_LIMITATIONS = (
    "Local SHA-256 hash chaining detects accidental corruption or single-line "
    "tampering, but is NOT cryptographically tamper-proof against an actor with "
    "write access to the host filesystem (who can recompute the full chain). "
    "External WORM log shipping or hardware-backed KMS signatures are required "
    "for tamper-proof compliance."
)

AUDIT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "INCIDENT_CREATED",
        "INCIDENT_STATUS_CHANGED",
        "STATE_TRANSITION_REJECTED",
        "DIAGNOSTIC_STARTED",
        "MODEL_IDENTITY_RECORDED",
        "TOOL_SELECTED",
        "TOOL_EXECUTED",
        "DIAGNOSTIC_EVIDENCE",
        "RECOMMENDATION_PROPOSED",
        "APPROVAL_REQUESTED",
        "APPROVAL_GRANTED",
        "APPROVAL_REJECTED",
        "APPROVAL_VALIDATION_FAILED",
        "REMEDIATION_EXECUTED",
        "REMEDIATION_BLOCKED",
        "REMEDIATION_FAILED",
        "VERIFICATION_COMPLETED",
        "INCIDENT_CLOSED",
        "EVALUATION_COMPLETED",
    }
)

_SENSITIVE_KEY_TOKENS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "hf_token",
        "huggingface_hub_token",
        "api_key",
        "apikey",
        "credential",
        "credentials",
        "private_key",
        "authorization",
        "postgres_password",
    }
)

_URL_SECRET_RE = re.compile(
    r"(?P<scheme>postgres(?:ql)?://[^:\s/@]+):(?P<pw>[^@\s/]+)@",
    re.IGNORECASE,
)
_TOKEN_VALUE_RE = re.compile(r"\bhf_[A-Za-z0-9]{10,}\b")


def redact_database_url(url: str) -> str:
    """Redact the password in a PostgreSQL connection URL while preserving host/port/db."""
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc and parsed.password is not None:
            username = parsed.username or ""
            hostname = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            safe_netloc = f"{username}:****@{hostname}{port}" if username else f"{hostname}{port}"
            return urlunparse(parsed._replace(netloc=safe_netloc))
    except Exception:
        pass
    return _URL_SECRET_RE.sub(r"\1:****@", url)


def redact_secrets(obj: Any, _depth: int = 0) -> Any:
    """Recursively redact sensitive keys, connection string passwords, and HF tokens."""
    if _depth > 15:
        return "[MAX_DEPTH_REDACTED]"

    if isinstance(obj, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, val in obj.items():
            key_str = str(raw_key)
            key_lower = key_str.lower()
            if any(tok in key_lower for tok in _SENSITIVE_KEY_TOKENS):
                if isinstance(val, bool) or val is None or val == "":
                    sanitized[key_str] = val
                else:
                    sanitized[key_str] = "***REDACTED***"
            elif key_lower in {"database_url", "lab_database_url", "dsn", "conn_str"} and isinstance(val, str):
                sanitized[key_str] = redact_database_url(val)
            else:
                sanitized[key_str] = redact_secrets(val, _depth + 1)
        return sanitized

    if isinstance(obj, list):
        return [redact_secrets(item, _depth + 1) for item in obj]

    if isinstance(obj, tuple):
        return [redact_secrets(item, _depth + 1) for item in obj]

    if isinstance(obj, str):
        cleaned = _URL_SECRET_RE.sub(r"\1:****@", obj)
        cleaned = _TOKEN_VALUE_RE.sub("hf_***REDACTED***", cleaned)
        return cleaned

    return obj


def _canonical_event_bytes(event_without_hash: dict[str, Any]) -> bytes:
    """Serialize event payload canonically for reproducible SHA-256 hashing."""
    return json.dumps(
        event_without_hash,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=False,
    ).encode("utf-8")


def compute_event_hash(event: dict[str, Any]) -> str:
    """Compute SHA-256 hex digest of an event dict (excluding ``event_hash``)."""
    body = {k: v for k, v in event.items() if k != "event_hash"}
    return hashlib.sha256(_canonical_event_bytes(body)).hexdigest()


class AuditLogger:
    """Append-only JSONL audit logger with SHA-256 chain integrity."""

    def __init__(self, log_path: Path | str | None = None) -> None:
        self.log_path = Path(log_path) if log_path is not None else DEFAULT_AUDIT_LOG_PATH
        self._lock = threading.Lock()

    def _read_tail_state(self) -> tuple[int, str]:
        """Return (next_sequence, prev_event_hash) from the current audit log."""
        if not self.log_path.exists():
            return 1, GENESIS_HASH

        last_event: dict[str, Any] | None = None
        last_index = 0
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                last_index += 1
                last_event = json.loads(stripped)

        if last_event is None:
            return 1, GENESIS_HASH

        seq = int(last_event.get("sequence", last_index)) + 1
        prev_hash = str(last_event.get("event_hash") or compute_event_hash(last_event))
        return seq, prev_hash

    def log_event(
        self,
        incident_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        actor: str = "dbmedic_system",
        execution_mode: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """
        Append one sanitized, SHA-256-chained audit event to the JSONL log.

        Parameters
        ----------
        incident_id:
            Incident identifier associated with this event (e.g. ``slow_query_001``).
        event_type:
            Event classification label.
        payload:
            Structured event details. Credentials/tokens are automatically redacted.
        actor:
            Originating component or operator identifier.
        execution_mode:
            ``live_tiny_aya``, ``deterministic``, ``simulated``, or ``system``.
        timestamp:
            Optional explicit ISO-8601 UTC timestamp (defaults to now).
        """
        if not isinstance(incident_id, str) or not incident_id.strip():
            raise ValueError("incident_id must be a non-empty string")
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")

        sanitized_payload = redact_secrets(payload or {})
        event_ts = timestamp or datetime.now(timezone.utc).isoformat()

        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            sequence, prev_hash = self._read_tail_state()

            event: dict[str, Any] = {
                "event_id": str(uuid.uuid4()),
                "sequence": sequence,
                "timestamp": event_ts,
                "incident_id": incident_id.strip(),
                "event_type": event_type.strip(),
                "actor": str(actor),
                "execution_mode": execution_mode or "system",
                "payload": sanitized_payload,
                "prev_event_hash": prev_hash,
                "hash_algorithm": "sha256",
            }
            event["event_hash"] = compute_event_hash(event)

            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, sort_keys=True, default=str, ensure_ascii=False) + "\n")

        return event

    def read_events(
        self,
        *,
        incident_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read audit events, optionally filtered by ``incident_id`` and/or ``event_type``."""
        if not self.log_path.exists():
            return []

        events: list[dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if incident_id is not None and record.get("incident_id") != incident_id:
                    continue
                if event_type is not None and record.get("event_type") != event_type:
                    continue
                events.append(record)
        return events

    def verify_integrity(self) -> dict[str, Any]:
        """
        Verify SHA-256 hash integrity and chain continuity across all log entries.

        Returns a structured integrity report explicitly documenting the security
        boundaries of local file-based hash chaining.
        """
        if not self.log_path.exists():
            return {
                "valid": True,
                "log_exists": False,
                "log_path": str(self.log_path),
                "total_events": 0,
                "broken_links": [],
                "hash_mismatches": [],
                "parse_errors": [],
                "algorithm": "sha256_chain",
                "limitations": HASH_CHAIN_LIMITATIONS,
            }

        total = 0
        broken_links: list[dict[str, Any]] = []
        hash_mismatches: list[dict[str, Any]] = []
        parse_errors: list[dict[str, Any]] = []
        expected_prev = GENESIS_HASH

        with self.log_path.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    parse_errors.append({"line": line_num, "error": str(exc)})
                    continue

                total += 1
                stored_prev = event.get("prev_event_hash")
                if stored_prev != expected_prev:
                    broken_links.append(
                        {
                            "line": line_num,
                            "event_id": event.get("event_id"),
                            "expected_prev_hash": expected_prev,
                            "actual_prev_hash": stored_prev,
                        }
                    )

                computed = compute_event_hash(event)
                stored_hash = event.get("event_hash")
                if computed != stored_hash:
                    hash_mismatches.append(
                        {
                            "line": line_num,
                            "event_id": event.get("event_id"),
                            "expected_event_hash": computed,
                            "actual_event_hash": stored_hash,
                        }
                    )

                expected_prev = str(stored_hash or computed)

        is_valid = not (broken_links or hash_mismatches or parse_errors)
        return {
            "valid": is_valid,
            "log_exists": True,
            "log_path": str(self.log_path),
            "total_events": total,
            "broken_links": broken_links,
            "hash_mismatches": hash_mismatches,
            "parse_errors": parse_errors,
            "algorithm": "sha256_chain",
            "limitations": HASH_CHAIN_LIMITATIONS,
        }


def log_audit_event(
    incident_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    actor: str = "dbmedic_system",
    execution_mode: str | None = None,
    log_path: Path | str | None = None,
) -> dict[str, Any]:
    """Convenience function wrapping :meth:`AuditLogger.log_event`."""
    return AuditLogger(log_path=log_path).log_event(
        incident_id=incident_id,
        event_type=event_type,
        payload=payload,
        actor=actor,
        execution_mode=execution_mode,
    )


def verify_audit_trail(log_path: Path | str | None = None) -> dict[str, Any]:
    """Convenience function wrapping :meth:`AuditLogger.verify_integrity`."""
    return AuditLogger(log_path=log_path).verify_integrity()


def read_audit_events(
    *,
    incident_id: str | None = None,
    event_type: str | None = None,
    log_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Convenience function wrapping :meth:`AuditLogger.read_events`."""
    return AuditLogger(log_path=log_path).read_events(
        incident_id=incident_id,
        event_type=event_type,
    )
