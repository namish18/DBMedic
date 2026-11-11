"""
demo/scripts/reset_demo_database.py
===================================
Reset ``dbmedic_lab`` back to the controlled slow-query incident baseline state
prior to running a reproducible DBMedic Phase 6 live demonstration.

Actions performed (only on dbmedic_lab):
  1. Verifies connectivity and expected schema tables.
  2. Drops ``idx_orders_customer_id`` if present (restoring Seq Scan incident).
  3. Runs ``ANALYZE orders`` to refresh planner statistics.
  4. Confirms EXPLAIN plans return Sequential Scan and prints verified state.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=False)

from agent.core.audit_logger import AuditLogger, redact_database_url, redact_secrets
from agent.tools.query_analyzer import (
    DEFAULT_CUSTOMER_ID,
    detect_sequential_scan,
    get_query_execution_plan,
    index_exists,
)


def reset_demo_database(*, customer_id: int = DEFAULT_CUSTOMER_ID) -> dict:
    lab_url = os.getenv("LAB_DATABASE_URL", "").strip()
    if not lab_url:
        raise RuntimeError("LAB_DATABASE_URL is not set in environment or .env.")

    with psycopg.connect(lab_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database();")
            db_name = str(cur.fetchone()[0])
            if db_name != "dbmedic_lab":
                raise RuntimeError(
                    f"Refusing demo reset on unexpected database '{db_name}' (expected 'dbmedic_lab')."
                )
            cur.execute("DROP INDEX IF EXISTS idx_orders_customer_id;")
            cur.execute("ANALYZE orders;")
        conn.commit()

        idx_present = bool(index_exists(conn, "idx_orders_customer_id"))
        plan_res = get_query_execution_plan(
            conn,
            params={"customer_id": customer_id},
            analyze=True,
            buffers=True,
        )
        scan = detect_sequential_scan(plan_res["raw_plan"])

    summary = {
        "database": "dbmedic_lab",
        "redacted_url": redact_database_url(lab_url),
        "idx_orders_customer_id_exists": idx_present,
        "scan_type": scan["primary_scan_type"],
        "has_sequential_scan": scan["has_sequential_scan"],
        "ready_for_demo": (not idx_present) and bool(scan["has_sequential_scan"]),
    }

    AuditLogger().log_event(
        incident_id="slow_query_001",
        event_type="INCIDENT_STATUS_CHANGED",
        actor="reset_demo_database",
        execution_mode="system",
        payload={"action": "reset_lab_to_baseline_incident", **summary},
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset dbmedic_lab to slow-query baseline incident.")
    parser.add_argument("--customer", type=int, default=DEFAULT_CUSTOMER_ID)
    args = parser.parse_args()

    try:
        res = reset_demo_database(customer_id=args.customer)
    except Exception as exc:
        print(f"[ERROR] Could not reset dbmedic_lab: {redact_secrets(str(exc))}")
        print("Ensure PostgreSQL is running (e.g. 'docker compose up -d') and schema is initialized.")
        return 2

    print("=== DBMedic Demo Database Reset ===")
    print(f"Target DB         : {res['redacted_url']}")
    print(f"Index Exists      : {res['idx_orders_customer_id_exists']}")
    print(f"Baseline Scan     : {res['scan_type']}")
    print(f"Demo Baseline Ready: {res['ready_for_demo']}")
    return 0 if res["ready_for_demo"] else 1


if __name__ == "__main__":
    sys.exit(main())

