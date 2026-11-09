"""
agent/tools/table_monitor.py
============================
Table statistics, row estimates, sequential scan metrics, and size monitoring tool.

Queries PostgreSQL system catalogs (pg_class, pg_namespace) and statistics
views (pg_stat_user_tables) to gather evidence regarding table volume, scan
patterns, and bloat.

Security constraints:
  - Table identifiers are strictly validated against an immutable allowlist.
  - Queries are executed using read-only database transactions.
  - Parameterized identifiers and views prevent SQL injection.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg

from agent.tools import get_lab_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowlist of permitted tables
# ---------------------------------------------------------------------------
ALLOWED_TABLES: frozenset[str] = frozenset(
    {"customers", "products", "orders", "order_items"}
)


def _validate_table_name(table_name: str) -> str:
    """Validate that table_name is in ALLOWED_TABLES. Raises ValueError otherwise."""
    if not isinstance(table_name, str):
        raise TypeError(f"table_name must be a str, got {type(table_name).__name__}")
    normalized = table_name.strip().lower()
    if normalized not in ALLOWED_TABLES:
        raise ValueError(
            f"Table '{table_name}' is not in permitted allowlist. "
            f"Allowed tables: {sorted(ALLOWED_TABLES)}"
        )
    return normalized


# ---------------------------------------------------------------------------
# Public Table Monitoring API
# ---------------------------------------------------------------------------

def get_table_statistics(
    table_name: str,
    conn: psycopg.Connection | None = None,
) -> dict[str, Any]:
    """
    Retrieve comprehensive table statistics from pg_stat_user_tables and pg_class.

    Returns
    -------
    dict:
      - table_name: str
      - estimated_row_count: int
      - sequential_scan_count: int
      - sequential_tuples_read: int
      - index_scan_count: int
      - index_tuples_fetched: int
      - table_size_bytes: int
      - table_size_pretty: str
      - dead_tuple_estimate: int
      - live_tuple_estimate: int
    """
    valid_table = _validate_table_name(table_name)

    sql = """
        SELECT
            c.relname                                    AS table_name,
            c.reltuples::BIGINT                          AS estimated_row_count,
            COALESCE(s.seq_scan, 0)                      AS sequential_scan_count,
            COALESCE(s.seq_tup_read, 0)                  AS sequential_tuples_read,
            COALESCE(s.idx_scan, 0)                      AS index_scan_count,
            COALESCE(s.idx_tup_fetch, 0)                 AS index_tuples_fetched,
            pg_total_relation_size(c.oid)                AS table_size_bytes,
            pg_size_pretty(pg_total_relation_size(c.oid)) AS table_size_pretty,
            COALESCE(s.n_dead_tup, 0)                    AS dead_tuple_estimate,
            COALESCE(s.n_live_tup, 0)                    AS live_tuple_estimate
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_stat_user_tables s
               ON s.relid = c.oid AND s.schemaname = 'public'
        WHERE c.relname = %s
          AND n.nspname = 'public'
          AND c.relkind = 'r'
    """

    with get_lab_connection(conn, read_only=True) as db_conn:
        with db_conn.cursor() as cur:
            cur.execute(sql, (valid_table,))
            row = cur.fetchone()

    if not row:
        raise ValueError(f"Table '{valid_table}' was not found in public schema")

    return {
        "table_name": row[0],
        "estimated_row_count": int(row[1]) if row[1] is not None else 0,
        "sequential_scan_count": int(row[2]),
        "sequential_tuples_read": int(row[3]),
        "index_scan_count": int(row[4]),
        "index_tuples_fetched": int(row[5]),
        "table_size_bytes": int(row[6]),
        "table_size_pretty": str(row[7]),
        "dead_tuple_estimate": int(row[8]),
        "live_tuple_estimate": int(row[9]),
    }


def get_table_size(
    table_name: str,
    conn: psycopg.Connection | None = None,
) -> dict[str, Any]:
    """
    Retrieve table, index, and total size breakdowns in bytes and human-readable form.

    Returns
    -------
    dict:
      - table_name: str
      - total_size_bytes: int
      - total_size_pretty: str
      - table_size_bytes: int
      - table_size_pretty: str
      - index_size_bytes: int
      - index_size_pretty: str
      - toast_size_bytes: int
    """
    valid_table = _validate_table_name(table_name)

    sql = """
        SELECT
            c.relname                                    AS table_name,
            pg_total_relation_size(c.oid)                AS total_size_bytes,
            pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size_pretty,
            pg_relation_size(c.oid)                      AS table_size_bytes,
            pg_size_pretty(pg_relation_size(c.oid))      AS table_size_pretty,
            pg_indexes_size(c.oid)                       AS index_size_bytes,
            pg_size_pretty(pg_indexes_size(c.oid))       AS index_size_pretty,
            pg_total_relation_size(c.oid)
                - pg_relation_size(c.oid)
                - pg_indexes_size(c.oid)                 AS toast_size_bytes
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = %s
          AND n.nspname = 'public'
          AND c.relkind = 'r'
    """

    with get_lab_connection(conn, read_only=True) as db_conn:
        with db_conn.cursor() as cur:
            cur.execute(sql, (valid_table,))
            row = cur.fetchone()

    if not row:
        raise ValueError(f"Table '{valid_table}' was not found in public schema")

    return {
        "table_name": row[0],
        "total_size_bytes": int(row[1]),
        "total_size_pretty": str(row[2]),
        "table_size_bytes": int(row[3]),
        "table_size_pretty": str(row[4]),
        "index_size_bytes": int(row[5]),
        "index_size_pretty": str(row[6]),
        "toast_size_bytes": max(0, int(row[7])),
    }


def get_sequential_scan_statistics(
    table_name: str,
    conn: psycopg.Connection | None = None,
) -> dict[str, Any]:
    """
    Retrieve scan ratio and sequential tuple metrics for table_name.

    Returns
    -------
    dict:
      - table_name: str
      - sequential_scan_count: int
      - sequential_tuples_read: int
      - index_scan_count: int
      - index_tuples_fetched: int
      - seq_scan_ratio: float
    """
    valid_table = _validate_table_name(table_name)

    sql = """
        SELECT
            COALESCE(seq_scan, 0)      AS seq_scan,
            COALESCE(seq_tup_read, 0)  AS seq_tup_read,
            COALESCE(idx_scan, 0)      AS idx_scan,
            COALESCE(idx_tup_fetch, 0) AS idx_tup_fetch
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
          AND relname = %s
    """

    with get_lab_connection(conn, read_only=True) as db_conn:
        with db_conn.cursor() as cur:
            cur.execute(sql, (valid_table,))
            row = cur.fetchone()

    seq_scan = int(row[0]) if row else 0
    seq_tup_read = int(row[1]) if row else 0
    idx_scan = int(row[2]) if row else 0
    idx_tup_fetch = int(row[3]) if row else 0

    total_scans = seq_scan + idx_scan
    seq_ratio = (seq_scan / total_scans) if total_scans > 0 else 0.0

    return {
        "table_name": valid_table,
        "sequential_scan_count": seq_scan,
        "sequential_tuples_read": seq_tup_read,
        "index_scan_count": idx_scan,
        "index_tuples_fetched": idx_tup_fetch,
        "seq_scan_ratio": round(seq_ratio, 4),
    }


def get_table_row_estimates(
    table_name: str,
    conn: psycopg.Connection | None = None,
) -> dict[str, Any]:
    """
    Retrieve planner row estimate and disk page counts from pg_class.

    Returns
    -------
    dict:
      - table_name: str
      - estimated_row_count: int
      - pages: int
    """
    valid_table = _validate_table_name(table_name)

    sql = """
        SELECT
            reltuples::BIGINT AS estimated_row_count,
            relpages          AS pages
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = %s
          AND n.nspname = 'public'
    """

    with get_lab_connection(conn, read_only=True) as db_conn:
        with db_conn.cursor() as cur:
            cur.execute(sql, (valid_table,))
            row = cur.fetchone()

    if not row:
        raise ValueError(f"Table '{valid_table}' was not found in public schema")

    return {
        "table_name": valid_table,
        "estimated_row_count": int(row[0]) if row[0] is not None else 0,
        "pages": int(row[1]) if row[1] is not None else 0,
    }

