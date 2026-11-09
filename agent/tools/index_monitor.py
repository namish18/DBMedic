"""
agent/tools/index_monitor.py
============================
Index metadata, definition, and usage monitoring tool for DBMedic.

Inspects PostgreSQL system catalogs (pg_indexes, pg_index, pg_class, pg_attribute)
and statistics views (pg_stat_user_indexes) to determine whether appropriate
indexes exist on query filter columns.

Security constraints:
  - Table identifiers are strictly validated against an immutable allowlist.
  - Index names are validated against safe identifier patterns.
  - Strictly read-only operations — NO DDL (CREATE/DROP INDEX) is executed here.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import psycopg

from agent.tools import get_lab_connection

logger = logging.getLogger(__name__)

# Allowlist of permitted tables
ALLOWED_TABLES: frozenset[str] = frozenset(
    {"customers", "products", "orders", "order_items"}
)

# Safe SQL identifier pattern (alphanumeric + underscores)
SAFE_IDENTIFIER_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")


def _validate_table_name(table_name: str) -> str:
    """Validate table_name against ALLOWED_TABLES."""
    if not isinstance(table_name, str):
        raise TypeError(f"table_name must be a str, got {type(table_name).__name__}")
    normalized = table_name.strip().lower()
    if normalized not in ALLOWED_TABLES:
        raise ValueError(
            f"Table '{table_name}' is not in permitted allowlist. "
            f"Allowed tables: {sorted(ALLOWED_TABLES)}"
        )
    return normalized


def _validate_index_identifier(index_name: str) -> str:
    """Validate index_name against safe SQL identifier pattern."""
    if not isinstance(index_name, str):
        raise TypeError(f"index_name must be a str, got {type(index_name).__name__}")
    normalized = index_name.strip()
    if not SAFE_IDENTIFIER_REGEX.match(normalized):
        raise ValueError(f"Invalid index identifier: '{index_name}'")
    return normalized


# ---------------------------------------------------------------------------
# Public Index Monitoring API
# ---------------------------------------------------------------------------

def get_table_indexes(
    table_name: str,
    conn: psycopg.Connection | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve all index definitions and specifications for table_name.

    Returns
    -------
    list[dict]:
      - index_name: str
      - table_name: str
      - indexed_columns: list[str]
      - index_definition: str
      - index_type: str
      - is_unique: bool
      - index_scan_count: int
      - index_tuples_read: int
      - index_tuples_fetched: int
    """
    valid_table = _validate_table_name(table_name)

    sql = """
        SELECT
            i.indexname                                  AS index_name,
            i.tablename                                  AS table_name,
            i.indexdef                                   AS index_definition,
            ix.indisunique                               AS is_unique,
            am.amname                                    AS index_type,
            array_agg(a.attname ORDER BY array_position(ix.indkey, a.attnum)) AS indexed_columns,
            COALESCE(sui.idx_scan, 0)                    AS index_scan_count,
            COALESCE(sui.idx_tup_read, 0)                AS index_tuples_read,
            COALESCE(sui.idx_tup_fetch, 0)               AS index_tuples_fetched
        FROM pg_indexes i
        JOIN pg_class c   ON c.relname = i.indexname
        JOIN pg_index ix  ON ix.indexrelid = c.oid
        JOIN pg_am am     ON am.oid = c.relam
        JOIN pg_class tc  ON tc.relname = i.tablename
        JOIN pg_attribute a ON a.attrelid = tc.oid
                            AND a.attnum = ANY(ix.indkey)
                            AND a.attnum > 0
        LEFT JOIN pg_stat_user_indexes sui
               ON sui.indexrelname = i.indexname AND sui.schemaname = 'public'
        WHERE i.schemaname = 'public'
          AND i.tablename  = %s
        GROUP BY
            i.indexname,
            i.tablename,
            i.indexdef,
            ix.indisunique,
            am.amname,
            sui.idx_scan,
            sui.idx_tup_read,
            sui.idx_tup_fetch
        ORDER BY i.indexname
    """

    with get_lab_connection(conn, read_only=True) as db_conn:
        with db_conn.cursor() as cur:
            cur.execute(sql, (valid_table,))
            rows = cur.fetchall()

    return [
        {
            "index_name": r[0],
            "table_name": r[1],
            "index_definition": r[2],
            "is_unique": bool(r[3]),
            "index_type": str(r[4]),
            "indexed_columns": list(r[5]) if r[5] else [],
            "index_scan_count": int(r[6]),
            "index_tuples_read": int(r[7]),
            "index_tuples_fetched": int(r[8]),
        }
        for r in rows
    ]


def get_index_usage(
    table_name: str,
    conn: psycopg.Connection | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve operational usage statistics for indexes on table_name from pg_stat_user_indexes.

    Returns
    -------
    list[dict]:
      - index_name: str
      - table_name: str
      - index_scan_count: int
      - index_tuples_read: int
      - index_tuples_fetched: int
    """
    valid_table = _validate_table_name(table_name)

    sql = """
        SELECT
            indexrelname                 AS index_name,
            relname                      AS table_name,
            COALESCE(idx_scan, 0)        AS index_scan_count,
            COALESCE(idx_tup_read, 0)    AS index_tuples_read,
            COALESCE(idx_tup_fetch, 0)   AS index_tuples_fetched
        FROM pg_stat_user_indexes
        WHERE schemaname = 'public'
          AND relname = %s
        ORDER BY indexrelname
    """

    with get_lab_connection(conn, read_only=True) as db_conn:
        with db_conn.cursor() as cur:
            cur.execute(sql, (valid_table,))
            rows = cur.fetchall()

    return [
        {
            "index_name": r[0],
            "table_name": r[1],
            "index_scan_count": int(r[2]),
            "index_tuples_read": int(r[3]),
            "index_tuples_fetched": int(r[4]),
        }
        for r in rows
    ]


def check_index_exists(
    index_name: str,
    conn: psycopg.Connection | None = None,
) -> bool:
    """
    Check whether index_name exists in the public schema.

    Returns True if present, False otherwise.
    """
    valid_name = _validate_index_identifier(index_name)

    sql = """
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname = %s
    """

    with get_lab_connection(conn, read_only=True) as db_conn:
        with db_conn.cursor() as cur:
            cur.execute(sql, (valid_name,))
            return cur.fetchone() is not None


def inspect_customer_id_index(
    conn: psycopg.Connection | None = None,
) -> dict[str, Any]:
    """
    Inspect the orders table specifically for an index covering customer_id.

    Returns
    -------
    dict:
      - table_name: 'orders'
      - target_column: 'customer_id'
      - index_exists: bool
      - index_name: str | None
      - index_definition: str | None
      - is_suitable_for_filter: bool
      - index_scan_count: int
      - index_tuples_read: int
      - index_tuples_fetched: int
    """
    indexes = get_table_indexes("orders", conn=conn)

    # An index is suitable for "WHERE customer_id = %s" if customer_id is its leading column
    matching_idx = next(
        (
            idx
            for idx in indexes
            if idx["indexed_columns"] and idx["indexed_columns"][0] == "customer_id"
        ),
        None,
    )

    if matching_idx:
        return {
            "table_name": "orders",
            "target_column": "customer_id",
            "index_exists": True,
            "index_name": matching_idx["index_name"],
            "index_definition": matching_idx["index_definition"],
            "is_suitable_for_filter": True,
            "index_scan_count": matching_idx["index_scan_count"],
            "index_tuples_read": matching_idx["index_tuples_read"],
            "index_tuples_fetched": matching_idx["index_tuples_fetched"],
        }

    return {
        "table_name": "orders",
        "target_column": "customer_id",
        "index_exists": False,
        "index_name": None,
        "index_definition": None,
        "is_suitable_for_filter": False,
        "index_scan_count": 0,
        "index_tuples_read": 0,
        "index_tuples_fetched": 0,
    }

