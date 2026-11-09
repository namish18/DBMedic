"""
agent/tools/query_analyzer.py
==============================
Read-only diagnostic tool and gated remediation executor for DBMedic.

Design constraints (enforced structurally):
  - Diagnosis functions are strictly read-only (no DDL / DML).
  - Remediation (CREATE INDEX) requires explicit caller approval via
    ``approved=True``; it is never triggered automatically.
  - No arbitrary SQL can be executed — only pre-approved, parameterised
    statements from the APPROVED_REMEDIATIONS registry.
  - Credentials are never written to log output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import psycopg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Approved remediation registry
# ---------------------------------------------------------------------------
# Each entry maps a remediation key → the exact DDL string the agent is
# permitted to run.  Keeping DDL here (not in caller code) prevents arbitrary
# SQL execution by the model layer.

APPROVED_REMEDIATIONS: dict[str, str] = {
    "create_idx_orders_customer_id": (
        "CREATE INDEX IF NOT EXISTS idx_orders_customer_id "
        "ON orders(customer_id)"
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IndexInfo:
    """Describes one index on a table."""

    index_name: str
    table_name: str
    column_names: list[str]
    index_type: str
    is_unique: bool


@dataclass
class TableStats:
    """Basic statistics for a table."""

    table_name: str
    row_estimate: int
    total_size_bytes: int
    index_size_bytes: int


# ---------------------------------------------------------------------------
# Diagnostic helpers (read-only)
# ---------------------------------------------------------------------------

def get_table_stats(conn: psycopg.Connection, table_name: str) -> TableStats:
    """
    Return row estimate and size statistics for *table_name*.

    Read-only: uses pg_class / pg_total_relation_size.
    """
    sql = """
        SELECT
            reltuples::BIGINT                        AS row_estimate,
            pg_total_relation_size(c.oid)            AS total_size_bytes,
            pg_indexes_size(c.oid)                   AS index_size_bytes
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = %s
          AND n.nspname = 'public'
    """
    with conn.cursor() as cur:
        cur.execute(sql, (table_name,))
        row = cur.fetchone()

    if row is None:
        raise ValueError(f"Table '{table_name}' not found in public schema")

    return TableStats(
        table_name=table_name,
        row_estimate=int(row[0]),
        total_size_bytes=int(row[1]),
        index_size_bytes=int(row[2]),
    )


def list_indexes(
    conn: psycopg.Connection, table_name: str
) -> list[IndexInfo]:
    """
    List all indexes defined on *table_name* in the public schema.

    Read-only: queries pg_indexes.
    """
    sql = """
        SELECT
            i.indexname,
            i.tablename,
            ix.indisunique,
            am.amname                            AS index_type,
            array_agg(a.attname ORDER BY a.attnum) AS column_names
        FROM pg_indexes i
        JOIN pg_class c  ON c.relname = i.indexname
        JOIN pg_index ix ON ix.indexrelid = c.oid
        JOIN pg_am am    ON am.oid = c.relam
        JOIN pg_class tc ON tc.relname = i.tablename
        JOIN pg_attribute a ON a.attrelid = tc.oid
                            AND a.attnum = ANY(ix.indkey)
                            AND a.attnum > 0
        WHERE i.schemaname = 'public'
          AND i.tablename  = %s
        GROUP BY i.indexname, i.tablename, ix.indisunique, am.amname
        ORDER BY i.indexname
    """
    with conn.cursor() as cur:
        cur.execute(sql, (table_name,))
        rows = cur.fetchall()

    return [
        IndexInfo(
            index_name=r[0],
            table_name=r[1],
            is_unique=r[2],
            index_type=r[3],
            column_names=list(r[4]),
        )
        for r in rows
    ]


def index_exists(conn: psycopg.Connection, index_name: str) -> bool:
    """Return True if *index_name* exists in the public schema."""
    sql = """
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public' AND indexname = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (index_name,))
        return cur.fetchone() is not None


def diagnose_missing_indexes(
    conn: psycopg.Connection, table_name: str
) -> dict[str, Any]:
    """
    Identify columns that have foreign-key constraints but no index.

    Returns a dict describing the table, its current indexes, and any
    un-indexed FK columns.

    Read-only: queries information_schema and pg_*.
    """
    # Gather FK columns for the table
    fk_sql = """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema    = kcu.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema    = 'public'
          AND tc.table_name      = %s
        ORDER BY kcu.column_name
    """
    with conn.cursor() as cur:
        cur.execute(fk_sql, (table_name,))
        fk_columns = {row[0] for row in cur.fetchall()}

    indexes = list_indexes(conn, table_name)

    # Determine which FK columns are covered by an index
    indexed_columns: set[str] = set()
    for idx in indexes:
        # An index covers a column if that column is the leading key
        if idx.column_names:
            indexed_columns.add(idx.column_names[0])

    missing = sorted(fk_columns - indexed_columns)

    return {
        "table": table_name,
        "fk_columns": sorted(fk_columns),
        "indexed_leading_columns": sorted(indexed_columns),
        "missing_indexes_on": missing,
        "recommendation": (
            f"Consider adding indexes on: {missing}" if missing else "All FK columns are indexed"
        ),
    }


# ---------------------------------------------------------------------------
# Remediation (gated — never automatic)
# ---------------------------------------------------------------------------

def apply_remediation(
    conn: psycopg.Connection,
    remediation_key: str,
    *,
    approved: bool = False,
) -> dict[str, Any]:
    """
    Execute a pre-approved remediation statement.

    Parameters
    ----------
    conn:
        An open psycopg connection to dbmedic_lab with appropriate privileges.
    remediation_key:
        Key into ``APPROVED_REMEDIATIONS``.  Only statements in that registry
        can be executed — no arbitrary SQL is permitted.
    approved:
        **Must be True** for the remediation to proceed.  Callers must
        explicitly set this flag; it is never set automatically by the agent.

    Returns
    -------
    dict describing the outcome (success, index existence, timing_ms).

    Raises
    ------
    PermissionError  if ``approved`` is False.
    KeyError         if ``remediation_key`` is not in APPROVED_REMEDIATIONS.
    """
    if not approved:
        raise PermissionError(
            "Remediation requires explicit approval. "
            "Call apply_remediation(..., approved=True) to proceed."
        )

    if remediation_key not in APPROVED_REMEDIATIONS:
        raise KeyError(
            f"Unknown remediation key '{remediation_key}'. "
            f"Valid keys: {list(APPROVED_REMEDIATIONS)}"
        )

    ddl = APPROVED_REMEDIATIONS[remediation_key]
    logger.info("Applying remediation '%s'", remediation_key)
    # Log key but never the connection string / credentials
    logger.debug("DDL: %s", ddl)

    import time
    start = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    elapsed_ms = (time.perf_counter() - start) * 1_000

    # Derive the expected index name from the DDL
    # Convention: "CREATE INDEX IF NOT EXISTS <name> ON …"
    index_name = _parse_index_name(ddl)
    exists = index_exists(conn, index_name) if index_name else False

    logger.info(
        "Remediation '%s' complete in %.1f ms — index exists: %s",
        remediation_key,
        elapsed_ms,
        exists,
    )

    return {
        "remediation_key": remediation_key,
        "ddl_applied": ddl,
        "elapsed_ms": round(elapsed_ms, 1),
        "index_name": index_name,
        "index_exists": exists,
        "success": exists,
    }


def _parse_index_name(ddl: str) -> str | None:
    """Extract the index name from a 'CREATE INDEX … ON …' statement."""
    tokens = ddl.split()
    # CREATE INDEX [IF NOT EXISTS] <name> ON …
    for i, tok in enumerate(tokens):
        if tok.upper() == "EXISTS":
            if i + 1 < len(tokens):
                return tokens[i + 1]
    # Fallback: 4th token when no IF NOT EXISTS clause
    if len(tokens) >= 3 and tokens[0].upper() == "CREATE":
        idx = 2
        return tokens[idx] if idx < len(tokens) else None
    return None

