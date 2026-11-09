"""
agent/tools/query_analyzer.py
==============================
Read-only diagnostic tool and gated remediation executor for DBMedic.

Phase 2 capabilities:
  - get_query_execution_plan(): EXPLAIN (FORMAT JSON) plan retrieval using registered templates.
  - analyze_query_performance(): EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) for controlled benchmark.
  - detect_sequential_scan(): Deterministic identification of scan types across plan tree.
  - extract_plan_metrics(): Extraction of planning/execution time, rows, buffer hit/reads.

Design constraints (enforced structurally):
  - Diagnosis functions are strictly read-only (no DDL / DML).
  - Remediation (CREATE INDEX) requires explicit caller approval via
    ``approved=True``; it is never triggered automatically.
  - No arbitrary SQL can be executed — only pre-approved, parameterised
    statements from registered templates.
  - Credentials are never written to log output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import psycopg

from agent.tools import get_lab_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registered Query Templates & Constants
# ---------------------------------------------------------------------------
# Arbitrary SQL is rejected to prevent SQL injection or destructive operations.
# Only pre-registered, parameterised templates are permitted for diagnosis.

DEFAULT_CUSTOMER_ID = 4242
DEFAULT_QUERY_KEY = "slow_query_customer_orders"

REGISTERED_QUERY_TEMPLATES: dict[str, dict[str, Any]] = {
    "slow_query_customer_orders": {
        "sql": "SELECT COUNT(*) FROM orders WHERE customer_id = %(customer_id)s",
        "default_params": {"customer_id": DEFAULT_CUSTOMER_ID},
        "description": "Controlled slow-query incident on orders.customer_id",
        "read_only": True,
        "table_name": "orders",
        "filter_column": "customer_id",
    },
}

# Approved remediation registry (Phase 1)
APPROVED_REMEDIATIONS: dict[str, str] = {
    "create_idx_orders_customer_id": (
        "CREATE INDEX IF NOT EXISTS idx_orders_customer_id "
        "ON orders(customer_id)"
    ),
}

# Scan type classification
SCAN_NODE_TYPES = {
    "Seq Scan": "Sequential Scan",
    "Index Scan": "Index Scan",
    "Index Only Scan": "Index Only Scan",
    "Bitmap Heap Scan": "Bitmap Heap Scan",
    "Bitmap Index Scan": "Bitmap Index Scan",
}


# ---------------------------------------------------------------------------
# Data classes (Phase 1 preserved)
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
# Phase 2: Plan parsing and extraction helpers
# ---------------------------------------------------------------------------

def _normalize_plan_dict(
    plan_data: dict[str, Any] | list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Normalize EXPLAIN JSON output into (top_level_dict, root_plan_node).

    Handles both a list-wrapped result ([{"Plan": ...}]) and a raw dictionary.
    """
    if isinstance(plan_data, list):
        if not plan_data:
            return {}, {}
        top = plan_data[0]
    elif isinstance(plan_data, dict):
        top = plan_data
    else:
        return {}, {}

    plan_node = top.get("Plan", top)
    return top, plan_node


def _collect_all_scans(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Recursively collect all scan nodes from an execution plan subtree."""
    scans: list[dict[str, Any]] = []
    node_type = node.get("Node Type", "")

    if "Scan" in node_type or node_type in SCAN_NODE_TYPES:
        scans.append({
            "node_type": node_type,
            "scan_type_display": SCAN_NODE_TYPES.get(node_type, node_type),
            "relation_name": node.get("Relation Name"),
            "index_name": node.get("Index Name"),
            "plan_rows": node.get("Plan Rows", 0),
            "actual_rows": node.get("Actual Rows", 0),
            "plan_width": node.get("Plan Width", 0),
            "startup_cost": node.get("Startup Cost", 0.0),
            "total_cost": node.get("Total Cost", 0.0),
            "filter": node.get("Filter"),
            "index_cond": node.get("Index Cond"),
        })

    for child in node.get("Plans", []):
        scans.extend(_collect_all_scans(child))

    return scans


def detect_sequential_scan(
    plan_data: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Inspect an execution plan and identify scan types across the plan tree.

    Parameters
    ----------
    plan_data:
        The execution plan structure returned by EXPLAIN (FORMAT JSON).

    Returns
    -------
    dict:
        has_sequential_scan: bool
        primary_scan_type: str (e.g. "Sequential Scan", "Index Scan")
        raw_node_type: str (e.g. "Seq Scan", "Index Scan")
        scanned_relation: str | None
        scanned_index: str | None
        all_scans: list[dict]
    """
    _, plan_node = _normalize_plan_dict(plan_data)
    all_scans = _collect_all_scans(plan_node)

    has_seq_scan = any(s["node_type"] == "Seq Scan" for s in all_scans)

    primary_scan = "unknown"
    raw_node_type = "unknown"
    scanned_relation = None
    scanned_index = None

    if all_scans:
        # Prefer the scan on the primary relation (e.g. orders) or the first scan node
        target_scan = next(
            (s for s in all_scans if s.get("relation_name") == "orders"),
            all_scans[0],
        )
        raw_node_type = target_scan["node_type"]
        primary_scan = SCAN_NODE_TYPES.get(raw_node_type, raw_node_type)
        scanned_relation = target_scan.get("relation_name")
        scanned_index = target_scan.get("index_name")

    return {
        "has_sequential_scan": has_seq_scan,
        "primary_scan_type": primary_scan,
        "raw_node_type": raw_node_type,
        "scanned_relation": scanned_relation,
        "scanned_index": scanned_index,
        "all_scans": all_scans,
    }


def extract_plan_metrics(
    plan_data: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Extract key performance metrics from an EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) result.

    Returns:
      - planning_time_ms: float
      - execution_time_ms: float
      - actual_rows: int
      - estimated_rows: int
      - shared_hit_blocks: int
      - shared_read_blocks: int
      - relation_name: str | None
      - index_name: str | None
      - scan_type: str
    """
    top, plan_node = _normalize_plan_dict(plan_data)

    planning_time = top.get("Planning Time", 0.0)
    execution_time = top.get("Execution Time", 0.0)

    # Walk to get relation, index, and scan type
    scan_info = detect_sequential_scan(plan_data)

    estimated_rows = plan_node.get("Plan Rows", 0)
    actual_rows = plan_node.get("Actual Rows", 0)

    # Buffer accounting: check top level and plan node
    shared_hit = top.get("Shared Hit Blocks", 0)
    shared_read = top.get("Shared Read Blocks", 0)

    # Check planning buffers
    if "Planning" in top:
        shared_hit += top["Planning"].get("Shared Hit Blocks", 0)
        shared_read += top["Planning"].get("Shared Read Blocks", 0)

    # Accumulate child node buffers if top level lacked them
    def _accumulate_buffers(node: dict[str, Any]) -> tuple[int, int]:
        hits = node.get("Shared Hit Blocks", 0)
        reads = node.get("Shared Read Blocks", 0)
        for child in node.get("Plans", []):
            ch_h, ch_r = _accumulate_buffers(child)
            hits += ch_h
            reads += ch_r
        return hits, reads

    node_hits, node_reads = _accumulate_buffers(plan_node)
    if shared_hit == 0 and node_hits > 0:
        shared_hit = node_hits
    if shared_read == 0 and node_reads > 0:
        shared_read = node_reads

    # If top node is Aggregate (e.g. COUNT(*)), find child scan rows
    if scan_info["all_scans"]:
        first_scan = scan_info["all_scans"][0]
        if actual_rows <= 1 and first_scan.get("actual_rows", 0) > 0:
            actual_rows = first_scan["actual_rows"]
        if estimated_rows <= 1 and first_scan.get("plan_rows", 0) > 0:
            estimated_rows = first_scan["plan_rows"]

    return {
        "planning_time_ms": round(float(planning_time), 3),
        "execution_time_ms": round(float(execution_time), 3),
        "actual_rows": int(actual_rows),
        "estimated_rows": int(estimated_rows),
        "shared_hit_blocks": int(shared_hit),
        "shared_read_blocks": int(shared_read),
        "relation_name": scan_info.get("scanned_relation"),
        "index_name": scan_info.get("scanned_index"),
        "scan_type": scan_info.get("primary_scan_type", "unknown"),
        "raw_scan_type": scan_info.get("raw_node_type", "unknown"),
    }


# ---------------------------------------------------------------------------
# Phase 2: Query Execution Plan Retrieval & Analysis
# ---------------------------------------------------------------------------

def get_query_execution_plan(
    conn: psycopg.Connection | None = None,
    *,
    template_key: str = DEFAULT_QUERY_KEY,
    params: dict[str, Any] | None = None,
    analyze: bool = False,
    buffers: bool = False,
) -> dict[str, Any]:
    """
    Retrieve PostgreSQL execution plan using EXPLAIN (FORMAT JSON).

    Security Rules:
      - Only registered query templates are allowed. Arbitrary SQL is rejected.
      - EXPLAIN ANALYZE is restricted to read-only registered templates.
      - Connects via LAB_DATABASE_URL with read-only transaction defaults.

    Parameters
    ----------
    conn:
        Optional active psycopg connection. If None, connects to LAB_DATABASE_URL.
    template_key:
        Identifier of registered query template in REGISTERED_QUERY_TEMPLATES.
    params:
        Optional parameters dictionary overriding template defaults.
    analyze:
        If True, executes EXPLAIN (ANALYZE, ...). Allowed only for read-only templates.
    buffers:
        If True, appends BUFFERS option.

    Returns
    -------
    dict: Parsed top-level dictionary from EXPLAIN JSON output.
    """
    if template_key not in REGISTERED_QUERY_TEMPLATES:
        raise ValueError(
            f"Query template '{template_key}' is not registered. "
            f"Permitted templates: {list(REGISTERED_QUERY_TEMPLATES.keys())}"
        )

    template_entry = REGISTERED_QUERY_TEMPLATES[template_key]
    if analyze and not template_entry.get("read_only", False):
        raise PermissionError(
            f"EXPLAIN ANALYZE refused: template '{template_key}' is not verified read-only."
        )

    options: list[str] = ["FORMAT JSON"]
    if analyze:
        options.insert(0, "ANALYZE")
    if buffers:
        options.insert(1, "BUFFERS")

    sql = f"EXPLAIN ({', '.join(options)}) {template_entry['sql']}"
    bind_params = params if params is not None else template_entry["default_params"]

    logger.debug("Executing plan retrieval for template '%s' (analyze=%s)", template_key, analyze)

    with get_lab_connection(conn, read_only=True) as db_conn:
        with db_conn.cursor() as cur:
            cur.execute(sql, bind_params)
            row = cur.fetchone()

    if not row or not row[0]:
        raise RuntimeError("EXPLAIN returned empty output")

    raw_plan_list = row[0]
    return raw_plan_list[0] if isinstance(raw_plan_list, list) else raw_plan_list


def analyze_query_performance(
    conn: psycopg.Connection | None = None,
    *,
    template_key: str = DEFAULT_QUERY_KEY,
    customer_id: int = DEFAULT_CUSTOMER_ID,
) -> dict[str, Any]:
    """
    Execute controlled EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) benchmark analysis.

    Combines plan retrieval, metric extraction, and scan type detection.

    Returns
    -------
    dict with:
      - template_key
      - customer_id
      - metrics (execution_time_ms, planning_time_ms, rows, buffer blocks)
      - scan_analysis (has_sequential_scan, primary_scan_type, relation, index)
      - raw_plan
    """
    plan_dict = get_query_execution_plan(
        conn=conn,
        template_key=template_key,
        params={"customer_id": customer_id},
        analyze=True,
        buffers=True,
    )

    metrics = extract_plan_metrics(plan_dict)
    scan_analysis = detect_sequential_scan(plan_dict)

    return {
        "template_key": template_key,
        "customer_id": customer_id,
        "metrics": metrics,
        "scan_analysis": scan_analysis,
        "raw_plan": plan_dict,
    }


# ---------------------------------------------------------------------------
# Phase 1: Diagnostic helpers (preserved for backwards compatibility)
# ---------------------------------------------------------------------------

def get_table_stats(conn: psycopg.Connection, table_name: str) -> TableStats:
    """Return row estimate and size statistics for *table_name*."""
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
    """List all indexes defined on *table_name* in the public schema."""
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
    """Identify columns that have foreign-key constraints but no index."""
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
    indexed_columns: set[str] = set()
    for idx in indexes:
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
# Remediation (gated — Phase 1 preserved)
# ---------------------------------------------------------------------------

def apply_remediation(
    conn: psycopg.Connection,
    remediation_key: str,
    *,
    approved: bool = False,
) -> dict[str, Any]:
    """Execute a pre-approved remediation statement."""
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
    logger.debug("DDL: %s", ddl)

    import time
    start = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    elapsed_ms = (time.perf_counter() - start) * 1_000

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
    for i, tok in enumerate(tokens):
        if tok.upper() == "EXISTS":
            if i + 1 < len(tokens):
                return tokens[i + 1]
    if len(tokens) >= 3 and tokens[0].upper() == "CREATE":
        idx = 2
        return tokens[idx] if idx < len(tokens) else None
    return None
