"""
scripts/generate_dataset.py
============================
Populate dbmedic_lab with synthetic e-commerce data via generate_data.sql.

Usage:
    python scripts/generate_dataset.py [--customers N] [--products N] [--orders N]

The underlying SQL seed uses PostgreSQL generate_series() and setseed() so
results are deterministic.  Re-running the script on an already-populated
database is safe — the SQL uses ON CONFLICT DO NOTHING.

Configurable row counts:
  --customers   default 10 000
  --products    default 50 000
  --orders      default 200 000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_dataset")

SEED_FILE = PROJECT_ROOT / "database" / "seeds" / "generate_data.sql"


def _get_lab_url() -> str:
    url = os.getenv("LAB_DATABASE_URL")
    if not url:
        raise EnvironmentError("LAB_DATABASE_URL is not set. Check your .env file.")
    return url


def _row_count(conn: psycopg.Connection, table: str) -> int:
    """Return the exact row count for *table*."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 — table name is internal
        return cur.fetchone()[0]  # type: ignore[index]


def already_populated(conn: psycopg.Connection, min_customers: int) -> bool:
    """Return True if the customers table already has at least *min_customers* rows."""
    count = _row_count(conn, "customers")
    return count >= min_customers


def generate_data(
    lab_url: str,
    n_customers: int,
    n_products: int,
    n_orders: int,
) -> None:
    """
    Execute generate_data.sql with the given row counts.

    psql \set variables are replaced inline before execution so we can drive
    the SQL from Python without shelling out to psql.
    """
    raw_sql = SEED_FILE.read_text(encoding="utf-8")

    # Replace psql \set-style placeholders used in the SQL file
    populated_sql = (
        raw_sql
        .replace(":n_customers", str(n_customers))
        .replace(":n_products", str(n_products))
        .replace(":n_orders", str(n_orders))
    )

    with psycopg.connect(lab_url, options="-c statement_timeout=600000") as conn:
        if already_populated(conn, n_customers):
            logger.info(
                "Database already contains data (customers ≥ %d). Skipping generation.",
                n_customers,
            )
            _log_counts(conn)
            return

        logger.info(
            "Generating: %d customers, %d products, %d orders …",
            n_customers,
            n_products,
            n_orders,
        )
        t0 = time.perf_counter()

        with conn.cursor() as cur:
            cur.execute(populated_sql)
        conn.commit()

        elapsed = time.perf_counter() - t0
        logger.info("Data generation complete in %.1f seconds.", elapsed)
        _log_counts(conn)


def _log_counts(conn: psycopg.Connection) -> None:
    for table in ("customers", "products", "orders", "order_items"):
        count = _row_count(conn, table)
        logger.info("  %-15s %10d rows", table, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic DBMedic lab data")
    parser.add_argument("--customers", type=int, default=10_000, metavar="N")
    parser.add_argument("--products",  type=int, default=50_000, metavar="N")
    parser.add_argument("--orders",    type=int, default=200_000, metavar="N")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lab_url = _get_lab_url()
    generate_data(
        lab_url,
        n_customers=args.customers,
        n_products=args.products,
        n_orders=args.orders,
    )


if __name__ == "__main__":
    main()

