"""
scripts/setup_lab.py
=====================
Create the dbmedic_lab database and apply the schema.

Usage:
    python scripts/setup_lab.py

The script connects to the *primary* DATABASE_URL first (to issue CREATE
DATABASE), then reconnects to LAB_DATABASE_URL to run the schema DDL.

It is safe to rerun — all DDL uses IF NOT EXISTS.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap — ensure the project root is on sys.path regardless of CWD
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("setup_lab")

SCHEMA_FILE = PROJECT_ROOT / "database" / "schemas" / "schema.sql"
LAB_DB_NAME = "dbmedic_lab"


def _get_admin_url() -> str:
    """Return DATABASE_URL (the primary DB) for admin operations."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "DATABASE_URL is not set.  Check your .env file."
        )
    return url


def _get_lab_url() -> str:
    """Return LAB_DATABASE_URL for lab-specific operations."""
    url = os.getenv("LAB_DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "LAB_DATABASE_URL is not set.  Check your .env file."
        )
    return url


def create_lab_database(admin_url: str) -> None:
    """
    Create the dbmedic_lab database if it does not already exist.

    Connects in autocommit mode (required for CREATE DATABASE).
    """
    # psycopg requires autocommit=True for CREATE DATABASE
    with psycopg.connect(admin_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (LAB_DB_NAME,),
            )
            if cur.fetchone():
                logger.info("Database '%s' already exists — skipping creation.", LAB_DB_NAME)
                return
            # Identifiers cannot be parameterised, but we control this value
            cur.execute(f"CREATE DATABASE {LAB_DB_NAME}")
            logger.info("Database '%s' created successfully.", LAB_DB_NAME)


def apply_schema(lab_url: str) -> None:
    """Apply schema.sql to dbmedic_lab."""
    schema_sql = SCHEMA_FILE.read_text(encoding="utf-8")
    with psycopg.connect(lab_url) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
    logger.info("Schema applied to '%s'.", LAB_DB_NAME)


def verify_schema(lab_url: str) -> None:
    """Log the tables that now exist in the lab database."""
    sql = """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename
    """
    with psycopg.connect(lab_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            tables = [r["tablename"] for r in cur.fetchall()]
    logger.info("Tables in '%s': %s", LAB_DB_NAME, tables)
    expected = {"customers", "products", "orders", "order_items"}
    missing = expected - set(tables)
    if missing:
        raise RuntimeError(f"Schema verification failed — missing tables: {missing}")
    logger.info("Schema verification passed.")


def main() -> None:
    admin_url = _get_admin_url()
    lab_url = _get_lab_url()

    logger.info("Step 1/3  Creating lab database …")
    create_lab_database(admin_url)

    logger.info("Step 2/3  Applying schema …")
    apply_schema(lab_url)

    logger.info("Step 3/3  Verifying schema …")
    verify_schema(lab_url)

    logger.info("Lab setup complete.  Run 'python scripts/generate_dataset.py' next.")


if __name__ == "__main__":
    main()

