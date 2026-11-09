"""
tests/test_database_setup.py
==============================
Automated tests for DBMedic lab database setup.

Tests verify:
  - LAB_DATABASE_URL is resolvable and connectable.
  - All four tables exist in the public schema.
  - Expected constraints and column types are present.
  - No secondary index on orders.customer_id (pre-remediation state).

Run:
    pytest tests/test_database_setup.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from agent.tools.query_analyzer import index_exists, list_indexes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lab_url() -> str:
    url = os.getenv("LAB_DATABASE_URL")
    if not url:
        pytest.skip("LAB_DATABASE_URL not set — skipping database tests")
    return url


@pytest.fixture(scope="module")
def conn(lab_url: str):
    with psycopg.connect(lab_url) as connection:
        yield connection


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

class TestConnectivity:
    def test_connection_succeeds(self, conn: psycopg.Connection) -> None:
        """Database must be reachable and return a valid version string."""
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
        assert "PostgreSQL" in version

    def test_database_name(self, conn: psycopg.Connection) -> None:
        """Must be connected to dbmedic_lab, not any other database."""
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            db_name = cur.fetchone()[0]
        assert db_name == "dbmedic_lab"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    EXPECTED_TABLES = {"customers", "products", "orders", "order_items"}

    def _get_tables(self, conn: psycopg.Connection) -> set[str]:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            return {row[0] for row in cur.fetchall()}

    def test_all_tables_exist(self, conn: psycopg.Connection) -> None:
        tables = self._get_tables(conn)
        missing = self.EXPECTED_TABLES - tables
        assert not missing, f"Missing tables: {missing}"

    def test_customers_has_unique_email(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT constraint_name
                FROM information_schema.table_constraints
                WHERE table_schema = 'public'
                  AND table_name   = 'customers'
                  AND constraint_type = 'UNIQUE'
                """
            )
            names = {row[0] for row in cur.fetchall()}
        assert names, "customers.email should have a UNIQUE constraint"

    def test_orders_has_fk_on_customer_id(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema    = kcu.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema    = 'public'
                  AND tc.table_name      = 'orders'
                """
            )
            fk_columns = {row[0] for row in cur.fetchall()}
        assert "customer_id" in fk_columns

    def test_order_items_has_fk_on_order_id(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema    = kcu.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema    = 'public'
                  AND tc.table_name      = 'order_items'
                """
            )
            fk_columns = {row[0] for row in cur.fetchall()}
        assert "order_id" in fk_columns

    def test_no_index_on_orders_customer_id_initially(
        self, conn: psycopg.Connection
    ) -> None:
        """
        Pre-remediation: orders.customer_id must NOT have a secondary index.
        If this test fails, the remediation was already applied — drop the
        index manually or recreate the lab to restore the incident state.
        """
        exists = index_exists(conn, "idx_orders_customer_id")
        assert not exists, (
            "idx_orders_customer_id already exists — incident state has been "
            "remediated.  Drop the index to restore the baseline."
        )

