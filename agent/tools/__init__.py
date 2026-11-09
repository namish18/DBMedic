"""
agent/tools/__init__.py
=======================
DBMedic agent diagnostic tools package.

Provides database connection management with read-only transaction defaults,
and exports diagnostic tools for query, table, and index analysis.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Ensure .env is loaded
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_lab_database_url() -> str:
    """Return the LAB_DATABASE_URL from environment or .env."""
    url = os.getenv("LAB_DATABASE_URL")
    if not url:
        # Retry with explicit path reload
        load_dotenv(PROJECT_ROOT / ".env")
        url = os.getenv("LAB_DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "LAB_DATABASE_URL is not set. Check your .env file."
        )
    return url


@contextmanager
def get_lab_connection(
    conn: psycopg.Connection | None = None,
    *,
    read_only: bool = True,
    timeout_sec: int = 5,
) -> Iterator[psycopg.Connection]:
    """
    Provide an active psycopg connection to dbmedic_lab.

    If an existing connection is passed, it is yielded as-is and NOT closed.
    If None is passed, a new connection is established with read-only safety
    defaults (options="-c default_transaction_read_only=on" when read_only=True)
    and closed automatically upon context exit.
    """
    if conn is not None:
        yield conn
        return

    url = get_lab_database_url()
    conn_options = "-c default_transaction_read_only=on" if read_only else ""

    logger.debug("Opening new connection to dbmedic_lab (read_only=%s)", read_only)
    with psycopg.connect(url, options=conn_options, connect_timeout=timeout_sec) as connection:
        yield connection
