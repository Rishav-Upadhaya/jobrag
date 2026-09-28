"""Runs the real migration and keyword_search against Postgres + pgvector.

Skipped unless TEST_DATABASE_URL points at a throwaway database, e.g.
    docker run -d -p 5433:5432 -e POSTGRES_PASSWORD=test pgvector/pgvector:pg16
    TEST_DATABASE_URL=postgresql://postgres:test@localhost:5433/postgres pytest
"""
import os
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from app.infrastructure.db import repository

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")

MIGRATION = Path(__file__).resolve().parents[1] / "app/infrastructure/db/migrations/001_init.sql"

JOBS = [
    ("J1", "Senior Python Backend Engineer", "Acme Payments", "Engineering", "Senior", "Kathmandu"),
    ("J2", "Frontend Developer", "Pixel Studio", "Engineering", "Mid", "Remote"),
    ("J3", "Data Analyst", "Numbers Co", "Data", "Junior", "Lalitpur"),
]
CHUNKS = [
    ("J1", 0, "Build FastAPI services and PostgreSQL schemas for payments."),
    ("J2", 0, "Ship React interfaces; some Python scripting is a plus."),
    ("J3", 0, "Build dashboards in SQL and Excel for the finance team."),
]


@pytest.fixture()
def conn():
    c = psycopg2.connect(URL)
    with c.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS job_chunks, jobs CASCADE")
        cur.execute(MIGRATION.read_text())
        cur.executemany(
            "INSERT INTO jobs (id, job_title, company_name, job_category, job_level, job_location)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            JOBS,
        )
        cur.executemany(
            "INSERT INTO job_chunks (job_id, chunk_index, chunk_text) VALUES (%s, %s, %s)",
            CHUNKS,
        )
    c.commit()
    yield c
    c.close()


def ids(rows):
    return [r["job_id"] for r in rows]


def test_title_match_outranks_description_mention(conn):
    rows = repository.keyword_search(conn, "python engineer")
    assert ids(rows)[0] == "J1"      # "Python" + "Engineer" in the title
    assert "J2" in ids(rows)         # "Python" only in the description
    assert "J3" not in ids(rows)


def test_terms_are_ored_not_anded(conn):
    # No single job contains all of these words.
    assert set(ids(repository.keyword_search(conn, "react dashboards"))) == {"J2", "J3"}


def test_metadata_fields_are_searchable(conn):
    assert ids(repository.keyword_search(conn, "kathmandu")) == ["J1"]


def test_whitelist_restricts_results(conn):
    assert ids(repository.keyword_search(conn, "python", job_id_whitelist=["J2"])) == ["J2"]
    assert repository.keyword_search(conn, "python", job_id_whitelist=[]) == []


def test_stopword_only_query_returns_nothing(conn):
    assert repository.keyword_search(conn, "the and of") == []


def test_migration_is_idempotent(conn):
    with conn.cursor() as cur:
        cur.execute(MIGRATION.read_text())
    conn.commit()
    assert ids(repository.keyword_search(conn, "kathmandu")) == ["J1"]
