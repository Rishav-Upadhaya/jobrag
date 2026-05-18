from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.pool import ThreadedConnectionPool

from app.core.config import settings


_POOL: ThreadedConnectionPool | None = None
_MIGRATION_PATH = Path(__file__).resolve().parent / "migrations" / "001_init.sql"


def _ensure_pool() -> ThreadedConnectionPool:
	global _POOL
	if _POOL is None:
		# Keepalives prevent "connection reset" errors after idle periods
		_POOL = ThreadedConnectionPool(
			minconn=1,
			maxconn=5,
			dsn=settings.DATABASE_URL,
			keepalives=1,
			keepalives_idle=30,
			keepalives_interval=10,
			keepalives_count=5,
		)
	return _POOL


def initialize_pool() -> None:
	_ensure_pool()


def initialize_schema() -> None:
	conn = get_connection()
	try:
		with _MIGRATION_PATH.open("r", encoding="utf-8") as migration_file:
			migration_sql = migration_file.read()
		with conn.cursor() as cursor:
			cursor.execute(migration_sql)
		conn.commit()
	finally:
		_ensure_pool().putconn(conn)


def close_pool() -> None:
	global _POOL
	if _POOL is not None:
		_POOL.closeall()
		_POOL = None


def _register_pgvector(conn: psycopg2.extensions.connection) -> None:
	register_vector(conn)


def _ensure_vector_extension(conn: psycopg2.extensions.connection) -> None:
	with conn.cursor() as cursor:
		cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
	conn.commit()


def get_connection() -> psycopg2.extensions.connection:
	conn = _ensure_pool().getconn()
	_ensure_vector_extension(conn)
	_register_pgvector(conn)
	return conn


@contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
	conn = get_connection()
	try:
		yield conn
	finally:
		_ensure_pool().putconn(conn)
