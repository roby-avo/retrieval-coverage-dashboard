from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from .env import load_dotenv


load_dotenv()


def _database_conninfo() -> str:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if database_url:
        return database_url

    database = os.environ.get("POSTGRES_DB", "coverage_dashboard").strip()
    user = os.environ.get("POSTGRES_USER", "coverage").strip()
    password = os.environ.get("POSTGRES_PASSWORD", "").strip()
    host = os.environ.get("POSTGRES_HOST", "localhost").strip()
    port = os.environ.get("POSTGRES_PORT", "5432").strip()

    kwargs = {"dbname": database, "user": user, "host": host, "port": port}
    if password:
        kwargs["password"] = password
    return make_conninfo("", **kwargs)


DATABASE_URL = _database_conninfo()
LEGACY_DATABASE_URL = os.environ.get("LEGACY_DATABASE_URL", "").strip()


def connect() -> psycopg.Connection[Any]:
    try:
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)
    except psycopg.OperationalError as exc:
        if not LEGACY_DATABASE_URL or "no password supplied" not in str(exc).casefold():
            raise
        return psycopg.connect(LEGACY_DATABASE_URL, row_factory=dict_row)


def _database_missing_error(exc: BaseException) -> bool:
    return "does not exist" in str(exc).casefold() and "database" in str(exc).casefold()


def ensure_database_exists() -> None:
    try:
        with connect():
            return
    except psycopg.OperationalError as exc:
        if not _database_missing_error(exc):
            raise

    conninfo = conninfo_to_dict(DATABASE_URL)
    database_name = conninfo.get("dbname")
    if not database_name:
        raise RuntimeError("DATABASE_URL does not include a database name")

    admin_conninfo = make_conninfo(DATABASE_URL, dbname=os.environ.get("POSTGRES_ADMIN_DB", "postgres"))
    try:
        with psycopg.connect(admin_conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
                if cur.fetchone():
                    return
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
    except psycopg.OperationalError:
        admin_conninfo = make_conninfo(DATABASE_URL, dbname="template1")
        with psycopg.connect(admin_conninfo, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
                if not cur.fetchone():
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))


def init_database() -> None:
    ensure_database_exists()
    schema_path = Path(__file__).with_name("schema.sql")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_path.read_text())
        conn.commit()
