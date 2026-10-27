"""Koneksi SQLAlchemy dan inisialisasi database SQLite."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config import settings

connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False, "timeout": 30}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)

Base = declarative_base()


if engine.dialect.name == "sqlite":

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def _migrate_sqlite_schema(connection) -> None:  # noqa: ANN001
    """Tambahkan kolom baru ke database SQLite lama tanpa menghapus data.

    `create_all` tidak mengubah tabel yang sudah ada, sehingga instalasi lama
    akan gagal saat model memakai kolom baru. Migrasi ini idempotent.
    """
    additions: dict[str, dict[str, str]] = {
        "jobs": {
            "mode": "TEXT NOT NULL DEFAULT 'legacy'",
            "mock_ocr": "INTEGER NOT NULL DEFAULT 0",
            "mock_pipeline": "INTEGER NOT NULL DEFAULT 0",
            "imgsz": "INTEGER NOT NULL DEFAULT 640",
            "model_weights": "TEXT NOT NULL DEFAULT ''",
            "alpr_engine": "TEXT NOT NULL DEFAULT ''",
        },
        "violations": {
            "plate_source": "TEXT NOT NULL DEFAULT 'unreadable'",
            "plate_raw": "TEXT NOT NULL DEFAULT ''",
            "plate_format": "TEXT NOT NULL DEFAULT ''",
            "plate_region": "TEXT NOT NULL DEFAULT ''",
            "ocr_engine": "TEXT NOT NULL DEFAULT ''",
            "plate_crop_path": "TEXT",
            "vehicle_confidence": "REAL NOT NULL DEFAULT 0",
            "needs_verification": "INTEGER NOT NULL DEFAULT 0",
        },
    }
    for table, columns in additions.items():
        existing = {
            str(row[1])
            for row in connection.exec_driver_sql(f"PRAGMA table_info({table})")
        }
        if not existing:
            continue
        for name, ddl in columns.items():
            if name in existing:
                continue
            connection.exec_driver_sql(
                f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"
            )


def init_db() -> None:
    # Import di dalam fungsi agar model terdaftar sebelum create_all.
    import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    if engine.dialect.name == "sqlite":
        with engine.begin() as connection:
            _migrate_sqlite_schema(connection)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
