# backend/app/db.py
from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


def _ensure_sqlite_dir(db_url: str) -> None:
    if db_url.startswith("sqlite:///"):
        path = db_url.replace("sqlite:///", "")
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)


_ensure_sqlite_dir(settings.DB_URL)

engine = create_engine(
    settings.DB_URL,
    connect_args={"check_same_thread": False} if settings.DB_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Creates tables + runs lightweight schema upgrades (SQLite-friendly).
    """
    from . import models  # noqa: F401
    from .migrations import ensure_schema

    Base.metadata.create_all(bind=engine)
    ensure_schema(engine)
