"""Database connection utilities."""
from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


DEFAULT_DB_URL = "sqlite:///./sportsee.db"


@lru_cache(maxsize=1)
def get_engine(url: str | None = None):
    db_url = url or os.environ.get("DATABASE_URL", DEFAULT_DB_URL)
    return create_engine(db_url, future=True)


def get_session(url: str | None = None):
    engine = get_engine(url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)
