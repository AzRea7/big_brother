from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Alembic Config object
config = context.config

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---- IMPORTANT: import your models metadata ----
# Adjust import path if your package layout differs.
from app.db import Base  # noqa: E402
from app.config import settings  # noqa: E402

target_metadata = Base.metadata


def get_database_url() -> str:
    """
    Prefer the same DB URL the app uses.
    Fall back to alembic.ini sqlalchemy.url if not set.
    """
    # Your app uses settings; most likely something like settings.DATABASE_URL
    # If your setting name is different, change it here.
    url = getattr(settings, "DATABASE_URL", None) or getattr(settings, "DB_URL", None)
    if url:
        return url

    # fallback to config ini url
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
