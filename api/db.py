from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column, DateTime, String, Text, func, select, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL")

engine: Optional[Any] = None
AsyncSessionLocal: Optional[Any] = None

if DATABASE_URL:
    try:
        engine = create_async_engine(DATABASE_URL, future=True, echo=False)
        AsyncSessionLocal = async_sessionmaker(
            engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to create async engine from DATABASE_URL: %s", exc
        )


class DecisionLog(Base):
    __tablename__ = "decision_logs"

    # ``timestamp`` is part of the primary key so TimescaleDB can partition
    # ``decision_logs`` on it (a hypertable's unique indexes must include the
    # partitioning column).
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    timestamp = Column(
        DateTime(timezone=True),
        primary_key=True,
        default=func.now(),
        server_default=func.now(),
    )
    run_id = Column(String, index=True, nullable=False)
    source = Column(String, nullable=False, default="system")
    action = Column(String, nullable=False)
    instrument = Column(String, nullable=True)
    underlying = Column(String, nullable=True)
    direction = Column(String, nullable=True)
    payload = Column(JSON, nullable=True)
    risk_result = Column(JSON, nullable=True)
    order_result = Column(JSON, nullable=True)
    rationale = Column(Text, nullable=True)


async def init_db() -> None:
    """Create all tables and, if the DB is TimescaleDB, the hypertable.

    Safe to call when ``DATABASE_URL`` is unset or the DB is unreachable:
    warnings are logged and no exception is raised.
    """
    if engine is None:
        logger.warning("DATABASE_URL not set; skipping database initialization")
        return

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database initialization failed (create_all): %s", exc)
        return

    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "SELECT create_hypertable('decision_logs', 'timestamp', if_not_exists => TRUE)"
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not create TimescaleDB hypertable (expected if not TimescaleDB): %s",
            exc,
        )


async def log_decision(session: AsyncSession, **kwargs: Any) -> DecisionLog:
    """Persist a decision log entry and return the refreshed ORM object."""
    entry = DecisionLog(**kwargs)
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


async def get_decision(session: AsyncSession, run_id: str) -> Optional[DecisionLog]:
    """Fetch the most recent ``DecisionLog`` for a given ``run_id``."""
    result = await session.execute(
        select(DecisionLog)
        .where(DecisionLog.run_id == run_id)
        .order_by(DecisionLog.timestamp.desc())
    )
    return result.scalars().first()
