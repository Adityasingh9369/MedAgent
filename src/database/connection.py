from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from loguru import logger
from src.config import settings
from src.database.models import Base


engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,     # Verify connection health before using from pool
    pool_size=10,
    max_overflow=20,
    echo=False,             # Set True to log all SQL queries (debug only)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """
    Enable pgvector extension and create all tables.
    Safe to run multiple times — fully idempotent.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        logger.info("pgvector extension enabled")

    Base.metadata.create_all(bind=engine)
    logger.success("All database tables created")


@contextmanager
def get_db() -> Session:
    """
    Context manager for database sessions.
    Auto-commits on success, rolls back on any error.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error, rolling back: {e}")
        raise
    finally:
        db.close()
