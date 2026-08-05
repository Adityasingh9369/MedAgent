from fastapi import APIRouter
from sqlalchemy import text
from loguru import logger
import redis

from src.database.connection import get_db
from src.config import settings
from src.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Checks that Postgres and Redis are both reachable.
    Used by Docker/Render to confirm the service is actually ready,
    not just that the FastAPI process is running.
    """
    db_ok = False
    redis_ok = False

    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.error(f"Health check | database unreachable: {e}")

    try:
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        redis_ok = True
    except Exception as e:
        logger.error(f"Health check | redis unreachable: {e}")

    status = "ok" if (db_ok and redis_ok) else "degraded"

    return HealthResponse(status=status, database=db_ok, redis=redis_ok)