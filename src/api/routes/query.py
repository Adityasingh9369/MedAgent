from fastapi import APIRouter, HTTPException
from loguru import logger
import hashlib
import json
import redis

from src.config import settings
from src.agents.pipeline import run_pipeline
from src.api.schemas import QueryRequest, QueryResponse

router = APIRouter()

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

CACHE_TTL_SECONDS = 3600  # 1 hour


def _cache_key(question: str) -> str:
    normalized = question.strip().lower()
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return f"clinicalagent:query:{digest}"


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Runs a clinical question through the full 4-agent pipeline
    (Retrieval -> NER -> Reasoning -> Critique).
    Caches responses in Redis to avoid re-running expensive LLM calls
    for repeated questions.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    cache_key = _cache_key(question)

    try:
        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Query | cache HIT | {question[:60]}...")
            result = json.loads(cached)
            result["cached"] = True
            return QueryResponse(**result)
    except Exception as e:
        logger.warning(f"Query | redis read failed, proceeding without cache: {e}")

    logger.info(f"Query | cache MISS | {question[:60]}...")

    try:
        result = run_pipeline(question)
    except Exception as e:
        logger.error(f"Query | pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    response_data = {
        "question": result["question"],
        "answer": result["answer"],
        "citations": result["citations"],
        "retrieved_chunks": result["retrieved_chunks"],
        "entities": result["entities"],
        "critique": result["critique"],
        "confidence_score": result["confidence_score"],
        "is_reliable": result["is_reliable"],
        "final_response": result["final_response"],
        "cached": False,
    }

    try:
        redis_client.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(response_data))
    except Exception as e:
        logger.warning(f"Query | redis write failed, response not cached: {e}")

    return QueryResponse(**response_data)