from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request body for POST /query"""
    question: str = Field(..., min_length=3, description="Clinical question to answer")


class ChunkResponse(BaseModel):
    content: str
    doc_id: str
    score: float
    source: str
    journal: Optional[str] = ""


class QueryResponse(BaseModel):
    """Full response for POST /query — mirrors the agent pipeline's final state"""
    question: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[ChunkResponse]
    entities: dict
    critique: str
    confidence_score: float
    is_reliable: bool
    final_response: str
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    database: bool
    redis: bool


class EvalMetric(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


class EvalResponse(BaseModel):
    phase: str
    metrics: EvalMetric
    notes: Optional[str] = ""