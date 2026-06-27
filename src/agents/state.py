from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    """
    Shared state passed between all agents in the pipeline.
    Each agent reads from it and adds its own output.
    """
    # Input
    question: str

    # Retrieval Agent output
    retrieved_chunks: List[dict]

    # NER Agent output
    entities: dict  # {diseases: [], drugs: [], symptoms: []}

    # Reasoning Agent output
    answer: str
    citations: List[str]

    # Critique Agent output
    critique: str
    confidence_score: float
    is_reliable: bool

    # Final response
    final_response: str