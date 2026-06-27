from langgraph.graph import StateGraph, END
from loguru import logger
from src.agents.state import AgentState
from src.agents.retrieval_agent import retrieval_agent
from src.agents.ner_agent import ner_agent
from src.agents.reasoning_agent import reasoning_agent
from src.agents.critique_agent import critique_agent


def build_pipeline() -> StateGraph:
    """
    Builds the LangGraph multi-agent pipeline.

    Flow:
    retrieval_agent -> ner_agent -> reasoning_agent -> critique_agent -> END

    Each agent receives the full state, adds its output,
    and passes the enriched state to the next agent.
    """
    graph = StateGraph(AgentState)

    # Register all agents as nodes
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("ner", ner_agent)
    graph.add_node("reasoning", reasoning_agent)
    graph.add_node("critique", critique_agent)

    # Define the flow — linear pipeline
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "ner")
    graph.add_edge("ner", "reasoning")
    graph.add_edge("reasoning", "critique")
    graph.add_edge("critique", END)

    return graph.compile()


def run_pipeline(question: str) -> dict:
    """
    Run the full multi-agent pipeline for a clinical question.
    Returns the complete state including answer, citations,
    critique, and confidence score.
    """
    logger.info("=" * 55)
    logger.info(f"PIPELINE START | {question[:60]}...")
    logger.info("=" * 55)

    pipeline = build_pipeline()

    # Initial state — only question is set, agents fill the rest
    initial_state = {
        "question": question,
        "retrieved_chunks": [],
        "entities": {},
        "answer": "",
        "citations": [],
        "critique": "",
        "confidence_score": 0.0,
        "is_reliable": False,
        "final_response": "",
    }

    final_state = pipeline.invoke(initial_state)

    logger.info("=" * 55)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Confidence: {final_state['confidence_score']}")
    logger.info(f"Reliable: {final_state['is_reliable']}")
    logger.info("=" * 55)

    return final_state