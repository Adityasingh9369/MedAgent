from loguru import logger
from src.database.connection import get_db
from src.database.models import Chunk
from src.ingestion.embedder import Embedder
from src.agents.state import AgentState


embedder = Embedder()


def retrieval_agent(state: AgentState) -> AgentState:
    """
    Retrieval Agent — searches pgvector for relevant chunks.
    Takes the user question, embeds it, runs cosine similarity
    search, returns top 5 most relevant chunks.
    """
    question = state["question"]
    logger.info(f"Retrieval Agent | query: {question[:60]}...")

    query_embedding = embedder.embed_query(question)

    with get_db() as db:
        results = db.query(
            Chunk,
            Chunk.embedding.cosine_distance(query_embedding).label("distance")
        ).order_by("distance").limit(5).all()

        retrieved_chunks = []
        for chunk, distance in results:
            retrieved_chunks.append({
                "content": chunk.content,
                "doc_id": chunk.doc_id,
                "score": round(1 - distance, 4),
                "source": chunk.meta.get("source", "unknown") if chunk.meta else "unknown",
                "journal": chunk.meta.get("journal", "") if chunk.meta else "",
            })

    logger.info(f"Retrieval Agent | found {len(retrieved_chunks)} chunks")
    logger.info(f"Top score: {retrieved_chunks[0]['score'] if retrieved_chunks else 0}")

    return {**state, "retrieved_chunks": retrieved_chunks}