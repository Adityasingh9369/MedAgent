"""
Semantic search smoke test.
Run after ingestion to verify everything is working:

    python scripts/test_search.py

Expected: top-3 relevant chunks per query with similarity scores.
Scores above 0.75 = strong semantic match.
Scores below 0.50 = ingest more articles on that topic.
"""
from src.database.connection import get_db
from src.database.models import Chunk
from src.ingestion.embedder import Embedder


def semantic_search(query: str, top_k: int = 3):
    embedder = Embedder()
    query_embedding = embedder.embed_query(query)

    with get_db() as db:
        # pgvector <=> operator = cosine distance
        # 0 = identical vectors, 2 = opposite vectors
        results = db.query(
            Chunk,
            Chunk.embedding.cosine_distance(query_embedding).label("distance")
        ).order_by("distance").limit(top_k).all()

        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        for i, (chunk, distance) in enumerate(results, 1):
            score = 1 - distance
            source = chunk.meta.get("source", "unknown") if chunk.meta else "unknown"
            print(f"\n  [{i}] Score: {score:.4f} | {chunk.doc_id} | {source}")
            print(f"      {chunk.content[:280]}...")


if __name__ == "__main__":
    queries = [
        "symptoms and diagnosis of heart failure",
        "first-line treatment for type 2 diabetes",
        "sepsis management intensive care unit",
        "antibiotic selection community acquired pneumonia",
    ]
    for q in queries:
        semantic_search(q)
