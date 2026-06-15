from sentence_transformers import SentenceTransformer
from typing import List
from loguru import logger
from tqdm import tqdm
from src.config import settings
from src.ingestion.chunker import TextChunk


class Embedder:
    """
    Generates dense vector embeddings using a biomedical-specific model.

    Model: NeuML/pubmedbert-base-embeddings
      - Trained on PubMed biomedical abstracts
      - 768-dimensional output vectors
      - Significantly outperforms general models on clinical text
      - Free, runs locally — zero API cost for embedding

    Embeddings are L2-normalized so cosine similarity == dot product,
    enabling faster pgvector ANN queries.
    """

    def __init__(self):
        logger.info(f"Loading: {settings.EMBEDDING_MODEL}")
        logger.info("First run downloads ~420MB model weights...")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.success(f"Embedder ready | dim={settings.EMBEDDING_DIMENSION}")

    def embed_chunks(
        self, chunks: List[TextChunk], batch_size: int = 32
    ) -> List[List[float]]:
        """Embed list of chunks in batches. Returns list of 768-dim vectors."""
        texts = [chunk.content for chunk in chunks]
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string for retrieval."""
        return self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()
