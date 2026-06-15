from loguru import logger
from tqdm import tqdm
from typing import List

from src.config import settings
from src.database.connection import get_db, init_db
from src.database.models import Document, Chunk as ChunkModel
from src.ingestion.pubmed_fetcher import PubMedFetcher, PubMedArticle
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import Embedder


DEFAULT_QUERIES = [
    "heart failure diagnosis management clinical guidelines",
    "type 2 diabetes mellitus treatment evidence based",
    "sepsis diagnosis criteria treatment protocol",
    "community acquired pneumonia antibiotic treatment",
    "hypertension cardiovascular risk reduction",
    "chronic kidney disease management progression",
    "acute myocardial infarction treatment outcomes",
    "stroke diagnosis thrombolysis management",
]


class IngestionPipeline:
    """
    Orchestrates Phase 1 data ingestion:
    PubMed API -> PDF Parser -> Chunker -> Embedder -> PostgreSQL + pgvector
    """

    def __init__(self):
        self.fetcher = PubMedFetcher()
        self.parser = PDFParser()
        self.chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.embedder = Embedder()

    def _exists(self, db, doc_id: str) -> bool:
        return db.query(Document).filter(
            Document.doc_id == doc_id
        ).first() is not None

    def _ingest(self, db, doc_id, source, title, content,
                meta, chunk_meta, **doc_fields):
        """Save document + chunk + embed + store. Core ingestion unit."""
        doc = Document(
            source=source, doc_id=doc_id, title=title,
            full_text=content, meta=meta, **doc_fields
        )
        db.add(doc)
        db.flush()  # Get auto-generated ID without committing

        chunks = self.chunker.chunk_text(content, doc_id, chunk_meta)
        if not chunks:
            return

        embeddings = self.embedder.embed_chunks(chunks)

        for chunk, embedding in zip(chunks, embeddings):
            db.add(ChunkModel(
                document_id=doc.id,
                doc_id=doc_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=embedding,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
            ))

        logger.debug(f"Ingested {len(chunks)} chunks for {doc_id}")

    def run_pubmed(self, queries: List[str], max_results: int = 50):
        init_db()
        logger.info("=" * 55)
        logger.info("PUBMED INGESTION STARTED")
        logger.info("=" * 55)

        all_articles = []
        for q in queries:
            all_articles.extend(self.fetcher.search_and_fetch(q, max_results))

        # Deduplicate by PMID
        seen, unique = set(), []
        for a in all_articles:
            if a.pmid not in seen:
                seen.add(a.pmid)
                unique.append(a)

        logger.info(f"Unique articles to process: {len(unique)}")
        skipped = 0

        with get_db() as db:
            for article in tqdm(unique, desc="PubMed ingestion"):
                doc_id = f"pubmed_{article.pmid}"
                if self._exists(db, doc_id):
                    skipped += 1
                    continue
                self._ingest(
                    db=db, doc_id=doc_id, source="pubmed",
                    title=article.title,
                    content=f"{article.title}\n\n{article.abstract}",
                    meta={"keywords": article.keywords, "pmid": article.pmid},
                    chunk_meta={"source": "pubmed", "pmid": article.pmid,
                                "journal": article.journal},
                    authors=article.authors,
                    abstract=article.abstract,
                    publication_date=article.publication_date,
                    journal=article.journal,
                    doi=article.doi,
                )

        logger.success(f"PubMed done. Skipped {skipped} existing.")

    def run_pdf(self):
        logger.info("=" * 55)
        logger.info("PDF INGESTION STARTED")
        logger.info("=" * 55)

        documents = self.parser.parse_directory()
        skipped = 0

        with get_db() as db:
            for doc in tqdm(documents, desc="PDF ingestion"):
                if self._exists(db, doc.doc_id):
                    skipped += 1
                    continue
                self._ingest(
                    db=db, doc_id=doc.doc_id, source="pdf",
                    title=doc.title, content=doc.content,
                    meta=doc.metadata,
                    chunk_meta={"source": "pdf",
                                "filename": doc.metadata.get("filename", "")},
                )

        logger.success(f"PDF done. Skipped {skipped} existing.")

    def run(self, queries: List[str] = None):
        """Full pipeline: init DB -> PubMed -> PDFs."""
        logger.info("Initializing database...")
        init_db()
        self.run_pubmed(queries or DEFAULT_QUERIES)
        self.run_pdf()
        logger.success("Full ingestion pipeline complete!")
