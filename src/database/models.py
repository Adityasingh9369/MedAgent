from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Document(Base):
    """
    Stores metadata for each source document.
    Relationship: one Document -> many Chunks.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)         # 'pubmed' or 'pdf'
    doc_id = Column(String(100), unique=True, nullable=False)  # e.g. 'pubmed_12345678'
    title = Column(Text)
    authors = Column(JSON)                              # List of author name strings
    abstract = Column(Text)
    publication_date = Column(String(20))
    journal = Column(String(500))
    doi = Column(String(200))
    full_text = Column(Text)
    meta = Column(JSON)                             # keywords, pmid, filename etc.
    created_at = Column(DateTime, default=datetime.utcnow)


class Chunk(Base):
    """
    Stores individual text chunks with 768-dim vector embeddings.
    This is the primary table queried during semantic search.
    """
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, nullable=False)       # FK -> documents.id
    doc_id = Column(String(100), nullable=False)        # Denormalized for fast lookup
    chunk_index = Column(Integer, nullable=False)       # Position within document
    content = Column(Text, nullable=False)              # The actual text
    embedding = Column(Vector(768))                     # Biomedical embedding vector
    token_count = Column(Integer)
    meta = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
