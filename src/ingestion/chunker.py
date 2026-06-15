import re
import tiktoken
from typing import List, Dict, Any
from pydantic import BaseModel
from loguru import logger


class TextChunk(BaseModel):
    doc_id: str
    chunk_index: int
    content: str
    token_count: int
    metadata: Dict[str, Any] = {}


class TextChunker:
    """
    Splits documents into overlapping token-based chunks.

    Why token-based (not character-based)?
      Embedding models have token limits, not character limits.
      Token counting gives accurate, consistent chunk sizes.

    Why overlap?
      Prevents losing context at chunk boundaries.
      A sentence spanning a boundary appears in both chunks,
      improving retrieval of relevant passages.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # cl100k_base = same tokenizer as GPT-4, good proxy for most models
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split on sentence boundaries.
        Only splits when next word starts with capital letter —
        avoids splitting on medical abbreviations like 'Dr.' or 'mg.'
        """
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = {}
    ) -> List[TextChunk]:
        sentences = self._split_sentences(text)
        chunks = []
        current = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            s_tokens = self._count_tokens(sentence)

            # Flush when adding this sentence exceeds the limit
            if current_tokens + s_tokens > self.chunk_size and current:
                chunks.append(TextChunk(
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    content=" ".join(current),
                    token_count=current_tokens,
                    metadata=metadata
                ))
                chunk_index += 1

                # Carry forward last N tokens as overlap
                overlap, overlap_tokens = [], 0
                for s in reversed(current):
                    t = self._count_tokens(s)
                    if overlap_tokens + t <= self.chunk_overlap:
                        overlap.insert(0, s)
                        overlap_tokens += t
                    else:
                        break

                current = overlap
                current_tokens = overlap_tokens

            current.append(sentence)
            current_tokens += s_tokens

        # Final chunk
        if current:
            chunks.append(TextChunk(
                doc_id=doc_id,
                chunk_index=chunk_index,
                content=" ".join(current),
                token_count=current_tokens,
                metadata=metadata
            ))

        logger.debug(f"Chunked '{doc_id}' -> {len(chunks)} chunks")
        return chunks
