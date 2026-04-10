from __future__ import annotations
# from fastembed import TextEmbedding

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = [sub.strip() for sub in re.split(sentence_endings, text) if sub.strip()]

        chunks = []
        for index in range(0, len(sentences), self.max_sentences_per_chunk):
            batch = sentences[index: index + self.max_sentences_per_chunk]
            chunks.append(" ".join(batch))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        if not remaining_separators:
            return [current_text[i:i+self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]
        
        if sep == "":
            parts = list(current_text)
        else:
            parts = current_text.split(sep)
            
        final_chunks = []
        buffer = []
        buffer_len = 0
        
        for part in parts:
            part_len = len(part) + (len(sep) if buffer else 0)
            
            if buffer_len + part_len <= self.chunk_size:
                buffer.append(part)
                buffer_len += part_len
            else:
                if buffer:
                    joined = sep.join(buffer)
                    final_chunks.extend(self._split(joined, next_seps))
                
                buffer = [part]
                buffer_len = len(part)
                
        if buffer:
            joined = sep.join(buffer)
            final_chunks.extend(self._split(joined, next_seps))
            
        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula
    dot_prod = _dot(vec_a, vec_b)
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
        
    return dot_prod / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # engine = TextEmbedding()
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=20),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=2),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
            # "custom": HybridSemanticChunker(embedding_fn=engine.embed, chunk_size=chunk_size)
        }
        
        results = {}
        for name, strategy in strategies.items():
            chunks = strategy.chunk(text)
            results[name] = {
                "count": len(chunks),
                "avg_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                "chunks": chunks
            }
            
        return results

#My custom chunker
class HybridSemanticChunker:
    """
    Kết hợp Recursive và Semantic:
    1. Chia văn bản thành các đoạn lớn dựa trên cấu trúc (Paragraphs).
    2. Nếu đoạn văn vẫn vượt quá chunk_size, sử dụng Semantic Similarity để tìm 
       điểm cắt tự nhiên thay vì cắt cứng theo số ký tự.
    """
    def __init__(
        self, 
        embedding_fn: Callable, 
        chunk_size: int = 600, 
        threshold: float = 0.82,
        separators: list[str] = ["\n\n", "\n"]
    ):
        self.embedding_fn = embedding_fn
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.separators = separators

    def chunk(self, text: str) -> list[str]:
        initial_chunks = self._recursive_split(text, self.separators)
        
        final_chunks = []
        for chunk in initial_chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                semantic_sub_chunks = self._semantic_split(chunk)
                final_chunks.extend(semantic_sub_chunks)
        
        return final_chunks

    def _semantic_split(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2:
            return [text]

        embeddings = self.embedding_fn(sentences)
        chunks = []
        buffer = [sentences[0]]
        
        for i in range(len(sentences) - 1):
            sim = compute_similarity(embeddings[i], embeddings[i+1])
            if sim < self.threshold or len(" ".join(buffer)) > self.chunk_size:
                chunks.append(" ".join(buffer))
                buffer = [sentences[i+1]]
            else:
                buffer.append(sentences[i+1])
        
        chunks.append(" ".join(buffer))
        return chunks

    def _recursive_split(self, text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text]
        sep = seps[0]
        parts = [p for p in text.split(sep) if p.strip()]
        return parts 