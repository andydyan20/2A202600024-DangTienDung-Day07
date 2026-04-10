from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            from chromadb.config import Settings

            # Initialize ephemeral client (in-memory persistent equivalent)
            client = chromadb.Client(Settings(allow_reset=True))
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id or f"chunk_{self._next_index}",
            "content": doc.content,
            "metadata": doc.metadata or {},
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records:
            return []
            
        query_vec = self._embedding_fn(query)
        scored_records = []
        
        for rec in records:
            # Sử dụng hàm compute_similarity đã implement ở bài trước
            score = compute_similarity(query_vec, rec["embedding"])
            scored_records.append((score, rec))
            
        # Sắp xếp giảm dần theo similarity score
        scored_records.sort(key=lambda x: x[0], reverse=True)
        
        # Trả về top_k kết quả, gắn kèm score vào record
        results = []
        for score, rec in scored_records[:top_k]:
            rec_with_score = rec.copy()
            rec_with_score["score"] = score
            results.append(rec_with_score)
            
        return results

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        if self._use_chroma and self._collection:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for doc in docs:
                idx = doc.id or f"id_{self._next_index}"
                self._next_index += 1
                
                ids.append(idx)
                documents.append(doc.content)
                embeddings.append(self._embedding_fn(doc.content))
                metadatas.append(doc.metadata or {})
                
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            # Fallback to in-memory
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)
                self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            # Reformat Chroma output to match our internal dict structure
            formatted = []
            for i in range(len(results['ids'][0])):
                formatted.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": results['distances'][0][i] # Chroma uses distance (lower is closer)
                })
            return formatted
        
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._use_chroma and self._collection:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if not metadata_filter:
            return self.search(query, top_k)

        if self._use_chroma and self._collection:
            query_embedding = self._embedding_fn(query)
            # ChromaDB uses MongoDB-style filter syntax
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter
            )
            # Formatting (similar to search)
            formatted = []
            for i in range(len(results['ids'][0])):
                formatted.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i]
                })
            return formatted

        # In-memory pre-filtering
        filtered_records = []
        for rec in self._store:
            match = True
            for key, value in metadata_filter.items():
                if rec["metadata"].get(key) != value:
                    match = False
                    break
            if match:
                filtered_records.append(rec)
        
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        initial_size = self.get_collection_size()
        
        if self._use_chroma and self._collection:
            self._collection.delete(where={"doc_id": doc_id})
            return self.get_collection_size() < initial_size
        
        # In-memory delete
        self._store = [
            rec for rec in self._store 
            if rec["id"] != doc_id and rec["metadata"].get("doc_id") != doc_id
        ]
        return len(self._store) < initial_size
