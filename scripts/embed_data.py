#!/usr/bin/env python3
"""
Script to scan the data/ folder and embed all .txt and .md files.
Usage:
    python3 scripts/embed_data.py
"""

import os
import sys
from pathlib import Path

# Add project root to path so we can import src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models import Document
from src.store import EmbeddingStore
from src.chunking import SentenceChunker
from src.embeddings import _mock_embed
# Import helper from main.py if available, else define locally
try:
    from main import load_documents_from_files
except ImportError:
    def load_documents_from_files(file_paths: list[str]) -> list[Document]:
        documents: list[Document] = []
        for raw_path in file_paths:
            path = Path(raw_path)
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            if not path.exists() or not path.is_file():
                continue
            content = path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    id=path.stem,
                    content=content,
                    metadata={"source": str(path), "extension": path.suffix.lower()},
                )
            )
        return documents

def main():
    data_dir = project_root / "data"
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return 1

    # Scan for files
    all_files = [str(f) for f in data_dir.glob("*") if f.suffix.lower() in {".txt", ".md"}]
    print(f"Found {len(all_files)} files in {data_dir}")

    # Load documents
    docs = load_documents_from_files(all_files)
    print(f"Loaded {len(docs)} documents")

    # Initialize chunker
    chunker = SentenceChunker(max_sentences_per_chunk=3)
    chunked_docs = []
    
    print("\nChunking documents...")
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for i, chunk_text in enumerate(chunks):
            chunked_docs.append(
                Document(
                    id=f"{doc.id}_chunk_{i}",
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "doc_id": doc.id,
                        "chunk_index": i
                    }
                )
            )
    print(f"Split {len(docs)} documents into {len(chunked_docs)} chunks")

    # Initialize store and add chunks
    store = EmbeddingStore(collection_name="data_embedding_test", embedding_fn=_mock_embed)
    store.add_documents(chunked_docs)

    print(f"\nStored {store.get_collection_size()} chunks in EmbeddingStore")
    print("Embedding complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
