#!/usr/bin/env python3
"""
Evaluation script for RAG retrieval.
Calculates Hit Rate @ k and MRR (Mean Reciprocal Rank).
Usage:
    python3 scripts/evaluate_retrieval.py [top_k]
"""

import sys
import json
from pathlib import Path

# Add project root to path so we can import src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models import Document
from src.store import EmbeddingStore
from src.chunking import SentenceChunker
from src.embeddings import _mock_embed
from main import load_documents_from_files

def evaluate():
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    
    # Paths
    eval_file = project_root / "scripts" / "golden_eval.json"
    incoterm_file = project_root / "data" / "incoterm.md"
    
    if not eval_file.exists():
        print(f"Error: Golden eval file not found at {eval_file}")
        return
        
    with open(eval_file, "r") as f:
        golden_data = json.load(f)
        
    # Setup Store
    print("Initializing Store and processing incoterm.md...")
    docs = load_documents_from_files([str(incoterm_file)])
    chunker = SentenceChunker(max_sentences_per_chunk=3)
    store = EmbeddingStore(collection_name="eval_store", embedding_fn=_mock_embed)
    
    # Manual chunking (since we reverted internal chunking in store)
    chunked_docs = []
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for i, chunk_text in enumerate(chunks):
            chunked_docs.append(
                Document(
                    id=f"{doc.id}_chunk_{i}",
                    content=chunk_text,
                    metadata={**doc.metadata, "doc_id": doc.id}
                )
            )
    store.add_documents(chunked_docs)
    print(f"Store ready with {store.get_collection_size()} chunks.\n")
    
    # Run Evaluation
    hits = 0
    mrr_sum = 0.0
    total = len(golden_data)
    
    print(f"{'Question':<50} | {'Hit?':<5} | {'Rank':<4} | {'Recip Rank'}")
    print("-" * 80)
    
    for item in golden_data:
        question = item["question"]
        expected_keywords = item["expected_keywords"]
        expected_source = item["expected_source"]
        
        results = store.search(question, top_k=top_k)
        
        found_rank = 0
        for i, res in enumerate(results):
            content = res["content"].lower()
            # Simple heuristic: check if at least 2 keywords match or source matches
            keyword_matches = sum(1 for k in expected_keywords if k.lower() in content)
            
            # If we match enough keywords AND the source matches
            if (keyword_matches >= 1) and (res["metadata"].get("source", "").endswith(expected_source)):
                found_rank = i + 1
                break
        
        hit = 1 if found_rank > 0 else 0
        reciprocal_rank = 1.0 / found_rank if found_rank > 0 else 0.0
        
        hits += hit
        mrr_sum += reciprocal_rank
        
        q_preview = (question[:47] + '..') if len(question) > 47 else question
        print(f"{q_preview:<50} | {'YES' if hit else 'NO':<5} | {found_rank if found_rank else '-':<4} | {reciprocal_rank:.3f}")
        
    hit_rate = (hits / total) * 100
    mrr = mrr_sum / total
    
    # Calculate a score out of 10
    # Formula: (HitRate * 0.4 + MRR * 10 * 0.6)
    score = (hit_rate / 10) * 0.4 + (mrr * 6)
    
    print("-" * 80)
    print(f"Total Questions: {total}")
    print(f"Hit Rate @ {top_k}: {hit_rate:.1f}%")
    print(f"MRR: {mrr:.3f}")
    print(f"\nFINAL RETRIEVAL SCORE: {score:.1f} / 10")
    
if __name__ == "__main__":
    evaluate()
