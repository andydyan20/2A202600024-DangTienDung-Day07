#!/usr/bin/env python3
"""
Utility to compare chunking strategies on a specific file.
Usage:
    python3 scripts/compare_strategies.py <file_path> [chunk_size]
"""

import sys
import re
from pathlib import Path

# Add project root to path so we can import src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.chunking import ChunkingStrategyComparator

def calculate_context_score(chunks: list[str]) -> float:
    """
    Heuristic: percentage of chunks that end with sentence-ending punctuation.
    """
    if not chunks:
        return 0.0
    
    clean_chunks = [c.strip() for c in chunks if c.strip()]
    if not clean_chunks:
        return 0.0
        
    sentence_endings = tuple(".!?")
    preserved_count = sum(1 for c in clean_chunks if c.endswith(sentence_endings))
    
    return (preserved_count / len(clean_chunks)) * 100

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/compare_strategies.py <file_path> [chunk_size]")
        return 1

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return 1

    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    text = file_path.read_text(encoding="utf-8")

    comparator = ChunkingStrategyComparator()
    results = comparator.compare(text, chunk_size=chunk_size)

    print(f"=== Chunking Comparison for: {file_path.name} ===")
    print(f"Original Length: {len(text)} characters")
    print(f"Requested Chunk Size: {chunk_size}\n")

    print(f"{'Strategy':<15} | {'Count':<6} | {'Avg Len':<8} | {'Context Score':<13}")
    print("-" * 55)
    for name, stats in results.items():
        score = calculate_context_score(stats['chunks'])
        print(f"{name:<15} | {stats['count']:<6} | {stats['avg_length']:<8.1f} | {score:>12.1f}%")

    print("\n--- Visualizing Context Preservation (Chunk Ends) ---")
    print("A '...' at the end of a preview line indicates where a chunk was split.\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
