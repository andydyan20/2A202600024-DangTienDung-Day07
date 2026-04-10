#!/bin/bash
# Wrapper to run chunking strategy comparison

if [ -z "$1" ]; then
    echo "Usage: ./run_comparison.sh <file_path> [chunk_size]"
    exit 1
fi

python3 scripts/compare_strategies.py "$@"
