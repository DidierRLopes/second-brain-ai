# KV Caching Explained

**Source:** https://huggingface.co/blog/not-lain/kv-caching
**Authors:** Hugging Face

## Overview

Comprehensive blog post explaining KV cache fundamentals. During autoregressive generation, the model recomputes key and value projections for all previous tokens at each step. The KV cache stores these intermediate states so they don't need to be recomputed, turning O(n²) per-token generation into O(n).

## Topics Covered

- Why KV caching is necessary for efficient generation
- How K and V matrices are stored and reused across decoding steps
- Memory growth patterns and why KV cache becomes the bottleneck for long sequences
- Relationship to attention variants (MHA, MQA, GQA) and their KV cache implications
