# Building A Generative AI Platform

**Source:** https://huyenchip.com/2024/07/25/genai-platform.html
**Author:** Chip Huyen

## Overview

Practical perspective on building production RAG and LLM systems. Emphasizes that data preparation and retrieval quality matter far more than infrastructure choices. Key advice: try keyword retrieval (BM25) before investing in expensive vector databases — it often works surprisingly well as a baseline.

## Key Takeaways

- Start simple: BM25 + reranker before vector search
- Data quality trumps model quality
- Context construction (chunking, retrieval) is the hardest and most impactful part
- Evaluation is critical and underinvested
- Production LLM systems need guardrails, caching, and routing
