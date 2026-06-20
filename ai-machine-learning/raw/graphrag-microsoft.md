# GraphRAG: Retrieval-Augmented Generation with Knowledge Graphs

**Source:** https://arxiv.org/abs/2501.00309 / https://github.com/microsoft/graphrag
**Authors:** Microsoft Research
**Published:** 2024-2025

## Key Concepts

GraphRAG builds a knowledge graph from source documents, organizes entities into community hierarchies, and generates community summaries. For queries, it uses these structured summaries rather than raw text chunks, enabling global reasoning over entire document collections.

Superior to standard vector-search RAG for "global" questions (e.g., "What are the main themes across all documents?") where traditional chunk-based retrieval struggles because relevant information is spread across many documents.

## Architecture

1. Extract entities and relationships from text → knowledge graph
2. Detect communities via graph algorithms (e.g., Leiden)
3. Generate hierarchical community summaries
4. At query time: route to relevant communities, synthesize from summaries
