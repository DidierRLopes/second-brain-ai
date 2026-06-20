# Chunking Strategies for RAG

**Source:** https://weaviate.io/blog/chunking-strategies-for-rag
**Authors:** Weaviate Team

## Overview

Comprehensive guide to document chunking for RAG systems. Chunking strategy is one of the most impactful decisions in RAG pipeline design — it determines what the retriever can find and what context the LLM sees.

## Strategies Covered

- **Fixed-size chunking:** Simple, fast, but can split concepts mid-sentence
- **Recursive chunking:** Split by paragraphs, then sentences, then characters — respects natural boundaries
- **Document-based chunking:** Use document structure (headers, sections) as chunk boundaries
- **Semantic chunking:** Group sentences by embedding similarity — keeps related content together
- **Contextual chunking:** Add surrounding context (e.g., document title, section header) to each chunk

## Key Insight

There's no universally best strategy — the right choice depends on document type, query patterns, and embedding model. Semantic chunking preserves coherence best but requires more computation. Fixed-size is simplest and works surprisingly well with overlap.
