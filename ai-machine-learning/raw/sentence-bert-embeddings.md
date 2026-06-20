# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

**Source:** https://arxiv.org/abs/1908.10084
**Authors:** Nils Reimers, Iryna Gurevych (2019)

## Key Concepts

SBERT addresses BERT's computational inefficiency for semantic similarity by using siamese and triplet network structures with mean pooling. Reduces 65-hour pairwise similarity search (on 10K sentences) to ~5 seconds while maintaining accuracy. Foundational for sentence-level embeddings and the Sentence Transformers library.

## Impact

- Created the Sentence Transformers library (most-used embedding library)
- Enabled practical semantic search, clustering, and similarity at scale
- Foundation for all subsequent sentence embedding work
