# Corrective Retrieval Augmented Generation (CRAG)

**Source:** https://arxiv.org/abs/2401.15884
**Authors:** Shi-Qi Yan, et al.
**Published:** 2024

## Key Concepts

CRAG introduces a lightweight retrieval evaluator that assesses document quality and triggers adaptive retrieval actions. If retrieved documents are low quality, it falls back to web search. A decompose-then-recompose algorithm filters irrelevant information from retrieved passages.

CRAG is plug-and-play — it can improve any existing RAG system without modifying the underlying models. Addresses a key failure mode: when the retriever returns irrelevant documents, standard RAG generates confidently wrong answers.
