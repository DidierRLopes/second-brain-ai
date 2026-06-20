# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Source:** https://arxiv.org/abs/2005.11401
**Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (Meta AI / UCL)
**Published:** 2020

## Key Concepts

Seminal RAG paper combining generative models with retrieval-based knowledge access. Instead of relying solely on parametric knowledge, the model retrieves relevant documents at inference time and conditions generation on them.

## Why It Matters

- Foundation for the RAG paradigm used in virtually all production LLM systems
- Enables models to access current, domain-specific, or proprietary information
- Reduces hallucination by grounding generation in retrieved evidence
- Essential technique for building LLM agents that interact with knowledge bases
- Spawned many variants: Self-RAG, CRAG, Adaptive RAG, etc.
