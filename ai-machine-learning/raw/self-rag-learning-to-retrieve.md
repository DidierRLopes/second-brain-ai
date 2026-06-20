# Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

**Source:** https://arxiv.org/abs/2310.11511
**Authors:** Akari Asai, Zeqiu Wu, et al. (University of Washington, Meta)
**Published:** 2023

## Key Concepts

Self-RAG trains models to adaptively decide when retrieval is needed and generate special "reflection tokens" to self-assess the quality and relevance of retrieved passages. The model critiques its own generation, deciding whether retrieved content supports its claims.

7B and 13B Self-RAG models outperform ChatGPT on reasoning, open-domain QA, and fact verification tasks. The key insight: not every query needs retrieval, and the model should learn to judge retrieval quality.
