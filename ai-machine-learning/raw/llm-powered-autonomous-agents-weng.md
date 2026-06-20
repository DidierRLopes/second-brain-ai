# LLM Powered Autonomous Agents

**Source:** https://lilianweng.github.io/posts/2023-06-23-agent/
**Author:** Lilian Weng (OpenAI)
**Published:** June 2023

## Overview

Influential blog post conceptualizing the modern LLM agent architecture with the formula:

**Agent = LLM + Memory + Planning + Tool Use**

Breaks down key components including planning/decomposition, self-reflection, memory management (short-term working memory + long-term vector store), and tool integration.

## Key Framework

- **Planning:** Task decomposition (CoT, Tree of Thoughts), self-reflection (ReAct, Reflexion)
- **Memory:** Short-term (in-context), long-term (external vector DB with retrieval)
- **Tool Use:** API calls, code execution, search, calculators
- **Challenges:** finite context length, long-term planning difficulties, reliability of natural language interfaces

## Why It Matters

- Canonical reference for understanding LLM agent architecture
- Synthesizes dozens of papers into a coherent framework
- One of the most-cited blog posts in the AI agent space
