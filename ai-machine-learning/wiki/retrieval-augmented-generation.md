# Retrieval-Augmented Generation (RAG)

RAG combines a language model with a retrieval system, allowing the model to access external knowledge at inference time rather than relying solely on what it memorized during training. This is the standard pattern for building LLM applications that need accurate, up-to-date, or domain-specific information.

## How It Works

The basic RAG pipeline:

1. **Index:** Chunk your documents, embed each chunk using an embedding model, store in a vector database
2. **Retrieve:** When a query comes in, embed it and find the most similar document chunks via vector similarity search
3. **Generate:** Pass the retrieved chunks as context to the LLM along with the query, and generate an answer grounded in the evidence

## Why RAG Matters

LLMs have three fundamental limitations that RAG addresses:

**Knowledge cutoff:** Models only know what was in their training data. RAG provides access to current information.

**Hallucination:** Models can confidently generate false information. RAG grounds generation in retrieved evidence, reducing (but not eliminating) hallucination.

**Domain specificity:** No model knows your company's internal documents. RAG makes proprietary or specialized data accessible without fine-tuning.

## The Original Paper

Lewis et al. (2020, Meta/UCL) introduced RAG as a general-purpose architecture combining a pre-trained retriever (DPR) with a pre-trained generator (BART). They showed it outperformed purely parametric models on knowledge-intensive tasks like open-domain QA.

## Chunking: The Critical Decision

How you split documents into chunks determines what your retriever can find. There is no universally best strategy — the right choice depends on document type, query patterns, and embedding model:

**Fixed-size chunking:** Simple and fast, but can split concepts mid-sentence. Works surprisingly well with overlap (e.g., 512 tokens with 50-token overlap).

**Recursive chunking:** Splits by paragraphs, then sentences, then characters — respects natural boundaries. The default in most frameworks.

**Semantic chunking:** Groups sentences by [[embeddings|embedding]] similarity, keeping related content together. Higher quality but more expensive.

**Contextual chunking:** Adds surrounding context (document title, section header) to each chunk, improving retrieval accuracy for out-of-context queries.

## Advanced RAG Variants

**Self-RAG** (Asai et al., 2023): The model generates "reflection tokens" to decide when retrieval is needed and assess whether retrieved content supports its claims. 7B Self-RAG outperforms ChatGPT on reasoning and fact verification.

**CRAG (Corrective RAG)** (Yan et al., 2024): A lightweight evaluator assesses retrieval quality. If documents are poor, it falls back to web search. Plug-and-play improvement for any RAG system — addresses the failure mode where bad retrieval produces confidently wrong answers.

**GraphRAG** (Microsoft, 2024): Builds a knowledge graph from documents, detects community hierarchies, and generates structured summaries. Superior for "global" queries where information is spread across many documents (e.g., "What are the main themes?") — exactly where chunk-based RAG struggles.

**Adaptive RAG:** Routes queries to different strategies (no retrieval, single-step, or iterative) based on complexity, reducing compute for simple queries.

## Production Guidance

Chip Huyen's practical advice: start with keyword retrieval (BM25) before investing in vector databases — it works surprisingly well as a baseline. Data quality and chunking strategy matter more than infrastructure choices. Evaluation is critical and underinvested in most RAG systems.

## RAG vs Fine-Tuning

RAG and fine-tuning serve different purposes. RAG is best for providing access to knowledge (especially changing or domain-specific knowledge). Fine-tuning is best for teaching behaviors, formats, or styles. Many production systems use both: fine-tune for the desired behavior, RAG for the knowledge.

## Related Topics
- [[embeddings]] — The foundation of the retrieval step
- [[kv-cache]] — Long retrieved contexts impact KV cache memory
- [[llm-agents]] — RAG serves as the "long-term memory" component of agent architectures
- [[parameter-efficient-fine-tuning]] — Complementary to RAG for different purposes
- [[inference-optimization]] — RAG adds retrieval latency that must be optimized

## Sources
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (arxiv:2005.11401)
- Self-RAG (arxiv:2310.11511)
- CRAG (arxiv:2401.15884)
- GraphRAG — Microsoft Research (arxiv:2501.00309)
- Chunking Strategies for RAG — Weaviate
- Building A Generative AI Platform — Chip Huyen
- LLM Powered Autonomous Agents — Lilian Weng
