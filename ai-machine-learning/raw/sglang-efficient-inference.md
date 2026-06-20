# SGLang: Efficient Execution of Structured Language Model Programs

**Source:** https://arxiv.org/abs/2312.07104
**Authors:** Lianmin Zheng et al.
**Published:** 2023

## Key Concepts

SGLang is an inference framework introducing RadixAttention for KV cache reuse across requests sharing common prefixes, and compressed finite state machines for structured output decoding. Achieves up to 6.4× higher throughput on complex LLM programs (multi-turn chat, RAG, reasoning chains).

## Key Innovations

- **RadixAttention:** Stores KV cache in a radix tree, enabling automatic prefix sharing across requests
- **Compressed FSM:** Efficient constrained decoding for JSON, regex, or grammar-guided generation
- Strong alternative to vLLM for workloads with shared prefixes (e.g., system prompts, few-shot examples)
