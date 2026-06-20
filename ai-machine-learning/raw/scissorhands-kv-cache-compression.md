# Scissorhands: KV Cache Compression via Persistence of Importance

**Source:** https://arxiv.org/abs/2305.17118
**Authors:** Dang et al.
**Published:** 2023

## Key Concepts

Scissorhands exploits the "persistence of importance" hypothesis: tokens that are important for attention at one step tend to remain important at future steps. It dynamically evicts unimportant tokens from the KV cache at test time, achieving 5× compression without fine-tuning and up to 20× when combined with 4-bit quantization.

## Key Insight

Not all tokens in the KV cache are equally important. By tracking which tokens consistently receive high attention scores, you can safely evict the rest, dramatically reducing memory usage during generation.
