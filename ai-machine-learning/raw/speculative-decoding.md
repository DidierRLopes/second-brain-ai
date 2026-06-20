# Speculative Decoding: Fast Inference from Transformers

**Source:** https://arxiv.org/abs/2211.17192
**Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias (Google)
**Published:** 2022

## Key Concepts

Speculative decoding uses a small, fast "draft" model to generate multiple candidate tokens, then verifies them all in a single forward pass of the large "target" model. The target model accepts correct predictions and rejects incorrect ones, effectively running at the draft model's speed with the target model's quality.

Achieves 2-3× speedup with mathematically guaranteed identical output distribution — no quality compromise whatsoever. The key insight: verifying multiple tokens in parallel is nearly as fast as generating one, because LLM inference is memory-bandwidth bound.

## Variants

- Medusa: adds parallel prediction heads instead of using a separate draft model (2.2-3.6× speedup)
- EAGLE: uses early-exit features from the target model itself as the "draft"
- Self-speculative: the model drafts from its own early layers
