# Efficient Memory Management for Large Language Model Serving with PagedAttention

**Source:** https://arxiv.org/abs/2309.06180
**Authors:** Woosuk Kwon et al.
**Published:** 2023

## Key Concepts

PagedAttention is an attention algorithm inspired by OS virtual memory paging. It partitions the KV cache into fixed-size blocks stored in non-contiguous memory, eliminating memory fragmentation. vLLM, built on PagedAttention, achieves 2-4× throughput improvements with near-zero memory waste (under 4% vs 60-80% in previous systems).

## Why It Matters

- KV cache memory is the primary bottleneck in LLM serving (can consume 30%+ of GPU memory)
- Previous systems waste 60-80% of KV cache memory due to fragmentation and over-allocation
- PagedAttention enables efficient memory sharing across requests (e.g., for beam search, shared prefixes)
- vLLM became the de facto open-source LLM serving framework
