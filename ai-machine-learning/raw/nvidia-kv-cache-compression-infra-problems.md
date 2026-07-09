# KV Cache Compression and Its Infra Problems

**Source:** https://research.nvidia.com/labs/eai/blogs/kv-cache-compression-and-its-infra-problems/
**Authors:** Weian Mao, Yukang Chen, Wei Huang, Shuai Yang, Luozhou Wang, Song Han (NVIDIA — Efficient AI Lab)
**Published:** June 12, 2026
**Code:** github.com/WeianMao/triattention

## Core Problem

Long-context inference exhausts GPU memory because the KV cache grows unbounded — every generated token appends keys and values across all layers. Concrete failure: a **Qwen3-32B model with 4-bit quantized weights crashes after ~24,000 generated tokens on a 24GB GPU**, short of the 32K-token traces reasoning models routinely need. Compressing the KV cache is therefore a prerequisite for running long reasoning traces on commodity hardware.

The blog's central thesis: *"The hard part of KV cache compression is not choosing which tokens to keep — it is two collisions with production infrastructure."* Token-selection algorithms are only half the problem; the other half is that the two most effective infrastructure optimizations in production serving actively fight most published compression methods.

## The Two Infrastructure Problems

### Problem 1 — FlashAttention incompatibility

Production inference uses **FlashAttention**, which tiles the attention computation through SRAM and *never materializes the full N×N attention score matrix* in main memory. Any compression method that needs to observe historical per-token attention scores (to decide what to evict) therefore has nothing to read. The reference **H2O** implementation, for example, "falls back to eager attention, materializing the full score matrix and abandoning FlashAttention outright" — surrendering the very kernel that makes long-context serving fast.

### Problem 2 — Paged-attention fragmentation

Production systems (e.g. **vLLM**) use **paged attention** with fixed-size physical blocks (~16 tokens per block). A block is only freed when it becomes *completely empty*. After token eviction, the surviving tokens are scattered across blocks. Worked example: evicting **14,400 of 16,000 tokens** leaves 1,600 survivors spread across **~1,000 blocks** — nearly every block still retains at least one survivor, so the allocator reclaims almost nothing despite a massive *logical* deletion. Compression that looks great on paper frees no physical memory.

## Existing Methods (and why they hit the walls above)

- **StreamingLLM** — keeps "attention sinks" (the first tokens, which receive disproportionate attention) plus a sliding window of recent tokens; discards the middle context permanently.
- **H2O (Heavy-Hitter Oracle)** — maintains cumulative attention-score sums per cached token across decode steps and evicts the lowest-scoring tokens to hold a fixed budget. Requires per-step attention-score observation → incompatible with FlashAttention (Problem 1).
- **SnapKV** — scores tokens *once* using an "observation window" of the ~25 most recent tokens to pick keepable tokens. Removes continuous bookkeeping but can't capture information attended during *different* reasoning phases.
- **Variants** — Scissorhands, TOVA, PyramidKV, Ada-KV, R-KV, and Quest use similar attention-scoring approaches with layer/head-specific budgets or page-selection strategies.

## TriAttention (the proposed solution)

**TriAttention abandons attention-score dependency entirely.** Instead of observing scores, it uses the *geometric properties of the learned Q/K vector representation spaces* to predict token importance. Because it never needs score observation, it sidesteps Problem 1 (works with FlashAttention).

To address Problem 2 (fragmentation), it adds **Forward-Packing Compaction** — consolidation that runs roughly every **128 decoded tokens** using one of two strategies:

- **Order-preserving repack** — survivors slide forward keeping original token order; holes drift to the tail; whole blocks empty out and return to the allocator. Minimal position-tracking overhead.
- **Hole-filling variant** — new survivors drop directly into eviction-vacated slots; far less data movement (**~3 copies vs. 18**) but scrambles physical order, so it needs explicit position tracking.

## Performance Results

Reasoning-benchmark accuracy at constrained KV budget:

| Method | AIME 2024 | AIME 2025 | MATH 500 |
|--------|-----------|-----------|----------|
| Full Attention | 57.1% | 40.8% | 69.6% |
| SnapKV | 34.6% | 20.0% | 49.2% |
| R-KV | 25.4% | 17.5% | 46.4% |
| **TriAttention** | **42.1%** | **32.9%** | **56.0%** |

- At **KV budget 2,048 tokens** (1/16th of the full 32K), TriAttention nearly **doubles R-KV's AIME 2025 accuracy**.
- At **budget 3,072**, it **matches the full-attention baseline (40.8% AIME 2025)** while delivering **2.5× higher throughput (563 vs. 223 tokens/second)** and **10.7× KV memory reduction (9.3% relative usage)**.

## Video-Generation Extension

The same memory pressure appears in autoregressive video generation, where tokens are spatial patches:

- **Quant VideoGen** — 7× memory compression via 2-bit quantization, exploiting near-identical adjacent frames and storing residuals.
- **LongLive 2.0** — scales to production with parallel dequantization kernels and **4-bit NVFP4 quantization**, for **1.84× throughput** with negligible quality loss; a fused kernel keeps quantization/dequantization overhead **below 2%**.

## Key Takeaway

Algorithmic token selection is only half of KV-cache compression. The deciding factor for real deployment is *infrastructure integration* — a method must coexist with FlashAttention (never needs materialized scores) and with paged attention (must actually free physical blocks, not just logically delete tokens). TriAttention is designed around both constraints rather than around benchmark accuracy alone.
