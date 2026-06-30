# Attention Variants

Standard multi-head attention has O(N²) time and memory complexity in sequence length, making it the primary bottleneck for scaling transformers to long contexts. Variants address this along two distinct axes: **what the per-token attention computation looks like** (FlashAttention, GQA, MQA, MLA, gated attention) and **which tokens are allowed to attend to which** (full causal, document masking, sliding window, chunked, DCA, interleaved local/global). Picking the right combination is one of the few architecture decisions that has real first-order impact on cost; for most frontier-2026 models the answer is **GQA with small groups (2/4/8) for the per-token computation, document masking + RoPE/YaRN for long context, sometimes with sliding-window or chunked attention layered on top**.

## FlashAttention

FlashAttention (Tri Dao et al., 2022) is the most impactful attention optimization. Rather than changing the attention computation mathematically, it makes it **IO-aware** — computing attention in tiles that fit in GPU SRAM instead of materializing the full N×N attention matrix in slower HBM memory.

Results: 15% speedup on BERT-large, 3× on GPT-2 (1K sequences), 2.4× on long-range tasks. FlashAttention doesn't approximate — it computes exact attention, just more efficiently. FlashAttention-2 parallelized over the sequence dimension; FlashAttention-3 added warp specialization and FP8 for Hopper (H100). It's now standard in essentially all modern LLM training and inference.

The key insight: attention is **memory-bandwidth bound**, not compute-bound. Moving less data between HBM and SRAM matters more than reducing FLOPs.

Modern production-grade implementations like [[training-ops|ThunderKittens]] (Spector et al., 2410.20399) extend this principle with GPU-aware kernel abstractions: tile data structures with managed layouts to match hardware memory hierarchies, asynchronous producer-consumer templates for coordinating load/store/compute across thread blocks, and persistent grid scheduling to overlap HBM accesses with computation. ThunderKittens achieves 10-40% speedups over FlashAttention-3 on attention backward passes and matches CuBLAS on GEMM, demonstrating that simple, maintainable abstractions can deliver performance without hand-optimized low-level code.

Caveat: FlashAttention / SDPA fused kernels assume standard attention, so they are **incompatible with logit softcapping during training**. See [[training-stability]] for the workaround.

### The Online-Softmax Trick, Derived

"Online softmax" is the numerically-stable trick underlying FlashAttention's single-pass tiling, and it's worth deriving explicitly. The naive numerically-stable softmax needs **two passes** over the logits: one to find the max `x_max`, one to compute the shifted denominator `Σ e^{x_j - x_max}`. Online softmax fuses both into a single pass by maintaining a running max and a running, continuously-rescaled denominator as new logits arrive.

Maintain a running max `m_k = max(x_1, ..., x_k)` and running shifted denominator `d_k = Σ_{j≤k} e^{x_j - m_k}`. On encountering `x_{k+1}`:

```
m_{k+1} = max(m_k, x_{k+1})
d_{k+1} = d_k · e^{m_k - m_{k+1}} + e^{x_{k+1} - m_{k+1}}
```

The update for `d_{k+1}` falls out by splitting off the new term and rewriting each old exponent as `x_j - m_k + m_k - m_{k+1}`, pulling the constant `e^{m_k - m_{k+1}}` out of the sum, and recognizing what's left as `d_k`:

```
d_{k+1} = e^{x_{k+1} - m_{k+1}} + Σ_{j≤k} e^{x_j - m_{k+1}}
        = e^{x_{k+1} - m_{k+1}} + e^{m_k - m_{k+1}} · Σ_{j≤k} e^{x_j - m_k}
        = d_k · e^{m_k - m_{k+1}}  +  e^{x_{k+1} - m_{k+1}}
          \_____rescale old terms____/   \___new term___/
```

When the max doesn't change (`m_k = m_{k+1}`), the rescaling factor is exactly 1 — no correction needed. After the full pass, a second sweep over the logits computes `e^{x_i - m_N} / d_N` to materialize the actual softmax — but for a *weighted sum* (rather than the distribution itself), no second pass is needed at all, which is exactly FlashAttention's situation.

**Connection to FlashAttention**: attention output is a weighted sum over a stream of logits `x_i = q·k_i` and values `v_i`: `o = (Σ_i e^{x_i} v_i) / (Σ_i e^{x_i})`. FlashAttention extends the two running quantities above with a third, a running numerator `o_k ∈ R^H` (`H` = head dim), updated by the identical rule:

```
o_k = Σ_{i≤k} e^{x_i - m_k} v_i
o_{k+1} = o_k · e^{m_k - m_{k+1}} + e^{x_{k+1} - m_{k+1}} v_{k+1}
```

After streaming through all `N` key/value pairs, the triple `(m_N, d_N, o_N)` gives the exact attention output as `o_N / d_N` — computed in one tiled pass with no `T×T` matrix ever materialized. This is the precise mechanism behind the "running max and running sum for numerical stability" step in the three-step summary above.

### FlashAttention-4 (Blackwell)

[FlashAttention-4 (2603.05451)](../../papers/02-architecture/attention-variants/FlashAttention-4: Algorithm and Kernel Pipelining Co-Design - 2603.05451.pdf) targets NVIDIA Blackwell (B200/GB200) where the hardware scaling is now deeply **asymmetric** — tensor cores doubled to 2.25 PFLOPS FP16/BF16 (vs 1 PFLOPS on Hopper), but shared memory bandwidth and the exponential MUFU unit stayed flat. A roofline analysis shows shared-memory traffic and exponential ops now dominate execution time by 25–60%, exceeding MMA compute. The kernel co-designs around three new constraints:

1. **Redesigned ping-pong pipeline** for Blackwell's fully asynchronous MMA and 128×128 MMA tile size (double Hopper's 64×128), with two softmax warpgroups synchronized to never overlap in their critical section.
2. **Polynomial exponential emulation**: software-emulates `2^x` on FMA units in parallel with the MUFU, decomposing `2^x = 2^⌊x⌋ · 2^{x-⌊x⌋}` with a degree-3 polynomial that matches hardware to 1 BF16 ULP on 99% of inputs. Applied to 10–25% of softmax entries to alleviate the exponential bottleneck.
3. **Conditional softmax rescaling**: only rescale when `m_j - m_{j-1} > τ` (threshold `τ = log₂(256) = 8`), skipping the expensive vector multiply most of the time while preserving correctness via final renormalization.
4. **2-CTA MMA mode + tensor memory (TMEM)**: a 256 KB on-chip per-SM memory; CTA pairs share operands to halve shared-memory traffic for the dQ backward step.

Empirical: up to **1.3× over cuDNN 9.13 and 2.7× over Triton** on B200, reaching ~1600 TFLOPs/s (71% utilization). Implemented in CuTe-DSL (Python-embedded), giving 20–30× faster compile times than C++ templates while keeping full expressivity. The broader lesson: hardware scaling is asymmetric, so attention kernels increasingly need to mask non-matmul bottlenecks rather than just reduce HBM traffic.

## Sparse Attention: DeepSeek Sparse Attention and Indexer Reuse

Distinct from GQA/MQA/MLA (which shrink the KV cache by sharing or compressing K/V heads), **sparse attention** shrinks the attention computation itself by only attending to a subset of tokens. **DeepSeek Sparse Attention (DSA)** is the production-grade trainable instance of this idea: a lightweight "lightning indexer" runs at every layer, scoring all preceding tokens with a multi-head ReLU-gated dot product (few heads, low-rank projections, FP8 arithmetic — an order of magnitude cheaper per-FLOP than the main MLA computation) and selecting the top-k (k=2048) highest-scoring tokens. Core attention then runs only over that sparse subset, cutting per-layer cost from O(L²) to O(Lk). DSA is trained with a two-stage recipe: a dense warm-up that distills the indexer via KL-divergence against the aggregated full-attention distribution (all other weights frozen), followed by sparse training that jointly optimizes the whole model with the indexer receiving distillation gradients on a detached graph.

The catch: the indexer itself is still O(L²) and runs independently at every layer, so its total cost across N layers is O(NL²) — on a 30B DSA model this indexer overhead alone consumed 27-81% of total attention latency at long context (10K-200K tokens), rising sharply with sequence length during prefill.

[**IndexCache** (2603.12201)](../../papers/04-efficiency/inference-kernels/IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse - 2603.12201.pdf) (Bai, Dong, Jiang, Lv, Du, Zeng, Tang, Li — Tsinghua University / Z.ai) exploits the finding that top-k index selections are 70-100% overlapping between adjacent layers, by designating a small set of **Full (F) layers** that compute their own indexer and top-k set, and routing the rest as **Shared (S) layers** that just reuse the nearest preceding F layer's indices (one conditional branch, no extra memory). A **training-free** variant uses greedy search guided by calibration-set LM loss to pick which layers stay Full (plain uniform interleaving is measurably worse at the same retention ratio, since early/transitional layers are disproportionately sensitive to indexer removal); a **training-aware** variant distills each retained indexer against the *averaged* attention distribution of all layers it serves (provably gradient-equivalent to a single centroid target), letting even uniform interleaving match full-indexer quality. On a 30B DSA model, retaining only 1/4 of indexer computations gave **up to 1.82× prefill speedup and 1.48× decode speedup** at 200K context with negligible quality loss across nine long-context and reasoning benchmarks (prefill: 19.5s→10.7s; per-request decode: 58→86 tok/s); preliminary results on 744B-parameter GLM-5 removing half the indexers showed ~1.2-1.3× end-to-end speedup. Retention can't be pushed arbitrarily low: at 1/8 retention, training-free uniform interleaving collapses long-context quality (50.2→35.3 average) and even greedy search only partially recovers it (46.1) — the paper flags 1/8 as past the safe operating point without retraining. Training-aware IndexCache (retraining retained indexers with a multi-layer distillation loss against the centroid of served layers' attention) removes this layer-pattern sensitivity entirely: with retraining, even naive uniform interleaving at 1/2 retention (51.6 Long Avg) matches or beats the greedy-searched pattern (50.6), since the model jointly adapts to whichever sharing pattern it's trained on. See [[kv-cache]] for this alongside other recent cache/indexing-efficiency work (Cartridges, the "Sleep" SSM-consolidation mechanism).

## Grouped Query Attention (GQA)

GQA ([Ainslie et al., 2023 / 2305.13245](../../papers/02-architecture/transformers/GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints - 2305.13245.pdf)) partitions query heads into groups that share key and value heads. If you have 32 query heads and 8 KV groups, each group of 4 query heads shares one K and one V head. This dramatically reduces the KV-cache memory during inference (critical for serving) while maintaining nearly the same quality as full multi-head attention.

The paper's two-step "uptraining" recipe matters in practice: (1) **mean-pool** the original H key/value heads into G groups (mean-pool beats "pick the first" and "random init"); (2) continue pre-training for a small fraction `α` of the original steps on the original recipe. The authors report that **α = 0.05 (5% of pre-training) is sufficient** to recover quality, with diminishing returns past 10%. This makes GQA a cheap retrofit for existing MHA checkpoints rather than something requiring a from-scratch run.

Empirical numbers from the paper (T5-XXL): MHA-XXL averages 47.2 across the eval suite at 1.51 s/sample, MQA-XXL averages 46.6 at 0.24 s, **GQA-8-XXL averages 47.1 at 0.28 s** — essentially MHA quality at MQA speed. The 8-groups choice was empirically tuned: time-per-sample stays flat from 1 to 8 groups, then climbs sharply at 16 and beyond (KV cache becomes the bottleneck again).

Why GQA is favorable for *large* models specifically: model FLOPs scale with `d²` but KV-cache size scales with `d`, and standard tensor sharding replicates the single MQA KV head across model partitions anyway — so MQA's bandwidth win is smaller at scale, while GQA preserves capacity proportional to model dimension. The paper applies GQA only to decoder self-attention and cross-attention; encoder self-attention stays MHA (not memory-bandwidth-bound).

Adopted by Llama 2, Llama 3, DeepSeek-LLM-67B, and most modern open-source LLMs. It's a sweet spot between multi-head attention (all separate KV heads) and multi-query attention (single KV head for all queries).

**Frontier-2026 ablation result (Hugging Face SmolLM3)**: when ablating with adjusted hyperparameters to keep model sizes comparable, GQA with **2, 4, or 8 groups outperformed MHA**, while MHA outperformed both MQA and GQA with 16 groups. The signal is consistent across HellaSwag, MMLU, and ARC — small groups are the sweet spot, not a compromise. Trinity Large, gpt-oss-120b, OLMo 3, and SmolLM3 all use small-group GQA.

## Multi-Query Attention (MQA)

The extreme version of GQA: all query heads share a single key and value head. Maximum memory savings but **leaks attention capacity** — heads can't store information specialized to each head's role. Introduced by [Shazeer, 2019 (1911.02150)](../../papers/02-architecture/transformers/Fast Transformer Decoding: One Write-Head is All You Need - 1911.02150.pdf) — the original motivation was incremental-decoding latency: the per-step memory-access-to-arithmetic ratio in batched MHA is `Θ(n/d + 1/b)`, and the offending `n/d` term comes from reloading the K and V tensors of shape `bhmk = bn²` at every step. MQA removes the heads dimension from K and V, dropping that term by a factor of `h`. The original WMT14 EN-DE experiments showed MHA at 26.7 BLEU vs MQA at 26.5 BLEU but **inference time dropped from 46μs/token to 3.8μs/token on the decoder** — an order-of-magnitude latency win for less than 1 BLEU point.

Used in PaLM and Falcon. The GQA paper found MQA can also cause **training instability on long-input fine-tuning** — pre-training suffered frequent loss spikes and final models diverged immediately when fine-tuning on long-input tasks, which is partly why GQA (with its intermediate KV head count) became preferred. Generally underperforms GQA at comparable model sizes.

## Multi-Latent Attention (MLA)

MLA stores a **compressed latent variable** that gets decompressed/projected into K and V at runtime. The latent is typically much smaller than the full KV cache — often 4–8× compression — yielding KV-cache parameter counts comparable to GQA while maintaining performance stronger than MQA.

Used by **Kimi-K2** (1T parameters). The tradeoff is implementation complexity: keys aren't materialized directly but projected from the latent, which complicates anything that wants to operate on key matrices — like [[training-stability|QK-norm]] (incompatible) and [[optimizers|MuonClip]] (needs careful handling of latent-to-key projection weights and the latent variable itself).

## Gated Attention

Apply an elementwise gating mechanism to the scaled dot-product attention output before the output projection.

```
gate_t = σ(W_g · x_t)             # learned gate from input
gate_h = gate_t[h * d_h : (h+1) * d_h]   # slice for head h
out_h = Attention_h(x) ⊙ gate_h    # elementwise multiply
output = W_O · concat(out_1, ..., out_H)
```

Benefits:
- **Reduces attention sinks** — tokens that receive disproportionately high attention.
- **Reduces large activations** that destabilize training.
- **Improves evals and long-sequence generalization.**
- Critically, **stabilizes training and reduces loss spikes**, making it valuable for large-scale runs.

## Document Masking (Intra-Document Attention)

Pre-training uses fixed-length tensors `[batch, sequence_length, hidden]` for GPU efficiency. To avoid padding waste, **packing** concatenates documents within a sequence to fill the budget.

With **standard causal masking**, tokens from unrelated document `A` (packed alongside document `B`) can attend to tokens in `B` — which degrades performance. **Intra-document masking** restricts attention to tokens within the same document.

Hugging Face's SmolLM3 ablations: small PIQA improvements but no notable short-context impact. **Crucial when scaling from 4k to 64k tokens** — the bigger the context, the more the cross-document leakage matters. For smaller models, the overhead may not be worth it. Used in combination with [[positional-encodings|RNoPE]] as the long-context foundation.

## Attention Patterns for Long Contexts

Distinct from per-token attention variants above (and from [[positional-encodings|positional-encoding scaling]] methods like ABF/YaRN) — these modify *which tokens attend to which* to reduce computational cost.

### Sliding Window Attention (SWA)

Every token can see up to `w` positions back, creating a sliding window that maintains local context. Used in Mistral. **Gemma 3** combined SWA with full attention every other layer.

### Chunked Attention

Divides the sequence into fixed-size chunks where tokens can only attend within their chunk. **Llama 4** paired chunked attention with [[positional-encodings|RNoPE]] (specifically the RoPE layers), which also reduces per-layer KV cache, but its performance on long-context tasks degraded.

### Dual Chunk Attention (DCA)

`N` tokens are chunked into `K` groups. Within each group, tokens attend normally (like chunked attention). Between successive chunks there's a local window to preserve locality, and inter-chunk attention allows queries to attend to previous chunks with a capped relative position cap. **Qwen-2.5 used DCA to support context windows up to 1 million tokens.**

### Interleaving Local and Global Attention

Alternates between layers using local attention (restricted to nearby tokens) and global attention (full sequence). Local layers reduce quadratic complexity while preserving local context; global layers ensure distant relationships aren't lost.

When training encounters instability or loss spikes, adjusting the **ratio of global layers** (e.g., increasing their frequency) can result in quicker loss recovery — the model regains access to long-range information crucial for certain patterns. Particularly effective for long-context models where full global attention would be computationally prohibitive.

## Related Topics

- [[transformer-architecture]] — The base architecture these variants modify
- [[positional-encodings]] — RoPE/NoPE/RNoPE/YaRN choices interact with attention patterns; covers the orthogonal axis of long-context scaling
- [[long-context-training]] — Where attention patterns, document masking, and context-extension data recipes come together
- [[gpu-kernel-engineering]] — FlashAttention, ThunderKittens, and the hardware side of attention kernels
- [[mixture-of-experts]] — orthogonal to attention; many frontier models combine MoE with GQA/MLA
- [[hybrid-architectures]] — replace softmax attention entirely with linear/state-space variants
- [[training-stability]] — gated attention reduces loss spikes; QK-norm and logit softcapping
- [[optimizers]] — MuonClip handles MLA's specific stability challenges
- [[kv-cache]] — GQA, MQA, MLA all primarily exist to reduce KV cache; also covers Cartridges and the SSM "Sleep" mechanism
- [[inference-optimization]] — KV-cache reduction is critical for serving efficiency
- [[data-curation-mixtures]] — sequence composition (packing strategies) and intra-document masking interact with attention patterns
- [[frontier-training-playbook]] — where attention choices sit in the decision tree

## Sources

- FlashAttention: Fast and Memory-Efficient Exact Attention (arxiv:2205.14135)
- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design (2603.05451)](../../papers/02-architecture/attention-variants/FlashAttention-4: Algorithm and Kernel Pipelining Co-Design - 2603.05451.pdf) — Blackwell co-design, asymmetric hardware bottlenecks.
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels (arxiv:2410.20399)](../../papers/03-scaling/training-optimization/ThunderKittens: Simple, Fast, and Adorable AI Kernels - 2410.20399.pdf) — GPU kernel abstractions, tile data structures, LCSCF async template.
- [Fast Transformer Decoding: One Write-Head is All You Need (1911.02150)](../../papers/02-architecture/transformers/Fast Transformer Decoding: One Write-Head is All You Need - 1911.02150.pdf) — Shazeer's original MQA paper.
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (2305.13245)](../../papers/02-architecture/transformers/GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints - 2305.13245.pdf) — GQA + uptraining recipe (mean-pool, α=0.05).
- DeepSeek-V2 / Kimi-K2 (MLA in production at scale).
- [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse (2603.12201)](../../papers/04-efficiency/inference-kernels/IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse - 2603.12201.pdf) — cross-layer reuse of DeepSeek Sparse Attention's lightning-indexer top-k selection.
- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- SmolLM3 report (GQA group-count ablation). See `raw/smollm3-hugging-face-report.md`.
- Qwen-2.5 (DCA at 1M context), Gemma 3 (SWA + full alternation), Llama 4 (chunked + RNoPE).
- Alisa Liu, "Book of LLMs" (Notion, alisawuffles.notion.site/alisa-s-book-of-llms) — the online-softmax running-max/running-denominator derivation and its extension to a running numerator for FlashAttention. See [`raw/alisa-liu-book-of-llms.md`](../raw/alisa-liu-book-of-llms.md).
