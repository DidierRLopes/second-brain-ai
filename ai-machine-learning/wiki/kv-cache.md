# KV Cache

The KV (Key-Value) cache is the mechanism that makes autoregressive LLM generation efficient. Without it, generating each new token would require recomputing attention over the entire sequence from scratch. With it, only the new token's computation is needed, turning O(n²) per-step cost into O(n).

## How It Works

During the attention computation, each layer produces key (K) and value (V) matrices for every token. In autoregressive generation, previously computed K and V vectors don't change — they only depend on the input token and its position. The KV cache stores these vectors so they can be reused. At each generation step, only the new token's K and V are computed and appended to the cache.

The prefill phase processes the entire prompt at once (compute-bound). The decode phase generates tokens one at a time, reading the entire KV cache at each step (memory-bandwidth bound). This distinction is critical for optimization.

A concrete benchmark from [KV Caching Explained](../raw/kv-caching-explained-huggingface.md) (Hugging Face) puts a number on the difference: generating 300 new tokens from `HuggingFaceTB/SmolLM2-1.7B` on a single T4 GPU took **11.7s with KV caching enabled vs. 1min 1s without it — a ~5.21× speedup** — using nothing more than `transformers`' default `use_cache=True` behavior in `model.generate()`.

## Exact Memory Formula

From "How to Scale Your Model" (Austin et al., 2025). Source: https://jax-ml.github.io/scaling-book/training

KV cache size for a single sequence of length S:

```
KV_bytes = 2 × S × L × K × H × bytes_per_element
```

where L = num layers, K = head dim, H = num KV heads, 2 = (K + V).

**LLaMA 3-70B example** (L=80, K=128, H=8, int8 quantized):
- `2 × S × 80 × 128 × 8 × 1 byte = 163,840 × S bytes`
- Per token: **160 kB/token**
- At 32k context: 160 kB × 32k = **5.12 GB per sequence**
- At 128k context: **~20 GB per sequence** — exceeds single A100 (80GB) with only 4 sequences

For bf16 (2 bytes): double all above figures.

### Context Length vs Model Size

For long contexts, KV cache memory often **exceeds model weight memory**:
- LLaMA 3-70B weights in bf16: 140 GB
- LLaMA 3-70B KV cache in bf16, 128k ctx, batch=4: 4 × 20 × 2 = 160 GB (exceeds weights!)

This is the primary motivation for GQA (reducing H from 32→8 → 4× smaller cache), int8 KV quantization, and PagedAttention.

### Load Time per Decode Step

At each decode step, the entire KV cache must be loaded from HBM:
```
T_kv_load = (2 × S × L × K × H × bytes) / HBM_bandwidth
```

For LLaMA 3-70B at S=32k, int8, on A100 (2TB/s):
- KV bytes: 5.12 GB
- T_kv_load: 5.12 GB / 2 TB/s = **2.56ms per step**

At long context, KV cache load time can exceed weight load time — explaining why throughput degrades with context length.

## Memory: The Primary Bottleneck

KV cache memory grows with: sequence length × batch size × number of layers × number of KV heads × head dimension × 2 (for K and V) × bytes per element. For a 70B model with 128K context, the KV cache alone can consume 40+ GB — often more than the model weights themselves.

This is why [[attention-variants]] like **Grouped Query Attention (GQA)** are so impactful: reducing KV heads from 32 to 8 cuts cache memory by 4×. The original [MQA paper (1911.02150)](../../papers/02-architecture/transformers/Fast Transformer Decoding: One Write-Head is All You Need - 1911.02150.pdf) framed this precisely: in incremental decoding, the memory-access-to-arithmetic ratio is `Θ(n/d + 1/b)`, where the offending `n/d` term comes from reloading K and V at every step. Removing the heads dimension from K and V drops that term by a factor of `h`. MQA's measured per-token decoder latency went from 46μs (MHA) to 3.8μs — over an order of magnitude — for less than 1 BLEU loss on WMT14 EN-DE.

The [GQA paper (2305.13245)](../../papers/02-architecture/transformers/GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints - 2305.13245.pdf) showed inference time per sample stays flat from 1 to 8 GQA groups, then climbs sharply at 16+ — beyond a certain point, KV cache becomes the bottleneck again and you're paying the cost without the capacity benefit. The paper also documented that **MQA can cause training instability with long-input fine-tuning** (loss spikes, divergence), which GQA's intermediate KV head count avoids — another reason GQA dominates over MQA in current open-source models.

## PagedAttention and vLLM

PagedAttention (Kwon et al., 2023) solved the memory fragmentation problem. Previous serving systems pre-allocated contiguous memory for the maximum possible sequence length, wasting 60-80% of KV cache memory. PagedAttention partitions the cache into fixed-size blocks stored non-contiguously (like OS virtual memory pages), reducing waste to under 4%. vLLM, built on PagedAttention, achieves 2-4× throughput improvements and became the de facto LLM serving framework.

## KV Cache Compression

Not all cached tokens are equally important. **Scissorhands** (Dang et al., 2023) exploits the "persistence of importance" hypothesis: tokens that receive high attention at one step tend to remain important. By evicting unimportant tokens, it achieves 5× compression (20× with 4-bit quantization) without fine-tuning.

Other approaches include H2O (Heavy-Hitter Oracle) which keeps only the most-attended tokens, and StreamingLLM which maintains a sliding window plus "attention sink" tokens. SnapKV scores tokens once using an "observation window" of the ~25 most recent tokens to decide what to keep, removing continuous bookkeeping but missing information attended during different reasoning phases; further variants (TOVA, PyramidKV, Ada-KV, R-KV, Quest) apply similar attention-scoring with layer/head-specific budgets or page selection.

### The Infrastructure Problem: Why Compression Rarely Ships

NVIDIA's [KV Cache Compression and Its Infra Problems](../raw/nvidia-kv-cache-compression-infra-problems.md) (Mao, Chen, Huang, Yang, Wang & Han, NVIDIA Efficient AI Lab, June 2026) makes the case that *"the hard part of KV cache compression is not choosing which tokens to keep — it is two collisions with production infrastructure."* The motivating failure: a Qwen3-32B model with 4-bit-quantized weights **crashes after ~24,000 generated tokens on a 24GB GPU**, short of the 32K-token traces reasoning models need. Yet most published methods can't be deployed to fix it, for two reasons:

- **FlashAttention incompatibility.** FlashAttention tiles the attention computation through SRAM and never materializes the full N×N score matrix. Any method that needs historical per-token attention scores to decide evictions (like H2O, which tracks cumulative score sums per token) has nothing to read — the reference H2O implementation *falls back to eager attention, materializing the full score matrix and abandoning FlashAttention outright*.
- **Paged-attention fragmentation.** vLLM-style [[inference-optimization|paged attention]] stores the cache in fixed ~16-token physical blocks that free only when *completely* empty. After eviction, survivors scatter: evicting 14,400 of 16,000 tokens leaves 1,600 survivors spread across ~1,000 blocks, so nearly every block retains a survivor and the allocator reclaims almost nothing despite a massive *logical* deletion.

**TriAttention** is their answer to both. It abandons attention-score dependency entirely, using the *geometric properties of the learned Q/K representation spaces* to predict token importance (no score observation → FlashAttention-compatible). To recover physical memory it adds **Forward-Packing Compaction** roughly every 128 decoded tokens, in one of two variants: an *order-preserving repack* (survivors slide forward keeping token order, whole blocks empty and return to the allocator, minimal position tracking) or a *hole-filling* variant (new survivors drop into vacated slots — ~3 copies vs. 18, but scrambles physical order and needs explicit position tracking). At a KV budget of 2,048 tokens (1/16th of full 32K) TriAttention nearly doubles R-KV's AIME 2025 accuracy (32.9% vs. 17.5%, against 40.8% full-attention), and at budget 3,072 it matches the full-attention baseline while delivering **2.5× higher throughput (563 vs. 223 tok/s) and 10.7× KV memory reduction (9.3% relative usage)**. The same memory pressure recurs in autoregressive video generation, where Quant VideoGen reaches 7× compression via 2-bit residual quantization of near-identical adjacent frames, and LongLive 2.0 uses 4-bit NVFP4 with fused parallel dequantization kernels for 1.84× throughput at <2% quantization overhead.

The broader lesson generalizes past this one method: a KV-compression technique's viability is decided less by benchmark accuracy than by whether it coexists with FlashAttention (never needs materialized scores) and with paged attention (must actually free physical blocks, not just logically delete tokens).

## TurboQuant: Near-Optimal Vector Quantization for KV Cache

Most KV-cache quantization (int8, NF4-style) quantizes each scalar independently. **TurboQuant** (Zandieh, Daliri, Hadian & Mirrokni, Google Research/DeepMind, ICLR 2026) instead treats each K or V vector as a single object to compress, combining two pieces:

- **PolarQuant** (AISTATS 2026): a rotation-based coordinate transform that reshapes a vector's distribution before quantizing it, so the quantization grid wastes less precision on directions with little information.
- **QJL (Quantized Johnson-Lindenstrauss) residual correction**: a 1-bit correction term, layered on top of PolarQuant's output, that recovers most of the error a naive rounding step would otherwise leave on the table.

The combination is **data-oblivious** (no calibration data or fine-tuning needed — unlike GPTQ/AWQ) and reaches within roughly **2.7× of the information-theoretic distortion limit** for vector quantization, which is unusually close for a training-free method. Reported results: KV cache quantized to **3 bits with 6× memory reduction**, and up to **8× faster attention on H100s** since less data needs to move through HBM. Conceptually this sits next to [[quantization-methods|QuIP's incoherence processing]] — both fight quantization error with a structured transform before rounding — but TurboQuant is purpose-built for the KV cache's vector-per-token structure rather than weight matrices.

## Beyond Eviction and Quantization: Distillation, Index Reuse, and Sleep

Scissorhands, H2O, and TurboQuant all operate on an *existing* KV cache (evict tokens or shrink bits). Three more recent papers attack the problem from different angles: training a cheaper cache offline, reusing computation that *builds* the cache, and replacing the cache with weights entirely.

### Cartridges: Self-Study Cache Distillation

**Cartridges** (Eyuboglu, Ehrlich, Arora et al., Stanford, 2025) target the case where many queries repeatedly reference the *same* large corpus (a codebase, a 10-K filing, a patient record). Instead of re-running prefill on the full corpus for every request, a **Cartridge** is a small set of trainable KV pairs `z_k, z_v ∈ R^(p×d)` — parameterized identically to a KV cache, but with `p` (the number of "virtual tokens") far smaller than the corpus length — trained once offline per corpus and then loaded in place of the corpus's KV cache at inference time.

The naive way to train this — next-token prediction on the raw corpus — fails: it produces a Cartridge that memorizes the text (using 107× less memory than the full KV cache) but can't generalize to the diverse query types a real user sends (math reasoning, summarization, creative writing, structured extraction), because next-token prediction never trains the model to *use* the corpus this way. The paper's fix is **Self-Study**, a two-step recipe:
1. **Synthetic data generation**: prompt the model to generate synthetic conversations *quizzing itself* about the corpus (seeded with prompts biased toward global reasoning and document-structure questions), chunking the corpus as needed for corpora that exceed the model's effective context.
2. **Context distillation**: train the Cartridge on those synthetic conversations so the Cartridge-augmented model's next-token distribution matches the distribution the *same model* produces with the full corpus in context (rather than plain next-token prediction on raw text).

Results on long-context benchmarks (100K–484K token corpora): Cartridges trained with Self-Study **match in-context-learning (ICL) quality while using 38.6× less memory and enabling 26.4× higher peak throughput** — an order of magnitude beyond cache-compression baselines like DuoAttention. Broken out by dataset, the memory savings reach **up to 10× on LongHealth and up to 100× on QASPER** at comparable quality, while compression baselines (prompt truncation/summarization, DuoAttention) degrade noticeably past 2× compression. On MTOB (Kalamang-to-English translation from a 484K-token textbook), a Cartridge built from the textbook **outperforms ICL over the first 130K tokens by 11.0 chrF points** and extends the model's effective context from 128K to 484K tokens, matching ICL performance on a hand-curated 60K-token subset despite never fitting the full textbook in the base model's 128K context window. Cartridges are also **composable without joint training**: two independently-trained Cartridges (e.g., one per company 10-K) can be concatenated at inference time and substantially outperform both a single Cartridge and truncated-ICL on multi-document questions, emulating ICL over both source documents together.

The parameterization is a simplified prefix-tuning (trainable KV pairs prepended to the cache, with all base-model weights frozen and initialized from the KV vectors of the corpus's first `p` tokens — critical for stability: random-vector init reaches only 29.9% accuracy on LongHealth vs. 51.3% for random-token-KV init vs. 55.3% for first-`p`-token init). The paper found this KV-cache parameterization outperforms an equivalent memory-matched LoRA parameterization both in-domain (4.5 chrF points higher on MTOB) and especially out-of-domain: on MMLU (unrelated to the corpus), LoRA's accuracy collapses from 54.7% to 45.3% as Cartridge size grows from 0.15GB to 1.06GB, while prefix-tuning barely degrades (54.7%→54.3% over a comparable size range) — LoRA's weight-level edits leak into unrelated capabilities in a way a separate KV-cache prefix does not.

Ablations isolate what makes Self-Study work: using 5 diverse seed-prompt types (vs. a single generic seed prompt) improves MTOB chrF by 7.9 points (24.1→32.0) and LongHealth accuracy by 4.8 points (43.6→48.4), though it makes no difference on QASPER (whose queries are less reasoning-intensive). The context-distillation objective itself (vs. plain next-token prediction on the same synthetic conversations) improves MTOB chrF by 8.6 points (24.9→33.5). The naive baseline — next-token prediction directly on the raw corpus — achieves near-perfect memorization at **107× less memory than ICL**, but fails to generalize beyond corpus-regurgitation queries, motivating the whole Self-Study recipe. **Limitation the authors flag**: Self-Study is far more expensive than ICL prefill — training an ICL-quality Cartridge takes roughly 30 minutes on an 8×H100 node for Llama-8B, so it's not a drop-in replacement but a compute-for-memory tradeoff best suited to corpora queried repeatedly (e.g., training offline during off-peak hours).

### IndexCache: Cross-Layer Reuse of Sparse-Attention Indices

**IndexCache** (Bai, Dong, Jiang, Lv, Du, Zeng, Tang, Li — Tsinghua University / Z.ai, 2026) targets a cost that sits *upstream* of the KV cache itself: the per-layer indexing overhead in sparse attention. It builds directly on **DeepSeek Sparse Attention (DSA)**, which adds a lightweight "lightning indexer" at every layer that scores all preceding tokens and selects only the top-k (k=2048) most relevant ones for the actual attention computation — cutting core attention from O(L²) to O(Lk) per layer. The catch: the indexer itself is still O(L²) and must run independently at *every* layer, so its total cost across N layers is O(NL²) — on a 30B DSA model, profiling showed the indexer consuming **27–81% of total attention latency** during prefill as context length grows from 10K to 200K tokens.

The key empirical observation: top-k token selections from the indexer are **highly correlated across adjacent layers** (70-100% overlap), mirroring similar cross-layer stability findings in full-attention models — except DSA has no full-attention oracle layer to anchor against, since full attention has been replaced by the indexer entirely. IndexCache exploits this by splitting layers into a small set of **Full (F) layers** that compute their own indexer and top-k selection, and a majority of **Shared (S) layers** that simply reuse the nearest preceding F layer's top-k index set (one conditional branch added to the inference loop; the cached index tensor requires no extra memory beyond what DSA already allocates). Two ways to pick the F/S pattern:
- **Training-free IndexCache**: a greedy layer-selection search that, starting from all-F, iteratively flips whichever F layer's removal least increases LM loss on a calibration set, until the target retention ratio is reached. Uniform interleaving (e.g., every 4th layer) is shown to be suboptimal — some layers (especially early/transitional ones) are much more sensitive to indexer removal than others — so greedy search consistently beats it at the same retention ratio.
- **Training-aware IndexCache**: a multi-layer distillation loss that trains each retained F-layer indexer against the *averaged* attention distribution of all S layers it serves (the paper proves this is gradient-equivalent to distilling against a single centroid target), letting even a naive uniform interleaving pattern match full-indexer accuracy.

On a 30B DSA model (GLM-4.7-Flash base, MoE, 47 layers, MLA) across nine long-context and reasoning benchmarks, retaining only **1/4 of indexer computations** gave negligible quality degradation and **up to 1.82× prefill speedup and 1.48× decode speedup** at 200K context. In absolute terms: prefill time at 200K drops from 19.5s to 10.7s, and per-request decode throughput rises from 58 to 86 tok/s at 1/4 retention; full-batch decode throughput (KV cache saturated, ~800K tokens/GPU) improves 22-51% across context lengths, peaking at 197→297 tok/s (1.51×) at 200K. Pushing retention down to **1/8 is where things break**: training-free uniform interleaving collapses Long-context average from 50.2 to 35.3, and even the greedy-searched pattern only partly recovers it (to 46.1) — the paper is explicit that 1/8 retention without retraining is past the safe operating point. Training-aware IndexCache (retraining the retained indexers with the multi-layer distillation loss) closes this gap differently: it shows that the layer-pattern sensitivity seen in the training-free setting **vanishes once the model is retrained** — at 1/2 retention, plain uniform interleaving (51.6 Long Avg) matches or slightly beats the greedy-searched pattern (50.6), because the Shared layers and retained indexers jointly adapt to the sharing pattern during training. Ablating the cross-layer loss itself (training each retained indexer only against its own layer, as in standard DSA) drops Long Avg from 51.6 to 49.8, confirming the multi-layer distillation objective — not just retraining per se — drives the improvement. Preliminary results on the 744B-parameter GLM-5 model removing 50% of indexer computations showed comparable benchmark performance with roughly 1.2-1.3× end-to-end speedup, suggesting the cross-layer redundancy holds at production scale.

### Do Language Models Need Sleep? Offline Recurrence as a KV-Cache Replacement

This paper (Lee, McLeish, Goldstein, Fanti — CMU / University of Maryland, 2026) attacks the KV cache from the architecture side: instead of compressing or reusing it, **periodically convert it into persistent weights and throw it away**. The model is an SSM-attention hybrid (e.g., Gated Delta Networks interleaved with attention blocks, in the spirit of [[hybrid-architectures]]). Once the attention context window fills up (a fixed window size L), instead of immediately evicting the oldest tokens the model enters a **"sleep"**: it performs N additional offline forward passes over the about-to-be-evicted context, using each pass to further update the SSM blocks' fast weights (`S_t = α_t·S_{t-1} + β_t·v_t k_t^T`, a gated Hebbian/delta-rule update) before the KV cache for that window is finally cleared. Wake-time (online) prediction still costs exactly one forward pass — the extra N-pass compute is paid offline, during sleep, not during latency-sensitive generation.

The motivating failure mode: on a synthetic Rule-110 cellular-automaton task (P-complete; no known parallel shortcut), a 4-layer GDN-attention hybrid with hard KV-cache eviction every 24 tokens degrades to near-random guessing as the required reasoning depth `t` increases, **even though the amount of information to store is held fixed** — showing the bottleneck isn't fast-weight *capacity* but the *amount of computation* available to convert evicted context into useful state. Adding sleep loops fixes this directly: at the hardest setting tested (t=32), the no-loop baseline plateaus near 10% exact accuracy after ~5B training tokens, while **2 sleep loops reach ~20%, and 3-4 loops exceed 30%**, with identical context length, eviction rule, and wake-time compute across all conditions. On Depo (a k-hop graph-traversal task from Allen-Zhu et al. where the relevant edges are scattered across multiple already-evicted cache windows), increasing N similarly accelerates learning specifically on the harder multi-hop queries: the 1-loop model stalls on 4+-hop queries, the 2-loop model stalls on 8+-hop queries, and only the 4-loop model makes progress on the hardest 16-hop setting.

The effect holds on a realistic task too: fine-tuning pretrained **Jet-Nemotron 2B** and **Ouro 1.4B** (a depth-recurrent model augmented with inserted SSM/Jet layers) on **GSM-Infinite** (procedurally generated math reasoning with a context window of L=2000 forcing eviction mid-problem), more sleep loops widen the gap specifically on harder problems: for Jet-Nemotron, 6 loops raised six-operation-problem accuracy from 0.742→0.812 and eight-operation accuracy from 0.351→0.388; for Ouro, 4 loops raised six-operation accuracy from 0.419→0.615 and eight-operation accuracy from 0.210→0.272. Easy 2-4-operation problems saturate regardless of loop count — the benefit is concentrated on deeper reasoning, consistent with the synthetic-task results. Training backpropagates end-to-end through the entire sleep-then-predict computation graph (gradient flows through the refined fast weights, not through cached features, distinguishing this from other depth-recurrent models). The authors explicitly position this against Cartridges (above): both spend offline compute to turn context into a compact reusable representation, but Cartridges shorten what stays *in the attention KV cache*, while the sleep mechanism moves evicted context *out of the KV cache entirely* and into SSM weight-space.

The method also works with **sliding-window eviction** (retaining the most recent L-1 tokens rather than hard-clearing the whole window): fine-tuning Ouro 1.4B with window L=512 (so the full problem spans 4-6× the window), sleep loops raise GSM-Infinite two-operation accuracy from 0.596 to 0.905 — a **52% relative improvement** — showing sleep helps not just multi-step reasoning but also basic retrieval/compression when the active window is much smaller than the sequence. **Training-cost caveat**: because window j+1 can't be processed until window j finishes its N sleep passes, training is sequential across context windows (unlike standard parallel teacher-forcing), and cost grows roughly linearly with N; the authors note this hurts wall-clock time only when the window size L is too small to keep the GPU saturated — at large L the lost sequence-parallelism doesn't translate into a real slowdown. The paper is explicit that its "failure" claims for vanilla SSM-attention hybrids are budget-controlled (fixed training-token budget), not a claim that hybrids could never learn the task with unlimited data/compute.

## Scaling Context: Ring Attention

Ring Attention (Liu et al., 2023) distributes the KV cache across GPUs in a ring topology with overlapped computation and communication. This makes context length scale linearly with the number of devices — enabling 1M+ token contexts by using more GPUs rather than bigger GPUs.

### Ring Attention vs. Ulysses: Context Parallelism in RL Training

Prime Intellect's prime-rl 0.6.0 (see `raw/primeintellect-rl-at-1t-scale.md` and [[rl-training-systems]] § prime-rl 0.6.0) frames the same idea as a training-time concern: at 131k+ sequence length, it's *intermediate activations*, not parameters, that dominate GPU memory, and Context Parallelism (CP) is the lever that shards the sequence dimension to control that cost. prime-rl supports two general CP schemes:

- **Ring Attention** — the sequence stays sharded throughout the forward pass; at the attention layer, each rank holds its own Q/K/V shard and exchanges K/V with neighbors in a ring, overlapping communication with compute (as above).
- **Ulysses** — the sequence also stays sharded through the forward pass, but at the attention layer an all2all flips the layout from sequence-sharded to head-sharded, attention is computed across the head dimension, and another all2all flips the layout back. This is prime-rl's default CP scheme and works well with non-standard attention (linear attention, Mamba-style SSMs, etc.).

Neither scheme applies directly to **DeepSeek Sparse Attention (DSA)**, used by GLM-5 — see [[mixture-of-experts]] and the IndexCache discussion above for DSA's lightweight per-layer indexer. prime-rl's custom CP for DSA keeps sequences sharded through the projections, then gathers K/V across ranks (cheap, since K and V are already projected into a compressed latent space) so the sparse-attention indexer can see the *full* sequence and compute global top-k indices; core attention then runs only over that fixed-size index set, so its cost stays constant regardless of context length, and the whole scheme needs only one all-gather collective per attention layer — this is exactly the kind of context-parallel design that lets GLM-5 train on SWE-agent rollouts at up to 131k sequence length on 28 H200 nodes with sub-5-minute step times.

## KV Cache Behaviour at Long Context: Lessons from RULER

[RULER (2404.06654)](../../papers/04-efficiency/context-extension/RULER: What's the Real Context Size of Your Long-Context Language Models - 2404.06654.pdf) benchmarked 17 long-context LLMs on retrieval, multi-hop tracing, aggregation, and QA tasks across 4K–128K. The headline finding: **only half of models claiming 32K+ context windows can actually maintain quality at 32K**, and almost all degrade well before their advertised limit. The gap between "claimed length" and "effective length" (length passing a Llama-2-7B@4K quality threshold) is brutal:

| Model | Claimed | Effective |
| --- | --- | --- |
| Gemini-1.5-Pro | 1M | >128K |
| GPT-4 | 128K | 64K |
| Llama3.1 (70B) | 128K | 64K |
| Qwen2 (72B) | 128K | 32K |
| Command-R-plus | 128K | 32K |
| Yi-34B | 200K | 32K |

Implications for KV cache design:

- **Larger model size correlates with better long-context performance** (Yi-34B beats Yi-9B beats Yi-6B at all lengths when trained identically) — so don't expect a small-model KV-cache optimisation to recover quality on its own.
- **Training context length isn't everything.** Top-ranked open-source models include both Llama3.1 (trained at 128K) and Qwen2 (trained at 32K with inference-time extrapolation). LWM-1M is worse than LWM-512K at length 256K — a longer training context can hurt if it under-trains RoPE base.
- **Non-Transformer KV alternatives lag.** RWKV-v5 and Mamba-2.8B-slimpj degrade significantly at 8K, underperform Llama2-7B baseline up to 4K, and degenerate after — so SSM-based "infinite context" is not yet a working KV-cache replacement.

The [ProLong paper (2410.02660)](<../../papers/04-efficiency/context-extension/How to Train Long-Context Language Models (Effectively) - 2410.02660.pdf>) reinforces this with KV-cache-relevant findings: training **longer than the evaluation context** (e.g. train at 512K, eval at 64K) materially improves long-context performance — likely because the model learns to manage longer-range dependencies that the KV cache must hold. **Disabling cross-document attention** (intra-document masking) during continued long-context training improves both short and long-context performance and also boosts training throughput, since attention skips across packed-document boundaries. See [[attention-variants|document masking]].

## KV Cache Reuse: SGLang

SGLang's RadixAttention stores KV caches in a radix tree, enabling automatic prefix sharing across requests. If multiple requests share a system prompt or few-shot examples, their KV cache is computed once and reused, dramatically improving throughput for workloads with common prefixes.

## Related Topics
- [[attention-variants]] — GQA and MQA directly reduce KV cache size; IndexCache builds on DeepSeek Sparse Attention's top-k token selection
- [[hybrid-architectures]] — SSM fast-weight blocks are the substrate the "Sleep" mechanism consolidates evicted KV-cache context into
- [[long-context-training]] — Context extension only works if KV memory and effective retrieval both hold up
- [[inference-optimization]] — KV cache optimization is central to serving efficiency
- [[quantization-fundamentals]] — KV cache can be quantized to reduce memory further
- [[rl-training-systems]] — prime-rl's Ring Attention/Ulysses/custom-DSA context parallelism trains GLM-5 at 131k+ sequence length
- [[mixture-of-experts]] — DSA's per-layer indexer and Wide EP are the MoE-side counterpart to CP's sequence-side sharding

## Sources
- [How to Scale Your Model — Austin et al. (2025)](https://jax-ml.github.io/scaling-book/training) — exact KV size formula, LLaMA 70B example, decode step load time
- PagedAttention / vLLM (arxiv:2309.06180)
- Scissorhands (arxiv:2305.17118)
- [KV Cache Compression and Its Infra Problems — Mao, Chen, Huang, Yang, Wang & Han, NVIDIA Efficient AI Lab (June 2026)](../raw/nvidia-kv-cache-compression-infra-problems.md) — the two infrastructure collisions (FlashAttention score materialization, paged-attention fragmentation), TriAttention + Forward-Packing Compaction, 10.7× KV memory reduction at full-attention accuracy
- TurboQuant: Near-Optimal Vector Quantization for Memory-Constrained Attention — Zandieh, Daliri, Hadian, Mirrokni, ICLR 2026 (Google Research / DeepMind)
- PolarQuant — AISTATS 2026
- Ring Attention (arxiv:2310.01889)
- SGLang (arxiv:2312.07104)
- [Fast Transformer Decoding: One Write-Head is All You Need / MQA (1911.02150)](../../papers/02-architecture/transformers/Fast Transformer Decoding: One Write-Head is All You Need - 1911.02150.pdf)
- [GQA (2305.13245)](../../papers/02-architecture/transformers/GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints - 2305.13245.pdf) — uptraining recipe, head-count vs latency curve.
- [RULER (2404.06654)](../../papers/04-efficiency/context-extension/RULER: What's the Real Context Size of Your Long-Context Language Models - 2404.06654.pdf) — claimed vs effective context length for 17 long-context LMs.
- [ProLong (2410.02660)](<../../papers/04-efficiency/context-extension/How to Train Long-Context Language Models (Effectively) - 2410.02660.pdf>) — train longer than eval length; disable cross-document attention.
- [Cartridges: Lightweight and general-purpose long context representations via self-study (2506.06266)](../../papers/04-efficiency/context-extension/Cartridges: Lightweight and general-purpose long context representations via self-study - 2506.06266.pdf) — self-study context distillation into a trainable KV cache; 38.6× memory reduction, 26.4× throughput.
- [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse (2603.12201)](../../papers/04-efficiency/inference-kernels/IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse - 2603.12201.pdf) — cross-layer reuse of DeepSeek Sparse Attention's top-k indexer; up to 1.82× prefill speedup.
- [Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference (2605.26099)](../../papers/04-efficiency/context-extension/Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference - 2605.26099.pdf) — sleep-time consolidation of evicted KV-cache context into SSM fast weights.
- [KV Caching Explained — Hugging Face (Not Lain)](../raw/kv-caching-explained-huggingface.md) — step-by-step process, PyTorch `KVCache` pseudocode, `use_cache`/`cache_implementation` GenerationConfig example, 5.21× T4 benchmark on SmolLM2-1.7B.
- Coding the KV Cache from Scratch — Sebastian Raschka
- "RL at 1T Scale: prime-rl Performance Deep Dive" — Prime Intellect Team, Matej Sirovatka (June 21, 2026), `raw/primeintellect-rl-at-1t-scale.md` — Ring Attention vs. Ulysses context parallelism, and GLM-5's custom DSA context-parallel scheme, used to train at 131k+ sequence length.
