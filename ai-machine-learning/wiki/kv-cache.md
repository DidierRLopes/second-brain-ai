# KV Cache

The KV (Key-Value) cache is the mechanism that makes autoregressive LLM generation efficient. Without it, generating each new token would require recomputing attention over the entire sequence from scratch. With it, only the new token's computation is needed, turning O(n²) per-step cost into O(n).

## How It Works

During the attention computation, each layer produces key (K) and value (V) matrices for every token. In autoregressive generation, previously computed K and V vectors don't change — they only depend on the input token and its position. The KV cache stores these vectors so they can be reused. At each generation step, only the new token's K and V are computed and appended to the cache.

The prefill phase processes the entire prompt at once (compute-bound). The decode phase generates tokens one at a time, reading the entire KV cache at each step (memory-bandwidth bound). This distinction is critical for optimization.

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

Other approaches include H2O (Heavy-Hitter Oracle) which keeps only the most-attended tokens, and StreamingLLM which maintains a sliding window plus "attention sink" tokens.

## TurboQuant: Near-Optimal Vector Quantization for KV Cache

Most KV-cache quantization (int8, NF4-style) quantizes each scalar independently. **TurboQuant** (Zandieh, Daliri, Hadian & Mirrokni, Google Research/DeepMind, ICLR 2026) instead treats each K or V vector as a single object to compress, combining two pieces:

- **PolarQuant** (AISTATS 2026): a rotation-based coordinate transform that reshapes a vector's distribution before quantizing it, so the quantization grid wastes less precision on directions with little information.
- **QJL (Quantized Johnson-Lindenstrauss) residual correction**: a 1-bit correction term, layered on top of PolarQuant's output, that recovers most of the error a naive rounding step would otherwise leave on the table.

The combination is **data-oblivious** (no calibration data or fine-tuning needed — unlike GPTQ/AWQ) and reaches within roughly **2.7× of the information-theoretic distortion limit** for vector quantization, which is unusually close for a training-free method. Reported results: KV cache quantized to **3 bits with 6× memory reduction**, and up to **8× faster attention on H100s** since less data needs to move through HBM. Conceptually this sits next to [[quantization-methods|QuIP's incoherence processing]] — both fight quantization error with a structured transform before rounding — but TurboQuant is purpose-built for the KV cache's vector-per-token structure rather than weight matrices.

## Scaling Context: Ring Attention

Ring Attention (Liu et al., 2023) distributes the KV cache across GPUs in a ring topology with overlapped computation and communication. This makes context length scale linearly with the number of devices — enabling 1M+ token contexts by using more GPUs rather than bigger GPUs.

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

The [ProLong paper (2401.02954)](<../../papers/04-efficiency/context-extension/How to Train Long-Context Language Models (Effectively) - 2401.02954.pdf>) reinforces this with KV-cache-relevant findings: training **longer than the evaluation context** (e.g. train at 512K, eval at 64K) materially improves long-context performance — likely because the model learns to manage longer-range dependencies that the KV cache must hold. **Disabling cross-document attention** (intra-document masking) during continued long-context training improves both short and long-context performance and also boosts training throughput, since attention skips across packed-document boundaries. See [[attention-variants|document masking]].

## KV Cache Reuse: SGLang

SGLang's RadixAttention stores KV caches in a radix tree, enabling automatic prefix sharing across requests. If multiple requests share a system prompt or few-shot examples, their KV cache is computed once and reused, dramatically improving throughput for workloads with common prefixes.

## Related Topics
- [[attention-variants]] — GQA and MQA directly reduce KV cache size
- [[long-context-training]] — Context extension only works if KV memory and effective retrieval both hold up
- [[inference-optimization]] — KV cache optimization is central to serving efficiency
- [[quantization-fundamentals]] — KV cache can be quantized to reduce memory further

## Sources
- [How to Scale Your Model — Austin et al. (2025)](https://jax-ml.github.io/scaling-book/training) — exact KV size formula, LLaMA 70B example, decode step load time
- PagedAttention / vLLM (arxiv:2309.06180)
- Scissorhands (arxiv:2305.17118)
- TurboQuant: Near-Optimal Vector Quantization for Memory-Constrained Attention — Zandieh, Daliri, Hadian, Mirrokni, ICLR 2026 (Google Research / DeepMind)
- PolarQuant — AISTATS 2026
- Ring Attention (arxiv:2310.01889)
- SGLang (arxiv:2312.07104)
- [Fast Transformer Decoding: One Write-Head is All You Need / MQA (1911.02150)](../../papers/02-architecture/transformers/Fast Transformer Decoding: One Write-Head is All You Need - 1911.02150.pdf)
- [GQA (2305.13245)](../../papers/02-architecture/transformers/GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints - 2305.13245.pdf) — uptraining recipe, head-count vs latency curve.
- [RULER (2404.06654)](../../papers/04-efficiency/context-extension/RULER: What's the Real Context Size of Your Long-Context Language Models - 2404.06654.pdf) — claimed vs effective context length for 17 long-context LMs.
- [ProLong (2401.02954)](<../../papers/04-efficiency/context-extension/How to Train Long-Context Language Models (Effectively) - 2401.02954.pdf>) — train longer than eval length; disable cross-document attention.
- KV Caching Explained — Hugging Face
- Coding the KV Cache from Scratch — Sebastian Raschka
