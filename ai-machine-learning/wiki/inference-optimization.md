# Inference Optimization

Making large language models fast and cost-effective at inference time is critical for deployment. LLM inference is primarily **memory-bandwidth bound** — the bottleneck is loading model weights from memory, not computing with them. This insight drives most optimization strategies.

## Prefill vs Decode: Roofline Analysis

From "How to Scale Your Model" (Austin et al., 2025). Source: https://jax-ml.github.io/scaling-book/training

LLM inference has two phases with fundamentally different compute profiles:

**Prefill** (processing the input prompt): arithmetic intensity ≈ `2N / (context_length × 4)` where N = num params. For short prompts, this is very high (>2500 on TPUv5p) — **compute-bound**. The entire prompt is processed in parallel, so throughput is high.

**Decode** (generating one token at a time): each step processes only 1 token. Arithmetic intensity = `2N / (4 · batch_size)` — at small batch sizes this is much lower than hardware's roofline. **Memory-bandwidth bound.**

### Step Time Formula

At each decode step (generating one token), time is:

```
T_step = max(T_param_load, T_flops) + T_kv_load
```

where:
- `T_param_load = param_size_bytes / (N_chips × HBM_bandwidth)` — load weights from HBM
- `T_flops = 2N / (N_chips × FLOPs_per_chip)` — compute matmuls  
- `T_kv_load = kv_cache_bytes_per_token × sequence_length / (N_chips × HBM_bandwidth)`

### Decode Latency Lower Bound

Even with infinite parallelism, decode latency can't be faster than the time to stream model weights through the chip:

> **Lower bound:** `T_decode_per_step ≥ param_size_bytes / (N_chips × HBM_bandwidth)`

For LLaMA 3-70B in bf16 on 8× A100s (2 TB/s bandwidth each):
- Param bytes: 70B × 2 = 140 GB
- T_param_load = 140 GB / (8 × 2 TB/s) = **8.75ms per step**
- At 60ms target latency, ~7 tokens/step is the max (beam search or speculative decoding can help)

### Critical Batch Size

The crossover point where inference becomes compute-bound (not memory-bound) happens when arithmetic intensity exceeds hardware capacity:

`B_critical = C / (2 × HBM_BW)` where C = FLOPs/s, HBM_BW = bandwidth.

**TPUv5p example:**
- C = 4.59e14 FLOPs/s, HBM_BW = 2 TB/s (approximate)
- B_critical ≈ 4.59e14 / (2 × 2e12) ≈ **~240 tokens/batch**

For int8 weights + bf16 FLOPs (2× memory savings): B_critical ≈ **120**
For int8 weights + int8 FLOPs (same memory, more FLOPs): B_critical ≈ **240**

Implication: to saturate the chip, you need batch size ≥ 240 active sequences simultaneously.

### Disaggregated Serving (Prefill-Decode Separation)

Modern high-throughput serving separates prefill and decode onto **different hardware pools**:

- Prefill servers: compute-bound workload; need high FLOPs
- Decode servers: memory-bandwidth-bound; need high HBM bandwidth  

**Optimal ratio of prefill to decode servers:**

```
N_prefill / N_decode = prefill_time / (median_output_tokens × step_time)
```

Example: 200 token prompt, 500 token output, 5ms prefill/step, 10ms decode/step:
- Prefill time: 200 × 5ms = 1000ms
- Decode time: 500 × 10ms = 5000ms
- Ratio: 1000/5000 = **1:5 prefill to decode servers**

Benefits over co-located serving:
- Each pool optimized for its workload (batching strategy, KV cache size)
- Prefill doesn't interrupt decode (reduces latency variance)
- Scales prefill and decode capacity independently

**Mooncake in production.** [Mooncake (2407.00079)](../../papers/04-efficiency/serving-systems/Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving - 2407.00079.pdf) extends phase separation into a KV-cache-centric architecture: CPU DRAM/SSD form a distributed prefix-cache tier, prefill streams cache state layer-by-layer to decode, and a conductor chooses nodes by predicted TTFT/TBT rather than simple cache-hit length. It improved SLO-compliant throughput 20-40% on two public long-context datasets, 50-525% on simulated 16k-128k prompts, and handled about 75% more requests than vLLM on a real trace. See [[kv-cache]] for the cache hierarchy and overload policy.

### Quantization Effects on Inference

| Precision | Effective weight size | B_critical (TPUv5p) |
|---|---|---|
| bf16 | 2 bytes/param | ~240 |
| int8 weights + bf16 FLOPs | 1 byte/param | ~120 |
| int8 weights + int8 FLOPs | 1 byte/param | ~240 |
| int4 weights | 0.5 bytes/param | ~60 |

Lower B_critical means you hit compute-bound operation with smaller batches → easier to achieve maximum throughput.

### MoE at Inference

For Mixture-of-Experts with E experts, k activated per token:
- Effective batch size for compute-bound: `B > 120 × E / k` (int8 weights)
- Example: E=64, k=2 → need B > 3840 tokens for compute-bound generation
- This is why MoE models require very high request throughput to be efficient at inference

## KV-Cache Optimization

The [[kv-cache]] is often the primary memory bottleneck in LLM serving. It stores key and value projections from previous tokens to avoid recomputation, but grows linearly with sequence length and batch size.

Key techniques: [[attention-variants|Grouped Query Attention (GQA)]] reduces cache size by sharing KV heads. **PagedAttention** (vLLM) manages cache memory like OS virtual memory, cutting waste from 60-80% to under 4%. **Scissorhands** and H2O compress the cache by evicting unimportant tokens. **SGLang's RadixAttention** enables prefix sharing across requests for up to 6.4× throughput on shared-prefix workloads.

## Quantization

The most impactful single optimization. Reducing weights from 16-bit to 4-bit gives ~4× memory reduction and often a large throughput improvement when inference is memory-bandwidth bound. See [[quantization-fundamentals]] for the mechanics, [[quantization-methods]] for GPTQ/AWQ/QuIP/AQLM tradeoffs, and [[practical-quantization]] for deployment formats.

## Speculative Decoding

Use a small, fast "draft" model to generate candidate tokens, then verify them in a single forward pass of the large model. The large model can accept or reject multiple tokens at once, effectively running the small model's speed with the large model's quality. Achieves 2-3× speedup with mathematically guaranteed identical output — no quality compromise. Variants include **Medusa** (parallel prediction heads, 2.2-3.6× speedup without a separate draft model) and **EAGLE** (using early-exit features as the draft).

### Acceptance/Rejection Rule and Why the Output Is Exact

Using target model \(M_p\) (distribution \(p\)) and draft model \(M_q\) (distribution \(q\)), the drafter proposes \(\gamma\) tokens autoregressively; \(M_p\) then verifies all \(\gamma\) in one parallel forward pass structurally identical to prefill — the prefix plus the \(\gamma\) draft tokens is fed in as if it were a prompt, producing \(\gamma+1\) next-token distributions in a single pass (the last one a "bonus" distribution available for free if every draft is accepted).

The per-token rule (speculative sampling): sample \(x \sim q\). If \(q(x) \le p(x)\), accept unconditionally. If \(q(x) > p(x)\), reject with probability \(1 - p(x)/q(x)\) and discard every token drafted after this position. On rejection, resample the replacement from the **adjusted distribution** \(p'(x) = \mathrm{norm}(\max(0, p(x) - q(x)))\) rather than from raw \(p(x)\).

The reason for the adjusted distribution is a probability-accounting argument: the acceptance path alone delivers \(P(\text{accepted}, x) = q(x)\cdot\min(1, p(x)/q(x)) = \min(p(x), q(x))\) of the true probability mass for token \(x\). For the total output probability to equal \(p(x)\) exactly, the resample-on-rejection path must supply only the remainder \(p(x) - \min(p(x),q(x)) = \max(0, p(x)-q(x))\) — normalizing this gives \(p'(x)\). Resampling from raw \(p(x)\) instead would double-count mass already paid out by acceptance and bias the output (e.g., over-represent tokens where \(q(x) > p(x)\)). This is what makes speculative decoding's output distribution **exactly identical** to the target model's, not an approximation.

### Acceptance Rate, Expected Tokens, and the Total-Variation Connection

Define the position-level acceptance rate \(\beta = \sum_x \min(p(x), q(x))\) (equivalently \(\mathbb{E}_{x\sim q}[\min(1, p(x)/q(x))]\)), and \(\alpha = \mathbb{E}(\beta)\) as a single scalar measuring how well the drafter approximates the target on average. With i.i.d. per-position acceptance, the number of tokens \(N\) produced by one iteration with budget \(\gamma\) follows \(P(N=k) = \alpha^{k-1}(1-\alpha)\) for \(k = 1,\dots,\gamma\) and \(P(N=\gamma+1) = \alpha^\gamma\) (the bonus-token case), giving expected tokens per iteration:
\[ E(N) = \sum_{j=0}^{\gamma} \alpha^j = \frac{1-\alpha^{\gamma+1}}{1-\alpha} \]

\(\alpha\) connects directly to total variation distance: defining the midpoint distribution \(M(x) = (p(x)+q(x))/2\), the divergence \(D_{LK}(p,q) = \sum_x |p(x)-M(x)|\) reduces to \(\sum_x |p(x)-q(x)|/2\) — exactly the total variation distance, and equal to \(1 - \sum_x \min(p(x),q(x))\). So \(\alpha = 1 - \mathbb{E}[D_{LK}(p,q)]\): the acceptance rate is one minus the expected distributional distance between drafter and target. At \(\alpha=0\) the drafter provides zero benefit (same throughput as standard decoding, pure loss from drafting overhead); at \(\alpha=1\) every iteration yields the full \(\gamma+1\) tokens.

### Wall-Time Improvement Formula

With cost ratio \(c = T_{M_q}/T_{M_p}\) (drafter forward-pass time relative to target forward-pass time — in the original paper, always < 0.05), the wall-time improvement factor over standard decoding is:
\[ \text{Improvement} = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)} \]
For the minimal case \(\gamma=1\) this simplifies to \(\frac{1+\alpha}{1+c}\) — a guaranteed speedup whenever \(\alpha > c\), i.e., whenever the drafter's acceptance rate exceeds its relative cost. High \(\alpha\) alone does not guarantee speedup; if \(\alpha < c\), verification overhead from a too-expensive drafter can erase the gains entirely.

Compute overhead works in the opposite direction: with \(\mu = F_{M_q}/F_{M_p}\) (drafter-to-target ops ratio per token), total arithmetic operations increase by a factor of \(\frac{(1-\alpha)(\gamma\mu+\gamma+1)}{1-\alpha^{\gamma+1}}\) versus standard decoding — speculative decoding trades **more total FLOPs for less wall time**, which is favorable specifically because inference is memory-bandwidth-bound rather than FLOP-bound (evaluating \(M_p\) on \(\gamma+1\) positions costs the same single memory read as evaluating it on one).

### Choosing \(\gamma\) and the Oracle Bound

Larger \(\gamma\) increases expected tokens per iteration with diminishing returns, while iteration cost \(\gamma c + 1\) grows linearly — so when \(c=0\) (free drafter) larger \(\gamma\) is always at least as good, but for \(c>0\) there's an interior optimum. A worked example at \(\alpha=0.8\), \(c=0.02\) finds the speedup peaks near \(\gamma=10\) (improvement factor ≈3.81) and then declines as drafting overhead dominates (e.g., 3.32× by \(\gamma=25\)). There's a hard theoretical ceiling regardless of \(\gamma\): removing the per-step cap entirely (an oracle that drafts an unbounded number of tokens) gives \(E(N_{\text{oracle}}) = 1/(1-\alpha)\) — 5 tokens/iteration at \(\alpha=0.8\) — which no finite \(\gamma\) can exceed. This oracle is unrealizable (knowing the position-level acceptance rate in advance would itself require running \(M_p\)), but it bounds adaptive-\(\gamma\) schemes.

### DSpark: Semi-Autoregressive Drafting and Load-Aware Verification

[DSpark (2607.05147)](../../papers/04-efficiency/serving-systems/DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation - 2607.05147.pdf) combines a deep parallel draft backbone with a lightweight sequential output head. The backbone preserves nearly constant draft latency as proposal length grows; the sequential head conditions later draft tokens on the sampled prefix, reducing the suffix-decay and multimodal-collision problem of independent parallel prediction.

A confidence head estimates per-position prefix survival, and a hardware-aware scheduler selects the verification length per request using current engine throughput. This matters because long proposals are valuable under light load but occupy scarce batch capacity under concurrency if their suffix is unlikely to survive.

Across Qwen3-4B/8B/14B targets, DSpark improved macro-average accepted length by 26.7-30.9% over autoregressive Eagle3 and 16.3-18.4% over parallel DFlash. Extending proposals from 4 to 16 tokens added only 0.2-1.3% round latency over DFlash at batch size 128. In DeepSeek-V4 production traffic, DSpark increased per-user generation speed by 60-85% for V4-Flash and 57-78% for V4-Pro at matched aggregate throughput. Because rejection sampling still uses the target probabilities, the acceleration preserves the target distribution rather than trading away output quality.

## Continuous Batching

Instead of processing batches of fixed size, continuously add new requests and remove completed ones. This maximizes GPU utilization since different requests finish at different times. Implemented in vLLM, TGI, and most modern serving frameworks.

## Model Parallelism

**Tensor parallelism:** Split individual layers across multiple GPUs. Essential for models too large for single GPU memory.

**Pipeline parallelism:** Split different layers across GPUs, processing different micro-batches simultaneously.

**Expert parallelism:** For [[mixture-of-experts]] models, different experts live on different GPUs.

## Distillation

[[knowledge-distillation|Knowledge distillation]] trains a smaller "student" model to mimic a larger "teacher." Not strictly an inference optimization but achieves the same goal: smaller, faster models with similar quality. Often combined with [[quantization-fundamentals|quantization]] — e.g., distill 70B → 7B, then quantize to 4-bit.

## Small Model Stacks and Hybrid Routing

[Interfaze: The Future of AI is Built on Task-Specific Small Models (2602.04101)](../../papers/04-efficiency/inference-kernels/Interfaze: The Future of AI is Built on Task-Specific Small Models - 2602.04101.pdf) proposes a fundamentally different inference architecture: instead of a single monolithic LLM for all tasks, use a **heterogeneous stack** of small, specialized models for perception and context construction, plus a user-selected LLM for final answer generation. This stack includes: (1) small DNNs and SLMs for OCR, document layout, charts/diagrams, ASR, and classification, (2) a context-construction layer for crawling, indexing, parsing, and retrieval, (3) an action layer that can browse, retrieve, execute code, and drive a headless browser, and (4) a thin controller that chooses which tools and small models to run before forwarding distilled context to the final LLM. Interfaze-Beta reports 83.6% on MMLU-Pro, 81.3% on GPQA-Diamond, 90.0% on AIME-2025, and strong multimodal scores, but the important systems point is the compute shift: most raw perception and context-building work happens outside the expensive generalist model. The paper also calls out practical limitations — delay from tool/model fan-out and over-building context — so hybrid stacks need cost-aware routing, not just more tools.

## Key Serving Frameworks

- **vLLM:** PagedAttention + continuous batching. De facto open-source serving standard
- **SGLang:** RadixAttention for prefix caching + compressed FSMs for structured output
- **TGI (Text Generation Inference):** Hugging Face's production serving solution
- **llama.cpp:** CPU/GPU inference, excellent for local deployment with GGUF models
- **TensorRT-LLM:** NVIDIA's low-level GPU-optimized inference library

## Related Topics
- [[kv-cache]] — Dedicated deep dive on KV cache mechanics and optimization
- [[knowledge-distillation]] — Compressing models for cheaper inference
- [[quantization-fundamentals]] — The most impactful single optimization
- [[attention-variants]] — FlashAttention and GQA reduce compute and memory
- [[scaling-laws]] — Inference cost drives model size decisions
- [[practical-quantization]] — Choosing quantization formats for serving
- [[applied-ml-systems]] — SLOs, overload, and production scheduler design

## Sources
- [How to Scale Your Model — Austin et al., Google DeepMind (2025)](https://jax-ml.github.io/scaling-book/training) — prefill/decode roofline, step time formula, disaggregated serving ratio, MoE inference thresholds
- Large Transformer Model Inference Optimization — Lilian Weng
- PagedAttention / vLLM (arxiv:2309.06180)
- Speculative Decoding (arxiv:2211.17192)
- SGLang (arxiv:2312.07104)
- Continuous Batching — Hugging Face
- The Llama Hitchhiking Guide to Local LLMs — Omar Sanseviero
- [Interfaze: The Future of AI is Built on Task-Specific Small Models (2602.04101)](../../papers/04-efficiency/inference-kernels/Interfaze: The Future of AI is Built on Task-Specific Small Models - 2602.04101.pdf)
- **Speculative Decoding - The Bits and the Bytes! (Part 1) — Aakash Kumar Nain (June 19, 2026)**, https://aakashkumarnain.github.io/posts/ml_dl_concepts/specdec_part1.html (`raw/speculative-decoding-aakashkumarnain-part1.md`). Source for the acceptance/rejection rule, the adjusted-distribution exactness proof, the acceptance-rate/total-variation-distance connection, the wall-time improvement and compute-overhead formulas, and the γ-choice/oracle-bound analysis above.
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (2407.00079)](../../papers/04-efficiency/serving-systems/Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving - 2407.00079.pdf) — SLO-aware prefill/decode separation, cache scheduling, and real/simulated throughput results.
- [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation (2607.05147)](../../papers/04-efficiency/serving-systems/DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation - 2607.05147.pdf) — semi-autoregressive drafts, survival confidence, engine-aware verification, and production speedups.
