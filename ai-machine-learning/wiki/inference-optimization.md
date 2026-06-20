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

## Sources
- [How to Scale Your Model — Austin et al., Google DeepMind (2025)](https://jax-ml.github.io/scaling-book/training) — prefill/decode roofline, step time formula, disaggregated serving ratio, MoE inference thresholds
- Large Transformer Model Inference Optimization — Lilian Weng
- PagedAttention / vLLM (arxiv:2309.06180)
- Speculative Decoding (arxiv:2211.17192)
- SGLang (arxiv:2312.07104)
- Continuous Batching — Hugging Face
- The Llama Hitchhiking Guide to Local LLMs — Omar Sanseviero
- [Interfaze: The Future of AI is Built on Task-Specific Small Models (2602.04101)](../../papers/04-efficiency/inference-kernels/Interfaze: The Future of AI is Built on Task-Specific Small Models - 2602.04101.pdf)
