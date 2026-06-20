# How to Scale Your Model: A Systems View of LLMs on TPUs

A free, open online textbook by researchers from Google DeepMind (now also at MatX) covering the systems science of scaling language models — how TPUs and GPUs work, how LLMs run on real hardware, and how to parallelize training and inference efficiently at massive scale. Published February 4, 2025.

Source: [How to Scale Your Model — jax-ml.github.io/scaling-book](https://jax-ml.github.io/scaling-book/)

---

## Authors and Affiliation

| Author | Affiliation |
|---|---|
| [Jacob Austin](https://www.jacobaustin.org/) | Google DeepMind |
| [Sholto Douglas](https://x.com/_sholtodouglas) | — |
| [Roy Frostig](https://cs.stanford.edu/~rfrostig/) | — |
| [Anselm Levskaya](https://anselmlevskaya.com/) | — |
| [Charlie Chen](https://x.com/charliexychen) | — |
| [Sharad Vikram](https://sharadvikram.com/) | — |
| [Federico Lebron](https://fedelebron.com/) | — |
| [Peter Choy](https://x.com/pchoy95) | — |
| [Vinay Ramasesh](https://x.com/vinayramasesh) | — |
| [Albert Webson](https://representation.ai/) | — |
| [Reiner Pope*](https://x.com/reinerpope) | — |

*Many of these are now at MatX. Work originally done at Google DeepMind.

**Acknowledgments:** James Bradbury and Blake Hechtman derived many of the foundational ideas.

**Citation:**
```
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
```
```bibtex
@article{scaling-book,
  title = {How to Scale Your Model},
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
  publisher = {Google DeepMind},
  howpublished = {Online},
  note = {Retrieved from https://jax-ml.github.io/scaling-book/},
  year = {2025}
}
```

---

## Why This Book Matters

> "A 20% win on benchmarks is irrelevant if it comes at a 20% cost to roofline efficiency. Promising model architectures routinely fail either because they can't run efficiently at scale or because no one puts in the work to make them do so."

Three or four years ago, ML researchers didn't need to understand hardware-level scaling. Today, even small models run so close to hardware limits that novel research requires thinking about efficiency at scale. Scaling laws have pushed models perpetually to the frontier of hardware — efficient scaling is now inextricably tied to cutting-edge research.

**The goal of "model scaling"** is to increase the number of chips used while achieving a proportional, linear increase in throughput — known as **strong scaling**. Adding chips reduces compute time but adds inter-chip communication cost. When communication takes longer than computation, you are **communication-bound** and cannot scale strongly. Per-chip bottlenecks are equally critical: memory bandwidth, total memory, and compute interact in ways that require careful design.

**Co-design challenge:** hardware designers must bet on what algorithms will look like when chips become available, 2–3 years in advance. TPUs were designed for ML workloads (matrix multiplication is unique: N FLOPs per byte — high arithmetic intensity). GPUs with Tensor Cores are evolving to fill the same niche.

**Expected background:** Basic understanding of LLMs and the Transformer architecture, basic familiarity with JAX. Useful pre-reading: [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), [original Transformer paper](https://arxiv.org/abs/1706.03762).

---

## Book Structure (12 Parts)

### Part 0: Introduction
[https://jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/)

Overview and motivation; this page.

---

### Part 1: A Brief Intro to Roofline Analysis
[https://jax-ml.github.io/scaling-book/roofline](https://jax-ml.github.io/scaling-book/roofline)

Algorithms are bounded by three resources: **compute**, **communication**, and **memory**. Roofline analysis estimates how fast an algorithm will run by identifying which resource is the bottleneck.

For a matmul [B,D]×[D,F]:
- Arithmetic intensity scales as ≈B for B ≪ D, F
- Below hardware's peak arithmetic intensity: **memory-bound** (compute idles waiting for data)
- Above it: **compute-bound** (hardware is fully utilized)
- Critical batch size B* separates the two regimes

This is the same framework referenced in [[frontier-async-rl]] to describe the "rollout-bound vs training-bound" analogy for async RL policy lag.

Referenced externally: [ScaleRL](https://arxiv.org/abs/2510.13786), the async RL paper, cites §1 of this book for the roofline model.

---

### Part 2: How to Think About TPUs
[https://jax-ml.github.io/scaling-book/tpus](https://jax-ml.github.io/scaling-book/tpus)

How TPUs work as individual chips and as interconnected systems:
- How long should a matmul of a given size take? When is it compute-bound vs memory-bound vs communication-bound?
- How TPU clusters are wired: inter-chip links, bandwidth, latency topology
- How long does AllGather, AllReduce, scatter, or redistribute take across multiple TPUs?
- How to multiply matrices distributed differently across devices

Key concept: the interplay of per-chip compute, memory bandwidth, and total memory is the central scaling story.

---

### Part 3: Sharded Matrices and How to Multiply Them
[https://jax-ml.github.io/scaling-book/sharding](https://jax-ml.github.io/scaling-book/sharding)

Explains model sharding and multi-TPU parallelism through the lens of the most fundamental operation: sharded matrix multiplication.

Core distinction: sharding splits parameters or activations across devices. Different sharding strategies have very different communication costs and efficiency profiles.

---

### Part 4: All the Transformer Math You Need to Know
[https://jax-ml.github.io/scaling-book/transformers](https://jax-ml.github.io/scaling-book/transformers)

Exhaustive "Transformer math" — counting parameters, FLOPs, and memory for both training and inference. Every matrix, every normalization, every attention head.

Key questions answered:
- How much memory does the model use?
- How much time is spent on compute vs communication?
- When does attention become important relative to feed-forward blocks?
- Exact sizes of every matrix; where normalization occurs; how many parameters and FLOPs in each part

Note on FLOPs: defined as total number of adds and multiplies required (not "operations per second" — that's FLOPs/s).

Useful for: estimating cost of a training run, sizing hardware requirements, comparing architectural choices.

---

### Part 5: How to Parallelize a Transformer for Training
[https://jax-ml.github.io/scaling-book/training](https://jax-ml.github.io/scaling-book/training)

The core of the book. Given a model of some size and some number of chips, how do you parallelize to stay in the strong scaling regime?

**Four primary parallelism techniques:**
1. **Data parallelism** — replicate model across devices, split batch
2. **Tensor parallelism** (Megatron-style) — split individual weight matrices across devices
3. **Pipeline parallelism** — split model layers across devices, pipeline micro-batches
4. **Expert parallelism** — specific to MoE architectures; route tokens to different devices

**Memory reduction techniques:**
- **Rematerialization (activation checkpointing)** — recompute activations during backward pass instead of storing them
- **Optimizer/model sharding (ZeRO)** — distribute optimizer states, gradients, and parameters across data-parallel ranks (FSDP/ZeRO)
- **Host offload** — move optimizer states to CPU memory
- **Gradient accumulation** — simulate larger batch sizes by accumulating gradients over multiple micro-batches

Goal: by the end of this section, you can choose the right parallelism scheme for a new architecture or hardware setting.

---

### Part 6: Training LLaMA 3 on TPUs
[https://jax-ml.github.io/scaling-book/applied-training](https://jax-ml.github.io/scaling-book/applied-training)

Applied tutorial: concretely apply Part 5 to LLaMA 3.
- How would you set up training?
- How long would it take?
- How much would it cost?
- What parallelism configuration is optimal?

---

### Part 7: All About Transformer Inference
[https://jax-ml.github.io/scaling-book/inference](https://jax-ml.github.io/scaling-book/inference)

Inference adds a new consideration — **latency** — and changes the memory landscape relative to training. Topics:
- How disaggregated serving works (prefill vs decode separation)
- KV cache mechanics and memory implications
- Latency/throughput tradeoffs in serving
- How inference parallelism differs from training parallelism
- Continuous batching implications

---

### Part 8: Serving LLaMA 3 on TPUs
[https://jax-ml.github.io/scaling-book/applied-inference](https://jax-ml.github.io/scaling-book/applied-inference)

Applied tutorial: LLaMA 3 inference on TPU v5e.
- How much does it cost to serve?
- What are the latency/throughput tradeoffs?
- How to configure serving for different SLAs?

---

### Part 9: How to Profile TPU Code
[https://jax-ml.github.io/scaling-book/profiling](https://jax-ml.github.io/scaling-book/profiling)

Real LLMs are never as simple as the theory. The JAX + XLA stack, and how to use the JAX/TensorBoard profiler to debug and fix real issues:
- Common profiling patterns
- Reading TensorBoard traces
- Identifying compute vs memory vs communication bottlenecks in real code
- Fixing inefficiencies discovered in production traces

---

### Part 10: Programming TPUs in JAX
[https://jax-ml.github.io/scaling-book/jax-stuff](https://jax-ml.github.io/scaling-book/jax-stuff)

JAX's magical APIs for parallelizing computation, with worked examples:
- `jit`, `vmap`, `pmap`, `shard_map`, `jax.lax.with_sharding_constraint`
- How JAX's functional programming model interacts with XLA compilation
- Fun examples and worked problems
- Functional transformations for ML

---

### Part 11: Conclusions and Further Reading
[https://jax-ml.github.io/scaling-book/conclusion](https://jax-ml.github.io/scaling-book/conclusion)

Closing thoughts and a curated list of further reading on TPUs and LLMs. Good entry point for finding primary sources on specific topics.

---

### Part 12: How to Think About GPUs (bonus)
[https://jax-ml.github.io/scaling-book/gpus](https://jax-ml.github.io/scaling-book/gpus)

A bonus section covering:
- How NVIDIA GPUs work (architecture vs TPUs)
- How they are networked (NVLink, NVSwitch, InfiniBand)
- How their rooflines differ from TPUs
- When to prefer GPUs vs TPUs for different workloads

---

## Key Numbers and Formulas (Quick Reference)

These are the concrete numbers from the book. Full derivations in the linked wiki pages.

### Hardware (TPUv5p)
- FLOPs/s: **4.59×10¹⁴** (bf16)
- HBM bandwidth: ~2 TB/s
- ICI bandwidth: 90 GB/s bidirectional (1.8×10¹¹ effective)
- ICI arithmetic intensity α = FLOPs/ICI_BW = **2550**
- DCN bandwidth: 6.25 GB/s per chip → DCN arithmetic intensity = **73,440**

### Parallelism Thresholds (TPUv5p)
| Strategy | Comm-bound when |
|---|---|
| Data Parallelism (DP) | batch/X < 2550 |
| FSDP | batch/X < 2550 |
| Tensor Parallelism | Y > M_Y · F / 2550 (≈8–16 way max) |
| FSDP+TP combined | batch/N < 100 (vs 850 for FSDP alone) |
| DCN scaling | batch/pod < 73,440 tokens |

Optimal FSDP degree: `X_opt = sqrt((B/F) · (M_X/M_Y) · N)`

### Transformer Math
- Total parameters: `N ≈ L · (4D² + 3DF)` (L=layers, D=d_model, F=d_ff)
- Training FLOPs: **6 · N · T_tokens** (Chinchilla rule)
- Inference FLOPs: **2 · N · T_tokens** (forward-only)
- Attention dominates MLP when: **T > 8D** (tokens > 8 × d_model)
- Block remat: 80% activation memory saved; ~8NT total FLOPs (vs 6NT)

### Inference
- Decode latency lower bound: `param_bytes / (N_chips × HBM_BW)`
- LLaMA 3-70B on 8× A100: ~**8.75ms per decode step** minimum
- Critical batch size (bf16): **~240 tokens** (TPUv5p)
- Critical batch size (int8 weights + bf16 FLOPs): **~120 tokens**

### KV Cache
- Per-sequence: `2 × S × L × K × H × bytes_per_element`
- LLaMA 3-70B (int8): **160 kB/token**, **5.12 GB at 32k context**
- At 128k context per sequence: ~20 GB (exceeds weights on single chip)

### Applied Example: LLaMA 3-70B on 8960 TPUv5p chips
- Optimal: **2048-way FSDP + 4-way TP** (from X_opt ≈ 1618)
- Memory per chip: ~2.4 GB activation checkpoints → **not memory-bound**
- Total training compute (15T tokens): 6 × 70B × 15T = **6.3×10²⁴ FLOPs**

See [[training-ops]] for full parallelism math, [[transformer-architecture]] for FLOPs tables, [[inference-optimization]] for prefill/decode analysis, [[kv-cache]] for memory formulas, [[gpu-kernel-engineering]] for hardware specs.

## Key Concepts and Takeaways

**Roofline model**: every compute primitive is either compute-bound, memory-bound, or communication-bound. Identifying which regime you're in is the first step to optimization.

**Arithmetic intensity**: FLOPs per byte of memory accessed. High arithmetic intensity (like matmul with large matrices) = can be compute-bound. Low arithmetic intensity (like layernorm, elementwise ops) = typically memory-bound.

**Strong vs weak scaling**: Strong scaling = same problem, more chips, proportionally faster. Weak scaling = scale problem size with chips, maintain same efficiency. ML research mostly cares about strong scaling.

**Why Transformers won**: matrix multiplication has N FLOPs per byte — one of the highest arithmetic intensities of any operation. This is uniquely suited to TPU/GPU tensor core architecture. Alternative architectures must match this arithmetic intensity to compete on real hardware.

**Parallelism tradeoffs**: Tensor parallelism requires high bandwidth (every layer needs AllReduce); Pipeline parallelism reduces bandwidth but adds pipeline bubbles; Data parallelism scales batch size (but has optimizer memory cost); Expert parallelism works for MoE but requires specialized routing.

---

## Practical Use Cases

- **"How expensive should this LLM be to train?"** → Part 4 (Transformer math) + Part 5 (training parallelism)
- **"How much memory do I need to serve this model?"** → Part 7 (inference) + Part 8 (applied serving)
- **"What's an AllGather?"** → Part 2 (TPUs) + Part 3 (sharded matmuls)
- **"Why is my training slow?"** → Part 9 (profiling)
- **"What parallelism should I use for this model size and chip count?"** → Part 5 (training) or Part 7 (inference)

---

## Related Topics
- [[gpu-kernel-engineering]] — FlashAttention-4, ThunderKittens, hardware-aware kernel design; the practical implementation layer below this book's abstractions
- [[inference-optimization]] — KV cache, speculative decoding, disaggregated serving; applied inference at scale
- [[training-ops]] — Practical training operations: throughput monitoring, parallel debugging, checkpoint management
- [[mixture-of-experts]] — Expert parallelism as the fourth parallelism technique; MoE-specific hardware challenges
- [[frontier-async-rl]] — References §1 of this book (roofline model) for the async RL staleness analogy
- [[frontier-training-playbook]] — Where hardware-aware training fits in the larger frontier training picture
- [[kv-cache]] — Deep dive into KV cache mechanics covered at a systems level in Part 7

## Sources
- [How to Scale Your Model — Austin et al., Google DeepMind (February 4, 2025)](https://jax-ml.github.io/scaling-book/)
- Full chapter links:
  - [Part 1: Rooflines](https://jax-ml.github.io/scaling-book/roofline)
  - [Part 2: TPUs](https://jax-ml.github.io/scaling-book/tpus)
  - [Part 3: Sharding](https://jax-ml.github.io/scaling-book/sharding)
  - [Part 4: Transformers](https://jax-ml.github.io/scaling-book/transformers)
  - [Part 5: Training](https://jax-ml.github.io/scaling-book/training)
  - [Part 6: Training LLaMA](https://jax-ml.github.io/scaling-book/applied-training)
  - [Part 7: Inference](https://jax-ml.github.io/scaling-book/inference)
  - [Part 8: Serving LLaMA](https://jax-ml.github.io/scaling-book/applied-inference)
  - [Part 9: Profiling](https://jax-ml.github.io/scaling-book/profiling)
  - [Part 10: JAX](https://jax-ml.github.io/scaling-book/jax-stuff)
  - [Part 11: Conclusions + Further Reading](https://jax-ml.github.io/scaling-book/conclusion)
  - [Part 12: GPUs](https://jax-ml.github.io/scaling-book/gpus)
- [Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Attention is All You Need — Vaswani et al.](https://arxiv.org/abs/1706.03762)
