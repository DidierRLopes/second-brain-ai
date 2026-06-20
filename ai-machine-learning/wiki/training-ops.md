# Training Ops

The least-celebrated part of frontier training: keeping the cluster running, the throughput high, and the loss curve honest. Across the 2026 frontier reports, **most failure modes are operational rather than algorithmic** — bad data shards being evicted from cache, dataloader bugs that grow lookup tables until allocations fail, tensor parallelism ranks sharing the same seed, or inference clusters bottlenecking on a single shared queue. The fix is rarely subtle once it's diagnosed; the hard part is noticing fast enough that you haven't burned a week of compute on a regression you could have caught with better logging.

## Pre-flight

Before the main run:

- **Cluster reservations**: Slurm time, no surprises.
- **Stress test GPUs**: GPU Fryer or DCGM to catch flaky nodes before they take down the run.
- **Storage hygiene**: upload checkpoints to third parties (S3 etc.) and delete local copies after saving the next checkpoint. Storage bloat is a real cause of throughput collapse.
- **Checkpoint + auto-resume** systems so a flaky node doesn't kill a 24-day run.

Allen Institute reports ~20% of compute is spent on evals. Automate logging not just for evaluation scores, but also throughput, loss, gradient norm, and node health — these are how you notice regressions early.

## Vanishing Throughput (SmolLM3)

Hugging Face observed a ~40% throughput drop (14k → 8k tokens/sec/GPU) within hours of starting the main SmolLM3 run.

**Cause**: cluster's network-attached storage used a "keep-hot" caching model — frequently accessed files stayed hot, "cold" files were evicted to third-party S3. With 24TB of training data, the storage was pushed to its limit and evicted dataset shards mid-training. The trainer then had to fetch them back, creating stalls.

**Fix 1**: swap storage method. Reserve a spare node with the dataset preloaded; use `fpsync` to copy (was 2× faster than `s5cmd`). This also fixed a related problem — when a node died and a replacement GPU had no data, swapping in the preloaded spare let training continue. The "new spare" then ran evals or dev jobs to not be wasted.

**Fix 2**: smaller residual throughput drops remained. Smaller step counts → smaller drops, which pointed at the dataloader. The nanotron dataloader was growing a lookup table per training step (mapping training step → next chunk of tokens to read) instead of keeping it bounded or precomputed. Stored in global memory, the growing table caused allocation failures, page faults, and worse cache locality. Switched to the `Tokenizedbytes` dataloader. Resolved.

## Noisy Loss (SmolLM3)

The SmolLM3 loss curve looked noisier than expected.

**Cause**: the dataloader read sequences sequentially for each document — no shuffling. So batches were no longer representative of the overall data distribution, increasing gradient variance. A single long file (e.g., a big code source file) would supply many consecutive sequences, spiking loss.

**Fix**: reshuffle tokenized sequences offline. An alternative was switching to random access in the dataloader, which has both higher memory usage and slower runtime — not preferred.

## Tensor Parallelism Seeds (SmolLM3)

After two days and 1T tokens, evals showed SmolLM3 (3B) underperforming SmolLM2 (1.7B) at the same training stage — despite a similar recipe.

**Cause**: SmolLM2's weights fit on a single GPU; SmolLM3's had to be sharded across 2 GPUs with tensor parallelism. The two TP ranks were initialized with the **same random seed**, which caused similar activations and gradients between the ranks, loss of feature diversity, and slower convergence.

**Fix**: distinct seeds per TP rank.

Generalizable lesson: **seed handling in parallelism setups is a high-leverage detail**. Verify it early, before the main run.

## Multi-client Orchestrator (Prime / Intellect-3)

Inference throughput should scale linearly with the number of nodes used. Prime found that the standard multi-node data-parallel strategy in vLLM didn't — as nodes increased, throughput plateaued.

**Fix**: abstract a multi-client orchestrator. Each inference node is deployed on an independent server — its own vLLM engine and scheduler, manages its own KV cache, batches its own requests. The orchestrator maintains **one client per node** (avoiding a single shared queue bottleneck), and distributes rollout requests across clients via round-robin scheduling.

This is also the underlying mechanism that enables **in-flight weight updates** for RL — the orchestrator continuously polls the trainer to update the inference pool once a new policy is available, the inference pool temporarily halts generation to update weights, then continues with rollouts. See [[alignment-methods]] for the RL side.

## PipelineRL for On-Policy RL Throughput

Conventional RL alternates between generation and training: generate a batch with the behavior policy, then run one or more optimizer steps. That creates a bad scaling tradeoff for long-sequence LLM RL. Large generation batches are needed to keep vLLM/SGLang-style engines efficient, but doing many optimizer steps per generated batch makes the data stale and off-policy.

[PipelineRL: Faster On-policy Reinforcement Learning by Leveraging Pipeline Parallelism (2509.19128)](../../papers/03-scaling/training-optimization/PipelineRL: Faster On-policy Reinforcement Learning by Leveraging Pipeline Parallelism - 2509.19128.pdf) runs generation and training concurrently and uses **in-flight weight updates**: the generation engine briefly receives newer model weights during ongoing sequence generation, then continues generating in-progress sequences. The resulting sequence can contain tokens sampled under slightly different weight versions, but the paper reports that the effective sample size stays close to near-on-policy baselines.

The reported setup trained Qwen 2.5 7B on long-form math reasoning with 128 H100s. PipelineRL reached the same reward about **2x faster** than the best stable conventional baseline because sample throughput was about 2x higher, while sample efficiency stayed comparable. The implementation pattern is worth remembering: distributed vLLM actor engines, a preprocessing stage for reference logprobs, DeepSpeed trainers, streaming queues, and explicit endpoints for generation, process-group initialization, and weight updates. This is the RL version of the same ops lesson as pre-training: utilization and freshness must be optimized together, not separately.

## MoE Training Systems

MoE training adds an ops layer that dense training does not have. [Scalable Training of Mixture-of-Experts Models with Megatron Core (2603.07685)](../../papers/03-scaling/training-optimization/Scalable Training of Mixture-of-Experts Models with Efficient Sparsity - 2603.07685.pdf) frames the bottlenecks as the memory wall, communication wall, and compute wall. All experts' parameters and optimizer states must live somewhere, expert parallelism needs all-to-all token dispatch/combine, and fine-grained experts turn big dense GEMMs into many smaller operations plus router/permutation overhead.

The operational workflow is: first find a memory-feasible parallelism configuration; then tune the parallelism strategy for the topology; then profile the actual bottleneck instead of assuming it is GEMM. Megatron-Core's production recipe combines Expert Parallelism, Parallel Folding, grouped GEMM, DeepEP/HybridEP dispatchers, communication overlap, recomputation/offload, CUDA Graphs where dynamic shapes allow it, and low-precision FP8/FP4 paths. The important planning point: increasing MoE sparsity can improve the scaling law while simultaneously making communication and small-GEMM utilization worse, so architecture and cluster layout have to be co-designed.

## The Usual Suspects

When the loss spikes mysteriously, check these first:

- **High learning rate.** First thing to look at.
- **Bad data batches.** Specific batches can spike loss; data-parameter state interactions are a real category.
- **MoE load imbalance.** Routing collapse, unbalanced experts. See [[mixture-of-experts]].
- **Storage / infra.** As in vanishing throughput above.
- **Poor initialization.** OLMo 2: `σ = 0.02` more stable than scaled init.
- **Precision.** Avoid fp16 — use bf16 or mixed precision.

Mitigations beyond the above:
- Stability techniques: [[training-stability|logit softcapping, z-loss, QK-norm]].
- **Data filtering**: OLMo 2 removed documents with 32+ repetitions of 1–13 token spans, significantly reducing spike frequency.
- If spikes persist: retrain around the spike (skip problematic batches), tighten gradient clipping.

## Training Parallelism (Systems View)

From "How to Scale Your Model" (Austin et al., Google DeepMind, 2025). Source: https://jax-ml.github.io/scaling-book/training

The goal of *strong scaling*: increase chip count, get proportional linear throughput increase. Performance at the cluster level depends on **hiding inter-chip communication by overlapping it with useful FLOPs**.

**Notation:** B = total batch tokens, D = d_model, F = d_ff, L = layers, C = FLOPs/s/chip, W = bidirectional ICI bandwidth, X/Y/Z = number of chips along mesh axes.

A Transformer layer approximated as two matmuls: **Win**: `bf16[D, F]` and **Wout**: `bf16[F, D]` with input **In**: `bf16[B, D]`.

### 4 Parallelism Schemes

| Strategy | Formula | What's sharded |
|---|---|---|
| Data Parallelism (DP) | In[B_X, D] · Win[D, F] · Wout[F, D] → Out[B_X, D] | Activations along batch; weights replicated |
| FSDP (ZeRO-3) | In[B_X, D] · Win[D_X, F] · Wout[F, D_X] → Out[B_X, D] | Activations + weights + optimizer state along batch |
| Tensor Parallelism (TP) | In[B, D_Y] · Win[D, F_Y] · Wout[F_Y, D] → Out[B, D_Y] | Weights along F; activations along D |
| FSDP + TP | In[B_X, D_Y] · Win[D_X, F_Y] · Wout[F_Y, D_X] → Out[B_X, D_Y] | Both above combined |

### Data Parallelism

- Forward pass: **no communication** (each chip sees a shard of the batch, full replica of weights)
- Backward pass: **AllReduce** on gradients (2 AllReduces per layer, each of size 2DF bytes)
- AllReduces are **off the critical path** — can overlap with next layer's backward
- Pure DP is rarely useful at scale because parameters + optimizer = 10 bytes/param → model must fit on one chip (e.g. ≤9B params on TPUv5p with 96GB HBM)
- **Becomes comm-bound when:** `B/X < C/W_ici` (per-device batch size below ICI arithmetic intensity)
- On TPUv5p: `C/W_ici = 4.59e14 / 1.8e11 = 2550`, so need B/X > 2550 (or ~850 with 3 mesh axes)

### FSDP (Fully-Sharded Data Parallelism / ZeRO-3)

Same communication cost as DP (AllReduce = AllGather + ReduceScatter), but **drastically reduces per-device memory** — weights, gradients, and optimizer state all sharded.
- Forward: AllGather weights before each layer, discard after use
- Backward: ReduceScatter gradients instead of AllReduce
- **Becomes comm-bound at same threshold as DP:** B/X < C/W_ici = 2550 (TPUv5p)
- For DeepSeek-V2's 40M token batch → can scale to ~47,000 chips before hitting bandwidth limit

### Tensor Parallelism (Megatron Sharding)

- Moves **activations** across chips (not weights). AllGather before Win, ReduceScatter after Wout
- Both ops are **on the critical path** — cannot overlap with compute
- Cost formula: T_comms = 4BD / W_ici; T_math = 4BDF / (Y·C)
- **Becomes comm-bound when:** `Y > M_Y · F / 2550` (TPUv5p)
- For most models this limits TP to **8–16 way**. LLaMA 3-70B (F≈30k) → max Y=11·M_Y
- Independent of batch size! TP bound depends on F not B

### Combined FSDP + Tensor Parallelism

FSDP moves weights (size ∝ DF/Y), TP moves activations (size ∝ BD/X). Combined they minimize communication jointly.

**Optimal FSDP amount:** `X_opt = sqrt((B/F) · (M_X/M_Y) · N)` where N = total chips.

**Minimum batch per chip to stay compute-bound:** `B/N > α² / (M_X · M_Y · F)` where α = C/W_ici = 2550.
- Plugging in F=32k, M_X·M_Y=2: **B/N > ~100 tokens/chip** (vs 850 with FSDP alone)
- This is ~8× lower minimum batch size than pure FSDP

**Example (LLaMA 3-70B, 4M token batch, 8960 TPUv5p chips):**
- Pure FSDP: need B/N > 850 → comms-bound at 8960 chips with 4M batch
- FSDP+TP: X_opt = sqrt(2 · 4.19e6 · 8960 / 28672) ≈ 1618 → use 2048-way FSDP + 4-way TP ✓

### Pipeline Parallelism

- Layers split across devices; activations passed between pipeline stages
- Low communication cost (just activations between adjacent stages, minimal bandwidth)
- **Problem: pipeline bubbles** — early stages idle while later stages compute; mitigated by microbatching
- DeepSeek-V3's "bubble-free" schedule: interleaves forward dL/dx, and dL/dW matmuls across pipeline stages
- Less critical on TPUs (dense ICI pods) than GPUs (which lack equivalent interconnect)

### Scaling Across Pods (DCN)

When scaling beyond a single ICI pod, use **pure data parallelism across pods**, model+FSDP within each pod.
- DCN bandwidth per TPU: ~6.25 GB/s (vs 90 GB/s ICI)
- ICI arithmetic intensity: C/W_ici = 2550 (TPUv5p)
- DCN arithmetic intensity: C/W_dcn = 4.59e14 / 6.25e9 = **73,440**
- **Becomes DCN-bound when batch per pod < 73k tokens**
- Crossing pods is fine for large batches; most models with BS≥1M easily satisfy this

### Compute and Comms Per Layer

| Strategy | Compute (fwd+bwd) | Comms (fwd+bwd) |
|---|---|---|
| DP | 12BDF/X | 0 + 8DF |
| FSDP | 12BDF/X | 4DF + 8DF |
| TP | 12BDF/Y | 4BD + 4BD |
| FSDP+TP | 12BDF/(XY) | (4BD/X + 4DF/Y) × 3/2 |

### Memory During Training

With bf16 params + fp32 optimizer (Adam) = 10 bytes/param. Gradient checkpointing dominates activations:
- **Block remat** (1 ckpt/layer): adds ~2× forward FLOPs, saves ≈ 80% activation memory → FLOPs ≈ 8·N·T (not 6·N·T)
- **Big matmuls only** (7 ckpts/layer): avoids large matmul recomputation, ~3× memory reduction

Example LLaMA-3 70B at 4M tokens on 8960 chips:
- Parameters (bf16): ~140 GB total, ~16 KB/chip
- Optimizer (fp32): ~560 GB total
- Gradient checkpoints (4×/layer): ~20.9 TB total
- → 8960 chips → ~2.4 GB/chip. **Not memory-bound at all; the big cluster is for FLOPs.**

## Training Ops Takeaways

- Throughput failures are usually data pipeline or storage issues, not model code.
- Dataloader behavior (shuffling, packing, access patterns) silently changes training dynamics.
- Seed handling in parallelism setups is a high-leverage detail; verify it early.
- Treat evals and logging as first-class citizens; they are how you notice regressions.

## Related Topics

- [[training-stability]] — the algorithmic side of loss spike prevention
- [[mixture-of-experts]] — load balancing, the most common MoE failure mode
- [[alignment-methods]] — in-flight weight updates and the multi-client orchestrator on the RL side
- [[rl-training-systems]] — how rollout freshness, pipeline RL, and verifier RL fit together
- [[gpu-kernel-engineering]] — kernel-level throughput constraints behind attention, MoE, and inference
- [[frontier-training-playbook]] — where training ops sits in the broader recipe
- [[diffusionblocks-blockwise-training]] — block-wise/local-loss training as a B× activation-memory reduction, an algorithmic alternative to remat for the same memory constraint

## Sources

- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- SmolLM3 report (vanishing throughput, noisy loss, TP seeds). See `raw/smollm3-hugging-face-report.md`.
- Prime Intellect, Intellect-3 (multi-client orchestrator).
- OLMo 2 report (init and data filtering for stability).
- [PipelineRL: Faster On-policy Reinforcement Learning by Leveraging Pipeline Parallelism (2509.19128)](../../papers/03-scaling/training-optimization/PipelineRL: Faster On-policy Reinforcement Learning by Leveraging Pipeline Parallelism - 2509.19128.pdf) — concurrent RL generation/training and in-flight weight updates.
- [Scalable Training of Mixture-of-Experts Models with Megatron Core (2603.07685)](../../papers/03-scaling/training-optimization/Scalable Training of Mixture-of-Experts Models with Efficient Sparsity - 2603.07685.pdf) — MoE parallelism, dispatch, memory, and throughput best practices.
