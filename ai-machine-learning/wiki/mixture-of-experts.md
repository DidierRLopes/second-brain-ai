# Mixture of Experts (MoE)

Mixture of Experts is a technique for scaling transformer models to massive parameter counts while keeping computation manageable. Instead of activating all parameters for every token, MoE routes each token to a subset of "expert" sub-networks. The 2026 frontier picture: MoE is **efficient when load-balanced**; routing, auxiliary or bias-based balancing, and global statistics (not local-batch statistics, which can mislead) are non-negotiable. Recent models trend toward **high sparsity** — over 100 total experts with ~10 active per token — because higher sparsity yields substantial performance improvements for fixed FLOPs.

## How It Works

In a standard transformer, the feed-forward network (FFN) in each layer processes every token identically. In an MoE layer, the FFN is replaced by multiple parallel "expert" FFNs, and a **router** (small learned linear projection + softmax) computes affinity scores for each expert per token. The top-k experts are selected — typically k ≪ total experts (e.g., 8 out of 384). Each selected expert processes the token; outputs are weighted by routing scores and summed.

The original modern recipe was [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (1701.06538)](../../papers/03-scaling/sparse-moe/Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer - 1701.06538.pdf): a trainable noisy top-k gate routes each example to a small number of experts, while auxiliary balancing losses keep both **importance** (probability mass) and **load** (actual token assignments) from collapsing onto a few experts. This paper established the core bargain that still defines MoE: huge total capacity, bounded per-token compute, and routing/balancing complexity as the price.

The foundational approach — **position-wise MoE with capacity constraints** — was established by [GShard (2006.16668)](../../papers/03-scaling/sparse-moe/GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding - 2006.16668.pdf). GShard introduced the key mechanisms: (1) **expert capacity thresholding** to prevent load imbalance (tokens exceeding threshold are "overflowed" to residual connections), (2) **local group dispatching** for efficient parallel execution (batches partitioned into groups, each group assigned fractional expert capacity), and (3) **auxiliary loss balancing** to enforce even load across all experts. The auxiliary loss is critical: it must use **global statistics aggregated across batches**, not local-batch statistics which can be misleading. GShard demonstrated that a 600B parameter model with position-wise sparse gating could achieve sub-linear scaling when trained with these balancing constraints.

## Key Architectures

- **Switch Transformer** (Google, 2021): [assigns each token to exactly one expert (2101.03961)](../../papers/03-scaling/sparse-moe/Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity - 2101.03961.pdf). Top-1 routing removes the weighted multi-expert combine and reduces communication, but makes capacity factors, token dropping, router stability, and load balancing more visible. The practical stability lessons were reduced initialization scale, router precision care, and a small auxiliary load-balancing loss.
- **Mixtral 8x7B** (Mistral AI): 8 experts of 7B each (46.7B total, ~13B active per token). Demonstrated strong quality-to-cost ratios competitive with much larger dense models.
- **GLaM** (Google): 1.2T parameters, ~96.6B active per token.
- **DeepSeek-V3**: 256 experts; the lineage Kimi K2 builds on.
- **Kimi K2** (2025): **384 experts, 8 active per token, sparsity 48, MLA attention**. Pushes the sparsity-driven design point. 1.06T total parameters.
- **gpt-oss-120b** (OpenAI 2025): MoE with gated SwiGLU. 116.83B parameters with GQA(8).
- **Intellect-3**: 106B total, 12B active. Post-trained on GLM-4.5-Air base.

## Unified Scaling Laws for Routed Language Models

Recent work ([Unified Scaling Laws for Routed Language Models, 2202.01169](../../papers/03-scaling/sparse-moe/Unified Scaling Laws for Routed Language Models - 2202.01169.pdf)) established a unified **bilinear scaling law** that relates routing architecture performance to both model size and expert count:

```
log L(N, E) = a log N + b log E + c log N log E + d
```

where `N` is the dense model size and `E` is the number of experts. This generalizes across three routing techniques (Sinkhorn-BASE, RL-R via Reinforcement Learning, and HASH deterministic routing), showing that **all routing methods scale as a unified power law in N and E**, with architecture choice affecting only the intercept and slope terms. The framework introduces **Effective Parameter Count (EPC)** — a mapping that equates the performance of a routed network with N and E to an equivalent dense model size. Key finding: routing is beneficial when N < ~1.3B; above that, dense models scale more efficiently unless routing overhead is negligible.

## Sparsity, Granularity, and Shared Experts

**Sparsity** = total experts / active experts. Higher sparsity improves loss and benefits more from increasing compute. Kimi K2's analysis showed substantial performance gains from increased sparsity at fixed FLOPs — motivating their move to 384 total experts. The Ant Group scaling study [Towards Greater Leverage (2507.17702)](../../papers/03-scaling/scaling-laws/Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models - 2507.17702.pdf) frames this as **Efficiency Leverage**: the dense compute divided by the MoE compute needed to reach the same loss. Their main finding is that activation ratio drives leverage by a power law, and the leverage gets stronger at larger compute budgets.

**Granularity** = (intermediate dim per expert) / (dense MLP intermediate dim). Higher granularity means more experts each with a smaller dimension. Recent models range from 2 (gpt-oss-120b) to 8 (qwen3-next-80b-a3b). Ant Group showed granularity doesn't significantly change loss monotonically but does drive **efficiency leverage** — the FLOP ratio for an MoE to match dense model loss — with an empirical sweet spot around 8-12 under their load-balancing setup.

**Shared experts**: always-on experts that absorb basic recurring patterns, freeing other experts to specialize more aggressively. One shared expert is often enough; DeepSeek-V2 uses two at some implementation complexity cost.

## Systems Reality: The Three Walls

Sparse activation creates a **parameter-compute mismatch**: an MoE has to store and update all experts, but each token only computes through a few of them. [Scalable Training of Mixture-of-Experts Models with Megatron Core (2603.07685)](../../papers/03-scaling/training-optimization/Scalable Training of Mixture-of-Experts Models with Efficient Sparsity - 2603.07685.pdf) describes the resulting systems problem as three coupled walls:

- **Memory wall**: all expert weights, gradients, and optimizer states must fit somewhere even when only `k` experts activate per token.
- **Communication wall**: expert parallelism requires all-to-all token dispatch and combine. If experts span nodes, the all-to-all can dominate step time.
- **Compute wall**: fine-grained experts create many small GEMMs, routing/permutation overhead, dynamic shapes, and load imbalance.

The practical stack is therefore more than "add experts." Megatron-Core's answer is Expert Parallelism plus **Parallel Folding**, which decouples attention-layer and MoE-layer parallelism so each can use a topology that fits its communication pattern. It then layers in grouped GEMM, fused router/permutation kernels, DeepEP/HybridEP dispatchers, activation recomputation/offload, CUDA Graphs for static parts, and FP8/FP4 recipes. This is why MoE architecture choices and training ops cannot be separated: higher sparsity improves the scaling law only if the dispatch, memory, and small-GEMM overheads are kept under control.

### Wide Expert Parallelism in RL Training and Inference

Prime Intellect's prime-rl 0.6.0 (see `raw/primeintellect-rl-at-1t-scale.md` and [[rl-training-systems]] § prime-rl 0.6.0) gives the memory wall a concrete number on the training side: for an 800B-parameter, 78-layer MoE with FP32 master weights, all-gathering a single full layer under FSDP costs roughly `(800B × 4) / 78 ≈ 40GB`, and with one layer of FSDP prefetch overlap that's ~80GB just for active-layer weights — untenable at 1T+ scale. Expert Parallelism sidesteps this by never gathering the full layer: at EP=8, tokens are dispatched and combined via all2all instead, since experts (not attention weights) dominate layer memory. prime-rl supports two EP backends with opposite scaling behavior — **torch-native all2all** is slightly faster within a single node (e.g. EP=8), but **DeepEP** wins by a large margin once EP spans multiple nodes.

On the inference side, RL rollout generation optimizes for *throughput* rather than latency (unlike user-facing serving), which favors **Wide EP** — large-scale expert parallelism spanning ≥32 GPUs, combined with a large data-parallel rank (e.g. 32) so that a large group of GPUs, each holding a different subset of experts, each acts as its own serving endpoint, synchronized per-layer through dispatch/combine. This is the same communication-wall tradeoff as training-side EP, but tuned for rollout throughput instead of memory ceiling.

## Load Balancing (Non-Negotiable)

If load balancing fails, training and inference efficiency plummet — and so does effective learning capacity (a few popular experts absorb most updates; others go untrained). Several strategies:

### Loss-Based Load Balancing (LBL)

Add an auxiliary loss term:

```
L_balance = α * E * sum_e (f_e * p_e)
```

where `E` is the number of experts, `α` controls balancing strength, `f_e` is the fraction of tokens routed to expert `e`, and `p_e` is the average routing probability for expert `e`. In perfect balance, `f_e = p_e = 1/E`.

**Must use global statistics aggregated across batches**, not local-batch statistics — local batches can be narrow and mislead routing decisions. `α` must not be so large that uniformity overwhelms the primary training objective.

### DeepSeek-V3 Loss-Free Load Balancing

Instead of adding a loss term that creates interference gradients, DeepSeek-V3 adds a **bias term** to affinity scores going into the routing softmax. The bias is updated decoupled from gradients based on observed load imbalance:

```
b_e <- b_e + u * sign(c_e - c̄)
```

where `c_e` is the load (tokens routed) for expert `e`, `c̄` is the mean across experts, and `u` is the bias update speed (a kind of learning rate). This version includes recentering of bias updates.

### Sequence-Wise Auxiliary Loss

Extends auxiliary balancing to promote balance within a single sequence (rather than just across batches). For each token position `t` in a sequence of length `T`, each expert `e` is assigned a normalized routing score `s_{t,e}`. Averaging over positions gives `P_e` = how often expert `e` is considered. The term `f_e` reflects the fraction of times expert `e` is actually selected (top-k after bias). The loss encourages `f_e * P_e` to be similar across experts — uneven utilization within sequences increases the loss.

### SMEBU (Sequence-wise MoE Balancing with Uniformity)

Sequence-level balancing using a normalized per-expert violation `v_e = (load_e - mean_load) / std_load` — scale-independent of sequence length and batch size. Maintains a momentum buffer with factor `β`, plus soft-clamping `tanh` with tunable scale `α`:

```
v_e <- (load_e - mean_load) / std_load
b_e <- b_e + α * tanh(β * (v_e - momentum_buffer))
```

The `tanh` provides continuity and stability (vs `sign` which forces ±α updates that oscillate). Momentum dampens noise — analogous to momentum SGD reducing variance in noisy gradient updates.

### Beyond Bias and Auxiliary Losses

Some implementations use **learnable routing functions** that adapt during training; others incorporate **expert capacity constraints** that prevent any single expert from being overwhelmed. The key insight across methods: effective load balancing must operate using **global statistics across multiple batches**.

### Hash Routing in Early Layers (DeepSeek-V4)

DeepSeek-V4 (see [[model-report-case-studies]] § DeepSeek-V4) departs from learned routing in its first several Transformer blocks: those early MoE layers use **Hash routing**, assigning each token to an expert via a predefined hash of the token ID rather than a learned router affinity score. This sidesteps router-driven load imbalance and instability at the layers closest to the embedding table, where DeepSeek-V4's authors observed MoE outliers were most disruptive to training. Later layers keep the standard DeepSeekMoE learned router with auxiliary-loss-free bias balancing, augmented by the sequence-wise balance loss described above — so the model mixes deterministic and learned routing by depth rather than using one scheme uniformly.

## MuonClip for MoE Stability

MoE models compound the attention-logit-explosion problem. **MuonClip** (Kimi K2) was developed in part because MoE expert-routing nondeterminism makes other stability techniques less reliable. See [[optimizers]] and [[training-stability]] for the details.

## When MoE Is the Right Call

From the [[frontier-training-playbook|architecture decision tree]]: choose **dense** if memory-constrained (MoEs must keep all experts loaded for routing), new to LLM training (simpler basics), or working under a tight timeline (well-documented dense recipes). Choose MoE when you need inference efficiency at scale and can manage the routing complexity.

## Related Topics

- [[transformer-architecture]] — MoE modifies the FFN layers
- [[attention-variants]] — many MoE models pair MoE FFN with GQA or MLA attention
- [[optimizers]] — MuonClip stabilizes MoE training at scale
- [[training-stability]] — MoE load imbalance is one of the "usual suspects" for training instability
- [[scaling-laws]] — MoE offers a different scaling trajectory than dense; Kimi K2's sparsity-driven design
- [[inference-optimization]] — MoE serving needs care due to large total parameter counts
- [[frontier-training-playbook]] — where MoE sits in the architecture decision tree
- [[rl-training-systems]] — prime-rl's Wide EP and FSDP+EP memory math for training/serving trillion-parameter MoE models in RL
- [[kv-cache]] — Context Parallelism (Ring Attention/Ulysses/custom DSA) as the sequence-side counterpart to EP's expert-side sharding
- [[model-report-case-studies]] — DeepSeek-V4's hash-routed early layers and sequence-wise balance loss in full model context

## Sources

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (1701.06538)](../../papers/03-scaling/sparse-moe/Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer - 1701.06538.pdf) — Foundational sparsely-gated MoE with noisy top-k gating and auxiliary loss balancing.
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (2101.03961)](../../papers/03-scaling/sparse-moe/Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity - 2101.03961.pdf) — Top-1 routing simplification, capacity factors, reduced init scale.
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (arxiv:2006.16668)](../../papers/03-scaling/sparse-moe/GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding - 2006.16668.pdf) — Position-wise MoE, expert capacity constraints, SPMD partitioning.
- [Unified Scaling Laws for Routed Language Models (arxiv:2202.01169)](../../papers/03-scaling/sparse-moe/Unified Scaling Laws for Routed Language Models - 2202.01169.pdf) — Bilinear scaling law across routing techniques, Effective Parameter Count.
- [Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models (2507.17702)](../../papers/03-scaling/scaling-laws/Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models - 2507.17702.pdf) — Efficiency Leverage, activation ratio, expert granularity.
- [Scalable Training of Mixture-of-Experts Models with Megatron Core (2603.07685)](../../papers/03-scaling/training-optimization/Scalable Training of Mixture-of-Experts Models with Efficient Sparsity - 2603.07685.pdf) — MoE systems stack, parallel folding, memory/communication/compute walls.
- Mixture-of-Experts (MoE) LLMs — Cameron R. Wolfe (cameronrwolfe.substack.com)
- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- DeepSeek-V3 technical report (loss-free balancing).
- Kimi K2 technical report (sparsity 48, 384 experts, MuonClip).
- Ant Group MoE study (granularity vs efficiency leverage).
- GLaM (arxiv:2112.06905).
- "RL at 1T Scale: prime-rl Performance Deep Dive" — Prime Intellect Team, Matej Sirovatka (June 21, 2026), `raw/primeintellect-rl-at-1t-scale.md` — Wide EP for RL inference throughput, and the FSDP+EP memory math (800B params/78 layers/~40GB all-gather buffer) for RL training.
- [DeepSeek-V4: Architecture and Training Breakdown](https://www.k-a.in/DeepSeek-V4.html) — third-party writeup of DeepSeek's technical report; not currently in the repo as a PDF. See `raw/deepseek-v4-analysis.md` — hash-routed early MoE layers and sequence-wise balance loss.
