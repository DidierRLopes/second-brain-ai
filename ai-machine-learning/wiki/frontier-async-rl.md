# Is Frontier Asynchronous RL Solved?

Post by Luke J. Huang (May 31, 2026). A comprehensive survey of async RL as it has become the standard for large-scale post-training, covering the staleness problem, every algorithmic and systems fix frontier labs are using, why they still fail at high policy lag, and what the real open questions are.

Source: [Is Frontier Asynchronous RL Solved? — Luke J. Huang](https://luk-huang.github.io/personal-website/blog/is-frontier-asynchronous-rl-solved.html)

## TL;DR

- Async RL decouples rollout and training, yielding 2–3× throughput over synchronous pipelines.
- Stale trajectories introduce off-policy instability; the degree is captured by **policy lag K**.
- Every frontier open-weights lab uses async RL, each with its own fixes for two distinct problems: algorithmic (policy lag → IS ratio explosions) and systems (numerical mismatch between rollout/training engines).
- Standard clipping/masking methods stabilize at low K but all collapse at high K — they delay, not prevent, instability.
- **Sequence-level IS** is the estimator that scales with compute; token-level IS is structurally inconsistent at high policy lag.
- The bias-vs-variance tradeoff between estimators, the right collapse diagnostics, and practical variance control at scale are still open.

---

## The Larger Landscape: Frontier Post-Training Stack

The emerging pattern for frontier post-training is a stack:

```
SFT / offline distillation
→ on-policy distillation / self-distillation
→ specialist RL
→ multi-teacher OPD
→ unified model
```

- **SFT / offline distillation:** fast and stable, ceiling limited to teacher trajectories.
- **On-Policy Distillation (OPD):** model generates its own trajectories; ceiling set by teacher capability.
- **Self-distillation ([OPSD](https://arxiv.org/abs/2601.18734) / [SDPO](https://arxiv.org/abs/2601.20802)):** removes external teacher; model supervises itself under richer context (hint, verified answer, or environment feedback). Applied Compute's [RMSD](https://appliedcompute.com/research/relevance-masked-self-distillation) extends this with a filtered loss mask for OOD tasks.
- **RL:** model searches its own policy space; improvements compound; ceiling set by verifier, not data.
- **MIMO / MOPD:** [MIMO (arxiv:2601.02780)](https://arxiv.org/abs/2601.02780) trains domain specialists independently, then merges via On-Policy Distillation. Later adopted by DeepSeek V4.

See [Will Brown's comparison of modern post-training methods](https://x.com/willccbb/status/2050038277454143918) for a broader view.

---

## What Is Async RL and Why Use It?

In synchronous on-policy RL, you generate responses with the current model, then train on them. The loop is conceptually simple but wastes GPU time waiting for the slowest trajectories to finish.

**Asynchronous RL** disentangles generation and training into two independent loops: rollout workers keep producing trajectories while the trainer keeps updating the policy, communicating through periodic weight syncs and a trajectory buffer.

Frontier labs reporting 2–3× throughput gains from async RL:
- [GLM-5](https://arxiv.org/pdf/2602.15763) (Zhipu AI)
- [Ring 1T](https://arxiv.org/pdf/2510.18855)
- [DeepSeek V3.2](https://arxiv.org/pdf/2512.02556)
- [Minimax M2.5](https://www.minimax.io/news/minimax-m25)
- [Qwen 3.5](https://qwen.ai/blog?id=qwen3.5)
- [Intellect-3](https://arxiv.org/pdf/2512.16144)
- [Nemotron-3 Super](https://arxiv.org/pdf/2604.12374)
- [Laguna-M.1](https://poolside.ai/assets/laguna/laguna-m1-xs2-technical-report.pdf)

**The catch:** generated data is now "stale" — the policy that produced a trajectory is no longer exactly the policy being trained. Async RL is inherently *off-policy*.

**Off-policy correction via importance sampling:**

$$\mathcal{J}_{\text{off-policy}}(\theta) = \mathbb{E}_{x,\tau\sim\mu}\left[\underbrace{\frac{\pi_\theta(\tau|x)}{\mu(\tau|x)}}_{\text{IS ratio}} A(\tau,x)\log\pi_\theta(\tau|x)\right]$$

As the policy drifts from the behavior policy, IS ratios become extreme → instability.

### Open-Source Frameworks

| Framework | Notes |
|---|---|
| [AsyncRLHF](https://arxiv.org/abs/2410.18252) | First formal study of async RL for LLMs |
| [AReaL](https://github.com/inclusionAI/AReaL) | Widely cited research implementation; full paper at [arxiv:2505.24298](https://arxiv.org/abs/2505.24298) — decoupled PPO clipped against a proximal policy, not the stale behavior policy; full coverage in [[rl-training-systems]] |
| [LlamaRL](https://github.com/facebookresearch/llama-rl) | Meta Research |
| [PipelineRL](https://arxiv.org/abs/2509.19128) | ServiceNow; adds in-flight weight syncing |
| [AsyncFlow](https://arxiv.org/abs/2507.01663) | Huawei/MindSpeed-RL, Ascend NPUs; TransferQueue data-plane split, delayed parameter update; full coverage in [[rl-training-systems]] |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | NovaSky / UC Berkeley; modular for research |
| [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | Intellect-3 framework |
| [VeRL](https://github.com/verl-project/verl) | ByteDance; easy to use, highly featured; foundational paper is HybridFlow ([arxiv:2409.19256](https://arxiv.org/abs/2409.19256)), 2026 low-precision update is FP8-RL ([arxiv:2601.18150](https://arxiv.org/abs/2601.18150)) — full coverage in [[rl-training-systems]] |
| [TorchForge](https://github.com/meta-pytorch/torchforge) | Meta/PyTorch; uses TorchTitan + Monarch for fault-tolerant scheduling |
| [Slime](https://github.com/THUDM/slime) | Zhipu/GLM-5; battle-tested production |
| Forge | Minimax; battle-tested production |

---

## The Staleness Problem: Policy Lag K

**Policy lag K** = number of optimization steps by which the training policy is ahead of the inference policy. K=0 is fully on-policy.

**Steady-state policy lag:**

$$K_{\text{steady}} \approx \frac{\text{rollout latency per trajectory}}{\text{training step time}} \times \frac{N_{\text{rollout}}}{N_{\text{train}}}$$

**Roofline analogy** (from [JAX-ML scaling book §1](https://jax-ml.github.io/scaling-book/roofline)): when K_max < K_steady, the trainer exhausts the rollout buffer and stalls (rollout-bound regime, analogous to memory-bound). When K_max ≥ K_steady, training never waits (training-bound regime, full throughput).

**Long-horizon pressure:** rollout latency scales with sequence length. Longer reasoning tasks / more tool use → higher K_steady → more pressure to push K_max up → more instability risk. At short sequences, K=4 gets close to full async speedup. At long sequences, K=4 asymptotic speedup shrinks dramatically.

How to cap K: **FIFO windowed scheduling** (Minimax) — train on samples as they arrive; **policy-lag cutoff** (GLM-5) — discard samples with lag ≥ K_max. Capping K_max sacrifices throughput.

A third staleness granularity beyond trajectory- and token-level lag shows up specifically in agentic/sandboxed RL: **environment-state-level staleness**, where a partial-rollout resume continues from sandbox state (e.g. half-applied file edits) left by an older policy. This isn't fixed by any IS-reshaping method below — it lives in the environment layer, not the policy-gradient estimator. See [[rl-training-systems]] § Partial Rollout and Environment-State-Level Staleness for the full mechanism and the production case study (slime + GSPO) that surfaced it.

---

## Stabilizing Async RL: Algorithmic Controls

All approaches involve **reshaping the IS weights** to reduce variance, at the cost of introducing bias.

### IS Estimator Types

The off-policy objective uses the *sequence-level* IS ratio:

$$w(\tau) \triangleq \frac{\pi_\theta(\tau|x)}{\mu(\tau|x)} = \prod_{i=1}^{|\tau|} \underbrace{\frac{\pi_\theta(\tau_i|x,\tau^{<i})}{\mu(\tau_i|x,\tau^{<i})}}_{\triangleq \rho(\tau,i)}$$

Sequence-level ratio variance compounds with number of tokens — the **Curse of the Horizon** ([arxiv:1810.12429](https://arxiv.org/abs/1810.12429)).

- **Token-level IS** ρ(τ,i): used by original PPO and GRPO; avoids horizon curse but is biased off-policy
- **Sequence-level IS** w(τ): unbiased but high variance; scales well with compute
- **Geometric-mean IS** w(τ)^(1/|τ|): adopted by [GSPO](https://arxiv.org/abs/2507.18071) and GMPO; middle ground, but tracks closer to token IS than sequence IS at long horizons — not a reliable middle ground per simulations

### Algorithmic Methods Table

| Method | Mechanism | Used In | Biased? |
|---|---|---|---|
| **Truncated IS (TIS) / CISPO** | Clips IS ratio r into [r_low, r_high] — works at token, sequence, or geometric mean level | AReaL, [LlamaRL/AIPO](https://arxiv.org/abs/2505.24034), [Minimax M2.5](https://www.minimax.io/news/minimax-m25), [Laguna-M.1](https://poolside.ai/assets/laguna/laguna-m1-xs2-technical-report.pdf) | Yes |
| **IcePop / Masked IS (MIS)** | Masks (zeros out) tokens outside [r_low, r_high]; more aggressive than clipping | [GLM-5](https://arxiv.org/pdf/2602.15763), [Ring 1T](https://arxiv.org/pdf/2510.18855), [Intellect-3](https://arxiv.org/pdf/2512.16144), [Nemotron-3 Super](https://arxiv.org/pdf/2604.12374) | Yes |
| **DeepSeek Masking** | Masks negative-advantage trajectories when average log-ratio exceeds threshold; keeps all non-negative samples | [DeepSeek V3.2](https://arxiv.org/pdf/2512.02556) | Yes |
| **M2PO** | Drops tokens until (1/|τ|)Σ(log ρ(τ,i))² < threshold | [arxiv:2510.01161](https://arxiv.org/abs/2510.01161) | Yes |
| **ScaleRL** | Found TIS particularly effective for async RL because detached IS weights don't vanish when clipped | [arxiv:2510.13786](https://arxiv.org/abs/2510.13786) | Yes |

For more theoretical analysis see [Richard Li Part 1](https://richardli.xyz/post/rl-collapse-part1/) and [Part 2](https://richardli.xyz/post/rl-collapse-part2/).

---

## Stabilizing Async RL: Systems-Level Fixes

Policy lag is only one source of mismatch. A second class comes from the fact that rollout and training engines are fundamentally different systems: different parallelism, different kernel implementations, different MoE routing behavior. This mismatch exists even at K=0 and amplifies distributional gaps from policy lag.

### MoE Routing Replay
Identical weights still cause ~10% of MoE routing decisions to diverge per forward pass, causing collapse. Fix: record inference routing masks and replay them during training.
- [R3 (Rollout Routing Replay)](https://arxiv.org/abs/2510.11370)
- [GSPO](https://arxiv.org/abs/2507.18071) first mentioned this, targeting it from the algorithm side via geometric-mean IS
- **Production result (prime-rl 0.6.0):** Prime Intellect reports R3 reduces trainer/inference KL mismatch **by roughly an order of magnitude** in production GLM-5-scale training — a concrete number for this mechanism's effect, at the cost of routed-expert payloads reaching tens of Gbps (shape `[num_layers, top_k, seq_len]`, hundreds of GB), handled as opaque tensors processed only via PyTorch ops to avoid Python/event-loop overhead. See `raw/primeintellect-rl-at-1t-scale.md` and [[rl-training-systems]] § prime-rl 0.6.0. The same release also gives the "policy-lag cutoff (GLM-5)" row above a concrete name: `max_off_policy_steps`, paired with a KV-cache salt that forces new rollouts to populate a fresh KV cache rather than reuse one built by a mixture of policy versions.

### Token-In Token-Out (TITO)
Tokenizers have hysteresis — without TITO, tokenization discrepancies silently corrupt log-probability computation between rollout and trainer. References:
- [strands-sglang](https://github.com/horizon-rl/strands-sglang): first OSS implementation of TITO for agentic RL, tracks complete token trajectories with logprobs directly in SGLang
- [Prime Intellect Renderers](https://www.primeintellect.ai/blog/renderers): inference server handles only tokens; all templating and masking in client code

### Batch-Invariant Kernels
Batching changes floating-point reduction order, making log-probs nondeterministic across batch sizes.
- [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): developed batch-invariant kernels achieving *bitwise identical* train-inference behavior
- [TBIK](https://arxiv.org/abs/2511.17826): extends fix to tensor-parallel sizes (training TP=1, rollout TP>1)
- [DeepSeek V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf): dual-kernel strategy for attention (first kernel handles full sequences within one SM; second handles partial waves via distributed shared memory); replaces cuBLAS end-to-end with [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (split-k abandoned — neither cuBLAS nor split-k guarantee batch invariance)

### FP32 LM Head
bf16 rounding at the LM head distorts IS ratios.
- [Minimax M1](https://arxiv.org/abs/2506.13585): identified this, switched to fp32 LM-head
- [ScaleRL](https://arxiv.org/abs/2510.13786): confirmed it dramatically improves asymptotic performance. One-line fix.

### FP16
[Qi et al.](https://arxiv.org/abs/2510.26788): switching the full pipeline to FP16 reduces train-inference mismatch with only a few lines of code change. Particularly important on Ampere GPUs.

### Quantized Rollouts
Reduces rollout latency; first studied in [FlashRL](https://github.com/yaof20/Flash-RL) (INT8/FP8 rollouts with accurate log-probs, no accuracy drop on Qwen2.5-32B when combined with TIS). [Slime](https://github.com/THUDM/slime) adds int4 rollout quantization for long-horizon agentic tasks where generation latency dominates.

### Efficient Weight Sync
- **[PipelineRL](https://arxiv.org/abs/2509.19128):** in-flight weight syncs — broadcast weights after each optimizer step without halting generation, keeping policy lag near zero
- **Kimi MoonCake / LMSYS [P2P weight update for SGLang](https://www.lmsys.org/blog/2026-04-29-p2p-update/):** 7× speedup over NCCL for Kimi K2 (53s → 7.2s) via RDMA P2P transfers through Mooncake TransferEngine
- **Perplexity [TransferEngine](https://research.perplexity.ai/articles/weight-transfer-for-rl-post-training-in-under-2-seconds):** 1.3-second trillion-parameter weight updates on Kimi K2 via RDMA WRITE with static transfer schedule and pipelined execution; zero-copy writes to remote inference GPU memory, no control plane overhead
- **[Composer 2.5](https://cursor.com/blog/composer-2-5):** delta-compression updates — only sync changed weight portions across clusters

### KV Cache Recomputation
Interestingly, **not** beneficial: [Magistral](https://arxiv.org/abs/2506.10910) and [Nemotron-3 Super](https://arxiv.org/pdf/2604.12374) both tested recomputing KV caches after weight syncs and found no benefit.

---

## Why Async RL Still Collapses at High K

Systems fixes get you further — deterministic kernels, right precisions, routing replay. But none address policy lag itself. Even a perfectly aligned rollout engine still produces stale trajectories, and at high K the IS ratio distribution becomes extreme.

At K=12, sweeping token TIS, sequence TIS, and token MIS on Qwen2-1.5B (GSM8k) and Llama 3.1 8B Instruct (Math-500):

> All methods collapse at high policy lag. Some crash outright. Others survive longer depending on thresholds. Sequence-level TIS holds up better than token-level variants, but even it eventually degrades. Clipping and masking *delay* collapse; they don't *prevent* it.

The asymmetry between token and sequence IS is the key clue.

---

## Bias or Variance Wall? The Core Estimator Question

In on-policy RL, per-token ratios ρ(τ,i) stay near 1 → sequence ratio w(τ) = Πρ(τ,i) is well-behaved. In async RL with high K and long horizons, state-occupancy mismatch makes this product extreme. **Token IS misses this; Sequence IS corrects it at the trajectory level.**

Sequence IS's unbiasedness costs variance. At small batch sizes, the sequence-level ratio is noisy — which is why token IS looks competitive at low compute. Past a critical batch size, bias in token IS becomes the performance ceiling.

### Simple Horizon Simulation

Setting: MDP where a policy chooses bit sequences over horizon H, succeeding if fraction f are 1; behavior policy constrained to lag K behind training. Findings:

- **As H grows:** Token IS and GeoMean IS fall sharply; Sequence IS degrades more robustly. GeoMean IS tracks closer to token IS than to sequence IS at long horizons — ruling it out as a reliable middle ground.
- **At small batch sizes:** High variance of Sequence IS dominates; truncated Sequence TIS (clip extreme ratios) is more robust by trading small bias for much lower variance.
- **By B ≥ 128:** Regular Sequence IS outperforms all truncated variants.
- **Policy lag sweep:** Token IS and GeoMean IS degrade monotonically with K regardless of batch size B. Sequence IS at large B is extremely robust even at very high K, matching fully synchronous training.

> As greater compute is poured into RL, token IS is not approximately biased in the async RL regime — it's **structurally inconsistent** as compute scales.

---

## The Low-Bias Compute Scaling Hypothesis

Confirmed via ablations: at B=32, Sequence TIS collapses even before Token TIS. By B=64, it matches the synchronous K=0 baseline. At B=128, it surpasses it. Token TIS continues to collapse regardless of batch size.

> **Low-bias methods** are often less efficient at low compute because they expose more variance, but they preserve the correct objective and have more room to improve as compute, batch size, and variance control scale. **High-bias methods** are often more efficient at small scale, but their bias becomes the bottleneck at high compute.

---

## Open Questions

1. How do we stabilize Sequence IS at low and moderate batch sizes without reintroducing bias? TIS variants help but are still patching variance, not solving it.
2. Is scaling batch size the right lever for variance control, or are there cheaper ways to reduce gradient noise without more compute?
3. Are there architectural changes to mitigate train-inference bias and allow policy lag to be pushed higher? MoE architectures suffer particularly due to expert routing.
4. Can we observe something in gradient statistics *before* a crash happens? Part 2 of this series will discuss a specific statistical cause and one potential solution.

---

## Related Topics
- [[alignment-methods]] — GRPO, PPO, RLVR, IcePop as the RL algorithms async pipelines execute
- [[rl-training-systems]] — The systems layer: PipelineRL, HybridFlow/verl, AReaL, AsyncFlow, FP8-RL, rollout/trainer architectures, KL control, stale policies
- [[rl-scaling-laws]] — Compute scaling laws (ScaleRL, ProRL, Polaris) for the recipes these frameworks execute
- [[icepop-stabilizing-rl-moe]] — IcePop/MIS masking technique, one of the main algorithmic controls surveyed here
- [[on-policy-distillation]] — OPD/OPSD/MOPD as alternatives/complements to pure RL in the post-training stack
- [[reasoning-models]] — What async RL is used to train: long-horizon reasoning models
- [[mixture-of-experts]] — MoE architectures that make async RL routing replay necessary
- [[gpu-kernel-engineering]] — Batch-invariant kernels, DeepGEMM, FlashAttention as enabling systems infrastructure
- [[inference-optimization]] — SGLang, vLLM, KV cache as the rollout infrastructure side
- [[safety-misalignment]] — Reward hacking as a failure mode that emerges in the same training regime

## Sources
- [Is Frontier Asynchronous RL Solved? — Luke J. Huang (May 31, 2026)](https://luk-huang.github.io/personal-website/blog/is-frontier-asynchronous-rl-solved.html)
- [AsyncRLHF — arxiv:2410.18252](https://arxiv.org/abs/2410.18252)
- [PipelineRL — arxiv:2509.19128](https://arxiv.org/abs/2509.19128)
- [AReaL paper — arxiv:2505.24298](https://arxiv.org/abs/2505.24298) — not currently in the repo as a PDF; arXiv link only.
- [AsyncFlow — arxiv:2507.01663](https://arxiv.org/abs/2507.01663) — not currently in the repo as a PDF; arXiv link only.
- [HybridFlow / verl — arxiv:2409.19256](https://arxiv.org/abs/2409.19256) — not currently in the repo as a PDF; arXiv link only.
- [FP8-RL — arxiv:2601.18150](https://arxiv.org/abs/2601.18150) — not currently in the repo as a PDF; arXiv link only.
- [AReaL](https://github.com/inclusionAI/AReaL) | [LlamaRL](https://github.com/facebookresearch/llama-rl) | [SkyRL](https://github.com/NovaSky-AI/SkyRL) | [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | [VeRL](https://github.com/verl-project/verl) | [Slime](https://github.com/THUDM/slime)
- [R3 (Rollout Routing Replay) — arxiv:2510.11370](https://arxiv.org/abs/2510.11370)
- [GSPO — arxiv:2507.18071](https://arxiv.org/abs/2507.18071)
- [TBIK — arxiv:2511.17826](https://arxiv.org/abs/2511.17826)
- [ScaleRL — arxiv:2510.13786](https://arxiv.org/abs/2510.13786)
- [M2PO — arxiv:2510.01161](https://arxiv.org/abs/2510.01161)
- [FlashRL](https://github.com/yaof20/Flash-rl)
- [strands-sglang](https://github.com/horizon-rl/strands-sglang)
- [Thinking Machines — batch-invariant kernels](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [Perplexity TransferEngine](https://research.perplexity.ai/articles/weight-transfer-for-rl-post-training-in-under-2-seconds)
- [LMSYS P2P SGLang weight update](https://www.lmsys.org/blog/2026-04-29-p2p-update/)
- [Curse of the Horizon — arxiv:1810.12429](https://arxiv.org/abs/1810.12429)
- [OPSD — arxiv:2601.18734](https://arxiv.org/abs/2601.18734) | [SDPO — arxiv:2601.20802](https://arxiv.org/abs/2601.20802)
- [MIMO/OPD — arxiv:2601.02780](https://arxiv.org/abs/2601.02780)
- [Richard Li on RL collapse Part 1](https://richardli.xyz/post/rl-collapse-part1/) | [Part 2](https://richardli.xyz/post/rl-collapse-part2/)
- [Applied Compute RMSD](https://appliedcompute.com/research/relevance-masked-self-distillation)
- [Minimax M1 — arxiv:2506.13585](https://arxiv.org/abs/2506.13585)
- [Magistral — arxiv:2506.10910](https://arxiv.org/abs/2506.10910)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [Composer 2.5 delta compression](https://cursor.com/blog/composer-2-5)
- [LlamaRL/AIPO — arxiv:2505.24034](https://arxiv.org/abs/2505.24034)
- [Qi et al. FP16 — arxiv:2510.26788](https://arxiv.org/abs/2510.26788)
- [Prime Intellect Renderers](https://www.primeintellect.ai/blog/renderers)
- "RL Systems Mind the Gap: Matching Trainer and Generator Throughput" — SemiAnalysis (June 16, 2026), `raw/semianalysis-rl-systems-mind-the-gap.md`. Source for environment-state-level staleness; full throughput-matching framework lives in [[rl-training-systems]].
- "RL at 1T Scale: prime-rl Performance Deep Dive" — Prime Intellect Team, Matej Sirovatka (June 21, 2026), `raw/primeintellect-rl-at-1t-scale.md`. Source for R3's order-of-magnitude KL-mismatch reduction and `max_off_policy_steps` as GLM-5's policy-lag cutoff; full inference/training stack lives in [[rl-training-systems]].
