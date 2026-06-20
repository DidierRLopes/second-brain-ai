# Optimizers and Training Hyperparameters

Choosing optimizers and tuning hyperparameters is notoriously time-consuming and significantly impacts convergence speed and training stability. While the temptation is to copy settings from larger labs, those choices are often tuned to a specific architecture and data mix and may not transfer. The frontier-2026 picture: **AdamW remains the default for both pre- and post-training, Muon is the most credible challenger for hidden layers, and MuonClip is the stabilization addition that made Muon viable at trillion-parameter MoE scale (Kimi K2).** Per-stage learning rates, schedule choice (WSD vs cosine vs multi-step), batch-size warmup, and the critical batch size all matter; the rest is in this article.

## AdamW

Despite being invented over 10 years ago, AdamW still stands the test of time. Adam updates weights individually based on EMA of gradients (`m_t`) and squared gradients (`v_t`), plus weight decay (the "W"):

```
m_t = β1 * m_{t-1} + (1 - β1) * g_t
v_t = β2 * v_{t-1} + (1 - β2) * g_t²
m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)
θ_t = θ_{t-1} - η * (m̂_t / (sqrt(v̂_t) + ε) + λ * θ_{t-1})
```

The EMAs provide adaptive per-parameter learning rates: consistently large gradients get smaller effective LRs (squared-gradient term), small/noisy gradients get larger effective LRs.

Modern hyperparameters remain largely unchanged from the original:
- Weight decay `λ = 0.1` or `0.01`
- `β1 = 0.9`, `β2 = 0.95–0.999`
- `ε = 1e-8`

## Muon

**Muon** ([original post](https://kellerjordan.github.io/posts/muon/)). Unlike AdamW which updates per-parameter, Muon treats the weight matrix as a singular object and updates based on matrix-level operations. This reduces axis-aligned bias (where optimization favors certain coordinate directions) and encourages exploration of suppressed directions.

```
M_t = μ * M_{t-1} + G_t              # momentum
G_t' = NewtonSchulz5(M_t)            # approximates matrix sign
θ_t = θ_{t-1} - η * G_t'
```

The Newton-Schulz iteration approximates the matrix-sign function via repeated `f(X) = aX + b(XX^T)X + c(XX^T)^2 X`. For SVD `G = U S V^T`, repeated application converges to `U V^T` — the orthogonalized gradient. The effect is to **normalize singular values** of the update, increasing the scale of "rare directions" that have small magnitude in the raw update but are important for learning.

The tuned coefficients are `(3.4445, -4.7750, 2.0315)` — chosen to maximize the leading coefficient `a` subject to `lim φ^N(x) ∈ [0.7, 1.3]` for all `x ∈ [0, 1]`. With these, 5 NS steps suffice for transformers. The FLOP overhead of Muon is at most `Tm/B` where `m` = model dim, `B` = batch tokens, `T` = NS steps. For typical LM training scenarios at any scale, this is below 1%.

**Muon is more sample-efficient than AdamW**, especially at large batch sizes where AdamW struggles.

### Hybrid Muon + AdamW (Trinity Large)

Some implementations use **Muon for hidden layers, AdamW for embedding and output layers**. The intuition: embeddings and output projections benefit from per-parameter adaptive LRs, while hidden layers benefit from Muon's matrix-level structure awareness. Also for transformer Q, K, V parameters, Muon works better when applied separately rather than as a joint QKV layer.

### Infrastructure: distributing Muon

Since Muon operates at the matrix level, NewtonSchulz needs access to the full gradient tensor. Two approaches at scale:

**Round-robin scheme**: each rank is responsible for gathering all gradient matrices corresponding to its index and applying Muon locally. Since FSDP expects sharded gradients/updates and each rank has its shard, optimizer step proceeds normally. But this issues many overlapping collectives across many matrices, which breaks at scale.

**All-to-all bulk permutation** (Prime's choice): each rank temporarily owns full gradients for its matrices, runs Muon, then bulk-permutes them back. Requires padding (tensors packed into contiguous buffers can change expected size), but uses fewer collectives and scales better.

## MuonClip (Kimi K2)

Built on Muon, MuonClip prevents exploding attention logits — a common failure mode in large-scale training. Standard mitigations have issues:
- **Logit soft-cap**: applies `tanh` clipping to pre-softmax logits, but the *scaled dot-product* can explode (making bounding too late) and gradients distort around unstable regions.
- **QK-norm**: key matrices aren't materialized during inference for MLA (projected from a latent variable).

MuonClip rescales weights when per-head max logits exceed a threshold:
- For each head `h`, define `S_max^h = max_{batch, i, j} (Q_i K_j / sqrt(d)) / τ` (the per-head max scaled dot-product).
- Set `S_max = max_h S_max^h`. When `S_max > τ`, rescale Q and K weight matrices for head `h` by `(τ/S_max^h)^α`.
- Commonly `α = 0.5` so Q and K are scaled equally.

For MHA, per-head clipping based on `S_max^h` is straightforward. For MLA, keys are projected from a latent variable, so clipping must apply to the latent-to-key projection weights and the latent variable itself, with separate scaling for head-specific Q, K, rotary components and the shared rotary.

The main Muon algorithm is also modified to match Adam RMS and enable weight decay. For each weight `W`:

```
W <- W - η * sqrt(d_out / d_in) * MuonUpdate(W) - η * λ * W
```

where the scaling factor `sqrt(d_out / d_in)` adapts the update magnitude to matrix size (matches Adam's RMS scaling behavior). Weight decay applied multiplicatively before the gradient update.

**Empirical result**: in a 9B-active 53B-total MoE training run, attention logits diverge quickly without MuonClip. With MuonClip and `τ = 100`, max logits decay to a stable range after ~30% of training steps.

## Learning Rate Schedules

Learning rates have a life cycle: warmup from zero to avoid chaos, then anneal after settling into a good minimum.

- **Warmup**: typically 1–5% of training steps for short trainings; large labs fix the step count.
- **Cosine annealing**: classic, but inflexible — the cosine period needs to match the total training duration, so re-running with a different token count requires restarting.
- **Warmup-Stable-Decay (WSD)**: constant LR through the bulk of training, then linear decay over the last 10–20% of tokens (matches cosine annealing). WSD's killer property is **enabling ablations across token counts without restarting**: you keep the stable portion fixed and re-run only the decay phase.
- **Multi-step**: discrete drops. 80/10/10 matches cosine; 70/15/15 and 60/20/20 can outperform it.
- **DeepSeek-V3**: cosine between decay drops + constant phase before the final sharp step.

[SGDR: Stochastic Gradient Descent with Warm Restarts (1608.03983)](../../papers/03-scaling/training-optimization/SGDR: Stochastic Gradient Descent with Warm Restarts - 1608.03983.pdf) is the classic source for **cosine warm restarts**. A restart is not a reset from scratch; it keeps the current weights and raises the learning rate again, then cosine-anneals it:

```
η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * T_cur / T_i))
```

The original result was about anytime performance in vision models: restarts with periods that grow by `T_mult` reached good error rates 2-4× faster than fixed step schedules, and snapshots right before restarts made cheap ensembles. For LLM pre-training, full SGDR restarts are less standard than one long cosine or WSD schedule, but the paper is still the reason "cosine cycle" means "smoothly decay, optionally jump back up, continue from the same weights."

**Hugging Face's WSD finding (SmolLM3-1B ablations)**: WSD underperformed cosine in the pre-decay phase, but once decay began, WSD showed nearly linear improvement and caught up to cosine by the end. They settled on `2e-4`; higher → instability.

**Kimi K2**: WSD with 10T tokens at `2e-4` after a 500-step warmup, then 5.5T tokens cosine decay from `2e-4` to `2e-5`.

For [[supervised-fine-tuning|SFT]]: LR ~10× smaller than pre-training (aggressive updates → catastrophic forgetting). Sequence packing → further reduce LR.

## Batch Size

There is a **critical batch size**: too small → underused compute; too large → model needs more tokens to reach the same loss. Larger batches give more efficient gradient estimates.

[An Empirical Model of Large-Batch Training (1812.06162)](../../papers/03-scaling/training-optimization/An Empirical Model of Large-Batch Training - 1812.06162.pdf) formalized this with the **gradient noise scale**. The simplified statistic is roughly:

```
B_noise ≈ tr(Σ) / |G|²
```

where `Σ` is the per-example gradient covariance and `G` is the true gradient. Below that scale, increasing batch size gives close to linear speedups because gradient estimates are still noisy. Above it, more examples mostly duplicate the same gradient signal, so wall-clock time may improve but compute efficiency falls. The paper validated this order-of-magnitude predictor across supervised learning, language modeling, generative modeling, Atari, and Dota, with critical batch sizes ranging from tens to millions.

**Scaling rule**: if batch size scales by `k`, scale LR by `sqrt(k)`. Larger batches → lower gradient variance → can afford larger step sizes. The SGD parameter update has variance `Var(Δθ) ∝ η²/B`, so to maintain the same update variance when scaling `B` by `k`, scale `η` by `sqrt(k)`.

The critical batch size grows during training:
- Early: model makes large updates, `∇L` is large, so the critical batch is small.
- Later: stabilization happens, larger batches become more effective.

This motivates **batch-size warmup**: start small, grow over training. It also explains why batch-size sweeps should be tied to the target loss region; a batch that is wasteful early can become efficient later as the noise scale rises. The large-batch paper's practical framing is a Pareto frontier: smaller batches are compute-efficient, larger batches are time-efficient, and the critical batch is the bend where both costs are about 2× their unattainable optimum.

### RSDB: Random Sequential Document Buffer (Arcee Trinity)

Imbalanced minibatches — created when sequence packing or data distribution produces highly variable sequence lengths or domain compositions — cause gradient variance that destabilizes training. Particularly bad when certain MoE experts receive disproportionately many or few tokens.

RSDB reduces intra-batch correlation:
1. Tokenize each document and load as an entry in the RSDB with a read head at index 0.
2. Repeat until RSDB is full.
3. Sample a random document and read-head position from RSDB, copy tokens to a separate sequence buffer.
4. Update read head; if the sequence buffer is full, return.
5. Otherwise, sample another random document index and continue reading tokens.

In Trinity Large: internal buffer 8192 per GPU (2× user-specified value of 4096), refilled when buffer reaches user threshold or when old documents need purging. Significantly improved dataloader performance.

## Optimizer for Pre vs Post-Training

AdamW is the default for both. When tested with Muon, using the same optimizer for both pre and post-training still yielded the best performance.

## Sources

- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- Keller Jordan, "Muon: An optimizer for hidden layers in neural networks" (Dec 2024). See `raw/muon-optimizer-keller-jordan.md`.
- Kimi K2 technical report (MuonClip and learning rate schedule).
- Bernstein & Newhouse, "Old Optimizer, New Norm: An Anthology" (arxiv:2409.20325).
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW, arxiv:1711.05101).
- [An Empirical Model of Large-Batch Training (1812.06162)](../../papers/03-scaling/training-optimization/An Empirical Model of Large-Batch Training - 1812.06162.pdf) — gradient noise scale and critical batch size.
- [SGDR: Stochastic Gradient Descent with Warm Restarts (1608.03983)](../../papers/03-scaling/training-optimization/SGDR: Stochastic Gradient Descent with Warm Restarts - 1608.03983.pdf) — cosine annealing with warm restarts.

## Related Topics

- [[training-stability]] — MuonClip is a stabilization technique
- [[mixture-of-experts]] — MuonClip helps stabilize MoE training in particular
- [[supervised-fine-tuning]] — LR sweeps and batch-size choices in post-training
- [[scaling-laws]] — batch-size scaling rules tie back to compute-optimal training
- [[frontier-training-playbook]] — where optimizers sit in the broader recipe
