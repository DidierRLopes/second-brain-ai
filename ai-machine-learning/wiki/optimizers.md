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

## Parametrization for Hyperparameter Transfer: µP, u-µP, CompleteP

Tuning learning rate, init scale, and weight decay directly on a frontier-scale model is prohibitively expensive. The **Maximal Update Parametrization (µP)** addresses this with a "tune small, train large" recipe: it defines width-dependent scaling rules for init variance, learning rate, and multipliers (the `abc`-parametrization: `A_W` for the forward multiplier, `B_W` for init std, `C_W` for the Adam LR multiplier) so that the optimal learning rate found on a small proxy model stays optimal as width grows. µP has been adopted by several open LLM efforts and is suspected to be behind GPT-4 and Grok's training (their technical reports/codebases reference or hint at it).

### u-µP: fixing µP's practical gaps and adding FP8 stability

[u-µP: The Unit-Scaled Maximal Update Parametrization (2407.17465)](../../papers/03-scaling/training-optimization/u-µP: The Unit-Scaled Maximal Update Parametrization - 2407.17465.pdf) (Aleph Alpha, Cohere, Graphcore) shows that vanilla µP has real failure modes once you leave the toy setup used to originally demonstrate it. Lingle (cited in the paper) had already shown µP fails to give LR transfer for standard decoder-LM training; the u-µP authors reproduce this and show the original Tensor Programs V demo only worked because of an unusually long-epoch, constant-LR setup that overfits and makes validation loss misleading. Switching to a standard Llama-style training setup (cosine LR, realistic epoch count) breaks transfer — it's recovered only after **removing learnable gain/bias parameters from LayerNorm/RMSNorm** and switching to **independent (decoupled) weight decay** in AdamW. The paper also documents that µP is, ironically, bad for low precision: in the original Tensor Programs V LLM runs, the *standard*-parametrized model trains fine in FP16 while the µP model diverges from gradient underflow — despite µP's stated goal of keeping activations at Θ(1) scale.

u-µP's fix is to combine µP with **Unit Scaling**: instead of just keeping activation scale independent of width (what µP gives you), Unit Scaling also fixes what that constant scale should be — variance ≈1 for activations, weights, and gradients at initialization, which centers values in a low-precision format's representable range. Concretely, u-µP sets `B_W ← 1` (unit initialization) and uses the Unit Scaling matmul factor `1/sqrt(fan_in)` in place of µP's own initialization rule, while keeping µP's width-scaling of the LR. The practical payoffs reported:
- **Cheaper sweeps**: u-µP supports independent (1-D) hyperparameter search — sweeping just the learning rate alone gets near-optimal loss — whereas µP needs a combined multiplier search because its `σ_W` (init scale) and `η_W` (LR) hyperparameters interact (their ratio, not either value alone, sets the effective update size).
- **No "base shape"**: µP requires picking an arbitrary base-width/base-depth reference model (the literature converged on base-width 256 with no principled justification) and re-initializing a throwaway base model just to compute scaling ratios. u-µP removes this HP entirely.
- **Out-of-the-box FP8**: because u-µP tensors sit near the center of a float format's range by construction, most matmuls can use a plain `.to(float8)` cast with no dynamic per-tensor rescaling (e.g. no Transformer Engine-style rescaling). Their proof-of-concept FP8 training run over 8k steps showed only minor degradation versus FP32 — a setting where the equivalent un-scaled cast would fail outright for other parametrizations.
- Across their sweep-strategy comparison, u-µP models reached validation loss equal to or lower than comparably-tuned µP models at widths from 128 to 4096.

### CompleteP: fixing µP's "lazy" deep layers

[Don't be lazy: CompleteP enables compute-efficient deep transformers (2505.01618)](../../papers/03-scaling/training-optimization/Don't be lazy: CompleteP enables compute-efficient deep transformers - 2505.01618.pdf) (Cerebras Systems, ETH Zurich, Princeton) tackles a different gap: µP transfers learning rate across *width*, but the paper shows standard µP (and the plain standard parametrization, SP) do **not** transfer hyperparameters across *depth* — the optimal LR/init-std drift as you add layers from 2 up to 128. Worse, even parametrizations that do achieve nominal depth transfer can still land in a **"lazy learning" regime**, where a layer's representation stays close to its linearization at initialization — effectively that layer is not learning meaningful nonlinear features, wasting the depth you paid compute for.

The family of depth-aware extensions to µP is indexed by a single exponent `α` controlling how the residual branch output is scaled before being added to the stream: `h_{l+1} = h_l + L^{-α} · F_l(h_l)`, with `α ∈ [0.5, 1]`. Prior work (Yang et al.) argued `α = 0.5` works best and that depth-wise HP transfer was impossible at any `α`; this paper shows that's wrong; **`α = 1` is the unique value that gives both depth-wise HP transfer and "complete" (non-lazy) feature learning in every layer**, which they name **CompleteP**. Realizing this in practice also requires extending the scaling rules to LayerNorm and bias learning rates, and to AdamW's weight decay `λ` and `ε` as explicit functions of depth and width (their Table 1) — without these extensions, `α = 0.5` is unstable.

Reported results, all on Cerebras CS-3 hardware with compute-optimal training (20 tokens-per-parameter):
- CompleteP shows stable optimal LR/init-std across depths 2 to 128 (exceeding LLaMA-70B's 80 layers and LLaMA-405B's 126 layers), where SP, µP, and `α = 0.5` all drift.
- At 1.5B (non-embedding) parameters, CompleteP gives **11.8% FLOP savings over µP at the compute-optimal width:depth ratio**, and **34.4% FLOP savings at the deepest (179-layer) setting** — the gap between CompleteP and µP widens as depth increases, because µP gets progressively more hyperparameter-detuned at depth.
- CompleteP widens the range of compute-efficient width:depth ratios: at 1.5B non-embedding params, a narrow-deep model with `N:L ≈ 11.8` stays within 1% of compute-optimal loss under CompleteP, versus `N:L ≈ 38.7` for µP — relevant for low-memory hardware that streams one layer at a time.
- Downstream zero-shot evals (HellaSwag, ARC-Easy, LAMBADA, RACE, PIQA, BoolQ) at 1.5B confirm the upstream loss gains transfer to task accuracy, not just validation perplexity.

Together, u-µP and CompleteP point at the same underlying lesson: µP's "maximal feature learning" guarantee is necessary but not sufficient — it needs the right *target* dynamics (Unit Scaling's variance-1 convention, for numerics) and the right *depth* scaling (CompleteP's `α = 1`, for feature learning at every layer) to deliver hyperparameter transfer that actually holds up in realistic training setups. See [[scaling-laws]] for how these parametrization choices interact with compute-optimal width:depth and token:parameter ratios, and [[training-stability]] for the broader numerical-stability context.

## Sources

- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- Keller Jordan, "Muon: An optimizer for hidden layers in neural networks" (Dec 2024). See `raw/muon-optimizer-keller-jordan.md`.
- Kimi K2 technical report (MuonClip and learning rate schedule).
- Bernstein & Newhouse, "Old Optimizer, New Norm: An Anthology" (arxiv:2409.20325).
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW, arxiv:1711.05101).
- [An Empirical Model of Large-Batch Training (1812.06162)](../../papers/03-scaling/training-optimization/An Empirical Model of Large-Batch Training - 1812.06162.pdf) — gradient noise scale and critical batch size.
- [SGDR: Stochastic Gradient Descent with Warm Restarts (1608.03983)](../../papers/03-scaling/training-optimization/SGDR: Stochastic Gradient Descent with Warm Restarts - 1608.03983.pdf) — cosine annealing with warm restarts.
- [u-µP: The Unit-Scaled Maximal Update Parametrization (2407.17465)](../../papers/03-scaling/training-optimization/u-µP: The Unit-Scaled Maximal Update Parametrization - 2407.17465.pdf) — combines µP with Unit Scaling for HP transfer + out-of-the-box FP8 training.
- [Don't be lazy: CompleteP enables compute-efficient deep transformers (2505.01618)](../../papers/03-scaling/training-optimization/Don't be lazy: CompleteP enables compute-efficient deep transformers - 2505.01618.pdf) — depth-wise HP transfer and non-lazy feature learning via α=1 residual scaling.

## Related Topics

- [[training-stability]] — MuonClip is a stabilization technique
- [[mixture-of-experts]] — MuonClip helps stabilize MoE training in particular
- [[supervised-fine-tuning]] — LR sweeps and batch-size choices in post-training
- [[scaling-laws]] — batch-size scaling rules tie back to compute-optimal training; CompleteP's width:depth findings revisit Kaplan-style compute-optimal aspect ratios
- [[frontier-training-playbook]] — where optimizers sit in the broader recipe
