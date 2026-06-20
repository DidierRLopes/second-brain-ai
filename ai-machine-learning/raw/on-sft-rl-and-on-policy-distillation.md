# On SFT, RL, and On-Policy Distillation

**Source:** Personal blog post (arguments by Will Brown, drafting by Claude Opus 4.7)
**Authors:** Will Brown & Claude Opus 4.7
**Published:** April 30, 2026

## Key Concepts

A unifying analysis of post-training methods — SFT, rejection-sampled SFT (SFT-RS / RFT), RL (GRPO), on-policy distillation (OPD), and self-distillation variants (SDFT, OPSD) — framed around two ideas: (1) which sampling distribution your method gets to compound with, which sets the performance ceiling, and (2) the geometry of the per-step gradient (sparse vs. dense, biased vs. unbiased, diffuse vs. concentrated), which sets the failure modes.

All four "dense, biased" methods (SFT, OPD, OPSD, SDFT) can be written as special cases of a single token-level policy-gradient meta-algorithm with two scalar knobs (α: how on-policy the sampling is; λ: how much per-token signal comes from a teacher KL vs. an outcome reward) and a teacher-policy choice π_T. The corners of that space are the named algorithms; the optimal-teacher question is the open research direction.

## The Compounding Argument (§1)

- **SFT** (incl. teacher SFT and instruction-tuning): the sampling distribution is fixed at dataset-construction time. As the student improves, the data does not. Ceiling ≈ teacher's.
- **RL**: student samples its own rollouts; improvements compound back into the sampling distribution. Ceiling ≈ verifier's.
- **Tipping point**: when the student is far below the teacher and teacher data is cheap, SFT bits are extremely cheap-per-improvement. As the student approaches the teacher, marginal SFT examples get less informative and rollout compute is better spent on RL.
- **SFT-RS / RFT** (sample → filter for correctness → train on survivors): strictly better than vanilla teacher SFT in expectation, but the sampling distribution is still pinned to whatever you're filtering. Same ceiling, shifted curve.

## Same-Family vs. Different-Family Teachers (§2)

- **Same-family** = tokenizer-matched + recipe-matched (Qwen3-32B → Qwen3-8B-Base is canonical). Teacher outputs sit in a distribution structurally close to what the student naturally produces; SFT signal is mostly about the capability gap, not stylistic differences. Per-token logprobs are directly comparable.
- **Different-family** = tokenizer mismatch and/or recipe mismatch. Two costs: (a) re-tokenization loses information at boundaries and breaks soft-target distillation; (b) the student absorbs the teacher's pipeline byproducts (formatting, register, when to use CoT) alongside actual content. A nontrivial fraction of bits go to surface form.
- **OPD essentially requires same-family.** Tokenizer match is needed to compute the per-token KL at all; recipe match is needed for the signal to be informative on what you care about, rather than dominated by "the teacher would have phrased this differently."

## On-Policy Distillation as the Same-Family Upgrade (§3)

OPD (Lu et al. 2025; Qwen3 technical report; earlier Agarwal et al. 2023). The student samples its own rollouts (RL-like compounding of distribution), and each token is graded by the teacher via per-token reverse KL:

```
∇_θ J_OPD(θ) = E_{x, ŷ ~ π_θ(·|x)} [ Σ_t (log π_T(ŷ_t|ŷ_<t) − log π_θ(ŷ_t|ŷ_<t)) · ∇_θ log π_θ(ŷ_t|ŷ_<t) ]
```

Each token's "advantage" is how much the teacher prefers this token relative to the student. Dense, on-policy, reverse-KL.

Reported gains: ~9–30× less compute than RL on AIME-style benchmarks, gap widens when teacher logprob calls can be parallelized. Why the practical ceiling sits above SFT-RS's despite targeting the same teacher: cheaper sampling (teacher only forward-passes student tokens — essentially prefill, much cheaper than generation) plus on-policy state coverage. SFT and SFT-RS train under the teacher's state distribution but evaluate the student under its own; OPD trains on student rollouts so the exposure-bias gap doesn't open.

Framing: "OPD beats RL" is wrong. Correct framing is "OPD gets you to the teacher's level much faster than RL would, and most of the time that's where you wanted to be anyway." OPD is bounded by the teacher; RL is bounded by the verifier.

## Self-Distillation When You Lack a Same-Family Teacher (§4)

When no same-family teacher exists, recent work uses the student itself as teacher with privileged info in the teacher's context that the student doesn't see at sampling time.

- **SDFT** (Shenfeld et al. 2026): teacher conditioned on an expert demonstration. Provides distributional pull without leaking the answer; modest distributional shift.
- **OPSD** (Zhao et al. 2026): teacher conditioned on the ground-truth answer. Sharper distributional shift.

Both have automatic tokenizer + recipe match (same model). Trade: privileged-info conditioning shifts the teacher's distribution away from the student's natural distribution.

## Gradient Geometry (§5)

The central technical contribution. Three shapes:

**RL: sparse, but saved by destructive interference.** In GRPO, group-relative advantages have ~zero mean by construction. Each per-token advantage × ∇ log π gives a parameter-space vector. Most are noise — reward is sparse and broadcast-assigned to tokens that didn't actually cause the outcome. The batch is a swarm of small mostly-random vectors with a small consistent bias along reward-correlated directions ("think longer," "double-check arithmetic"). Large-batch, low-LR RL is robust because uninformative gradients cancel; what survives is the small consistent component. Empirically consistent with the observation that RL updates modify small subnetworks (Mukherjee et al. 2025). Sparsity is the price for an unbiased estimator.

**SFT: dense, biased, but spread out.** Every token gets a one-hot label; gradient density is enormous. Gradient distribution does not have zero mean — biased toward the data. Constructive interference, not destructive. SFT doesn't blow up because (a) the data distribution is itself diverse so the bias decomposes across many slightly-different directions, drifting toward the data manifold rather than any one example, and (b) most of what an SFT step does is reinforce things the model already half-knew. The bias is real but unconcentrated.

**OPSD: dense, biased, AND concentrated.** Consider a long math rollout where the student fails because it missed a key observation at one *pivot token* — a token whose probability under the student is, say, 0.01 but under the teacher (conditioned on the answer) is 0.6. The per-token reverse KL there is ~log(0.6/0.01) ≈ 4.1, vs. ~0 for typical tokens. One pivot token contributes ~100× a typical one. The gradient is dominated by it — and unlike RL, the noise vectors don't cancel; unlike SFT, the bias isn't diffuse across many already-supported tokens. One concentrated tug toward a region the model didn't previously believe in.

OPSD ships with **per-token point-wise KL clipping** for exactly this reason — without it, performance collapses within ~100 steps. The fix works; the lesson is that the KL signal in self-with-hint distillation is concentrated enough that you have to budget it. Suggests: look for a teacher whose KL is *naturally* diffuse rather than relying on clipping to make a concentrated one tolerable.

## The Sparse/Dense × Biased/Unbiased × Concentration Taxonomy (§6)

Two main axes (sparse vs. dense; biased vs. unbiased) plus a third for distinguishing the dense-biased methods (concentration). SFT, OPD, and OPSD all live in "dense, biased" — what distinguishes them is *where* the bias points and *how* it's distributed across tokens. RL is sparse but unbiased. SFT and OPD are dense, biased, and diffuse. OPSD is the unique case of dense + biased + concentrated, which is why it needs explicit defenses.

## Meta-Algorithm (§7)

All four methods are special cases of a single token-level policy gradient with two knobs and a teacher-policy choice:

- **α ∈ [0,1]** — how on-policy the sampling distribution is.
- **λ ∈ [0,1]** — how much per-token advantage comes from teacher KL vs. sequence-level outcome reward.
- **π_T** — the teacher policy: which model, conditioned on what context c_T.

Drops out: SFT is "distillation from a degenerate teacher" π_T(y|x) = δ_{y_data}(y); RL is "no teacher" (λ=0 collapses to broadcast outcome reward); OPSD has identical (α, λ) to OPD and differs only in π_T.

Caveat from the author: the factorization is illustrative, not a recommendation to mix at intermediate (α, λ) — the clean corners are where the statistics work without importance-sampling corrections, and they correspond to qualitatively different KL-budget regimes. The interesting axis is β (KL budget), and "pick β, then find the best teacher for that β." The hard problem is the inner one: teacher optimization, which is typically discrete (which model? which prompt? which hint?) and doesn't decompose neatly into the gradient framing.

## Toward an Optimal Teacher (§8)

Cast as Lagrangian: maximize E[Δreward] − β · KL(π_T || π_θ) on the student's rollouts. Different methods are different points on this Pareto curve as β varies.

Critique of SDFT's framing: the ICL assumption (that conditioning on an expert demonstration approximates the optimal policy) gives a similar ceiling to SFT/SFT-RS — only as good as the demonstrations, can't bootstrap beyond. The author wants curves more like RL where the ceiling is only a function of verifier ability.

Five candidate directions (all aim to construct the teacher rather than fix it):

1. **Per-task prompt optimization over the Lagrangian** — use [[prompt-optimization|GEPA]] over E[Δreward] − β·KL on a per-task basis, with teacher sampling for reward estimates. Cheap inner loop, no retraining.
2. **Distribution-level prompt optimization** — train a hint-rewriter that turns large privileged-info hints (the answer, a known-good demo) into small ones that move the teacher distribution as little as possible while still improving reward.
3. **Self-prompt-optimization online RL** — hints as rollouts in a parallel environment co-evolving with the student via an adaptive minmax over reward-delta and KL.
4. **Train a hint-writing model directly with RL** — objective like correctness-delta × (1 − KL-delta), scoped per-model/distribution/task or generally.
5. **Borrow from "expert RL + OPD"** — recent models (e.g., DeepSeek V4) layer a teacher signal on top of locally-optimal RL.

Open question the author finds most interesting: a meta-algorithm that interpolates cleanly between distillation and RL without needing a real teacher, with compute-optimal learning at each point on the curve. RL may still be optimal in the infinite-compute limit on the hardest heavy-tail distributions, but for everything else the optimal-teacher construction is where the action is.

## Why It Matters

- Cleanest unifying framework I've seen for SFT vs. RL vs. OPD vs. self-distillation — distinguishes them by sampling distribution (compounding), gradient density, gradient bias, and gradient concentration, all in one pass.
- Reframes the practitioner question from "which method?" to "what KL budget β, and what teacher π_T for that β?"
- Identifies *concentration* as the third axis that explains why OPSD needs KL clipping while OPD doesn't, even though they have identical α and λ.
- Surfaces the optimal-teacher / hint-rewriter research direction as the most interesting open problem, with [[prompt-optimization|GEPA-style]] search as one plausible building block.

## References

- Lu, K. & Thinking Machines Lab. *On-Policy Distillation* (2025).
- Zhao, S. et al. *Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models* (OPSD, 2026).
- Shenfeld, I. et al. *Self-Distillation Enables Continual Learning* (SDFT, 2026).
- Agarwal, R. et al. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (2023).
- Mukherjee, A. et al. *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models* (2025).
- Qwen Team. *Qwen3 Technical Report* (2025).
- Ross, S. et al. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAGGER, 2010).
