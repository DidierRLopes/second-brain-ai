# On-Policy Distillation

On-policy distillation (OPD) sits between SFT and RL on the post-training algorithm tree. The student samples its own rollouts (so improvements compound back into the sampling distribution, like RL), but each token in the rollout is graded by a teacher via per-token reverse KL (a dense, on-policy supervised signal, like SFT). Reported gains over RL are roughly 9–30× less compute on AIME-style benchmarks. The catch is that OPD essentially requires a same-family teacher — tokenizer-matched, ideally recipe-matched — because the per-token KL has to be computed at the same positions over the same vocabulary, and because the signal needs to be informative on capability rather than dominated by stylistic differences.

## The Compounding Argument

Post-training methods differ in *which sampling distribution they get to compound with*, and that determines the ceiling.

- **SFT** (incl. teacher SFT and instruction-tuning): sampling distribution is fixed at dataset-construction time. Ceiling ≈ teacher's. As the student approaches the teacher, marginal SFT examples get less informative.
- **Rejection-sampled SFT (SFT-RS / RFT)**: sample → filter for correctness → train on survivors. Strictly better than vanilla SFT in expectation but the ceiling is unchanged — the sampling distribution is still pinned to whatever you're filtering.
- **RL** (see [[alignment-methods|GRPO and friends]]): student samples its own rollouts, gradient updates the policy, next batch is sampled from the improved policy. Improvements compound. Ceiling ≈ verifier's, not the teacher's.
- **OPD**: student samples its own rollouts (RL-style compounding) but the per-token signal is the teacher's preference (SFT-style density). Same nominal ceiling as the teacher, but the *practical* ceiling sits above SFT-RS's because you train on the student's own state distribution and the exposure-bias gap doesn't open.

The honest framing: OPD doesn't beat RL in the limit. It gets you to the teacher's level much faster than RL would, and most of the time that's where you wanted to be anyway.

## Same-Family vs. Different-Family Teachers

A teacher's relationship to the student is the major efficiency axis.

**Same-family** = tokenizer-matched + recipe-matched. Canonical example: Qwen3-32B → Qwen3-8B-Base (Qwen3 technical report; revisited in Thinking Machines' OPD post). Teacher outputs sit structurally close to what the student naturally produces, so the SFT/OPD signal is mostly about the capability gap rather than stylistic differences.

**Different-family** = tokenizer mismatch and/or recipe mismatch. Two costs: (a) every teacher completion has to be re-tokenized in the student's vocabulary, losing information at boundaries and breaking soft-target distillation; (b) the student absorbs the teacher's pipeline byproducts (formatting, when it uses CoT, register) alongside content — a nontrivial fraction of bits go to surface form rather than competence.

OPD essentially requires same-family. Tokenizer match is needed to compute the per-token KL at all (it's between teacher and student distributions over the same token positions). Recipe match is needed for the gradient to actually be informative on the things you care about, rather than dominated by "the teacher would have phrased this differently here."

## Foundational OPD Methods

Before self-distillation and multi-teacher variants existed, a first wave of methods established the core OPD recipe and its main knobs (synthesized from Karkar's 2026 survey, see Sources).

- **MiniLLM** (Gu et al.) — the foundational paper. Casts reverse-KL minimization as an RL problem: reward = negative per-token student/teacher log-ratio, decomposed into single-step terms to avoid unrolling the full sequence. Uses teacher-mixed sampling for stability and a length-normalization term so the implicit reward isn't biased toward shorter sequences. Its reverse-KL choice is mode-seeking — it diversifies less than forward-KL SFT but concentrates probability mass on modes the teacher endorses (see [[ml-theory-statistics]] for the forward/reverse-KL definitions, and "Why On-Policy Methods Forget Less" below).
- **GKD** (Generalized Knowledge Distillation, Agarwal et al.) — generalizes MiniLLM along two axes: a λ-mixing ratio between on-policy (student) and off-policy (teacher/dataset) sampling, and a choice of divergence (forward KL, reverse KL, or generalized JSD). This is essentially the (α, λ) knobs from the Unifying Meta-Algorithm below, made literal and independently tunable.
- **DistiLLM** (Ko et al.) — uses a skew-KL divergence (an interpolation between forward and reverse KL, less prone to reverse-KL's mode collapse) plus an adaptive off-policy schedule that anneals the on/off-policy mix over training instead of fixing it upfront.
- **G-OPD / ExOPD** — adds a λ-dial that interpolates the OPD objective toward a DPO-style preference objective, explicitly framed as a way to beat the teacher's ceiling: pure reverse-KL imitation is capped by teacher quality, while the DPO-adjacent end can exploit preference signal beyond the teacher's own greedy output. (ExOPD's reference-model corrective, used in MiMo-V2-Flash, is the production descendant — see Multi-Teacher OPD below.)
- **AOPD** (Asymmetric OPD) — observes that gradients on negative-advantage tokens become heavy-tailed and destabilizing, and fixes this with top-K forward-KL gating restricted to negative-advantage tokens, rather than applying reverse-KL uniformly across all tokens.
- **On-Policy Context Distillation** — teacher = the *same* weights as the student, but with extra context (a system prompt, worked examples) the student doesn't see. OPD then internalizes the behavior the context induces, without paying its inference-time token cost at deployment. The lightest-touch version of self-distillation, predating SDFT/OPSD's privileged-answer conditioning.

## Self-Distillation When You Lack a Same-Family Teacher

If no same-family teacher is available, recent work uses the student itself as teacher with privileged information in the teacher's context that the student doesn't see at sampling time. Tokenizer and recipe match become automatic; the trade is that privileged-info conditioning shifts the teacher's distribution away from the student's natural distribution.

- **SDFT** (Shenfeld et al. 2026): teacher conditioned on an *expert demonstration*. Modest distributional shift. Provides distributional pull without leaking the answer. The paper formalizes the in-context-learning assumption — that conditioning on a demonstration approximates the optimal policy — which means SDFT inherits an SFT-style ceiling (only as good as the demonstrations available).
- **OPSD** (Zhao et al. 2026): teacher conditioned on the *ground-truth answer*. Sharper distributional shift, more aggressive distributional pull.
- **GATES** — consensus-gated self-distillation for document QA: the privileged-information (PI) teacher is unreliable, so updates are gated on agreement across multiple PI-conditioned teacher samples rather than trusting a single privileged forward pass.
- **CRISP** — uses "be concise" as the privileged instruction itself, i.e. compressed-reasoning self-distillation where the PI is a behavioral instruction rather than a reference answer. Risk: compressing reasoning traces this aggressively can suppress hedging/uncertainty language even when the underlying uncertainty hasn't gone away ("epistemic suppression," see Failure Modes below).
- **RLSD** (Self-Distilled RLVR) — resolves "PI alone leaks unverified shortcuts" by taking the *magnitude* of the update from the self-distillation log-ratio but the *direction* from an external verifier: RLVR's reward gates which way the update goes, self-distillation's log-ratio sets how big the step is.

All five (SDFT, OPSD, GATES, CRISP, RLSD) share the same algorithmic shape as OPD; only the teacher choice and what gates the update differ.

## SDPG: Self-Distilled Policy Gradient

[Self-Distilled Policy Gradient (2606.04036)](../../papers/05-learning/reinforcement-learning/Self-Distilled Policy Gradient - 2606.04036.pdf) (Liu, Zhang, Zhang, Gu et al., UCLA/Princeton) makes explicit a connection the rest of this page treats implicitly: **on-policy self-distillation is itself a form of policy gradient.** The paper's central technical contribution is a gradient identity (Proposition 3.1): for a fixed sampled prefix with the teacher branch detached, the student-side gradient of the full-vocabulary reverse-KL OPD loss `D_KL(p_t || SG[q_t])` is *identical* to a detached-sampling policy-gradient update whose per-token advantage is a centered log teacher/student ratio, `SG[D̄_t − log(p̄_t(a)/q̄_t(a))]`, which is provably zero-mean under the detached student distribution. This is a gradient-equivalence result, not an implementation swap — SDPG still minimizes the explicit full-vocabulary KL because it gives a more accurate gradient estimate than the sampled-token policy-gradient surrogate would.

**The recipe.** SDPG combines three terms into one objective, `L_SDPG = L_out + β(k)·L⁺_OPD + α·L_K`:
- **L_out** — a standard RLVR/GRPO-style term: group-relative verifier advantages (binary correct/incorrect reward, normalized by group mean/std exactly as in GRPO — see [[alignment-methods]]), but optimized via a plain REINFORCE-style surrogate rather than PPO-style clipped importance ratios (the rollout policy is taken to be exactly the current policy, so no importance-ratio correction is needed).
- **L⁺_OPD** — the **exact full-vocabulary student-to-teacher reverse-KL** self-distillation loss described above, where the "teacher" `q_t = π_θ(· | c, x, y_<t)` is the *same* model additionally conditioned on privileged context `c`, and the "student" `p_t = π_θ(· | x, y_<t)` is the deployable, unprivileged model — the same teacher-as-self-with-privileged-context pattern as SDFT/OPSD above, but computed densely over the entire vocabulary at every token position rather than approximated by a sampled token.
- **L_K** — KL regularization against a fixed reference policy, using an *unnormalized* KL (UKL) rather than the standard normalized KL, to correctly handle the case where the reference/teacher distributions aren't perfectly normalized over the relevant sub-vocabulary.

**Why this matters for sparse rewards specifically.** RLVR's binary verifier reward is sequence-level and sparse: a single 0/1 signal gets broadcast across every token in a (possibly very long) reasoning trace, giving the model no information about *which* tokens were responsible for success or failure. SDPG's full-vocabulary OPD term supplies a dense, per-token, every-position supervision signal that piggybacks on the same rollouts, addressing the sparsity problem directly rather than via better credit assignment of the sparse signal (contrast with GAE/PPO's approach on [[policy-gradient-actor-critic]], which spreads a sparse reward via bootstrapped value estimates rather than adding a second dense signal).

**Two stabilizers** keep the dense teacher signal from overwhelming or destabilizing the sparse verifier signal:
- **Positive-advantage gating**: the OPD loss is masked to only the rollouts the verifier marks correct (`A_out > 0`). Rationale: on an incorrect rollout, the privileged teacher can still assign high probability to locally plausible tokens along a globally wrong trajectory — without gating, the distillation term would reinforce exactly the failure mode the verifier is trying to suppress.
- **β warmup-then-decay schedule**: the OPD coefficient starts at zero (so early training isn't destabilized by a still-noisy privileged signal before the policy has found any correct trajectories), ramps up once the verifier starts endorsing rollouts, then decays again later — because under an idealized privileged-information model, distilling from a teacher conditioned on information unavailable to the deployable student leaves an irreducible conditional-mutual-information gap that more OPD pressure can't close.

**How SDPG compares to the rest of this page's catalog.** SDFT, OPSD, GATES, CRISP, and RLSD (above) are all *pure* self-distillation — they replace the verifier signal with a self-distillation signal entirely, with RLSD as the partial exception (verifier sets the update's *direction*, self-distillation log-ratio sets its *magnitude*). SDPG instead keeps both signals as separate additive loss terms running concurrently — RLVR's sparse outcome reward plus dense full-vocabulary self-distillation plus reference-policy KL — and uses gating/scheduling rather than a verifier-gated direction/magnitude split to keep them from conflicting. It is also the only method in this group built specifically around the *exact* full-vocabulary KL (matching OPSD's "full, vocabulary-wise KL divergence" design point above, as opposed to MiniLLM/GKD-style sampled-token approximations) and around an explicit policy-gradient reading of what that full-vocabulary KL loss is actually doing under the hood.

## Why OPSD Needs KL Clipping (the gradient-geometry story)

The most useful analytical lens for distinguishing these methods is the *shape of the per-step gradient*. Two main axes (sparse vs. dense, biased vs. unbiased) plus a third — *concentration* — that distinguishes the dense-biased methods.

**RL: sparse, but saved by destructive interference.** GRPO group-relative advantages have ~zero mean by construction. Each per-token advantage × ∇ log π gives a parameter-space vector. Most are noise — reward is sparse and broadcast-assigned to tokens that didn't actually cause the outcome. The batch is a swarm of small mostly-random vectors with a small consistent bias along reward-correlated directions. Large-batch, low-LR RL is robust because the uninformative gradients cancel. Sparsity is the price for an unbiased estimator. Empirically consistent with the observation that RL updates modify small subnetworks (Mukherjee et al. 2025).

**SFT: dense, biased, but spread out.** Every token gets a one-hot label; gradient density is enormous. Bias points toward the data — constructive interference, not destructive. SFT doesn't blow up because (a) the data distribution is itself diverse so the bias decomposes across many directions, drifting toward the data manifold rather than any one example, and (b) most of what an SFT step does is reinforce things the model already half-knew. The bias is real but unconcentrated.

**OPD: dense, biased, diffuse.** Same shape as SFT, but the teacher distribution is calibrated to the student's family, so the per-token bias stays diffuse across many tokens.

**OPSD: dense, biased, AND concentrated.** Imagine a long math rollout where the student fails because it missed a key observation at one *pivot token* — a token whose probability under the student is, say, 0.01 but under the teacher (conditioned on the answer) is 0.6. The per-token reverse KL there is ~log(0.6/0.01) ≈ 4.1, vs. ~0 for typical tokens. One pivot token contributes ~100× a typical one. The gradient is dominated by it — and unlike RL, noise vectors don't cancel; unlike SFT, the bias isn't diffuse across many already-supported tokens. One concentrated tug toward a region the model didn't previously believe in.

OPSD ships with **per-token point-wise KL clipping** for exactly this reason — without it, the paper reports performance collapse within ~100 steps. The fix works; the lesson is that the KL signal in self-with-hint distillation is concentrated enough that you have to budget it. Suggests looking for a teacher whose KL is *naturally* diffuse rather than relying on clipping to make a concentrated one tolerable.

## Why On-Policy Methods Forget Less

A complementary lens to the gradient-geometry story above: *why* do RL and OPD preserve prior capabilities so much better than SFT, even when the KL penalty against a reference policy is weakened or removed entirely?

**Forward-KL vs. reverse-KL mode behavior.** SFT-via-cross-entropy is approximately forward-KL minimization, KL(data‖student) = −H(data) + H(data, student) — mode-covering, so it can sacrifice an existing mode (a capability the model already has) to cover a new target mode. RL behaves more like reverse-KL minimization — mode-seeking, so it tends to sharpen around modes the model already visits rather than chasing new ones (Chen et al. 2025, "Retaining by Doing," arXiv 2510.18874, toy multi-modal demonstration). Useful but incomplete on its own: it leans on an explicit KL penalty against a reference, yet RLVR pipelines often weaken or remove that penalty and still resist forgetting.

**RL's Razor: on-policy sampling as implicit KL-locality** (Shenfeld et al., "RL's Razor," arXiv 2509.04259) — the strongest single explanation on offer. With a binary 0/1 verifiable reward, REINFORCE behaves like rejection sampling: reward-1 rollouts contribute signal, reward-0 rollouts contribute none. There are typically *many* optimal (reward-1) policies, but because training data is on-policy, the update implicitly targets whichever optimal policy is *closest in KL to the current policy* — the nearest task-solving policy, not an arbitrarily distant external target. SFT's target distribution has no such proximity constraint; it pulls toward the dataset's distribution regardless of distance. This is also why OPD inherits RL's anti-forgetting behavior even when distilled from a degraded SFT teacher: the teacher supplies the per-token *signal*, but the *state distribution* — and hence the implicit KL-locality — comes from the student's own on-policy sampling, not from the teacher.

**Data-dependent regularization vs. uniform pressure.** SFT pushes up every demonstrated token's probability regardless of whether it's task-critical or stylistic filler; SFT training data contains many low-probability, low-entropy tokens — the model is already confident, but forced to fit a divergent label anyway, disrupting representations that weren't broken (Diao et al., arXiv 2601.02151). RL gets implicit data-dependent regularization for free: GRPO-style advantage normalization gives high-variance/high-diversity groups smaller updates and low-variance/high-reward groups larger ones (Lai et al., arXiv 2507.05386). Consistent with this, RL updates a small, full-rank subnetwork while SFT updates densely (Mukherjee et al. 2025, already the basis for the "RL is sparse" claim in the gradient-geometry section above), and SFT's updates are more redundant under parameter pruning than RL's (Yuan et al., arXiv 2510.04454).

**"SFT memorizes, RL generalizes"** (Chu et al., arXiv 2501.17161) is the umbrella empirical finding behind all of the above — and is also why SFT-after-pretrain remains necessary before RL: without it, RL is too inefficient at establishing basic instruction-following to bootstrap from. This is why almost every production pipeline still runs Pretrain → SFT → RL/OPD rather than skipping SFT.

**Direct evidence from a controlled teacher-swap experiment** (nrehiew, on the Minimal Code Editing task — see `raw/nrehiew-coding-models-overediting.md`): training an SFT teacher and an RL teacher on the same task, then OPD-distilling a student from each, produced two students that converged to nearly the same performance *regardless of which teacher supplied the signal* — both outperforming the RL teacher and substantially outperforming the SFT teacher, with **far less forgetting than the SFT teacher exhibited, even for the student distilled from the degraded SFT teacher**:

| Model | Pass@1 ↑ | Norm. Levenshtein ↓ | Added CC ↓ | LiveCodeBench v6 ↑ |
|---|---|---|---|---|
| SFT teacher | 0.775 | 0.450 | 0.450 | 0.286 |
| RL teacher | 0.792 | 0.063 | 0.206 | 0.320 |
| OPD student (SFT teacher) | 0.800 | 0.059 | 0.206 | 0.297 |
| OPD student (RL teacher) | 0.787 | 0.055 | 0.228 | 0.314 |

Takeaway: the *source of the data* (on-policy sampling) dominates over *which teacher* supplied the per-token signal. You can over-specialize a model via brute-force SFT and still recover most of its lost generality through OPD — the on-policy state distribution does the anti-forgetting work, not the teacher's quality.

**Why can a student surpass its teacher?** Not new to this corpus — Agarwal et al. (arXiv 2306.13649) first showed distilled students surpassing teachers on GSM8K. Two non-exclusive hypotheses: (a) OPD supervises the student's *own* error-prone states, which the teacher's own trajectories rarely visit and therefore never corrected; (b) KL-matching isn't reward-maximization — the teacher distribution carries style/uncertainty/structure beyond its own greedy output, and matching it can reshape sampling behavior favorably even when the teacher's own samples aren't better. A speculative data point in this direction: self-distilling on uncorrected, even high-temperature gibberish completions has been reported to still improve coding performance (Zhang et al., arXiv 2604.01193) — possibly because OPD's mode-seeking reverse-KL induces sharper, more sudden entropy collapse than RL's gradual reward/entropy curves, somewhat independent of how "good" the specific completions being matched are.

**Generalization, not just retention.** The same on-policy property that limits forgetting also limits the train/test state-distribution mismatch that drives compounding errors in imitation learning generally (Ross et al.'s DAgger analysis, arXiv 1011.0686, with a modern LLM-agent treatment in [Revisiting DAgger in the Era of LLM-Agents (2605.12913)](../../papers/05-learning/fine-tuning/Revisiting DAgger in the Era of LLM-Agents - 2605.12913.pdf)). In SFT, the model only ever trains on states the teacher visited, so one autoregressive mistake at test time can push it into territory it never trained on, compounding. RL and OPD attach supervision to the student's own visited states, so there's no such gap to compound.

## A Unifying Meta-Algorithm

SFT, RL, OPD, and OPSD can all be written as special cases of one token-level policy gradient with two scalar knobs and a teacher-policy choice:

- **α ∈ [0,1]** — how on-policy the sampling distribution is (RL: 1, SFT: 0, OPD/OPSD/SDFT: 1).
- **λ ∈ [0,1]** — how much per-token signal comes from a teacher KL vs. a sequence-level outcome reward (SFT/OPD/OPSD: 1, RL: 0).
- **π_T** — the teacher policy, including which model and what context it sees.

SFT is "distillation from a degenerate teacher" π_T(y|x) = δ_{y_data}(y). RL is "no teacher" (λ = 0 collapses to broadcast outcome reward). OPSD has identical (α, λ) to OPD and differs only in π_T (self with answer in context vs. larger same-family model). The interesting axis of variation across the corners is the KL budget β; the hard problem is the inner one — *which π_T for that β*.

## Toward an Optimal Teacher

Cast the question as a Lagrangian: maximize **E[Δreward] − β · KL(π_T || π_θ)** on the student's rollouts. Different methods are different points on this Pareto curve as β varies.

What you want is a teacher that produces a large reward improvement per step subject to a hard KL constraint that keeps updates stable — and that you can construct rather than fix. Open directions:

- **Per-task prompt optimization over the Lagrangian** — use [[prompt-optimization|GEPA]] over E[Δreward] − β·KL on a per-task basis, with teacher sampling for reward estimates.
- **Distribution-level prompt optimization** — train a hint-rewriter that turns large privileged-info hints (the answer, a known-good demo) into small ones that move the teacher distribution as little as possible while still improving reward.
- **Self-prompt-optimization online RL** — hints as rollouts in a parallel environment co-evolving with the student via an adaptive minmax over reward-delta and KL.
- **Train a hint-writing model directly with RL** — objective like correctness-delta × (1 − KL-delta), scoped per-model/distribution/task or generally.
- **Expert RL + OPD** — recent models (e.g., DeepSeek V4) layer a teacher signal on top of locally-optimal RL.

In the infinite-compute limit on the hardest heavy-tail distributions, RL may still be optimal — any teacher distribution adds bias the student would've eventually corrected anyway. For everything else, the optimal-teacher construction is where the open research lives.

## Failure Modes

A 2026 survey (Karkar, see Sources) catalogs the failure modes that recur across OPD and OPSD, mostly traceable to one root cause: **per-token reverse-KL is a leaky proxy for the thing actually wanted — faithful sequence-level behavior transfer.**

- **Token-level KL is a biased/fragile proxy for sequence-level KL** (Fu et al., arXiv 2603.25562) — per-token reverse-KL terms don't sum to a faithful estimate of the *sequence*-level divergence that actually governs forgetting and generalization, especially once autoregressive sampling drift sets in.
- **Prefix drift + gradient SNR collapse** — as the student's generated prefix diverges from anything the teacher was calibrated on, the teacher's per-token scores get noisier, and the resulting gradient's signal-to-noise ratio collapses.
- **Local teachability collapse** — some states are locally "unteachable" because teacher and student locally agree (≈0 local KL) even though both are still globally wrong, stalling learning in that region with no local signal to escape it.
- **Rock Tokens** — roughly 18% of tokens dominate the gradient norm in OPD/OPSD training despite not being the tokens that are actually learnable or informative (often the same style/pivot tokens discussed in the gradient-geometry section above); uncorrected, they crowd out the per-token budget that should go to genuinely teachable tokens.
- **Tokenizer mismatch causes silent corruption** — in cross-tokenizer OPD (below), naive alignment can silently misalign token boundaries, corrupting the KL computation without raising any error.
- **Diversity collapse** — the classic pass@1-up / pass@k-down tradeoff: OPD's mode-seeking reverse-KL sharpens the distribution, often at the cost of sample diversity (the same cost flagged for MiniLLM above).
- **OPSD-specific calibration gap (CaOPD)** (Zhang et al., arXiv 2604.16830) — a teacher's confidence when conditioned on privileged information is *not* the same distribution as the unconditioned, deployment-time confidence the student will actually need, creating a calibration mismatch unique to self-distillation setups.
- **Epistemic suppression** — compression-style self-distillation pressure (e.g. CRISP's "be concise" instruction) can strip out hedging/uncertainty language even when the underlying uncertainty hasn't gone away, suppressing the model's expression of its own epistemic state.

Every mitigation in the literature so far (per-token clipping, top-K gating, asymmetric treatment, calibration correction) is a local patch to one of these leaks rather than a structural fix — cross-tokenizer-safe, sequence-level-faithful OPD remains an open problem (see Toward an Optimal Teacher above).

## Multi-Teacher OPD (MOPD)

The natural extension when you've trained multiple capability specialists is to distill them all into one student at once. **Four 2026 frontier reports converge on MOPD** but deploy it differently. Recipe variants:

- **Final-stage consolidation.** MiMo-V2-Flash (Jan 2026), GLM-5 (Feb 2026), DeepSeek-V4 (Apr 2026) place MOPD at the end of post-training to merge specialists or recover from sequential RL forgetting.
- **Mid-pipeline stabilization.** Nemotron-Cascade 2 (Mar 2026) inserts MOPD *between* RL specialization stages as a periodic re-anchor against drift.

**Teacher composition is the main axis of variation.**
- **Types (MiMo-V2-Flash).** Three teacher types — domain SFT models, RL specialists, and **Self** (a snapshot of the student at the start of MOPD, acting as a stability anchor against drift toward unfamiliar teacher territory). Each prompt's domain label deterministically picks *one* teacher; multi-teacher aggregation is implicit via sample-level domain mixing.
- **Stages (GLM-5).** Each prior post-training stage's terminal checkpoint is a teacher; prompts come from each stage's own RL training set.
- **Capabilities (Nemotron-Cascade 2).** Math-as-SFT-init + RLHF teacher + multi-domain RL teacher. Math teacher is the SFT init itself when SFT data quality is high enough that further math RL would shorten reasoning traces.
- **Count (DeepSeek-V4).** 10+ RL specialists across math/coding/agent/IF, some split across three reasoning-effort modes (Non-think, Think High, Think Max).

**Train/inference mismatch (IcePop).** Training and inference engines produce different logits for the same input (different kernels, batch-invariance issues; MoE adds expert-routing nondeterminism). **IcePop** masks tokens whose train/infer probability ratio falls outside a tolerance band [α, β], dropping the noisiest updates entirely. Three of the four MOPD reports (MiMo, GLM-5, Nemotron) explicitly adopt it.

**Beating the teacher's ceiling.** Pure OPD inherits the teacher's mistakes, style biases, and calibration. MiMo-V2-Flash combines the OPD advantage with an outcome-reward-model advantage: `Â = Â^OPD + α · Â^ORM`. Ablation: MOPD > MOPD-w/o-ORM > pure ORM. Empirically the combined advantage lets the student *exceed* teacher accuracy on several benchmarks (Arena-Hard, etc.). ExOPD adds a similar corrective using a *reference* model (pre-RL checkpoint) and extrapolates in the (teacher − reference) direction. Same structural recipe — OPD as the dense imitation core plus a scaled corrective.

**Engineering scale (DeepSeek-V4 specifically).** Three infrastructure innovations push MOPD into a new regime:
- *Full-vocabulary logit distillation.* Estimating KL on the sampled token only is cheap but high-variance; DeepSeek preserves full logit distributions for stability. With vocab > 100K and 1M-token contexts, this requires careful teacher scheduling.
- *FP4 (MXFP4) for inference-only forwards.* All teacher and reference forwards run in MXFP4 with lossless FP4→FP8 dequantization for training. Critical when each batch fires 10+ teacher forwards.
- *Token-granular Write-Ahead Log (WAL) for preemptible rollouts.* Naive recovery with fresh randomness introduces length bias (short completions finish before interruption; long ones get re-sampled). WAL persists tokens + KV cache, so on resume decoding continues from the persisted state — preserving sample identity. Mathematically required for correctness, not just speed. The agentic-task analogue is DeepSeek Elastic Compute (DSec), which keeps a globally ordered trajectory log per sandbox.

**Who is the best teacher?** Across MiMo-V2-Flash's 12 benchmarks: RL wins 6, Self wins 5, SFT wins 1. RL teachers dominate verifiable-reward benchmarks (math, code, reasoning); Self wins where SFT/RL specialists distort calibration on broad/open-ended tasks. **Caveat:** best standalone teacher ≠ best distillation teacher. OPD cares about the teacher's conditional distribution over the *student's* on-policy samples, not just the teacher's final task accuracy. Higher benchmark score can come with worse calibration, worse log-prob support on student rollouts, or distribution mismatch.

**Industrial adoption beyond the four flagship reports.** Karkar's 2026 survey also names Baichuan-M3, KAT-Coder-V2, and CoPD as production systems applying the same multi-teacher merge-or-route pattern, suggesting MOPD has become a standard final-stage primitive well beyond the four technical reports analyzed in depth here.

**Teacher disagreement** is a distinct multi-teacher failure mode: when specialist teachers disagree on overlapping inputs, naive averaging produces a target distribution that doesn't correspond to *any* teacher's actual behavior, degrading the student below either specialist individually. This is plausibly part of why MiMo-V2-Flash and GLM-5 route to a *single* teacher per prompt (by domain or by stage) rather than always averaging across all teachers simultaneously.

## When to Use What (Hugging Face's framing)

A practical decision table from the SmolLM3 team (via Alex Wa's synthesis), useful as a high-level prior on which post-training algorithm to reach for:

| Algorithm | When to Use | Tradeoffs | Best Model Size |
|---|---|---|---|
| **Online DPO** | Preference labels are cheap; aligning behaviour with evolving distributions | Easy to scale iteratively, more stable than RL, depends on label quality and coverage. Supported in few training frameworks. | Any size, where preferences capture improvements beyond imitation |
| **On-policy distillation** | Stronger teacher available, want to transfer capabilities efficiently | Simple to implement, cheap, inherits teacher biases, ceiling limited by teacher. Supported in TRL and NemoRL | Most effective for small to mid-sized models (<30B) |
| **Reinforcement learning** | Verifiable rewards or tasks requiring multi-step reasoning / planning. Reward models work but face reward-hacking risk | Flexible and powerful, costly and harder to stabilise, requires careful reward shaping. Supported in most post-training frameworks | Mid to large models (20B+), where extra capacity lets them exploit structured reward signals |

Hugging Face also found that **semi-online DPO** (sync between trainer and generator every 100 steps) was generally best vs sync every 10 steps, online DPO, and GRPO — DPO variants can match GRPO with far less compute when applied thoughtfully.

## Cross-Tokenizer OPD

A limitation of standard OPD is that **teacher and student must share the same tokenizer** — the per-token KL has to be computed at the same positions over the same vocabulary (see Same-Family vs. Different-Family Teachers above). Removing this constraint would let any organization distill from any frontier teacher regardless of vocabulary — Karkar's 2026 survey calls it the single highest-leverage structural unlock available in the field. Four method families tackle it:

- **Optimal-transport-style** — ULD (Universal Logit Distillation), Multi-Level OT, and byte-level interfaces that sidestep vocabulary mismatch entirely by operating below the tokenizer, at the byte level.
- **Dual-space projection** — DSKD, which projects teacher and student into a shared latent space before computing the distillation loss.
- **Vocabulary-level alignment** — CDM, DWA-KD (dynamic weighted alignment), and SimCT, which explicitly map/align vocabulary items between teacher and student tokenizers.
- **Logit-level** — Hugging Face's **GOLD (General On-Policy Logit Distillation)** allows any teacher to be distilled into any student regardless of tokenization choices, significantly widening the teacher-pool available to small-model trainers. **CTPD** (cross-tokenizer preference distillation) extends the same idea to preference-style objectives rather than pure KL matching.

A practical gotcha across all of these: chat-template tokens need to be stripped or aligned consistently across tokenizers, or they silently corrupt the cross-tokenizer alignment (the same silent-corruption failure mode flagged above).

## PrefixRL: Conditioning On-Policy RL on Off-Policy Prefixes

**PrefixRL** ([Reuse Your FLOPs: Scaling RL on Hard Problems by Recycling Computation (2601.18795)](../../papers/05-learning/reinforcement-learning/Reuse your FLOPs: Scaling RL on Hard Problems by Recycling Computation - 2601.18795.pdf)) addresses a critical inefficiency in RL: on hard problems with low pass@k, the model rarely samples correct traces, so RL spends enormous FLOPs receiving no learning signal. The solution: run on-policy RL conditioned on prefixes of previously collected correct off-policy traces (from prior RL runs or inference on older models).

The mechanism: extract prefixes from successful off-policy trajectories, append them to the original problem, and run on-policy RL on both prefixed and unprefixed problems with gradients masked on the off-policy prefix. The prefixes place the current policy in higher-rewarding states early in the trajectory, reducing gradient variance and increasing learning signal strength.

**Theoretical backing**: PrefixRL is consistent with standard RL (Theorem 3.2: maximizers of the PrefixRL objective also maximize standard RL performance on unprefixed problems). Sample complexity improves by a factor of context length: PrefixRL achieves a suboptimality gap with KL(µ||π₀) dependence (where µ is the behavior policy realizing off-policy traces) that does not accumulate over iterations, unlike standard RL which suffers distribution shift penalties.

**Back-generalization**: empirically, training only on prefixed problems improves performance on unprefixed problems — a transfer that occurs via shared parameter updates. The model learns strategies different from those in the prefix and can even suppress suboptimal strategies that appear in it (Figure 5). This suggests the model is learning higher-level structure, not simply imitating the prefix.

**Empirical gains** (compute-matched, including rejection-sampling cost): 2× faster convergence on hard problems, >45% higher final training accuracy on unprefixed problems (3× relative improvement), and improvements transfer to held-out benchmarks (AIME'25: +12% over SFT+RL, HMMT'25: +20.2%). Pass@k improves at all k, suggesting PrefixRL expands the support of solvable problems rather than just sharpening the distribution on already-solvable ones.

## Related Topics

- [[alignment-methods]] — RLHF, DPO, GRPO, RLVR, APO; the broader post-training landscape
- [[supervised-fine-tuning]] — SFT is the on-ramp; OPD usually starts from an SFT'd model
- [[knowledge-distillation]] — Hinton-style soft-target distillation; OPD is the on-policy modern descendant
- [[reasoning-models]] — GRPO is the canonical RL recipe; reasoning distillation is the offline cousin of OPD
- [[prompt-optimization]] — GEPA-style search is one plausible building block for the optimal-teacher question
- [[parameter-efficient-fine-tuning]] — How OPD/SFT updates are typically applied in practice
- [[frontier-training-playbook]] — where OPD sits in the broader recipe
- [[coding-agent-over-editing]] — the minimal-editing benchmark and SFT/RL/OPD teacher-comparison experiment behind the forgetting evidence above
- [[knowledge-distillation#Memorization Dynamics in Knowledge Distillation|Memorization Dynamics in Knowledge Distillation]] — offline (teacher-trajectory) KD reduces training-data memorization vs. fine-tuning; a privacy-relevant data point for the broader distillation family OPD/SDPG belong to, even though SDPG's "teacher" is privileged-context self-distillation rather than a separate model with its own training set

## Sources

- [Self-Distilled Policy Gradient (2606.04036)](../../papers/05-learning/reinforcement-learning/Self-Distilled Policy Gradient - 2606.04036.pdf) — Liu, Zhang, Zhang, Gu et al. (UCLA/Princeton, 2026). Source for the SDPG section above: the policy-gradient-equivalence proof for full-vocabulary reverse-KL OPD, the L_out + β·L⁺_OPD + α·L_K objective, positive-advantage gating, and the β warmup-decay schedule.
- On SFT, RL, and On-Policy Distillation — Will Brown & Claude Opus 4.7 (April 2026)
- **On-Policy Distillation — Lu, K. & Thinking Machines Lab (October 27, 2025)** — full blog post added to raw/. Defines OPD, gives the chess-analysis analogy, walks through the Qwen3-8B reasoning recipe, the internal-assistant personalization recipe, the 50–100× compute-efficiency experiment, and the continual-learning property.
- **Multi-Teacher On-Policy Distillation: A New Post-Training Primitive — Yumo Xu (April 29, 2026)** — survey of MOPD in MiMo-V2-Flash, GLM-5, Nemotron-Cascade 2, DeepSeek-V4. Covers IcePop, ORM augmentation, Self-distillation as anchor, ExOPD, and the engineering scale at DeepSeek-V4 (full-vocab logits, FP4, WAL).
- **SFT, RL, and On-Policy Distillation Through a Distributional Lens — nrehiew** (`raw/nrehiew-sft-rl-on-policy-distillation.md`). Source for the forward/reverse-KL forgetting framing, RL's Razor, the SFT/RL-teacher OPD comparison table, and the data-dependent-regularization citations (Diao et al., Lai et al., Yuan et al.).
- **The Imitation Game: State of Policy Distillation in Language Model Training — Chinmay Karkar, 2026** (`raw/chinmaykarkar-imitation-game-opd-survey.md`). Source for the Foundational OPD Methods section (MiniLLM, GKD, DistiLLM, G-OPD/ExOPD, AOPD, On-Policy Context Distillation), the self-distillation-family additions (GATES, CRISP, RLSD), the Failure Modes section, the Cross-Tokenizer OPD taxonomy, and the MOPD industrial-adoption/teacher-disagreement additions.
- Minimal Editing: Measuring and Fixing Over-Editing in Coding LLMs — nrehiew (`raw/nrehiew-coding-models-overediting.md`). Source of the SFT-vs-RL-vs-OPD teacher-comparison experiment data.
- Self-Distilled Reasoner: On-Policy Self-Distillation for LLMs (OPSD) — Zhao et al. (2026)
- Self-Distillation Enables Continual Learning (SDFT) — Shenfeld et al. (2026)
- On-Policy Distillation of Language Models — Agarwal et al. (2023)
- Reinforcement Learning Finetunes Small Subnetworks in LLMs — Mukherjee et al. (2025)
- RL's Razor: On-Policy Sampling and Implicit KL-Regularization — Shenfeld et al., arXiv 2509.04259
- Retaining by Doing — Chen et al., arXiv 2510.18874
- SFT Memorizes, RL Generalizes — Chu et al., arXiv 2501.17161
- Qwen3 Technical Report — Qwen Team (2025)
- DAGGER — Ross et al. (2010); modern LLM-agent treatment: [Revisiting DAgger in the Era of LLM-Agents (2605.12913)](../../papers/05-learning/fine-tuning/Revisiting DAgger in the Era of LLM-Agents - 2605.12913.pdf)
- [Reuse Your FLOPs: Scaling RL on Hard Problems by Recycling Computation (2601.18795)](../../papers/05-learning/reinforcement-learning/Reuse your FLOPs: Scaling RL on Hard Problems by Recycling Computation - 2601.18795.pdf)
- DeepSeek-V4 Technical Report — DeepSeek (April 2026)
- MiMo-V2-Flash — Xiaomi (January 2026)
- GLM-5 — Zhipu (February 2026)
- Nemotron-Cascade 2 — NVIDIA (March 2026)
- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`. Source for the algorithm-selection table and GOLD.
