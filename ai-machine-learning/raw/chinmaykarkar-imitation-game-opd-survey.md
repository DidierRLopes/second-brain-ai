# The Imitation Game: State of Policy Distillation in Language Model Training (Chinmay Karkar)

**Source:** https://chinmaykarkar.com/blog/OPD_blog/
**Author:** Chinmay Karkar
**Filed under:** On-policy distillation survey

## Premise

A comprehensive survey of on-policy distillation (OPD) and on-policy self-distillation (OPSD): why OPD is the right way to do distillation when you have rollout access, a taxonomy of the foundational methods, the self-distillation family, the failure modes the field hasn't solved, open problems, and two appendices on cross-tokenizer distillation and multi-teacher OPD. This is the single richest source consulted for the [[on-policy-distillation]] wiki update.

## Why Only OPD

Off-policy / offline distillation (training on teacher-generated trajectories) puts the student in states it doesn't generate itself, so its own errors compound at test time in ways the teacher's trajectories never exposed it to (the DAgger problem). OPD instead samples from the student's own policy and asks the teacher only to *score* those student-generated tokens, training via reverse-KL between student and teacher next-token distributions. This keeps training data on-policy (implicit KL-locality / anti-forgetting, as in RL) while still getting a dense, per-token teacher signal instead of RL's sparse per-episode reward.

## A Survey on OPD (foundational methods)

- **MiniLLM (Gu et al.)** — the foundational OPD paper. Treats reverse-KL minimization as an RL problem: the reward is the (negative) per-token log-ratio between student and teacher, decomposed into a single-step term to avoid unrolling the full sequence. Uses **teacher-mixed sampling** (mixing teacher and student rollouts) for stability and a **length-normalization** term so reward isn't biased toward shorter sequences.
- **GKD (Generalized Knowledge Distillation, Agarwal et al.)** — introduces two knobs: a **λ-interpolation mixing ratio** between on-policy (student) and off-policy (teacher/dataset) sampling, and a **choice of divergence** (forward KL, reverse KL, or a generalized JSD) as a tunable rather than fixed reverse-KL. Lets practitioners dial between SFT-like and RL-like behavior along both axes.
- **DistiLLM (Ko et al.)** — uses a **skew-KL** divergence (a tunable interpolation between forward and reverse KL that's less prone to the failure modes of pure reverse-KL) and an **adaptive off-policy schedule** that anneals the on/off-policy sampling mix over training rather than fixing it.
- **G-OPD / ExOPD (Exploratory/Generalized OPD)** — adds a **λ-dial** that explicitly interpolates the OPD objective toward a DPO-style preference objective, connecting on-policy distillation to preference optimization as the dial moves; framed by the survey as a way to "beat the teacher's ceiling" since pure reverse-KL imitation is bounded by teacher quality while the DPO-adjacent end can exploit preference signal beyond the teacher's own greedy behavior.
- **AOPD (Asymmetric OPD)** — observes that gradients on negative-advantage tokens become heavy-tailed and destabilize training; fixes this with **top-K forward-KL gating** restricted to negative-advantage tokens (asymmetric treatment of positive vs. negative tokens) rather than applying reverse-KL uniformly.
- **On-Policy Context Distillation** — teacher = the *same* weights as the student but given extra context (e.g. a system prompt, worked examples, or instructions) the student doesn't see; OPD then teaches the student to internalize the behavior the extra context induces, without paying the context's inference-time cost at deployment.

A comparison table across these methods spans: sampling source (on/off/mixed-policy), divergence used (reverse/forward/skew/generalized-JSD), and whether the teacher is external, self, or self+context.

## On-Policy Self-Distillation (OPSD family)

- **Self-Distilled Reasoner / OPSD** — teacher = student + privileged reference-solution prefix (matches the mechanism in `raw/nrehiew-sft-rl-on-policy-distillation.md`).
- **SDFT (Self-Distillation Fine-Tuning)** and **SDPO (Self-Distillation Preference Optimization)** — variants applying the self-distillation idea inside SFT-style and preference-optimization-style update rules respectively, rather than a pure RL-style reverse-KL loss.
- **GATES** — consensus-gated self-distillation for document QA, where the privileged-information (PI) teacher is unreliable; gates updates on agreement across multiple PI-conditioned teacher samples rather than trusting a single privileged forward pass.
- **CRISP** — uses "be concise" as the privileged instruction, i.e. compressed-reasoning self-distillation where the PI is a *behavioral instruction* rather than a reference answer.
- **RLSD (Self-Distilled RLVR)** — resolves the instability of "PI alone causes leakage" (the privileged-information self-distillation signal alone can leak unverified shortcuts) by taking the **magnitude** of the update from the self-distillation log-ratio but the **direction** from an external verifier — i.e. RLVR's reward gates *which way* the update goes, self-distillation's log-ratio sets *how big* the step is.

A second comparison table spans these by: source of privileged information, failure mode each method patches, and whether an external verifier is required.

## Failure Modes (the survey's most novel contribution)

- **Token-level KL is a biased/fragile proxy for sequence-level KL** (cites Fu et al., arXiv 2603.25562) — per-token reverse-KL objectives don't sum to a faithful estimate of the *sequence*-level divergence that actually governs forgetting/generalization, especially under autoregressive sampling drift.
- **Prefix drift / unreliable teacher signal** combined with **gradient SNR collapse** — as the student's prefix diverges from anything the teacher was calibrated on, the teacher's per-token scores become noisier, and the resulting gradient's signal-to-noise ratio collapses.
- **Local teachability collapse** — some states are locally unteachable because the teacher and student locally agree (zero local KL) even when both are still globally wrong, stalling learning in that region.
- **Rock Tokens** — roughly 18% of tokens dominate the gradient norm in OPD/OPSD training despite not being the tokens that are actually learnable or informative; without correction they crowd out the per-token budget that should go to genuinely teachable tokens.
- **Tokenizer mismatch causes silent corruption** — when teacher and student tokenizers differ (cross-tokenizer OPD), naive alignment can silently misalign token boundaries, corrupting the KL computation without raising an error.
- **Diversity collapse** — the classic pass@1-up / pass@k-down tradeoff: OPD's mode-seeking reverse-KL sharpens the distribution, often at the cost of sample diversity.
- **OPSD-specific calibration gap (CaOPD)** — Zhang et al. (arXiv 2604.16830): a teacher's confidence when conditioned on privileged information is *not* the same distribution as the deployment-time (unconditioned) confidence the student will actually need, creating a calibration mismatch unique to self-distillation setups.
- **Epistemic suppression** — compression/self-distillation pressure (e.g. CRISP's "be concise" PI) can strip out hedging/uncertainty language, suppressing the model's expression of its own epistemic state even when the underlying uncertainty hasn't gone away.

The author's "My read" section frames most of these as symptoms of a single underlying issue: OPD's per-token reverse-KL training signal is a leaky proxy for the thing that's actually wanted (faithful sequence-level behavior transfer), and every patch in the literature (clipping, gating, asymmetric treatment, calibration correction) is a local fix to one leak rather than a structural solution.

## Open Problems

- Combining privileged-information self-distillation with a **soft external verifier** rather than treating PI and verification as mutually exclusive.
- **Distilling skills, not PI-shapes** — formulated as averaging over reasoning traces, $\bar\pi(\cdot|x) = \mathbb{E}_r[\pi_\theta(\cdot|x,r)]$, to extract the underlying skill rather than overfitting to one privileged-trace shape.
- **Uncertainty-aware / calibration-aware OPD objectives** that explicitly correct for the CaOPD gap rather than ignoring it.
- **Cross-tokenizer OPD** is called out as the single highest-leverage structural unlock, since it would let any organization distill from any frontier teacher regardless of vocabulary, not just same-tokenizer-family teachers.

## Appendix A: Cross-Tokenizer Distillation Taxonomy

- **Optimal-transport-style** — ULD (Universal Logit Distillation), Multi-Level OT, and byte-level interfaces that sidestep vocabulary mismatch entirely by operating at the byte level.
- **Dual-space projection** — DSKD, which projects teacher and student into a shared latent space before computing the distillation loss.
- **Vocabulary-level alignment** — CDM, DWA-KD (dynamic weighted alignment), SimCT — methods that explicitly map/align vocabulary items between teacher and student tokenizers.
- **CTPD (cross-tokenizer preference distillation)** — extends the cross-tokenizer idea to preference-style objectives rather than pure KL matching.
- **GOLD** (briefly) and practical **chat-template-stripping** guidance — a reminder that template tokens need to be excluded/aligned consistently across tokenizers or they silently corrupt the alignment.

## Appendix B: Multi-Teacher OPD (MOPD)

Names and frames the practice of merging multiple domain-specialist teachers into one OPD student — academically framed as either **weighted-average** distillation across teachers or **specialty-routing** (different teachers for different domains/prompts), with industrial citations (the post references production systems doing this, overlapping with the MiMo-V2-Flash/GLM-5/DeepSeek-V4 pattern already documented under Multi-Teacher OPD). Flags **teacher disagreement** as a distinct failure mode: when specialist teachers disagree on overlapping inputs, naive averaging produces a target distribution that doesn't correspond to *any* teacher's actual behavior, degrading the student below either specialist.

## References

The post includes a 44-entry numbered reference list (arXiv-linked) covering every method above plus 4 "suggested reads." Not reproduced here in full; consult the live post or the arXiv IDs cited inline above for primary sources when going deeper on a specific method.

## Related wiki pages

[[on-policy-distillation]], [[knowledge-distillation]], [[alignment-methods]], [[ml-theory-statistics]]
