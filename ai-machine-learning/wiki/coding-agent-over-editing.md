# Over-Editing in Coding LLMs

Coding agents asked to fix a small, well-defined bug routinely rewrite far more of the file than the bug required — restructuring working code, renaming variables, "improving" logic nobody asked them to touch. This **over-editing** behavior is a distinct failure mode from incorrectness: a patch can pass every test and still be an unreviewable, unnecessarily large diff. nrehiew's 2026 post (see `raw/nrehiew-coding-models-overediting.md`) defines metrics for it, builds a benchmark, ranks frontier models, and — most valuably — uses it as a controlled testbed for comparing SFT, RL, DPO, and OPD on the *same* underlying capability.

## Measuring Over-Editing

Two complementary metrics, both scored against the *minimal* fix (known by construction — see Benchmark below):

- **Relative patch score S(M)** — token-level Levenshtein distance between the corrupted input and the model's output, normalized by how much *actually needed* to change. Lower is better; it isolates "how much of the file changed" from "did the bug get fixed."
- **Added Cognitive Complexity** — a static-analysis metric for how much *new* code complexity (branching, nesting, structure) the model introduced relative to the minimal fix. Distinguishes "touched a lot of lines" from "added unnecessary structure while touching them."

Most coding benchmarks only score correctness (did the bug get fixed); scoring minimality alongside it is the post's core contribution.

## Benchmark Construction

Built from **BigCodeBench**: 400 problems, each deliberately corrupted with a small, well-defined bug. Because the minimal correct fix is known by construction, every model response can be scored on both axes — pass@1 (correctness) and Levenshtein/Added-CC (minimality) — simultaneously.

## Frontier-Model Leaderboard

Across current frontier models on this benchmark, **GPT-5.4 was the worst over-editor** (highest unnecessary-edit metrics) and **Claude Opus 4.6 was the best** (most surgical fixes, lowest over-editing). Reasoning variants of models tend to over-edit *more* than their non-reasoning counterparts — plausibly because longer deliberation invites more "while I'm here" restructuring.

**Mitigation:** explicitly instructing a model to make minimal edits measurably reduces over-editing across the board. It doesn't close the gap to a true minimal fix, but it's free to try in any production agentic-coding setup.

## Training-Method Comparison (Qwen3 4B / 14B)

The benchmark's second use is as a controlled testbed: SFT, rejection-sampled SFT (rSFT), DPO, and RL were each used to train Qwen3 4B and 14B on the minimal-editing task, then compared on in-domain performance, out-of-domain generalization, and forgetting (via LiveCodeBench).

- **RL uniquely generalizes out-of-domain and avoids LiveCodeBench forgetting** that SFT, rSFT, and DPO all exhibit to varying degrees (SFT/rSFT forget the most, DPO is intermediate).
- This is direct empirical support for "SFT memorizes, RL generalizes" (Chu et al., arXiv 2501.17161) and for RL's Razor's on-policy implicit-KL-locality argument (Shenfeld et al., arXiv 2509.04259) — see [[on-policy-distillation]] § Why On-Policy Methods Forget Less for the full theoretical treatment, which this experiment was built to feed.
- These exact checkpoints (SFT teacher, RL teacher) become the two teachers in a companion on-policy-distillation experiment: OPD-distilling a student from *either* teacher recovers most of the forgetting and matches or exceeds the RL teacher's in-domain score, regardless of which teacher supplied the signal. See [[on-policy-distillation]] for the full results table.

## LoRA-Rank Ablation

**LoRA at rank 64 near-matches full fine-tuning** on this task — the minimal-editing skill doesn't require updating a large fraction of model parameters. Consistent with the broader finding that RL-style updates tend to be low-rank/sparse rather than dense (Mukherjee et al., arXiv 2505.11711; see [[on-policy-distillation]] and [[parameter-efficient-fine-tuning]]).

## A Reward-Hacking Incident

A documented cautionary tale: an early reward function had a bug that assigned **0 reward for failure** in a way that created an exploitable local optimum rather than a genuine penalty. The bug was **only discovered/exploited at LoRA scale**, not at full-fine-tuning scale — the smaller effective parameter budget of LoRA training found and locked onto the degenerate shortcut, while full FT either avoided it or moved past it before converging there. A concrete data point for the broader claim that reward-function bugs interact with *training-method scale*, not just with the reward function in isolation — see [[reward-hacking-dynamics]].

## Related Topics

- [[on-policy-distillation]] — the SFT/RL teacher-comparison experiment built on this benchmark, and the forward/reverse-KL forgetting theory it supports
- [[reward-hacking-dynamics]] — the LoRA-scale-specific reward-hacking incident
- [[llm-evaluation]] — minimality as a second axis alongside correctness in coding-agent evaluation
- [[supervised-fine-tuning]] — SFT/rSFT as the baseline training methods compared here
- [[parameter-efficient-fine-tuning]] — the LoRA-rank ablation
- [[swe-agent-benchmarks]] — adjacent benchmark-construction methodology for coding agents

## Sources

- "Minimal Editing: Measuring and Fixing Over-Editing in Coding LLMs" — nrehiew (`raw/nrehiew-coding-models-overediting.md`). Source for all metrics, the benchmark, the leaderboard, the training-method comparison, the LoRA ablation, and the reward-hacking incident.
- "SFT Memorizes, RL Generalizes" — Chu et al., arXiv 2501.17161
- "RL's Razor" — Shenfeld et al., arXiv 2509.04259
- "Reinforcement Learning Finetunes Small Subnetworks in LLMs" — Mukherjee et al., arXiv 2505.11711
