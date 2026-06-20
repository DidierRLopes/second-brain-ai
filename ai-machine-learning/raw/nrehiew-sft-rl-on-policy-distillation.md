# SFT, RL, and On-Policy Distillation Through a Distributional Lens (nrehiew)

**Source:** https://nrehiew.github.io/blog/sft_rl_opd/
**Author:** wh. (nrehiew)
**Filed under:** Post-training theory, on-policy distillation

## Premise

Frames every post-training method by asking: *what is the target distribution being optimized toward, and how directly is it defined?* This single lens explains qualitative differences between SFT, RL, and on-policy distillation (OPD), including why RL/OPD forget less and generalize better than SFT.

## A Distributional View

- **SFT — fixed external target.** The dataset distribution exists before training; cross-entropy pulls the model toward it regardless of where the model started. Because NLL doesn't care about the starting policy, SFT has no built-in mechanism to spare existing capabilities — gradient pressure is broad and uniform across the whole distribution, not just task-relevant regions. Good for cold-start format changes; prone to catastrophic forgetting when the dataset is far from the model's native distribution.
- **RL — direction of greatest expected reward.** No externally fixed target; the model samples from itself, scores its own samples, and reshapes high-probability regions it already visits via policy gradient. Works well when the reward is a low-bias proxy for quality (RLVR); messier when the reward model is itself imperfect (RLHF).
- **On-Policy Distillation (OPD) — pseudo-RL.** Has a teacher signal like SFT, but the data comes from the student's own sampling like RL. The gradient pulls the student toward the teacher's distribution via reverse-KL, weighted analogously to an RL advantage but using the student/teacher log-ratio instead of a reward.

## On-Policy Self-Distillation (OPSD)

Teacher and student are the *same* model; the teacher is given the reference solution as a privileged prefix when computing its log-probabilities, and that privileged information becomes the learning signal (arXiv 2601.18734). Because teacher and student share weights, most tokens already agree — but per-token KL analysis shows **style/pivot tokens** ("wait," "alright") have much higher KL than task-critical tokens (math operators like "power," "exponent"). Aggressive updates on these unimportant high-KL tokens risk collapse, motivating per-token KL clipping. This makes OPSD feel closer to RLHF (high-bias signal → need KL penalties/trust-region clipping) than to RLVR (low-bias verifiable reward → comfortable removing explicit KL penalties, using GRPO over PPO).

## Experiment: OPD With Different Teachers (Minimal-Editing Task)

Using the Minimal Code Editing task from the author's companion post (see `raw/nrehiew-coding-models-overediting.md`), two teachers were trained (SFT and RL) and then distilled via OPD:

| Model | Pass@1 ↑ | Norm. Levenshtein ↓ | Added CC ↓ | LiveCodeBench v6 ↑ |
|---|---|---|---|---|
| SFT teacher | 0.775 | 0.450 | 0.450 | 0.286 |
| RL teacher | 0.792 | 0.063 | 0.206 | 0.320 |
| OPD student (SFT teacher) | 0.800 | 0.059 | 0.206 | 0.297 |
| OPD student (RL teacher) | 0.787 | 0.055 | 0.228 | 0.314 |

Counter to expectation, the two OPD students converged to nearly the same performance regardless of teacher, both slightly *outperforming* the RL teacher and substantially outperforming the SFT teacher — and crucially, **both students forgot far less than the SFT teacher did, even the one distilled from the degraded SFT teacher.** This suggests the *source of the data* (on-policy sampling) dominates over *which teacher* supplied the signal — i.e. you can "overtrain" a narrow specialist (even via brute-force SFT) and then recover lost generality through OPD.

## Why Does RL Forget Less? (four explanations, ranked by the author's preference)

1. **Forward-KL vs. reverse-KL.** SFT-via-cross-entropy ≈ minimizing forward KL, $D_{KL}(p\|q_\theta) = -H(p) + H(p,q_\theta)$, whose mode-covering behavior can sacrifice existing modes (pre-existing capabilities) to cover the new target. Chen et al. 2025 (arXiv 2510.18874, "Retaining by Doing") show RL behaves like reverse-KL minimization, which is mode-seeking and forgets less in toy multi-modal settings. The author flags this as useful but incomplete: it leans on explicit KL regularization against a reference, yet RLVR often removes or weakens that penalty and still resists forgetting.
2. **Uniform/aggressive SFT gradients vs. data-dependent RL regularization.** SFT pushes up every demonstrated token's probability regardless of whether it's task-critical or stylistic filler; Diao et al. (arXiv 2601.02151) find SFT contains many low-probability, low-entropy tokens — i.e. the model is confident but forced to fit a divergent label, disrupting existing representations. RL has implicit data-dependent regularization (Lai et al., arXiv 2507.05386): high-variance/high-diversity groups get smaller advantage-driven updates, low-variance high-reward groups get larger ones. Mukherjee et al. (arXiv 2505.11711) find RL updates a small, full-rank subnetwork while SFT updates densely; Yuan et al. (arXiv 2510.04454) show SFT's updates are more redundant — pruning parameters hurts RL much faster than SFT.
3. **The author's preferred explanation: on-policy data (Shenfeld et al., arXiv 2509.04259, "RL's Razor").** With a binary 0/1 reward, REINFORCE behaves like rejection sampling — reward 1 contributes signal, reward 0 contributes none. There exist multiple optimal (reward-1) policies, but because training data is on-policy, the update implicitly targets the optimal policy *closest in KL to the current policy* — the nearest task-solving policy, not an arbitrarily distant external target. SFT's target distribution has no such proximity constraint. This is why OPD inherits RL's anti-forgetting behavior even when distilled from an SFT teacher: the teacher supplies the *signal*, but the *state distribution* (and hence the implicit KL-locality) comes from the student's own on-policy sampling.
4. **Why can the student outperform the teacher?** Not new — Agarwal et al. (arXiv 2306.13649) showed distilled students surpassing teachers on GSM8K. Hypotheses: (a) OPD supervises the student's *own* mistake-prone states rather than teacher-generated trajectories the student rarely visits; (b) KL-matching ≠ reward maximization — the teacher distribution carries style/uncertainty/structure information beyond its own greedy output, and matching it can reshape sampling behavior favorably even if the teacher's own samples aren't better. A speculative aside: Zhang et al. (arXiv 2604.01193) report coding gains from self-distilling on *uncorrected, even high-temperature gibberish* completions, which the author connects to OPD inducing sharper entropy collapse (mode-seeking reverse-KL) than RL — visible in steeper, more sudden reward/entropy curves for OPD vs. gradual ones for RL.

## Why Do RL and OPD Generalize Better?

Builds on the above plus Ross et al.'s DAgger analysis (arXiv 1011.0686): in SFT the model only ever sees states the teacher visited, so at test time one autoregressive mistake can push it into unvisited states, compounding errors. On-policy data aggregation (RL, OPD) reduces this train/test state mismatch because supervision is attached to task success rather than to a fixed token sequence.

## The Full Pipeline and the Best Algorithm

Most open pipelines run Pretrain → SFT → RL → OPD. SFT-after-pretrain remains necessary for format adherence/instruction-following — without it RL would be inefficient (cites Chu et al., arXiv 2501.17161, "SFT memorizes, RL generalizes"). Newer frontier models (GLM-5, DeepSeek V4 per the post) use OPD as a final **expert-merging** stage, with the final checkpoint sometimes never touched by RL directly — citing the MiMo-V2-Flash technical report's domain-by-domain teacher comparison (math/code favor RL; creative writing/knowledge-heavy benchmarks favor self-distillation, consistent with noisier LLM-judge rewards in those domains). The piece closes by arguing that the active ingredient across RL and OPD is **on-policy data**, not RL's specific machinery or explicit KL penalties — and that a hypothetical more compute-optimal successor algorithm would need the *density* of distillation supervision, the *unbiasedness* of RL reward, and the *on-policy* property of both, which nobody currently has in one method.

## Related wiki pages

[[on-policy-distillation]], [[ml-theory-statistics]] (forward/reverse KL definitions), [[alignment-methods]], [[supervised-fine-tuning]], [[reward-hacking-dynamics]]
