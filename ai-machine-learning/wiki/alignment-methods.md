# Alignment Methods

Alignment is the process of teaching a language model to follow instructions, be helpful, and avoid harmful outputs. The modern alignment pipeline typically has two stages: supervised fine-tuning (SFT) on instruction-following examples, followed by preference optimization or RL to refine the model's behavior based on human judgments, verifiable rewards, or self-play-style feedback.

## RLHF: Reinforcement Learning from Human Feedback

The original alignment approach (used for InstructGPT, ChatGPT, Claude). The pipeline has three stages:

1. **SFT:** Fine-tune the base model on high-quality instruction-response pairs
2. **Reward model training:** Train a separate model to score responses based on human preference rankings
3. **RL optimization:** Use PPO (Proximal Policy Optimization) to optimize the SFT model against the reward model, with a KL penalty to prevent divergence from the SFT model

RLHF works well but is complex — it requires training and maintaining three models (policy, reward, reference) and PPO training can be unstable with reward hacking and mode collapse risks.

## KL-Regularized Policy Gradients

The technical core of RLHF-style training is not just "run PPO"; it is **policy gradient under a trust-region constraint**. The policy is rewarded for better outputs while being penalized for drifting too far from a reference model, usually through a KL term. [On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning (2505.17508)](../../papers/05-learning/alignment-preferences/ON THE DESIGN OF KL-REGULARIZED POLICY GRADIENT - 2505.17508.pdf) is useful because it names the design surface cleanly:

- **KL direction:** forward KL `KL(π_ref || π_θ)` is more coverage-seeking; reverse KL `KL(π_θ || π_ref)` is more mode-seeking.
- **KL form:** normalized KL vs unnormalized KL. The paper shows the common `k3(y)=y-log(y)-1` estimator corresponds to an unnormalized KL objective, not just a convenient penalty trick.
- **Estimator shape:** fully differentiable surrogate vs REINFORCE-style loss with stop-gradient. These can be gradient-equivalent only under the right weighting.
- **On-policy vs off-policy weighting:** if rollouts come from `π_old`, the KL term needs the matching importance weight `π_θ/π_old`; otherwise the optimized surrogate is not the gradient of the intended KL-regularized objective.

This matters for GRPO-family training because implementations often sample off-policy for throughput while writing objectives as if the data were on-policy. The paper argues that GRPO's KL penalty omits an essential importance weight under off-policy sampling. Its RPG framing adds a clipped-importance-sampling variant, RPG-Style Clip, and periodically updates the reference policy; the authors report up to +6 AIME points over DAPO and 52% on AIME25 at 8K context with Qwen3-4B-scale experiments. The practical takeaway: if you keep a KL term, specify the actual KL objective and the sampling policy before trusting the loss.

## DPO: Direct Preference Optimization

DPO ([Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2305.18290)](../../papers/05-learning/alignment-preferences/Direct Preference Optimization: Your Language Model is Secretly a Reward Model - 2305.18290.pdf)) showed that the optimal RLHF policy can be derived in **closed form**, eliminating the need for a separate reward model entirely. Instead, DPO uses a direct classification loss on preference pairs (chosen vs rejected responses). The language model itself becomes an implicit reward model.

The key derivation: the optimal policy can be extracted from a reward model via π*(a|x) = π_ref(a|x) exp(r(x,a)/β) / Z(x), where the implicit reward function becomes r(x,a) = β log(π(a|x)/π_ref(a|x)). This means supervised learning on preference pairs directly recovers the policy gradient direction without ever training a reward model. The loss becomes a cross-entropy classification: L_DPO = -log σ(β log[π_θ(y_w|x)/π_ref(y_w|x)] - β log[π_θ(y_l|x)/π_ref(y_l|x)]), where σ is sigmoid.

DPO is dramatically simpler — no reward model, no RL loop, just supervised training on preference data. It's more stable and won the **NeurIPS 2023 Outstanding Paper Award**. DPO spawned many variants:

- **IPO (Identity Preference Optimization):** Addresses DPO's overfitting to preference data by tightening the preference classification margin
- **KTO ([KTO: Model Alignment as Prospect Theoretic Optimization (2402.01306)](../../papers/05-learning/alignment-preferences/KTO: Model Alignment as Prospect Theoretic Optimization - 2402.01306.pdf)):** Per-sample desirable/undesirable labels (no pairs), takes ideas from human decision making (prospect theory) using a reference point to frame outcomes as gains vs losses. Computes desirability as log-ratio κ(x,a) = log[π_θ(a|x)/π_ref(a|x)] and learns from positive/negative samples using a loss that incorporates loss aversion (α weights losses higher than gains, typically 2:1)
- **ORPO ([ORPO: Monolithic Preference Optimization without Reference Model (2403.07691)](../../papers/05-learning/alignment-preferences/ORPO: Monolithic Preference Optimization without Reference Model - 2403.07691.pdf)):** Integrates preference optimization directly into the supervised learning objective via L = L_CE + λ·(-log σ(log ratio)) on preferred tokens. Eliminates the separate reference model entirely, reducing complexity. More computationally efficient since no forward pass through π_ref is needed
- **APO ([Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Direct Preference Optimization (2408.06266)](../../papers/05-learning/alignment-preferences/Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Direct Preference Optimization - 2408.06266.pdf)):** Two flavors addressing under-specification in preferences. **APO-zero** pushes the preferred response up and rejected down in probability space (standard contrastive). **APO-down** pushes both down — useful when the quality of the preferred response is below that of the current model (e.g., when both outputs contain errors, suppress both). Hugging Face's SmolLM3 ablations found **APO-zero had best overall out-of-domain performance**.
- **SimPO (Meng et al., NeurIPS 2024, arxiv:2405.14734):** Goes further than ORPO in removing the reference model — and also removes the reward formulation's dependence on conditioning on a prompt-specific normalizer. The implicit reward is simply the **average log-probability of the sequence** under the policy (length-normalized), which matches what's actually used at generation time (no need to evaluate a reference model at all). SimPO adds a **target reward margin** γ to the Bradley-Terry loss to push the winner/loser gap larger, not just positive. Net effect: simpler than DPO (no reference forward pass, so cheaper and more memory-efficient) while reporting up to **+6.4 AlpacaEval 2 and +7.5 Arena-Hard** over DPO on the same base models, without inflating response length (a common reward-hacking failure mode for length-unnormalized objectives).

## Self-Play RL Foundation: AlphaZero

Self-play RL predates LLM reasoning training by years. [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (1712.01815)](../../papers/05-learning/reinforcement-learning/Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm - 1712.01815.pdf) is the cleanest reference point: AlphaZero starts from random play, uses only the game rules, generates its own games through self-play, and trains a network `(p, v)=f_θ(s)` to predict both the MCTS search policy and the eventual game outcome with loss `(z-v)^2 - π^T log p + c||θ||²`.

The analogy to LLM alignment is structural, not literal. AlphaZero has perfect simulators and unambiguous win/loss rewards; LLMs have messy language states and often learned or rule-based rewards. But the important pattern is the same: the policy improves by sampling from its own current distribution, using search/verifiers/rewards to score outcomes, and feeding the improved policy back into the next round. That is the conceptual ancestor of self-play reasoning loops, RLVR, and GRPO/DAPO-style training.

## GRPO: Group Relative Policy Optimization

Used by DeepSeek-R1, GRPO is an RL-based method that samples multiple responses, scores them, and uses relative rankings within the group for optimization. Notable for enabling [[reasoning-models|reasoning capabilities to emerge]] purely from RL without any human-annotated reasoning demonstrations.

GRPO's core tradeoff: each rollout is a long, structured natural-language artifact (reasoning steps, tool calls, errors, judge rationales), but the optimizer compresses it to a single ±1 group-relative reward and back-propagates one bit per trajectory. That is why RL on LLMs typically needs tens of thousands of rollouts to converge — the signal was rich, the optimizer made it sparse. For compound multi-module systems where the base model already has the capability and the prompt is the bottleneck, [[prompt-optimization|GEPA-style reflective prompt evolution]] has been shown to beat GRPO by ~10 points with ~35× fewer rollouts and no GPU training. The two are increasingly combined (BetterTogether, mmGRPO) rather than treated as alternatives — use RL when the model needs new capabilities, prompt optimization when it needs new instructions.

**GRPO vs PPO** (per DeepSeek-R1 appendix): PPO has a per-token KL penalty (KL between sequence distributions decomposes into a sum over time of KL between tokens). Because RL enables longer reasoning over time, PPO implicitly penalizes response length. GRPO doesn't, and is less computationally expensive (no separate value model). On MATH tasks, GRPO consistently performed better than PPO with KL coefficient `β = 0.04`, which consistently outperformed PPO with `β = 0.0`.

**Practical stability issues in GRPO** (per [DAPO: An Open-Source LLM Reinforcement Learning System at Scale (2503.14476)](../../papers/05-learning/reinforcement-learning/DAPO: An Open-Source LLM Reinforcement Learning Framework - 2503.14476.pdf)): Naive GRPO suffers from entropy collapse, reward noise, and training instability at scale. DAPO removes the explicit KL penalty for long-CoT reasoning because the desired policy may need to move far from the initial model, then stabilizes training with four techniques: (1) **Clip-Higher**: decouple upper and lower clipping bounds (asymmetric, e.g., ε_low=0.2, ε_high=0.28) to allow low-probability exploration tokens to increase probability while constraining the usual lower side of the trust region — prevents early deterministic policy. (2) **Dynamic Sampling**: filter out prompts with group accuracy=0 or 1 and oversample until the effective batch has non-zero advantages. (3) **Token-Level Policy Gradient Loss**: aggregate over tokens rather than equal-weighting whole samples, so long high-quality traces can be learned from and long low-quality patterns can actually be penalized. (4) **Overlong Reward Shaping**: assign soft penalties to truncated long samples rather than hard penalties, signaling length control without confusing the model about reasoning validity. These techniques enable 50 points on AIME with Qwen2.5-32B vs 47 (DeepSeek-R1-Zero) using 50% fewer training steps.

## RLVR: RL with Verifiable Rewards

Popularized by DeepSeek-R1. Instead of training a reward model from human preferences, **verifiers check whether a model's output matches criteria** — math correctness, code unit-test pass/fail, instruction-following constraint adherence. The policy is fine-tuned to produce verifiably-correct outputs.

RLVR is particularly valuable when:
- **Reward drift** is a concern (verifiers provide very stable signals vs learned reward models)
- KL control is needed to prevent policy collapse
- Addressing stale-policy artifacts in multi-step reasoning tasks

For hybrid reasoning models, RLVR needs **length control** to prevent reward hacking — see [[reasoning-models]] for the overlong-completion-penalty mechanism.

## In-Flight Weight Updates and IcePop (Intellect-3)

In practice, even nominally on-policy algorithms like GRPO become slightly off-policy to maximize throughput. Without freezing the policy, generating multiple rollout batches and doing optimizer updates sequentially makes only the first batch truly on-policy — all subsequent batches are off-policy. These are **in-flight updates**.

In-flight updates matter most when:
- Throughput is critical (large-scale RL training)
- Reward drift could accumulate from stale policies
- KL divergence between inference and training policies needs monitoring
- Long rollouts span multiple policy updates

**Intellect-3's setup**: CPU orchestrator between training and inference clusters. The orchestrator continuously polls the trainer; on new policy, the inference pool temporarily halts generation, updates weights, then continues with rollouts. Long rollouts can span multiple policies, bounded by `max_off_policy_steps`. See [[training-ops]] for the multi-client orchestrator that underpins this.

**IcePop** stabilizes off-policy training via importance-sampling clipping. For each token, compute the importance weight `s = π_train(token) / π_inf(token)`. If `s` falls outside `[l, u]`, mask the token (drop it from the gradient):

```
A_ic = mask(s_ic) * A_i
mask = 1 if l ≤ s_ic ≤ u, else 0
```

Asymmetric bounds (e.g., `l = 0.5, u = 1.2`) are fine. A tighter `u` clips high-entropy tokens when `π_inf` is small. Prime uses default `l = 0.5, u = 1.2`.

## Rubric-Based Rewards (Kimi K2)

Kimi K2's self-critique rubric reward mechanism: the K2 actor generates `N` rollouts; the K2 critic ranks all results by performing pairwise evaluations against a combination of rubrics:

- **Core rubrics** (fundamental values)
- **Prescriptive rubrics** (aimed at eliminating reward hacking)
- **Human-annotated rubrics** (for specific instructional contexts)

The critic model is refined using verifiable signals — transfer learning grounds its more subjective judgments in verifiable data. This lets the critic recalibrate its evaluation standard in lockstep with the policy's evolution.

## PTX Loss and Temperature Decay (Kimi K2)

**PTX loss** = pre-training cross-entropy loss. During joint RL training, the model can catastrophically forget valuable high-quality data. Kimi K2 curates a dataset of hand-selected high-quality samples and integrates them into the RL objective via PTX loss. Twofold advantage: high-quality data continues to be leveraged, and the risk of overfitting to the RL tasks is mitigated.

**Temperature decay**: for tasks like creative writing and complex reasoning, high temperatures during initial stages generate diverse responses and avoid premature local-minimum convergence. At later stages, temperature is decayed via schedule to preserve reliability and consistency.

## Where SFT and RL Sit in the Post-Training Pipeline

The standard pipeline is "SFT first, then RL." There's a real argument for that ordering: SFT's sampling distribution is fixed at dataset-construction time, so its ceiling is roughly the teacher's; RL samples its own rollouts, so improvements compound back into the sampling distribution and the ceiling is set by the verifier. When current performance is far below the teacher and teacher data is cheap, SFT bits are extremely cheap-per-improvement; as the student approaches the teacher, marginal SFT examples get less informative, and rollout compute is better spent on RL. Rejection-sampled SFT (SFT-RS / RFT) shifts the SFT curve up but doesn't change its shape — the ceiling is still pinned to whatever you're filtering.

A third option, [[on-policy-distillation|on-policy distillation (OPD)]], sits between SFT and RL: the student samples its own rollouts (RL-style compounding) but each token is graded by a teacher via per-token reverse KL (SFT-style density). It's roughly 9–30× more compute-efficient than RL on AIME-style benchmarks, but requires a same-family (tokenizer- and recipe-matched) teacher.

## The Alignment-Beyond-Methods Question

Most of this page covers technical alignment methods (RLHF, DPO, GRPO). But the harder question — does training a reward model and optimizing against it actually align a sufficiently capable system? — is the subject of a different body of work and is more contested. The [[dwarkesh-podcast]] corpus is where the dominant positions are most clearly stated:

- **Paul Christiano** (inventor of RLHF, ex-OpenAI alignment lead) argues alignment is solvable but requires far more investment in evaluation, interpretability, and "stop-button" tractability than labs currently give it. See `raw/dwarkesh-paul-christiano-preventing-ai-takeover.md`.
- **John Schulman** (OpenAI co-founder) describes RLHF as "taming the shoggoth" — a working method, not a complete solution. See `raw/dwarkesh-john-schulman-openai-reasoning-rlhf-agi.md`.
- **Eliezer Yudkowsky** argues current methods cannot scale to superintelligent systems; we need to stop training frontier models entirely. See `raw/dwarkesh-eliezer-yudkowsky-ai-will-kill-us.md`.
- **Joe Carlsmith** offers a more philosophical framing of takeover risk and how techno-capital dynamics make alignment harder. See `raw/dwarkesh-joe-carlsmith-preventing-ai-takeover.md`.
- **Dario Amodei** (Anthropic) treats alignment as a research bet that compounds — Anthropic's safety + capabilities co-development. See `raw/dwarkesh-dario-amodei-anthropic-2023.md` and `raw/dwarkesh-dario-amodei-end-of-exponential.md`.
- **Sholto Douglas & Trenton Bricken** present the mechanistic-interpretability complement to RL-style alignment — if we can read the weights, we don't need to fully trust the reward signal. See `raw/dwarkesh-sholto-trenton-how-llms-actually-think.md`.
- **Dwarkesh's "Give AIs a stake in the future"** is an unusual proposal: skip training-based alignment in favor of giving AI systems property rights and legal standing. See `raw/dwarkesh-give-ais-a-stake-in-the-future.md`.

Where you fall on these questions depends largely on your [[agi-timelines]] view.

## Related Topics
- [[supervised-fine-tuning]] — SFT specifics (sequence packing, chat templates, learning rate, epochs)
- [[parameter-efficient-fine-tuning]] — LoRA/QLoRA are used in both SFT and DPO stages
- [[reasoning-models]] — GRPO enabled emergent reasoning in DeepSeek-R1; RLVR length-control
- [[rl-training-systems]] — rollout freshness, KL design, DAPO, PipelineRL, and verifier-backed RL infrastructure
- [[practical-fine-tuning]] — Applying alignment methods in practice
- [[prompt-optimization]] — Reflective prompt evolution (GEPA, MIPROv2) as an alternative to RL on compound systems
- [[on-policy-distillation]] — OPD, SDFT, OPSD; the same-family alternative to pure RL
- [[training-ops]] — multi-client orchestrator that underpins in-flight weight updates
- [[ai-rd-automation]] — Letting agents run the post-training pipeline themselves; misaligned-reward and eval-on-training-distribution are the dominant failure modes
- [[agi-timelines]] — Whether you think alignment is urgent depends on this
- [[dwarkesh-podcast]] — Primary source material for alignment positions across the field
- [[frontier-training-playbook]] — where alignment sits in the broader recipe

## Sources
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2305.18290)](../../papers/05-learning/alignment-preferences/Direct Preference Optimization: Your Language Model is Secretly a Reward Model - 2305.18290.pdf)
- SimPO: Simple Preference Optimization with a Reference-Free Reward — Meng, Xia, Chen, NeurIPS 2024 (arxiv:2405.14734)
- [KTO: Model Alignment as Prospect Theoretic Optimization (2402.01306)](../../papers/05-learning/alignment-preferences/KTO: Model Alignment as Prospect Theoretic Optimization - 2402.01306.pdf)
- [ORPO: Monolithic Preference Optimization without Reference Model (2403.07691)](../../papers/05-learning/alignment-preferences/ORPO: Monolithic Preference Optimization without Reference Model - 2403.07691.pdf)
- [Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Direct Preference Optimization (2408.06266)](../../papers/05-learning/alignment-preferences/Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Direct Preference Optimization - 2408.06266.pdf)
- [On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning (2505.17508)](../../papers/05-learning/alignment-preferences/ON THE DESIGN OF KL-REGULARIZED POLICY GRADIENT - 2505.17508.pdf)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale (2503.14476)](../../papers/05-learning/reinforcement-learning/DAPO: An Open-Source LLM Reinforcement Learning Framework - 2503.14476.pdf)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (1712.01815)](../../papers/05-learning/reinforcement-learning/Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm - 1712.01815.pdf)
- DPO Deep Dive — Cameron R. Wolfe (cameronrwolfe.substack.com)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.00656)](../../papers/01-models/gpt-deepseek-v2-v3/DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning - 2501.00656.pdf); see `raw/deepseek-r1-reasoning-via-rl.md`
- Dwarkesh Podcast — see `raw/dwarkesh-*` for primary interviews with Christiano, Yudkowsky, Schulman, Carlsmith, Amodei, Sholto/Trenton
- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`. Source for APO, RLVR, IcePop, rubric rewards, in-flight updates.
- Prime Intellect, Intellect-3 (in-flight updates, IcePop, pipeline RL).
- Kimi K2 technical report (rubric rewards, PTX loss, temperature decay).
