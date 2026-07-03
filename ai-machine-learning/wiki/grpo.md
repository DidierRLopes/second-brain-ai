# GRPO: Group Relative Policy Optimization

GRPO is the RL algorithm behind [[reasoning-models|DeepSeek-R1]] and most of the
2025 RL-for-reasoning wave. It replaces PPO's learned value network with a
dead-simple idea: sample a **group** of responses to the same prompt, score them,
and push the policy toward the responses that beat the group average and away
from the ones below it. No critic, no per-token value estimates — just
group-relative advantages.

Murali Manohar's code-first walkthrough
([raw note](../raw/grpo-intro-gitlostmurali.md)) frames it as the **"FAFO
principle" — Fool Around and Find Out**: generate multiple answers, compute an
advantage for each, and move probability mass toward the higher-advantage ones.
This page builds the objective up piece by piece the way that guide does, then
connects to how GRPO is actually run in [[alignment-methods]],
[[rl-training-systems]], and [[reasoning-models]].

## Why advantages, not raw rewards

A raw reward tells you whether a response is good; it does **not** tell you how
good *relative to its peers* on the same query. GRPO turns rewards into
advantages by normalizing within the group using the group's mean and standard
deviation:

```
A_hat_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
```

This is the "group relative" in the name. The critical consequence: the
advantage is computed **per answer (sequence-level)**, then broadcast to every
token in that answer — not estimated per token by a value model. That is exactly
what lets GRPO drop PPO's separate critic network and its associated compute.

## The atomic objective, stripped to a loop

Before any importance sampling or clipping, the core GRPO objective is
structurally just a double for-loop — outer over the `G` sampled answers, inner
over tokens within each answer:

```
L_GRPO = (1/G) * Σ_{i=1}^{G} (1/|o_i|) * Σ_{t=1}^{|o_i|}  π_θ(o_{i,t} | q, o_{i,<t}) · A_hat_{i,t}
```

- `G` — group size (number of answers sampled for one prompt)
- `o_i` — the i-th answer; `|o_i|` — its token count
- `π_θ(o_{i,t} | q, o_{i,<t})` — the policy's probability of token `t` given the prompt and preceding tokens
- `A_hat_{i,t}` — the (sequence-level) advantage broadcast to every token

In code — multiply each token's log-prob by its answer's advantage, average over
tokens, then average over answers:

```python
for answer_i in generated_answers:            # G answers
    advantage = calculate_advantage(answer_i)
    for token_t in answer_i:                   # |o_i| tokens
        token_loss = pi_theta(token_t) * advantage
    loss_i = sum(token_losses) / len(answer_i)

final_loss = sum(loss_i for all answers) / len(generated_answers)
```

That double average — over tokens within an answer, then over answers within the
group — is the whole objective in its simplest form.

## The staleness problem: training on off-policy data

Regenerating rollouts every gradient step is expensive, so real implementations
**reuse one batch of generations across several gradient steps** (e.g. generate 4
answers/prompt, then take ~10 steps on them). This is where the naive objective
breaks. By step 10, the log-probs in the loss are computed with the *current*
(already-updated) weights, but the tokens were actually sampled by the *step-0*
model. The data is now **off-policy** relative to the policy being updated.

> "It's like practicing basketball shots based on a video of yourself from last
> week. You've improved since then, so the video doesn't represent your current
> form anymore." — [gitlostmurali](../raw/grpo-intro-gitlostmurali.md)

The result is increasingly biased gradients, instability, and possible
*unlearning* of good behaviors because the gradient signal no longer matches the
data-generating process. This is the same on-policy/off-policy distinction
covered in [[rl-fundamentals]] — GRPO just runs into it in a very concrete way.

## The fix: importance-sampling ratio + clipping

Freeze the log-probs from the model that actually generated the answers
(`π_θ_old`) and reweight each token by the ratio between current and old policy
probabilities:

```
ratio_{i,t} = π_θ(o_{i,t}|·) / π_θ_old(o_{i,t}|·)   =   exp(current_log_probs - old_log_probs)
```

Computed as `exp(current − old)` in log-space for numerical stability. The
operational reading is clean:

- **ratio > 1** — current model likes the token *more* than the generator did → amplify its gradient
- **ratio < 1** — current model likes it *less* → shrink its gradient
- **ratio = 1** — models agree → gradient unchanged

If the ratio swings to an extreme (e.g. 100 or 0.01) training can explode or
collapse, so GRPO **clips** it to `[1−ε, 1+ε]` and takes the pessimistic
minimum of the clipped and unclipped objectives — the same clip trick PPO uses
(see [[policy-gradient-actor-critic]]):

```
L_GRPO = (1/G) Σ_i (1/|o_i|) Σ_t  min( ratio_{i,t}·A_hat_{i,t},  clip(ratio_{i,t}, 1−ε, 1+ε)·A_hat_{i,t} )
```

With the **typical ε = 0.2**, the ratio is pinned to `[0.8, 1.2]`. As the guide
puts it, clipping is "a conservative approach to prioritize stable training over
perfect gradient correction" — it trades exact-gradient fidelity for a
guaranteed-bounded update.

```python
importance_ratio = torch.exp(current_log_probs - old_log_probs)
clipped_ratio    = torch.clip(importance_ratio, 0.8, 1.2)   # epsilon = 0.2
loss = torch.min(importance_ratio * advantages, clipped_ratio * advantages)
```

## On-policy vs off-policy: the economic trade-off

| | On-policy (regenerate every step) | Off-policy (reuse a batch) |
|---|---|---|
| **Pros** | Mathematically clean — no correction factor, gradients are exactly what you'd expect | Sample-efficient; reuse expensive generations across steps ("~10× fewer generations"); better compute utilization |
| **Cons** | Expensive — a full forward pass + sampling per step, batch discarded after one update | Needs importance-sampling correction; risks instability if the policy moves too far; gradients become approximations |

The "~10× fewer generations" figure is the guide's own characterization, tied to
reusing one batch across ~10 gradient steps. In practice, even nominally
on-policy GRPO becomes slightly off-policy for throughput — see
[[alignment-methods|in-flight updates]] and the systems view in
[[rl-training-systems]].

## Where the KL term went

The classic PPO/GRPO objective adds a KL penalty against a reference policy
(coefficient β). Many recent GRPO setups set **β = 0 and drop it entirely**. The
reasoning (per gitlostmurali, citing Qingfeng's analysis): "the clipped objective
is designed as a replacement of constraint policy optimization in form of the KL
divergence term. Thus, adding a KL divergence term is not necessary
theoretically" — the clip range already bounds how far the policy can move per
step. [[alignment-methods|DAPO]] removes the KL penalty for the same reason:
long-CoT reasoning may need to move *far* from the base model, so a KL leash is
counterproductive.

Note the tension with the [[alignment-methods|KL-regularized policy gradients]]
critique: *if* you keep a KL term under off-policy sampling, it needs the
matching importance weight to be correct. The two views agree that GRPO's
relationship between clipping and KL is easy to get wrong.

## Not all tokens matter equally

"Beyond the 80/20 Rule" (Wang et al., 2025, arXiv:2506.01939) finds only ~20% of
tokens in a reasoning sequence — **forking tokens**, the actual decision points
that branch the reasoning — meaningfully drive learning. Training on just that
20% subset preserves or even *improves* performance, suggesting GRPO's token-level
credit assignment is heavily skewed toward a few high-leverage positions. This
motivates [[alignment-methods|DAPO's Token-Level Policy Gradient Loss]], which
reweights by token count rather than identifying *which* tokens matter.

## GRPO in practice: DeepSeek-R1-Zero

The reference recipe ([[reasoning-models]]): GRPO on DeepSeek-V3 Base with **no
SFT cold-start** — 10.4k GRPO steps, batch size 512, reference policy replaced
every 400 steps, learning rate 3e-6, KL coefficient 0.001. Two reward types:
accuracy (verified correctness) and format (enforced thinking tags).
Self-verification, backtracking, and "wait"/"mistake" markers **emerged
spontaneously** (rising 5–7× over training) with no chain-of-thought
demonstrations — the headline result that made GRPO famous.

**GRPO vs PPO** (DeepSeek-R1 appendix): PPO's per-token KL penalty implicitly
penalizes response length (KL between sequence distributions decomposes into a
sum over tokens, and RL lengthens reasoning). GRPO avoids this and is cheaper (no
value model). On MATH, GRPO consistently beat PPO with `β = 0.04`, which in turn
beat PPO with `β = 0.0`.

## Stabilizing GRPO at scale

Naive GRPO suffers entropy collapse, reward noise, and instability at scale. Two
families of fixes:

- **[[alignment-methods|DAPO]]** — Clip-Higher (asymmetric bounds, e.g. ε_low=0.2 / ε_high=0.28, to keep exploration tokens alive), Dynamic Sampling (drop prompts with all-correct or all-wrong groups so advantages aren't zero), Token-Level Policy Gradient Loss, and Overlong Reward Shaping. Reaches 50 on AIME with Qwen2.5-32B in 50% fewer steps.
- **[[icepop-stabilizing-rl-moe|IcePop]]** — token-discrepancy masking + per-expert-path gradient clipping for the MoE-specific routing instability that GRPO's plain clipping wasn't designed to handle.

For when GRPO is *not* the right tool — compound systems where the base model
already has the capability and the prompt is the bottleneck —
[[prompt-optimization|GEPA-style reflective prompt evolution]] has beaten GRPO by
~10 points with ~35× fewer rollouts and no GPU training. Use RL for new
capabilities, prompt optimization for new instructions.

## Related Topics
- [[alignment-methods]] — GRPO's place among RLHF/DPO/RLVR/DAPO; the KL-regularized-PG critique; in-flight updates
- [[reasoning-models]] — DeepSeek-R1 / R1-Zero recipe where GRPO produced emergent reasoning
- [[rl-training-systems]] — rollout/trainer systems that run GRPO off-policy (PipelineRL, verl/slime, staleness control)
- [[icepop-stabilizing-rl-moe]] — stabilizing GRPO on MoE architectures
- [[rl-fundamentals]] — on-policy vs off-policy and importance sampling, the foundations GRPO's correction rests on
- [[policy-gradient-actor-critic]] — REINFORCE → PPO clip objective that GRPO inherits and simplifies
- [[prompt-optimization]] — GEPA as a rollout-cheap alternative to GRPO for prompt-bottlenecked systems

## Sources
- Murali Manohar, "Lightweight Guide to Understanding GRPO and RL Principles" (gitlostmurali.com, Sep 13, 2025) — the code-level walkthrough this page is built on. See [`raw/grpo-intro-gitlostmurali.md`](../raw/grpo-intro-gitlostmurali.md).
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL — see [`raw/deepseek-r1-reasoning-via-rl.md`](../raw/deepseek-r1-reasoning-via-rl.md) and [[reasoning-models]].
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale (arXiv:2503.14476) — via [[alignment-methods]].
- "Beyond the 80/20 Rule" (Wang et al., 2025, arXiv:2506.01939) — forking tokens.
- "RL's Razor" (Shenfeld et al., 2025, arXiv:2509.04259) — why RL forgets less than SFT.
