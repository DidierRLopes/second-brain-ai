# Lightweight Guide to Understanding GRPO and RL Principles

**Source:** https://gitlostmurali.com/blog/grpo-intro/
**Author:** Murali Manohar
**Published:** September 13, 2025

## Framing / Motivation

The author calls this a "missing piece" written for readers with *no prior RL knowledge* — most GRPO explainers assume familiarity with PPO/DPO first. The post is explicitly beginner-oriented and code-first: every equation is immediately translated into a Python-pseudocode loop.

The author's tagline for GRPO: it works on the **"FAFO principle" — Fool Around and Find Out**. Concretely: generate multiple responses to the same prompt, calculate an advantage for each, then push the model toward higher-advantage responses and away from lower-advantage ones.

## Why Advantages, Not Just Rewards

Reward alone says whether a response is good; it doesn't say how good *relative to its peers* on the same query. Advantage is obtained by normalizing rewards within the group using the group's mean and standard deviation (standard GRPO group-relative normalization — the post doesn't spell out the exact formula numerically but frames it purely as "normalize rewards with mean and std").

## Core GRPO Objective (atomic unit, no clipping yet)

The post isolates the "atomic unit" of the full GRPO objective as:

```
L_GRPO = (1/G) * sum_{i=1}^{G} (1/|o_i|) * sum_{t=1}^{|o_i|} pi_theta(o_{i,t} | q, o_{i,<t}) * A_hat_{i,t}
```

Where:
- `G` = number of generated answers (group size) for a given prompt
- `o_i` = the i-th generated answer; `|o_i|` = number of tokens in it
- `pi_theta(o_{i,t} | q, o_{i,<t})` = the policy's probability (the post calls it "log probabilities" loosely) of token `t` given the query `q` and previous tokens in that answer
- `A_hat_{i,t}` = the advantage — note it is computed **per answer/sequence**, not per token (same advantage value broadcast to every token in that answer)

### Code translation given in the post

```python
for each_generated_answer_i in generated_answers: # G in the equation
    advantage = calculate_advantage(each_generated_answer_i)
    for each_token_o_t in each_generated_answer_i: # |o_i| in the equation
        token_loss = pi_theta(each_token_o_t) * advantage

    loss_of_each_answer = sum(tokens_loss_in_answer_i) / len(each_generated_answer_i)

final_loss = sum(loss_of_all_answers) / len(generated_answers) # G in the equation
```

The author's framing: this is just "a two-nested for-loop over `i` and `t` against `pi_theta`" — i.e., GRPO's loss, stripped of importance sampling/clipping, is structurally nothing more than a double loop: outer loop over the `G` sampled answers, inner loop over tokens within each answer, multiplying each token's log-prob by that answer's (sequence-level) advantage, then averaging twice (over tokens, then over answers).

## The Hidden Problem: Training on Stale Data

The post walks through *why* naive reuse of sampled data breaks, using a concrete training recipe:

1. Generate a batch of answers with the current model (example given: 4 answers per prompt)
2. Calculate advantages for those answers once
3. Train on that **same** batch for multiple gradient steps (example given: 10 steps)

This is efficient (avoids regenerating expensive LLM rollouts every step) but creates **distribution shift / off-policy training**: by gradient step 10 the model's weights have moved, but the log-probs being used in the loss are still being computed as if the *current* (already-updated) model had generated those tokens, when in fact the *step-0* model generated them.

**Author's analogy:** "It's like practicing basketball shots based on a video of yourself from last week. You've improved since then, so the video doesn't represent your current form anymore."

Broken code illustrating the bug:

```python
# Generate answers with initial model
answers = model.generate(prompt)  # Model at step 0
advantages = calculate_advantages(answers)

# Train for multiple steps on the SAME answers
for step in range(10):
    log_probs = model.get_log_probs(answers)  # Model at step 1, 2, ... 10
    loss = log_probs * advantages
    model.update(loss)
    # By step 10, we're calculating gradients as if the current model
    # generated these answers, but it didn't! The step-0 model did!
```

Stated consequence: increasingly biased gradients, unstable training, and possible "unlearning" of good behaviors because the gradient signal no longer matches the data-generating process.

## The Fix: Importance Sampling Ratio

Introduce `pi_theta_old` — the frozen log-probs from the model that *actually* generated the answers — and reweight the loss by the ratio between current and old policy probabilities:

```
token_loss = ( pi_theta(o_{i,t}|q, o_{i,<t}) / pi_theta_old(o_{i,t}|q, o_{i,<t}) ) * A_hat_{i,t}
```

This ratio `pi_theta / pi_theta_old` is the **importance sampling ratio**. The post gives a clean three-case operational reading:

- Ratio > 1: current model likes this token *more* than the old model did → amplify the gradient
- Ratio < 1: current model likes this token *less* than the old model did → reduce the gradient
- Ratio = 1: models agree → gradient unchanged

Code:

```python
# Generate answers ONCE with initial model
answers = old_model.generate(prompt)
old_log_probs = old_model.get_log_probs(answers)  # Store these!
advantages = calculate_advantages(answers)

# Now we can safely train for multiple steps
for step in range(10):
    current_log_probs = model.get_log_probs(answers)

    # The magic correction factor
    importance_ratio = exp(current_log_probs - old_log_probs)

    # Corrected loss that accounts for distribution shift
    loss = importance_ratio * advantages
    model.update(loss)
```

Note the ratio is computed as `exp(current_log_probs - old_log_probs)` — i.e., probabilities are tracked in log-space and the ratio is recovered via exponentiation of the log-prob difference, which is the standard numerically stable implementation pattern.

## Clipping for Safety

If the importance ratio swings to an extreme (the post's examples: ratio = 100 or ratio = 0.01), training can explode or collapse. GRPO clips the ratio:

```
ratio_clipped = clip(ratio, 1 - epsilon, 1 + epsilon)
```

With the **typical value epsilon = 0.2**, the ratio is constrained to the range **[0.8, 1.2]**.

Full clipped objective given in the post:

```
L_GRPO = (1/G) * sum_{i=1}^{G} (1/|o_i|) * sum_{t=1}^{|o_i|}
         min( ratio_{i,t} * A_hat_{i,t},
              clip(ratio_{i,t}, 1-epsilon, 1+epsilon) * A_hat_{i,t} )
```

Code:

```python
# With clipping for safety
importance_ratio = exp(current_log_probs - old_log_probs)
clipped_ratio = torch.clip(importance_ratio, 0.8, 1.2)  # epsilon = 0.2

# Take the minimum of clipped and unclipped objectives
loss_unclipped = importance_ratio * advantages
loss_clipped = clipped_ratio * advantages
loss = torch.min(loss_unclipped, loss_clipped)
```

Author's framing: clipping is "a conservative approach to prioritize stable training over perfect gradient correction" — it trades off exact-gradient fidelity for guaranteed bounded updates.

## On-Policy vs. Off-Policy: Why Not Regenerate Every Step?

The post poses this as the natural follow-up question and answers it with a direct comparison.

**On-policy (the "ideal" approach):** generate fresh data from the current model on every single gradient step.

```python
for step in range(training_steps):
    # Generate fresh data with current model
    answers = current_model.generate(prompt)
    advantages = calculate_advantages(answers)
    log_probs = current_model.get_log_probs(answers)

    # Simple, clean loss calculation
    loss = log_probs * advantages
    current_model.update(loss)
```

Mathematically clean — no importance-sampling correction needed since the data always matches the current policy. But each LLM rollout (full forward pass + sampling) is expensive, and the batch is thrown away after a single gradient step.

**Off-policy (the "economic" approach):** reuse a single batch of generations across multiple gradient steps (the scheme analyzed throughout the post), correcting via importance sampling + clipping.

```python
# Generate once with current model
answers = current_model.generate(prompt)
old_log_probs = current_model.get_log_probs(answers)
advantages = calculate_advantages(answers)

for step in range(10):  # Reuse same data for multiple steps
    new_log_probs = current_model.get_log_probs(answers)

    # Need importance sampling to correct for staleness
    importance_ratio = exp(new_log_probs - old_log_probs)
    loss = importance_ratio * advantages
    current_model.update(loss)
```

### Trade-off table from the post

| | On-policy | Off-policy |
|---|---|---|
| Pros | Mathematically cleaner (no correction factors needed); always trains on "fresh" data from current policy; gradients are exactly what you'd expect | Sample efficient — reuse expensive generations multiple times; "much faster in practice (10x fewer generations needed)"; better compute utilization |
| Cons | Extremely expensive (full forward pass + sampling per step); wastes compute (batch discarded after one gradient step); slower convergence in wall-clock time | Requires complex corrections (importance sampling); risk of instability if the model changes too much; gradients become approximations rather than exact |

The explicit "10x fewer generations needed" figure is the author's own characterization of the efficiency gain from off-policy reuse, tied directly to the example of reusing one batch across 10 gradient steps.

## Interesting Developments Section

### 1. KL divergence term disappears in practice

The post explicitly notes it *omitted* the KL-divergence term from the objective (the standard PPO/GRPO formulation usually includes a KL penalty against a reference policy). Reasoning given: recent GRPO implementations set **β = 0**, removing the KL term entirely, because the clipping mechanism already constrains how far the policy can move — making the KL term theoretically redundant. The post quotes/cites Qingfeng's blog post (lancelqf.github.io) on this point: "the clipped objective is designed as a replacement of constraint policy optimization in form of the KL divergence term. Thus, adding a KL divergence term is not necessary theoretically."

### 2. Why RL forgets less than SFT

Cites "RL's Razor" (Shenfeld et al., 2025, arXiv:2509.04259): RL fine-tuning — especially on-policy training — forgets less of the original model's capabilities than SFT does, even when both reach the same performance on the new target task. Practical implication noted by the author: this matters when you're fine-tuning for a new task but want to preserve standard-benchmark performance.

### 3. "Forking tokens" — the 80/20 rule for reasoning tokens

Cites "Beyond the 80/20 Rule" (Wang et al., 2025, arXiv:2506.01939): only ~20% of tokens in reasoning sequences actually matter for learning and exploration of reasoning paths. These are called **"forking tokens"** — decision points that drive nearly all of the performance gains. The paper's finding as relayed here: training on just this 20% subset of tokens not only maintains performance but actually *improves* it.

## Visual Aids Referenced

The post includes three figures (images hosted at gitlostmurali.com/assets/images/grpo-intro/): a horizontal "GRPO Training Workflow Overview" diagram (Figure 1, reused as Figure 3), and a "GRPO Objective Function" image (Figure 2) showing the full clipped-objective equation before it's decomposed piece by piece in text.
