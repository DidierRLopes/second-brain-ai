# Policy Gradient Methods and Actor-Critic

Policy gradient methods directly optimise the policy π(a|s; θ) by computing gradients of the expected return with respect to the policy parameters θ. This contrasts with value-based methods (Q-Learning) which learn a value function and derive a policy from it.

---

## Policy Gradient Theorem

The policy gradient theorem (Sutton et al., 1999) provides a closed-form expression for the gradient of expected return with respect to policy parameters, without needing to differentiate through the environment dynamics.

**Setup:** we want to maximise:

```
J(θ) = E_π[G_t] = E_π[Σ_{t=0}^T γ^t R_{t+1}]
```

**The theorem** states:

```
∇_θ J(θ) = E_π [ Σ_t ∇_θ log π(A_t | S_t; θ) · Q^π(S_t, A_t) ]
```

This is remarkable: the gradient depends on Q^π (the action-value function under the current policy) but **not on the derivative of the state distribution** — even though changing θ changes the distribution of states visited. The state distribution gradient terms cancel out.

**Intuition:** actions that lead to high returns get their log-probability increased; actions that lead to low returns get their log-probability decreased. The term ∇_θ log π(A_t | S_t; θ) is the **score function** — the direction to push θ to make the taken action more likely.

**REINFORCE (Williams, 1992)** is the simplest instantiation. Use Monte Carlo estimates of Q^π:

```
θ ← θ + α Σ_t G_t · ∇_θ log π(A_t | S_t; θ)
```

where G_t is the actual return from time t. High variance due to Monte Carlo sampling — the return varies a lot across trajectories.

---

## Variance Reduction in RL

High variance is the primary problem with policy gradient methods. Several techniques reduce it:

### Baseline Subtraction

Replace Q^π(s,a) with an **advantage** A^π(s,a) = Q^π(s,a) - b(s), where b(s) is any baseline that doesn't depend on the action. The gradient is unchanged (baselines have zero expectation), but variance is reduced.

**Proof**: E_π[∇_θ log π(a|s) · b(s)] = b(s) · Σ_a ∇_θ π(a|s) = b(s) · ∇_θ Σ_a π(a|s) = b(s) · ∇_θ 1 = 0

The optimal baseline is b*(s) = E[Q(s,a) · (∇ log π)² ] / E[(∇ log π)²], approximated in practice by V^π(s).

The **advantage function** A^π(s,a) = Q^π(s,a) - V^π(s) measures how much better action a is compared to average. Using advantages instead of returns dramatically reduces variance.

### Reward-to-Go

Instead of using the full return G_t = Σ_{k=0}^∞ γ^k R_{t+k}, only use the **future return** starting from time t. Rewards before taking action A_t don't depend on A_t, so they add variance without signal.

### Causality

Past rewards are not caused by future actions, so the gradient estimator is cleaner when we only weight ∇_θ log π(A_t|S_t) by rewards from t onwards.

### Normalise Advantages

Within a batch, normalise advantages: A_norm = (A - mean(A)) / std(A). Reduces sensitivity to reward scale, prevents updates being dominated by a few high-advantage samples.

---

## Generalised Advantage Estimation (GAE)

GAE (Schulman et al., 2016) provides a practical, variance-controlled advantage estimator that interpolates between TD(0) and Monte Carlo.

**TD residual (1-step advantage estimate):**

```
δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t)
```

This is a biased, low-variance estimate of A^π(S_t, A_t).

**GAE(λ)** is the exponentially-weighted average of k-step advantage estimates:

```
Â_t^GAE(γ,λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

Equivalently: Â_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...

- **λ=0**: Â_t = δ_t = R_{t+1} + γ V(S_{t+1}) - V(S_t) (pure TD, low variance, high bias)
- **λ=1**: Â_t = Σ_{l=0}^∞ γ^l δ_{t+l} = G_t - V(S_t) (Monte Carlo advantage, high variance, low bias)

**Practical implementation** (rollout of T steps):

```python
advantages = []
gae = 0
for t in reversed(range(T)):
    delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
    gae = delta + gamma * lam * (1-dones[t]) * gae
    advantages.insert(0, gae)
```

GAE(γ=0.99, λ=0.95) is the standard setting across most PPO implementations. The λ parameter provides a clean bias-variance knob that doesn't exist in simpler estimators.

---

## Actor-Critic

Actor-Critic combines:
- An **actor** (policy network π(a|s; θ)) that selects actions
- A **critic** (value network V(s; w) or Q(s,a; w)) that evaluates how good states/actions are

The critic provides low-variance advantage estimates to the actor's gradient updates. The actor-critic architecture is the basis of PPO, A3C, A2C, SAC, and most modern deep RL.

**Advantage Actor-Critic (A2C/A3C):**

```
∇_θ J ≈ E [ Â_t · ∇_θ log π(A_t | S_t; θ) ]
```

with Â_t estimated using the critic V(s; w). The critic is trained to minimise:

```
L_critic = E[(V(S_t; w) - G_t)²]   # or TD error
```

**A3C (Mnih et al., 2016)** asynchronously runs multiple workers in parallel environments, each computing gradients and pushing them to a shared parameter server. No replay buffer needed — parallelism provides decorrelated samples.

**Shared vs separate networks:** embedding the actor and critic in a shared trunk (shared layers + separate heads) allows feature reuse. Separate networks are stabler but less efficient.

---

## Proximal Policy Optimization (PPO)

PPO (Schulman et al., 2017) is the most widely used deep RL algorithm. It improves on TRPO (Trust Region Policy Optimization) by using a simpler clipped objective instead of a hard constraint.

### The PPO Objective

The clipped surrogate objective:

```
L^CLIP(θ) = E_t [ min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t) ]
```

where **r_t(θ) = π_θ(A_t | S_t) / π_θ_old(A_t | S_t)** is the importance sampling ratio between the current and old policy.

**Intuition of the clip:**
- If Â_t > 0 (action was better than average): we want to increase r_t, but clip it at 1+ε. The `min` ensures we don't benefit from pushing r_t beyond 1+ε.
- If Â_t < 0 (action was worse than average): we want to decrease r_t, but clip at 1-ε. The `min` ensures we don't benefit from pushing r_t below 1-ε.

The clipping effectively enforces a **trust region** — the policy can't move too far from the old policy — without solving a constrained optimisation problem.

### Full PPO Loss

```
L(θ) = L^CLIP(θ) - c_1 L^VF(θ) + c_2 H(π_θ)
```

- L^VF = (V_θ(S_t) - V_t^target)² — value function loss
- H(π_θ) — entropy bonus to encourage exploration
- Typical: c_1 = 0.5, c_2 = 0.01

### PPO Training Loop

1. Collect rollouts from current policy π_θ_old for T steps
2. Compute advantages Â_t using GAE
3. For K epochs, shuffle data into minibatches and optimise L(θ)
4. Update θ_old ← θ
5. Repeat

**PPO Hyperparameters:**
- ε = 0.1–0.2 (clipping range)
- K = 3–10 epochs per rollout
- T = 128–4096 steps per rollout
- λ = 0.95 (GAE)
- γ = 0.99 (discount)

### PPO vs GRPO (in LLM context)

For LLM alignment (see [[alignment-methods]]):
- PPO maintains a separate value model (critic) which is expensive
- PPO's per-token KL penalty implicitly penalises long sequences
- GRPO uses group-relative rewards (no separate critic needed)
- GRPO doesn't penalise length, which matters for reasoning tasks

**PPO remains standard** in classic RL domains (games, robotics) where a critic is affordable and a per-token discount makes sense. GRPO is preferred for LLM post-training at scale.

---

## Connections to LLM Training

The policy gradient theorem underpins all of modern LLM alignment:

- **REINFORCE loss in RLHF/GRPO**: each token is an "action," the reward at the end of generation is the "return," and we optimise log π(token) weighted by advantage
- **KL penalty as trust region**: `β KL(π || π_ref)` plays the role of PPO clipping — prevents the policy from drifting too far
- **Importance sampling in off-policy GRPO**: when rollouts come from a previous snapshot of the policy, we need to correct with r_t = π_θ(a) / π_old(a), which is exactly the PPO ratio
- **GAE for process reward models**: when you have dense per-step rewards (process reward model, tool call success/failure), GAE estimates the advantage of each reasoning step

See also: [[alignment-methods]], [[rl-training-systems]], [[frontier-async-rl]]
