# Reinforcement Learning Fundamentals

The core framework for learning from interaction. Unlike supervised learning (learn from labelled examples) or unsupervised learning (find structure in data), RL learns by trial and error: an agent takes actions, receives rewards, and updates its policy to maximise cumulative future reward.

---

## Markov Decision Process (MDP)

An MDP is the formal mathematical framework for RL problems. It is defined by the tuple **(S, A, P, R, γ)**:

- **S** — state space (all possible states the environment can be in)
- **A** — action space (all actions the agent can take)
- **P(s' | s, a)** — transition dynamics: probability of reaching state s' after taking action a in state s
- **R(s, a, s')** — reward function: scalar feedback signal
- **γ ∈ [0, 1)** — discount factor: how much to downweight future rewards

The agent's goal is to find a **policy** π(a | s) — a mapping from states to actions (or distributions over actions) — that maximises the expected discounted return:

```
G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ... = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

### The Markov Property

An MDP assumes the **Markov Property**: the future depends only on the current state, not on the history of how we got there.

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

This is crucial because it means the current state is a sufficient statistic — you don't need to remember the entire history to act optimally. In practice, many environments are only approximately Markov (e.g., Atari frames where velocity isn't captured in a single frame), which is why frame-stacking is used.

---

## Bellman Equations

The Bellman equations are the fundamental recursive relationships that define optimal value functions.

### Value Functions

The **state-value function** V^π(s) gives the expected return starting from state s under policy π:

```
V^π(s) = E_π[G_t | S_t = s] = E_π[R_{t+1} + γ V^π(S_{t+1}) | S_t = s]
```

The **action-value function** Q^π(s, a) gives the expected return from taking action a in state s, then following π:

```
Q^π(s, a) = E_π[G_t | S_t = s, A_t = a] = E_π[R_{t+1} + γ Q^π(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
```

The relationship: V^π(s) = Σ_a π(a|s) Q^π(s, a)

### Bellman Optimality Equations

The **optimal value functions** V*(s) and Q*(s, a) satisfy:

```
V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]

Q*(s, a) = R(s,a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q*(s', a')
```

The optimal policy is then: π*(s) = argmax_a Q*(s, a)

These equations have a unique solution (under mild conditions) and form the foundation of dynamic programming-based RL. The recursive structure — current value defined in terms of future value — is what makes Bellman equations powerful: you can solve for V* by iterating the Bellman operator until convergence (value iteration).

---

## Q-Learning and TD Learning

### Temporal Difference (TD) Learning

TD learning is the core idea unifying most modern RL: **bootstrap from your own estimates** rather than waiting for the episode to end (unlike Monte Carlo).

The TD(0) update for the value function:

```
V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]
```

The term `R_{t+1} + γ V(S_{t+1}) - V(S_t)` is the **TD error** (δ_t) — how wrong our current estimate was. TD converges to the correct value function under standard conditions (decaying learning rates, all state-action pairs visited infinitely often).

**TD vs Monte Carlo:**

| Property | Monte Carlo | TD |
|----------|------------|-----|
| Updates | End of episode | Each step |
| Bias | Zero bias | Biased (bootstrapping) |
| Variance | High | Lower |
| Works with | Episodic only | Episodic + continuing |
| Data efficiency | Low | High |

### Q-Learning

Q-Learning (Watkins, 1989) is the canonical off-policy TD method for learning Q*(s, a) directly:

```
Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)]
```

Key properties:
- **Off-policy**: the target uses `max_{a'} Q(S_{t+1}, a')` regardless of what action was actually taken. This means you can learn from data collected by any behaviour policy.
- **Convergence**: provably converges to Q* for any behaviour policy that visits all state-action pairs, with appropriate learning rates.
- **Tabular form**: when state-action space is small enough to enumerate.
- **Deep Q-Network (DQN)**: extends Q-Learning to large state spaces using a neural network to represent Q(s, a; θ), stabilised with experience replay and a target network.

**DQN stabilisation tricks:**
- **Experience replay**: store (s, a, r, s') transitions in a buffer, sample random minibatches. Breaks temporal correlations, improves data efficiency.
- **Target network**: use a separate, periodically-updated network Q(s, a; θ⁻) for computing targets. Prevents chasing a moving target.
- **Reward clipping**: clip rewards to [-1, 1] for stability across games.

---

## SARSA

SARSA is the **on-policy** counterpart to Q-Learning. The name comes from the quintuple (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}):

```
Q(S_t, A_t) ← Q(S_t, A_t) + α [R_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

The update uses the actual next action A_{t+1} taken by the policy, rather than the greedy max. This means SARSA learns the value of the **actual policy being followed**, including its exploration behaviour.

**When SARSA is preferred over Q-Learning:**
- Cliff-walking problems: Q-Learning learns the optimal risky path but the agent falls because it's still exploring; SARSA learns the safer path
- When you care about safe online learning (the policy you learn is the policy you're running)

---

## Monte Carlo vs TD

Both estimate value functions from sampled trajectories.

**Monte Carlo** waits until the end of an episode, then updates using the full actual return G_t:

```
V(S_t) ← V(S_t) + α [G_t - V(S_t)]
```

- Unbiased estimates (uses true returns)
- High variance (return is sum of many random rewards)
- Only works for episodic tasks
- Slow to propagate information (one update per episode)

**TD** bootstraps from the next-step estimate:

```
V(S_t) ← V(S_t) + α [R_{t+1} + γ V(S_{t+1}) - V(S_t)]
```

- Biased (V(S_{t+1}) is an estimate)
- Lower variance
- Works for continuing tasks
- Faster propagation (update every step)

**TD(λ)** unifies them via eligibility traces, interpolating between TD(0) and Monte Carlo with a parameter λ ∈ [0, 1]:

```
G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t^(n)
```

where G_t^(n) is the n-step return. λ=0 gives TD(0), λ=1 gives Monte Carlo.

---

## On-Policy vs Off-Policy

**On-policy** algorithms learn about the same policy that is used to collect data. The policy being evaluated and the policy generating behaviour are the same.

- Examples: SARSA, PPO (approximately), A3C
- More stable, simpler to implement
- Less data efficient — can't reuse old experience directly

**Off-policy** algorithms learn about a **target policy** different from the **behaviour policy** that collects data.

- Examples: Q-Learning, DQN, SAC, TD3
- Can reuse experience (replay buffers)
- More data efficient
- Requires importance sampling corrections if policies differ significantly
- Can learn optimal policy while exploring with a suboptimal behaviour policy

The key challenge in off-policy learning is distribution shift: your estimates are valid for the policy that collected the data, not necessarily the policy you're improving. This is why importance sampling (see below) appears in off-policy corrections and why GRPO-family algorithms carefully track whether rollouts are on- or off-policy.

---

## Exploration vs Exploitation Dilemma

One of the central tensions in RL: **exploit** what you know (take the action you currently believe is best) vs **explore** to gather information about actions you haven't tried (which might be better).

### ε-greedy

The simplest approach: take a random action with probability ε, take the greedy action with probability 1-ε.

```python
if random() < epsilon:
    action = random_action()
else:
    action = argmax(Q[state])
```

Usually decay ε over time as the agent learns. Simple and effective in practice.

### Upper Confidence Bound (UCB)

Select action based on an optimism-in-the-face-of-uncertainty principle — actions that have been tried fewer times get a bonus:

```
A_t = argmax_a [Q(a) + c * sqrt(ln(t) / N_t(a))]
```

where N_t(a) is the number of times action a has been taken. UCB1 achieves logarithmic regret in the multi-armed bandit setting.

### Thompson Sampling

Maintain a posterior distribution over Q values, sample from it to pick an action. Natural Bayesian approach to exploration.

### Intrinsic Motivation / Curiosity

Add an **intrinsic reward** based on novelty or prediction error:

- **Count-based exploration**: bonus inversely proportional to visit count √(1/N(s,a))
- **Random Network Distillation (RND)**: train a fixed random network and a predictor; high prediction error = novel state
- **ICM (Intrinsic Curiosity Module)**: prediction error in a learned feature space

In LLM RL, exploration manifests differently: entropy regularisation (`-β H(π)` term in the loss) prevents the policy from collapsing to deterministic behaviour, maintaining sufficient diversity in rollouts.

---

## Credit Assignment Problem

**Which actions in a trajectory caused the reward?** In many RL settings, rewards are sparse and delayed — you might only receive a reward at the end of a 100-step episode. Figuring out which of the 100 actions were responsible is the credit assignment problem.

**Temporal credit assignment**: attributing reward to actions across time. Solutions:
- **Eligibility traces / TD(λ)**: maintain a decaying trace of recently visited states/actions; propagate credit backwards in time
- **GAE (Generalised Advantage Estimation)**: variance-reduced estimate of advantage, effectively a soft n-step return
- **Reward shaping**: add intermediate rewards to provide more frequent signal

**Structural credit assignment**: in multi-agent or compositional settings, which *component* (agent, submodule) deserves credit? Relevant in hierarchical RL and multi-agent RL.

In LLM RL, credit assignment is particularly hard because the "action" is a token and the reward is at the end of a potentially 1000+ token response. GRPO addresses this with token-level policy gradient loss, but the fundamental problem remains — a single binary reward at the end provides very weak per-token credit.

---

## Importance Sampling

Importance Sampling (IS) is a technique for estimating expectations under one distribution using samples from another.

Given samples from behaviour policy b(a|s), to evaluate target policy π(a|s):

```
E_π[f(x)] = Σ_x π(x) f(x) = Σ_x b(x) [π(x)/b(x)] f(x) ≈ (1/n) Σ_i [π(x_i)/b(x_i)] f(x_i)
```

The ratio **ρ = π(a|s) / b(a|s)** is the importance weight.

**In RL:**
- **Off-policy TD**: weight each sample by the product of per-step importance weights ρ_t = π(A_t|S_t) / b(A_t|S_t)
- **PPO clipping**: clips the importance ratio ρ to [1-ε, 1+ε] to prevent large updates when the behaviour policy diverges from the current policy
- **Sequence-level IS**: for sequences, ρ is a product of per-token ratios — this can explode (high variance). Solutions: use per-token IS (DeepSeek), or restrict the staleness of off-policy data

**Variance issues**: IS can have very high variance when the target and behaviour policies diverge significantly. Clipped IS (PPO), normalised IS, and per-token IS are practical solutions.

In the context of GRPO off-policy training (discussed in [[frontier-async-rl]]), the distinction between **Sequence IS** (one weight per trajectory) and **Token IS** (per-token weight) is important. Token IS has lower variance and is what the Bellman equation requires when rewards are on individual tokens.

---

## Curriculum Learning

Training a model on tasks ordered by difficulty, starting easy and gradually increasing complexity.

**Why it works:**
- Easy examples provide stable gradients early in training
- Hard examples become tractable once the model has basic competence
- Prevents the model from getting stuck in local minima when all examples are hard
- Analogy: human learning — you don't learn calculus before arithmetic

**Approaches:**

**Self-paced learning**: let the model determine what's hard (based on current loss). Train on examples the model finds "appropriately challenging" — not too easy (no learning signal) or too hard (noisy gradients).

**Task curriculum**: in RL, sequence environments by difficulty. Start with shorter episodes, simpler physics, or reduced action spaces.

**Difficulty metrics**: loss/error magnitude, age of training examples, teacher model confidence, adversarial difficulty.

**In LLM RL (RLVR):** curriculum over problem difficulty is common — start with problems where the model can sometimes succeed (so there's reward signal), then increase difficulty as competence grows. Problems where the model always fails (reward=0) or always succeeds (reward=1) both provide zero gradient signal — the **Goldilocks difficulty zone** from [[reward-hacking-dynamics]].

**DAPO's dynamic sampling** is a form of implicit curriculum: filter out prompts with group accuracy=0 (too hard) or =1 (too easy), maintaining a rolling window of tractable problems.

**Anti-curriculum**: some work finds training on hard examples first then moving to easier ones can also help — forces the model to develop general strategies early.
