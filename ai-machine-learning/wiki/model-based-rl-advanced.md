# Model-Based RL and Advanced Algorithms

---

## Model-Based vs Model-Free

**Model-free RL** learns a policy (and/or value function) directly from environment interactions, without building an explicit model of the world.

- Examples: Q-Learning, DQN, PPO, SAC
- Simpler to implement, more broadly applicable
- Sample-inefficient: needs many real-environment interactions
- No planning ahead — the policy just reacts to states

**Model-based RL** learns a **world model** P(s' | s, a) and uses it for:
- **Planning**: simulate rollouts internally to evaluate actions (without real environment)
- **Generating synthetic data**: augment real experience with imagined rollouts
- **Lookahead search**: MCTS, beam search over model-predicted futures

Tradeoffs:

| | Model-Free | Model-Based |
|--|-----------|-------------|
| Sample efficiency | Low | High |
| Wall-clock efficiency | High (no model training) | Lower |
| Generalisation | Limited to seen experience | Can plan to unseen states |
| Model errors | N/A | Can compound ("model bias") |
| Best for | Cheap simulation, large data | Expensive real-world interaction |

**Model bias** is the key failure mode: if the world model is imperfect, optimising against it leads to exploiting model errors rather than solving the real problem. This is analogous to reward hacking in LLMs — the model learns to fool the learned model rather than actually succeed.

---

## AlphaGo

AlphaGo (Silver et al., 2016, DeepMind) was the first system to defeat a professional human Go player. It combined deep neural networks with Monte Carlo Tree Search (MCTS).

### The Problem

Go is a board game with ~10^170 possible positions — far too large for exhaustive search. The branching factor (~250 legal moves) makes traditional minimax search intractable even with alpha-beta pruning.

### AlphaGo Architecture

Two networks:
- **Policy network** p_σ(a|s): trained via SL on expert human moves, then fine-tuned via policy gradient RL (self-play). Outputs probability distribution over legal moves.
- **Value network** v_θ(s): predicts win probability from state s. Trained on positions from RL self-play games.

### MCTS with Neural Network Guidance

MCTS builds a search tree. At each node:
1. **Selection**: traverse tree using UCB score: a* = argmax_a [Q(s,a) + u(s,a)], where u ∝ P(s,a)/(1+N(s,a))
2. **Expansion**: add new leaf, evaluate with fast rollout policy and value network
3. **Simulation**: run fast rollout to terminal state (random play or fast policy)
4. **Backup**: update Q values and visit counts up the tree

The value network replaces expensive Monte Carlo rollouts — instead of simulating to terminal, we evaluate position directly.

### AlphaGo Zero / AlphaZero

AlphaGo Zero (2017) removed human knowledge entirely: no human games, just self-play from scratch. Single network outputs both policy and value. No fast rollout policy.

AlphaZero (2017) extended this to chess and shogi with identical architecture, training from scratch in hours to superhuman level. See also [[alignment-methods]] for the connection to LLM self-play reasoning.

**Key insight from AlphaZero:** given a perfect reward signal (win/loss), self-play + MCTS generates training data that improves the policy which improves MCTS which improves the data. This bootstrapping loop is the conceptual ancestor of RLVR for LLMs.

---

## MuZero

MuZero (Schrittwieser et al., 2020, DeepMind) extends AlphaZero to **unknown dynamics** — the agent learns its own model of the world rather than relying on a simulator.

### The Core Idea

Instead of learning the true game dynamics, MuZero learns three functions:
- **Representation function** h: s_t → s̃_t (encode observation to hidden state)
- **Dynamics function** g: (s̃_t, a_t) → (r_t, s̃_{t+1}) (predict reward and next hidden state)
- **Prediction function** f: s̃_t → (p_t, v_t) (policy and value from hidden state)

The hidden state s̃ doesn't need to match the true environment state — it only needs to be useful for predicting rewards and values. This is learned end-to-end.

### Planning with the Learned Model

At decision time, MuZero runs MCTS **entirely within the learned model**:
1. Start from encoded observation
2. Expand tree by simulating actions through the dynamics function
3. Use policy and value predictions to guide search
4. Execute the most-visited action

Training uses a joint loss: prediction loss (policy, value, reward) from MCTS targets.

### Significance

MuZero achieves superhuman performance on chess, shogi, Go, and Atari games — all with the same algorithm, no knowledge of the rules. It's arguably the most general RL system pre-LLM era. The learned latent dynamics model (not interpretable, but predictive) is the key advance over AlphaZero.

**Connection to World Models:** MuZero's learned dynamics function is a world model in the latent space. The representation + dynamics + prediction pipeline is exactly the structure Dreamer later adopted for continuous control.

---

## World Models / Dreamer

**World Models** (Ha & Schmidhuber, 2018) introduced the idea of learning a compressed latent representation of the environment and training a policy entirely in the "dream" (imagined rollouts).

### Architecture (World Models)

Three components:
- **V (Vision model)**: VAE that compresses observation o_t to latent z_t
- **M (Memory model)**: MDN-RNN predicting next latent states given actions
- **C (Controller)**: linear policy π(a|h,z) trained with CMA-ES (evolution strategy)

Training: first train V and M on environment interactions, then evolve C purely in imagination. The controller never needs to interact with the real environment after the world model is trained.

### Dreamer (Hafner et al., 2019) and DreamerV3 (2023)

Dreamer learns a **Recurrent State Space Model (RSSM)** with:
- Deterministic recurrent state: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
- Stochastic latent state: z_t ~ p(z_t | h_t) (prior) or q(z_t | h_t, o_t) (posterior)

The RSSM can generate trajectories purely in latent space. Actor and critic are trained via **analytic gradients through the imagined rollouts** (backprop through time in the latent space), which is much more data-efficient than policy gradients.

**DreamerV3** (2023) achieves:
- First algorithm to collect diamonds in Minecraft from scratch
- Positive returns on 150+ tasks across games, robotics, locomotion — without task-specific hyperparameters
- Single set of hyperparameters across all domains

Key innovation in DreamerV3: **symlog predictions** — apply symlog(x) = sign(x) log(|x|+1) to targets, making the model robust to wildly varying reward scales. This enables truly zero-configuration training.

### World Models for LLMs

The concept maps to LLMs: a language model that predicts next tokens is implicitly a world model for text. "Reasoning with a world model" is the intuition behind chain-of-thought — the model simulates possible futures in token space before acting. Explicit world models for LLM agents (predicting environment responses to actions) are an active research area.

---

## Soft Actor-Critic (SAC)

SAC (Haarnoja et al., 2018) is the dominant model-free algorithm for **continuous action spaces** (robotics, locomotion). It adds maximum entropy to the RL objective.

### Maximum Entropy RL

Instead of maximising expected return alone, SAC maximises expected return plus entropy:

```
J(π) = E_π [ Σ_t R(s_t, a_t) + α H(π(·|s_t)) ]
```

where H(π(·|s)) = -E[log π(a|s)] is the policy entropy and α is the temperature parameter controlling the entropy-return tradeoff.

**Why entropy?**
- Encourages exploration: policy remains stochastic, tries multiple solutions
- Robustness: entropy-maximising policies are more robust to perturbations
- Better exploration-exploitation: naturally balances committing to good actions vs maintaining variety

### SAC Algorithm

SAC maintains:
- **Actor** π_φ(a|s): policy network (Gaussian, outputs mean + log_std)
- **Two critics** Q_{θ1}(s,a), Q_{θ2}(s,a): Q-functions (twin-Q to reduce overestimation)
- **Target critics**: slowly-updated copies for stable training
- **Automatic temperature α**: tuned to maintain target entropy H_target (usually -dim(A))

**Critic update** (off-policy from replay buffer):

```
y = r + γ (min_i Q_{θ_i'}(s', ã') - α log π_φ(ã'|s'))
L_Q = (Q_θ(s,a) - y)²
```

where ã' ~ π_φ(·|s') is a new sample from the current policy.

**Actor update** (maximise Q - α log π):

```
L_π = E_s [ α log π_φ(ã|s) - min_i Q_{θ_i}(s, ã) ]
```

The reparameterisation trick: ã = f_φ(ε; s) = μ_φ(s) + σ_φ(s) ⊙ ε, ε ~ N(0,I). This allows backprop through the sampling operation.

**Temperature update:**

```
L_α = E_s [ -α (log π_φ(ã|s) + H_target) ]
```

SAC is remarkably robust: it works out-of-the-box on most continuous control tasks without careful hyperparameter tuning. The automatic entropy tuning (α adaptation) removes the need to set the most critical hyperparameter manually.

### SAC vs PPO (Continuous Control)

| | SAC | PPO |
|--|-----|-----|
| Sample efficiency | High (replay buffer) | Lower (on-policy) |
| Stability | High (off-policy stability tricks) | High (clipping) |
| Hyperparameters | Few (automatic α) | More |
| Parallelism | Replay buffer parallelises | Workers parallelise |
| Best for | Robotics, expensive envs | Game playing, cheap envs |

TD3 (Twin Delayed DDPG) is a deterministic alternative to SAC without entropy maximisation, useful when you want a deterministic policy at test time.

---

## EfficientZero: Sample-Efficient MuZero

[EfficientZero (2111.00210)](../../papers/05-learning/reinforcement-learning/Mastering Atari Games with Limited Data - 2111.00210.pdf) adapts MuZero to the low-data visual-control regime by addressing three failure modes that become severe when replay data is scarce:

1. A SimSiam-style temporal consistency loss trains the predicted latent next state to match the representation of the real next observation, supplying richer supervision than scalar reward/value/policy losses.
2. An LSTM predicts the discounted **value prefix** end-to-end instead of summing independently predicted rewards, reducing compounding error from aliased imagined states.
3. A model-based off-policy correction shortens the real replay prefix for older trajectories, then re-runs MCTS with the current policy at the bootstrap state.

On Atari 100k - 100,000 environment steps, roughly two hours of play - EfficientZero reported mean and median human-normalized scores of 1.943 and 1.090 and exceeded human performance on 14 of 26 games. DQN's comparable mean score used 500 times more frames. On three DMControl 100k image tasks, EfficientZero was competitive with or better than a SAC oracle trained from ground-truth states. The result is a useful design lesson: model-based RL becomes sample-efficient only when representation learning, model-error control, and replay-policy mismatch are handled together.

## Long-Horizon Q-Learning (LQL): Bounding Compounding TD Error

SAC and the other value-based methods above all train their critic with the same basic mechanism: bootstrapped temporal-difference (TD) updates, `Q(s,a) ← r + γ Q(s', a')`. [Long-Horizon Q-Learning: Accurate Value Learning via n-Step Inequalities (2605.05812)](../../papers/05-learning/reinforcement-learning/Long-Horizon Q-Learning: Accurate Value Learning via n-Step Inequalities - 2605.05812.pdf) (Shi & Finn, Stanford) tackles a specific failure mode of that mechanism: **bootstrapping makes long-horizon Q-learning brittle** because estimation errors at later states propagate backward through TD updates and compound over time — this is the "deadly triad" (off-policy learning + bootstrapping + function approximation) made concrete.

**Why the obvious fix (n-step returns) doesn't fully work.** The standard mitigation for compounding TD error is to use multi-step or λ-returns: replace the 1-step backup with a longer segment of *observed* rewards before bootstrapping. But in an off-policy setting this creates a new problem — an n-step target for `Q(s_t, a_t)` necessarily incorporates the rewards generated by whatever actions `a_{t+1}, ..., a_{t+n-1}` actually followed in the replayed trajectory. Even if `a_t` was a great decision, the n-step backup becomes pessimistic if the logged continuation was low-quality, because it implicitly assumes the agent keeps following the old behavior policy for n more steps rather than switching immediately to the policy being learned. Empirically, the paper shows n-step TD's success rate *degrades* as n grows on the hardest sparse-reward task in OGBench (`humanoidmaze-giant`), exactly because longer n means more contamination from off-policy continuations.

**The n-step-inequality mechanism.** LQL's fix is not a new target — it's a constraint. It builds on the "optimality tightening" observation (He et al., 2016) that *any realized action sequence lower-bounds what the optimal policy could have achieved* from the same starting state: acting optimally immediately can never be worse in expectation than committing to the observed actions for a while and only then switching to optimal behavior. Formally, for any `i < j` along a sampled trajectory:

```
Q*(s_i, a_i) ≥ E[ G_{i:j} + γ^(j-i) Q*(s_j, a_j) ]
```

LQL uses two practical variants of this inequality that substitute an optimal/bootstrapped action on one side to match what's already computed in a normal TD update — lower-bound (LB) and upper-bound (UB) constraints. When `j = i+1`, the inequality collapses to the ordinary Bellman optimality equation, so LQL's penalty is a strict generalization of standard TD, not a separate objective. Constraint violations are penalized with a **hinge loss** (Lagrangian-style relaxation) rather than enforced as hard constraints:
- **LB penalty**: pushes `Q(s_k, a_k)` *up* when it falls below the return realized by rolling the trajectory forward and then bootstrapping with an (approximately) optimal action at a later state.
- **UB penalty**: pushes `Q(s_k, a_k)` *down* when combining it with preceding observed rewards would imply outperforming an optimal action taken earlier than it should.

The key engineering property: **LQL computes these hinge penalties entirely from network outputs already produced for the ordinary TD error** (`Q(s_t, a_t)` and the bootstrapped value at later states) — no auxiliary networks, no extra forward passes versus vanilla Q-learning. This contrasts with the original optimality-tightening method (He et al., 2016), which needs at least 2x (and up to 4x) additional Q evaluations per update, eating into wall-clock efficiency.

**Results.** Combined with multiple state-of-the-art value-based methods across online and offline-to-online benchmarks (OGBench locomotion/manipulation, RoboMimic) and four policy-extraction families (Gaussian/RLPD-style actor, Best-of-N sampling, Flow Q-Learning, and action-chunked QC-FQL), LQL consistently outperforms both 1-step TD and length-matched n-step TD (n=8) at similar runtime cost. On `humanoidmaze-giant` — the longest-horizon task in OGBench, requiring thousands of environment steps per episode under a single sparse terminal reward — the gap is stark: **1-step TD solves zero of the five tasks (0% success)**, n-step TD partially helps but plateaus at n=4 (38.4% averaged) and *degrades* further out (n=64 achieves only 6.1%), while **LQL with trajectory length L=64 reaches 75.7%**, saturating two of the five tasks at 97.3% and 98.7%. The overhead is small — LQL costs 4.7% more per update on average than TD across policy classes, and this overhead doesn't scale with network size, so it shrinks further as critics/actors grow.

**Trajectory length as a scaling axis, where TD batch-size scaling fails.** Increasing TD's batch size is known to *not* reliably improve (and often hurts) value-based RL performance, hypothesized to reflect overfitting to inaccurate targets (Fu et al., 2025). LQL's trajectory length L offers an alternative scaling knob that doesn't share this pathology: sweeping L (1→16 or 1→32, holding trajectories-per-batch fixed so larger L means strictly more compute) monotonically improves final success rate across all three tested task/actor combinations, with the paper noting "substantial headroom" remains beyond L=64.

**Ablation isolating the hinge penalties from trajectory sampling.** LQL changes two things relative to 1-step TD simultaneously: it samples short trajectories instead of independent transitions, and it adds the hinge penalties. Zeroing out λ_LB = λ_UB = 0 (keeping trajectory-sampled minibatches but removing the constraints) substantially reduces performance versus full LQL, confirming the hinge penalties — not just trajectory-batched sampling — drive the gain (trajectory sampling alone sometimes helps a little, but nowhere near as much).

**Q-value divergence evidence.** On `humanoidmaze-giant`, where rewards are in {−1, 0} so Q* must stay non-positive everywhere, **14 of 15 1-step-TD training runs blow up**, with the average Q crossing zero and growing past magnitudes of 900. Under LQL, Q-values stay within the analytically valid (non-positive) range across all tasks and seeds — direct evidence that the upper-bound hinge penalty is preventing the runaway overestimation that destabilizes plain TD. The paper also reports LQL matching or beating 1-step TD across a swept range of action-noise levels (robustness to environment stochasticity, which in principle introduces bias into the inequality since it's stated in expectation), and shows LQL outperforming the original optimality-tightening method (He et al., 2016) directly in a controlled comparison using OT's public codebase.

**Stated limitation.** The authors flag one explicit theoretical caveat: the optimality inequality holds in expectation over dynamics, so the single-sampled-trajectory approximation introduces bias under *stochastic* transitions (not under deterministic ones, where it's unbiased). They derive a horizon-independent bound on this bias (tightening as behavior-data suboptimality grows) but acknowledge it as the main limitation of the method, alongside leaving the n-step-TD-plus-hinge-penalty combination to future work.

**Relation to other value-based methods on this page.** SAC's twin-critic trick addresses *overestimation bias* from the max operator; LQL addresses a different problem — *backward error propagation/compounding* over long horizons — and is agnostic to the policy-extraction mechanism, so it's complementary to (not a replacement for) tricks like twin-Q or target networks. Unlike MuZero/Dreamer's model-based approach to long-horizon credit assignment (learn a world model, plan inside it), LQL stays fully model-free and off-policy, fixing long-horizon TD learning from within Q-learning itself.

## Related Topics

- [[rl-fundamentals]] — Q-learning, Bellman equations, TD learning basics
- [[policy-gradient-actor-critic]] — the policy-gradient family that LQL's value-based approach contrasts with
- [[rl-training-systems]], [[agentic-rl]] — long-horizon credit assignment in LLM-agent RL settings (not covered here per scope)
- [[deep-learning-fundamentals]] — self-supervised representation learning used by EfficientZero

## Sources

- [Long-Horizon Q-Learning: Accurate Value Learning via n-Step Inequalities (2605.05812)](../../papers/05-learning/reinforcement-learning/Long-Horizon Q-Learning: Accurate Value Learning via n-Step Inequalities - 2605.05812.pdf) — Shi & Finn, Stanford
- [Mastering Atari Games with Limited Data / EfficientZero (2111.00210)](../../papers/05-learning/reinforcement-learning/Mastering Atari Games with Limited Data - 2111.00210.pdf) — temporal consistency, value-prefix prediction, model-based off-policy correction, and Atari/DMControl 100k results.
