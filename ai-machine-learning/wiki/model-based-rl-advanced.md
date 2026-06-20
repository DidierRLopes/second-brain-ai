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
