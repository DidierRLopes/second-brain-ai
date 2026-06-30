# Introduction to Policy Gradient for LMs

**Source:** https://ivison.id.au/2026/02/09/policy-gradient.html
**Author:** Hamish Ivison (NLP/RL researcher; RLHF/post-training, formerly Allen AI; PhD work at UW)
**Published:** February 9, 2026

Lightly adapted notes from a "RL101" lecture given for research group meetings, adapted from material Ivison took in an RL course at the University of Washington. The notes are scoped specifically to policy gradient for language models, going as far as CISPO but stopping short of PPO.

## MDP-to-LM Terminology Mapping

The post sets up the standard MDP tuple $\mathcal{M} = (\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0)$ and gives an explicit dictionary mapping each MDP concept to its LM analogue:

| MDP term | MDP notation | LM analogue | LM notation |
|---|---|---|---|
| State | $s\in\mathcal{S}$ | prompt + prefix tokens | $s_t \equiv x_{0:t}$ |
| Action | $a\in\mathcal{A}$ | next token | $a_t \equiv x_{t+1}$ |
| Transition | $P(s'\mid s,a)$ | append token (usually deterministic) | $s_{t+1}=(s_t,a_t)$ |
| Reward | $r(s,a)$ | terminal score / reward model | $r_T$ (often at EOS) |
| Discount | $\gamma\in[0,1)$ | (optional) token-position discount | $\gamma$ |
| Start state dist. | $\rho_0$ | prompt distribution | $p(\text{prompt})$ |
| Policy | $\pi_\theta(a\mid s)$ | LM next-token distribution | $p_\theta(x_{t+1}\mid x_{0:t})$ |
| Trajectory | $\tau=(s_0,a_0,\dots,s_T)$ | completion / rollout | $\tau \equiv x_{0:T}$ |
| Return | $R(\tau)=\sum_t \gamma^t r(s_t,a_t)$ | scalar score for completion | $R(x_{0:T})$ |

The policy is identified with the autoregressive next-token distribution: $\pi_\theta(a_t\mid s_t) \equiv p_\theta(x_{t+1}\mid x_{0:t})$, so the full trajectory probability factors as
$$\pi_\theta(\tau) \equiv p_\theta(x_{0:T}) = p(x_0)\prod_{t=0}^{T-1} p_\theta(x_{t+1}\mid x_{0:t}),$$
with $p(x_0)$ the prompt distribution (treated as fixed/conditioned on). Ivison explicitly flags that discounting is "optional" in this LM setting because most LM RL work sets $\gamma=1$ with a single terminal reward — discounting only resurfaces when dense, per-step reward is available.

## Deriving the Policy Gradient

Starting point: differentiable parameterized policy $\pi_\theta$, updated by gradient ascent $\theta \leftarrow \theta + \alpha\,\widehat{\nabla_\theta J(\theta)}$.

**Log-derivative trick** (stated as the key initial identity):
$$\nabla_\theta \log \pi_\theta(\tau) = \frac{\nabla_\theta \pi_\theta(\tau)}{\pi_\theta(\tau)} \implies \nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau)\,\nabla_\theta \log \pi_\theta(\tau)$$

**Objective:** $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)] = \int \pi_\theta(\tau)R(\tau)\,d\tau$.

**Derivative**, applying the log-deriv trick under the integral:
$$\nabla_\theta J(\theta) = \int \nabla_\theta\pi_\theta(\tau)R(\tau)\,d\tau = \int \pi_\theta(\tau)\,\nabla_\theta\log\pi_\theta(\tau)\,R(\tau)\,d\tau$$

**Expanding trajectory log-probability** into start-state, policy, and transition terms:
$$\log\pi_\theta(\tau) = \log\rho_0(s_0) + \sum_{t=0}^{T-1}\log\pi_\theta(a_t\mid s_t) + \sum_{t=0}^{T-1}\log P(s_{t+1}\mid s_t,a_t)$$
Taking $\nabla_\theta$, both the start-state and transition-dynamics terms vanish (they don't depend on $\theta$), leaving:
$$\nabla_\theta\log\pi_\theta(\tau) = \sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)$$
Ivison calls out explicitly that "the transition dynamics disappear completely! Although, of course, they still are affecting the trajectory distribution implicitly" — i.e., the policy gradient is model-free with respect to $P$, even though $P$ shapes which trajectories get sampled.

Substituting back gives the basic policy gradient:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,R(\tau)\right]$$

## Reward-to-Go and the Causality Argument

Defines reward-to-go $G_t \triangleq \sum_{k=t}^{T-1}\gamma^{k-t}r(s_k,a_k)$, then splits the full return into a past part and future part: $R(\tau) = R_{<t} + G_t$ where $R_{<t} \triangleq \sum_{k=0}^{t-1}\gamma^k r(s_k,a_k)$.

The proof that the past-reward term drops out: $\mathbb{E}[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,R_{<t}] = 0$ because, conditioned on $s_t$, $R_{<t}$ is a valid baseline (this is presented as a preview/special case of the general baselining argument done later). This yields the REINFORCE loss:
$$\mathcal{L}_{\text{REINFORCE}}(\theta) \triangleq -\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\log\pi_\theta(a_t\mid s_t)\,G_t\right]$$

**Algorithm as stated:**
1. Initialize $\theta$.
2. Repeat: sample $N$ trajectories from $\pi_\theta$; compute rewards and reward-to-go $G_t^{(i)}$ for each; form the Monte Carlo estimate $\widehat{\nabla_\theta J(\theta)} = \frac{1}{N}\sum_{i=1}^N\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)})\,G_t^{(i)}$; update $\theta \leftarrow \theta + \alpha\,\widehat{\nabla_\theta J(\theta)}$.

**Three named issues with REINFORCE:**
1. High variance — reward values can have arbitrary scale (e.g., a reward that's always negative breaks intuition); needs normalization.
2. Unknown step size — no good prior for learning rate, and it may need to change over training unlike standard supervised ML.
3. Data reuse — can't reuse old rollouts since the gradient is derived under the assumption $\tau\sim\pi_\theta$ for the *current* $\theta$.

Ivison notes step size is "less of a big issue for things like CISPO" but motivated the historical development of TRPO/PPO; the post explicitly limits its own scope to variance and data-reuse, not step size.

## Baselining

General baseline subtraction (function of state only, not action):
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,(G_t - b(s_t))\right]$$

**Unbiasedness proof** (full chain given): splits the expectation over $\tau$ into nested expectation over $s_t$ then $a_t\mid s_t$, pulls $b(s_t)$ out of the inner expectation, and uses $\sum_a \pi_\theta(a\mid s_t)\nabla_\theta\log\pi_\theta(a\mid s_t) = \sum_a \nabla_\theta\pi_\theta(a\mid s_t) = \nabla_\theta\sum_a\pi_\theta(a\mid s_t) = \nabla_\theta 1 = 0$. Key remark: this only works because the baseline depends on $s_t$ but not on the current action $a_t$ — "if the current action was included, then we couldn't move the baseline out of the expectation."

**Variance reduction (heuristic, not rigorous):** Ivison is explicit that he "couldn't find a clear explanation" of the optimal-baseline derivation in general and instead gives an approximate argument, citing Daniel Seita's blog post. The approximation: (i) drop cross-covariance between timesteps, (ii) treat the score term $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ and the centered return $(G_t-b(s_t))$ as approximately independent, giving
$$\mathrm{Var}(\cdot) \approx \sum_{t=0}^{T-1}\mathbb{E}[(\nabla_\theta\log\pi_\theta(a_t\mid s_t))^2]\,\mathbb{E}[(G_t-b(s_t))^2]$$
Since the score term's variance can't be controlled, the best lever is minimizing $\mathbb{E}[(G_t - b(s_t))^2]$ — i.e., the best baseline is the best predictor of $G_t$ given $s_t$, **which is exactly the value function $\hat V_{\pi_\theta}(s_t)$.** This motivates the advantage function:
$$A_{\pi_\theta}(s_t,a_t) = G_t - \hat V_{\pi_\theta}(s_t)$$
Framed intuitively as replacing "weighting by absolute reward" with "weighting by the improvement of the action over the average action taken by the policy."

## Re-using Old Data via Importance Weighting

Since the REINFORCE gradient assumes $\tau\sim\pi_\theta$, after one gradient step old rollouts come from $\pi_{\theta_{old}}$, a different distribution. Importance weighting fix:
$$\mathbb{E}_{\tau\sim\pi_\theta}[f(\tau)] = \int \pi_{\theta_{old}}(\tau)\frac{\pi_\theta(\tau)}{\pi_{\theta_{old}}(\tau)}f(\tau)\,d\tau = \mathbb{E}_{\tau\sim\pi_{\theta_{old}}}[w(\tau)f(\tau)]$$
where the trajectory-level importance weight factors into a per-token product:
$$w(\tau) \triangleq \frac{\pi_\theta(\tau)}{\pi_{\theta_{old}}(\tau)} = \prod_{t=0}^{T-1}\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$$
The token-level ratio is defined separately as $r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$ — this is the quantity that later appears (stop-gradiented) inside CISPO, and is the same object PPO clips.

## CISPO

Presented as "basically already have enough to get the current SOTA RL algorithm, CISPO" (citing MiniMax et al. 2025, MiniMax-M1, arXiv:2501.08313) — Ivison's own assessment is that CISPO is "pretty much one of the better algorithms out there right now." Loss:
$$J_{\mathrm{CISPO}}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\,\{o^i\}_{i=1}^G\sim\pi_{\theta_{old}}(\cdot\mid q)}\left[\frac{1}{\sum_{i=1}^G |o^i|}\sum_{i=1}^G\sum_{t=1}^{|o^i|}\mathrm{sg}(r_{i,t}(\theta))\,\hat A_{i,t}\,\log\pi_\theta(o^i_t\mid q,o^i_{<t})\right]$$
- Averages over both the group index $i$ (rollouts sharing a prompt) and timestep $t$.
- $\mathrm{sg}$ = "stop gradient," applied to the importance ratio $r_{i,t}(\theta)$ so gradients don't flow back through the logprobs used to *compute* the ratio (only through the $\log\pi_\theta$ term).
- Group-relative advantage (same form as GRPO):
$$\hat A_{i,t} = \frac{R_i - \mathrm{mean}(\{R_j\}_{j=1}^G)}{\mathrm{std}(\{R_j\}_{j=1}^G)}$$
- Ivison's framing: this is the same baselining idea as before, but the group's empirical mean/std *substitutes for* a learned value function — no critic needed.

## Other Advantage Estimators (survey list)

1. **VinePPO** (Kazemnejad et al. 2025, arXiv:2410.01679) — token-level Monte Carlo rollouts to estimate value directly; "incredibly expensive but arguably the best way to do it."
2. **GIGPO** (Feng et al. 2025, "Grouping in Group Policy Optimization for LLM," arXiv:2504.02763) — ties together similar states via token similarity; works well for agentic state-based tasks (e.g., web-browser navigation) where the underlying state is simple and accessible.
3. **REINFORCE++** (Hu et al. 2025, arXiv:2501.03262) — uses the average reward across the batch as the value estimate; "makes sense if your tasks are all similar."
4. **RLOO** (Ahmadian et al. 2024, "Back to Basics," arXiv:2402.14740) — described as "basically GRPO, but remove the standard deviation and use a leave-one-out approach instead of doing average over all samples."

## Final Note: Why Using the Trajectory's Own Reward in the Baseline Is Valid

Addresses an apparent paradox in GRPO/REINFORCE++-style baselines: doesn't averaging in the current sample's own reward violate the "baseline can't depend on the current action" rule? Worked resolution using leave-one-out decomposition. Let $b(s_t) \equiv \frac{1}{B}\sum_{n=1}^B G_t^n$ (batch-average reward-to-go) and let $x$ index the current sample:
$$G_t^x - b(s_t) = G_t^x - \frac{1}{B}\sum_{n=1}^B G_t^n = \frac{B-1}{B}G_t^x - \frac{1}{B}\sum_{n=1,n\ne x}^{B}G_t^n$$
The leave-one-out sum over $n\ne x$ is a valid baseline (independent of the current sample). The remaining $\frac{B-1}{B}G_t^x$ term just rescales the gradient by a constant factor $\frac{B-1}{B}$, which does not bias the estimator (only changes effective step size). Conclusion: using the *reward itself* in a batch-mean baseline is a special case that happens to be fine, but "strictly, using a leave-one-out estimate as in RLOO is better to avoid this scaling" — explaining why RLOO's leave-one-out is technically preferable to GRPO's plain batch mean. Ivison stresses this trick is special: "other functions involving $a_t$ would be invalid to use" as a baseline.

## Cited References

- Weng, L. (2018). [Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- Sutton & Barto (1998), *Reinforcement Learning: An Introduction*, Ch. 13
- Seita, D. (2017). [Going Deeper Into Reinforcement Learning: Fundamentals of Policy Gradients](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/) — source for the heuristic variance-reduction argument
- MiniMax et al. (2025). MiniMax-M1: Scaling Test-Time Compute, arXiv:2501.08313 — CISPO
- Kazemnejad et al. (2025). VinePPO: Refining Credit Assignment, arXiv:2410.01679
- Feng et al. (2025). Grouping in Group Policy Optimization for LLM, arXiv:2504.02763 — GIGPO
- Hu et al. (2025). REINFORCE++: Stabilizing Critic-Free Policy, arXiv:2501.03262
- Ahmadian et al. (2024). Back to Basics: Revisiting REINFORCE Style Optimization, arXiv:2402.14740 — RLOO

## Note on Scope

The post explicitly stops short of deriving TRPO/PPO's clipped surrogate objective in detail (despite introducing the token-level importance ratio $r_t(\theta)$ that PPO clips) — it stops at CISPO, which sidesteps the clipping/step-size problem differently (stop-gradient on the ratio rather than clipping it).
