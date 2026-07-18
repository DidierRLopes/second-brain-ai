# Online Learning and Dynamic Regret

Online learning studies sequential decisions made before the current loss function is known. In a changing environment, comparison with one fixed decision can hide important failures, so dynamic and strongly adaptive regret compare the learner against changing local or per-round optima.

## Static, Strongly Adaptive, and Dynamic Regret

At round `t`, a learner selects `w_t`, observes a loss function `f_t`, and incurs `f_t(w_t)`. Static regret compares cumulative loss with the best single `w` in hindsight. Strongly adaptive regret instead asks for low static regret on **every interval** of a given length, allowing the best comparator to change from interval to interval. Dynamic regret compares directly against a sequence `u_1, ..., u_T`:

```text
D-Regret(u_1:T) = sum_t f_t(w_t) - sum_t f_t(u_t)
```

Without a regularity assumption, sublinear dynamic regret is impossible. [Zhang et al. (1701.07570)](../../papers/05-learning/online-learning/Dynamic Regret of Strongly Adaptive Methods - 1701.07570.pdf) use the functional variation

```text
V_T = sum_{t=2}^T max_w |f_t(w) - f_{t-1}(w)|
```

to quantify how non-stationary the loss sequence is.

## Strong Adaptivity as a Route to Dynamic Regret

The paper's central result upper-bounds dynamic regret using strongly adaptive regret plus `V_T`. This makes a strongly adaptive method automatically useful for dynamic environments, without needing an advance estimate of how much the environment will drift. That is operationally important because earlier restarting procedures required an upper bound on `V_T` before training.

The derived rates are concrete:

- For convex losses, a strongly adaptive meta-algorithm obtains `O(T^(2/3) V_T^(1/3) log^(1/3) T)`, matching the minimax dependence up to logarithmic factors.
- For exp-concave losses, the proposed method obtains `O(d sqrt(T V_T) log T)`; the paper identifies this as the first dynamic-regret result exploiting exp-concavity.
- For strongly convex losses, the dimension-free construction obtains `O(sqrt(T V_T log T))`, again minimax-optimal up to logarithmic factors.

The key lesson is that local robustness over every interval is not merely another evaluation metric. It is a reusable algorithmic property that converts into guarantees against a drifting comparator.

## Related Topics

- [[rl-fundamentals]] — sequential decision-making and regret are related but distinct frameworks
- [[optimizers]] — online gradient and Newton methods are the subroutines used by adaptive-regret algorithms
- [[scaling-laws]] — a contrasting use of asymptotic laws to reason about learning behavior

## Sources

- [Dynamic Regret of Strongly Adaptive Methods (1701.07570)](../../papers/05-learning/online-learning/Dynamic Regret of Strongly Adaptive Methods - 1701.07570.pdf) — connection between strongly adaptive and dynamic regret, functional variation, and convex/exp-concave/strongly-convex bounds.
