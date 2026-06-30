# ML Theory and Statistics

The mathematical foundations underlying all of machine learning. Probability, information theory, and the key theoretical results that explain why ML works (and when it doesn't).

---

## Probability Fundamentals

### PDF and PMF

**Probability Mass Function (PMF)**: for discrete random variables. P(X = x) gives the probability of each exact value.

```
Σ_x P(X = x) = 1
P(X = x) ≥ 0 for all x
```

Examples: coin flips (Bernoulli), dice rolls (Uniform discrete), word counts (Poisson).

**Probability Density Function (PDF)**: for continuous random variables. f(x) is not a probability; it's a density. Probability of interval: P(a ≤ X ≤ b) = ∫_a^b f(x) dx.

```
∫_{-∞}^{∞} f(x) dx = 1
f(x) ≥ 0 for all x
```

Common PDFs: Normal N(μ, σ²), Uniform U(a,b), Exponential Exp(λ), Beta Beta(α,β).

**CDF (Cumulative Distribution Function)**: F(x) = P(X ≤ x). For continuous: F(x) = ∫_{-∞}^x f(t) dt. For discrete: F(x) = Σ_{t≤x} P(X=t).

### Expectation

The **expected value** (mean) of a random variable:

```
E[X] = Σ_x x · P(X=x)          (discrete)
E[X] = ∫ x · f(x) dx            (continuous)
```

**Key properties:**
- Linearity: E[aX + bY] = aE[X] + bE[Y] (no independence required)
- E[g(X)] = ∫ g(x) f(x) dx (LOTUS — Law of the Unconscious Statistician)
- E[XY] = E[X]E[Y] only if X, Y independent

**Variance** = E[(X - E[X])²] = E[X²] - (E[X])². Measures spread around the mean.

**Standard Deviation** = √Var(X). Same units as X.

---

## Common Distributions

A cheat-sheet for routing a problem to the right distribution: memoryless waiting time → geometric/exponential; trials with success/failure → Bernoulli/binomial; large-sample average → CLT (approximately Gaussian); bound from mean/variance alone → Markov/Chebyshev; estimating a parameter from data → MLE/MAP/Bayesian update. Distributions split along two axes: discrete vs. continuous, and counting vs. waiting — binomial counts successes in n trials, Poisson counts events in a fixed window; geometric counts trials until first success, exponential measures continuous waiting time until the first event.

### Discrete

**Bernoulli(p)**: single trial, P(X=1)=p. E[X]=p, Var(X)=p(1-p). Handy trick — **indicator squaring**: for binary X∈{0,1}, X²=X, so E[X²]=E[X]. A **categorical** RV generalises Bernoulli from 2 to k outcomes.

**Binomial(n,p)**: sum of n iid Bernoulli(p) trials, X=ΣXᵢ. P(X=k) = C(n,k)pᵏ(1-p)ⁿ⁻ᵏ. E[X]=np, Var(X)=np(1-p) (both follow immediately from linearity/independence of the sum). Sanity check via the binomial theorem: Σₖ C(n,k)pᵏ(1-p)ⁿ⁻ᵏ = (p+(1-p))ⁿ = 1.

**Poisson(λ)**: count of events in a fixed interval at average rate λ. P(X=k) = λᵏe⁻λ/k!. E[X]=Var(X)=λ. Derivable as the **limit of Binomial(n,p) as n→∞ with np=λ fixed** — split the interval into n tiny sub-intervals, each ≈Bernoulli(λ/n). Sums of independent Poissons are Poisson with rates adding: X+Y ~ Poisson(λ₁+λ₂).

**Geometric(p)**: number of Bernoulli(p) trials until the first success (inclusive). P(X=k)=(1-p)^(k-1)p, E[X]=1/p, Var(X)=(1-p)/p². **Memoryless**: P(X>m+n | X>m) = P(X>n) — failing m times already tells you nothing about the future. The geometric is the *only* memoryless distribution on the positive integers. E[X]=1/p has a clean **first-step-analysis** derivation: on the first trial, succeed w.p. p (X=1) or fail w.p. (1-p) and restart (X = 1+E[X]); solving E[X] = p·1 + (1-p)(1+E[X]) gives E[X]=1/p.

### Continuous

**Uniform(a,b)**: E[X]=(a+b)/2, Var(X)=(b-a)²/12. The n+1 gaps between n uniform points on an interval are identically distributed — a fact used repeatedly in order-statistics puzzles below.

**Exponential(λ)**: waiting time to the next event in a Poisson(λ) process. f(x)=λe^(-λx), P(X>x)=e^(-λx). E[X]=1/λ, Var(X)=1/λ². **Memoryless** (the continuous analogue of geometric, and the only continuous memoryless distribution on [0,∞)): P(X>s+t | X>s) = P(X>t). **Poisson↔exponential duality**: events/unit-time ~ Poisson(λ) ⟺ inter-event time ~ Exp(λ). **Minimum of independent exponentials**: min(X₁,X₂) ~ Exp(λ₁+λ₂) for X₁~Exp(λ₁), X₂~Exp(λ₂) independent — proof via the tail function: P(min>t) = P(X₁>t)P(X₂>t) = e^(-(λ₁+λ₂)t).

**Gaussian N(μ,σ²)**: standardise via Z=(X-μ)/σ ~ N(0,1). The α-percentile of N(μ,σ²) is x_α = μ+σ·z_α (z_α being the standard-normal percentile, symmetric: z_{1-α}=-z_α). **Sum of independent Gaussians is Gaussian**: N(μ₁,σ₁²)+N(μ₂,σ₂²) = N(μ₁+μ₂, σ₁²+σ₂²) — the closure property that makes CLT-style aggregation well-behaved.

---

## Variance and Covariance

**Variance**: Var(X) = E[(X - μ)²] = E[X²] - μ²

**Covariance**: measures linear dependence between two variables.

```
Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
```

- Cov > 0: X and Y tend to increase together
- Cov < 0: when X increases, Y tends to decrease
- Cov = 0: no linear relationship (but could still be nonlinearly dependent)

**Pearson Correlation**: normalises covariance to [-1, 1]:

```
ρ(X, Y) = Cov(X, Y) / (σ_X · σ_Y)
```

**Covariance Matrix** for a random vector X ∈ Rⁿ:

```
Σ = E[(X - μ)(X - μ)ᵀ]
Σ_{ij} = Cov(X_i, X_j)
```

Always positive semi-definite (PSD). The eigenvectors give principal components; eigenvalues give variance along each component.

---

## Entropy

**Shannon Entropy** measures the average uncertainty (information content) of a distribution:

```
H(p) = -Σ_x p(x) log p(x)        (discrete)
h(p) = -∫ p(x) log p(x) dx        (differential entropy, continuous)
```

- H(p) ≥ 0 for discrete distributions
- Maximised by the uniform distribution (maximum uncertainty)
- Minimised (= 0) by a deterministic distribution (no uncertainty)
- In bits: use log₂. In nats: use log_e

**Example**: fair coin has H = -2×(0.5 log 0.5) = 1 bit. A 99%-heads coin has H ≈ 0.08 bits.

**Alternate, logits-based form**: given logits `x_i` (pre-softmax scores) for a softmax distribution `p`, entropy can be written without ever materialising `p` explicitly:

```
H(p) = log Σ_i e^{x_i}  -  (Σ_i e^{x_i}·x_i) / (Σ_i e^{x_i})
```

The second term is `E[x]` under the softmax distribution itself — so this form expresses entropy as "log-sum-exp of the logits, minus the softmax-weighted average logit."

**Cross-Entropy**: expected information content of distribution p under model q:

```
H(p, q) = -Σ_x p(x) log q(x)
```

This is the loss minimised in classification (cross-entropy loss). It equals H(p) + KL(p||q) — minimising cross-entropy is equivalent to minimising KL divergence when the target distribution p is fixed.

**Proof of CE(p,q) = KL(p‖q) + H(p)**: expand KL and split the sum:

```
KL(p‖q) = Σ_x p(x)(log p(x) - log q(x))
        = Σ_x p(x) log p(x)  -  Σ_x p(x) log q(x)
        = -H(p)  +  CE(p,q)
```

Rearranging gives `CE(p,q) = KL(p‖q) + H(p)` directly. Since `H(p)` doesn't depend on `q`, gradients of `CE(p,q)` and `KL(p‖q)` with respect to `q`'s parameters are identical — which is the formal reason minimising cross-entropy loss against a fixed target distribution is exactly minimising KL divergence to it.

---

## KL Divergence

**Kullback-Leibler Divergence** measures how much one distribution P differs from another Q:

```
KL(P || Q) = Σ_x P(x) log (P(x) / Q(x))    (discrete)
KL(P || Q) = ∫ P(x) log (P(x) / Q(x)) dx    (continuous)
```

**Properties:**
- KL(P||Q) ≥ 0 always (Gibbs' inequality / Jensen)
- KL(P||Q) = 0 iff P = Q almost everywhere
- **Not symmetric**: KL(P||Q) ≠ KL(Q||P)
- Not a metric (triangle inequality doesn't hold)

**Forward KL KL(P||Q)** — where P is the target: minimise Q to cover P. If P has probability everywhere Q has low probability, KL explodes. Q is forced to be "mean-seeking" — doesn't miss any mode of P.

**Reverse KL KL(Q||P)** — where P is the target: minimise Q to be covered by P. Q can be 0 wherever P is small. Results in "mode-seeking" — Q concentrates on a single mode of P. Used in variational inference.

**In ML applications:**
- VAE: KL(q_φ(z|x) || p(z)) — reverse KL (q is the approximate posterior, p is the prior)
- RLHF: KL(π || π_ref) penalty — keeps trained policy close to reference
- Knowledge distillation: KL(p_teacher || p_student)

### Jensen-Shannon Divergence (JSD)

JSD is a **symmetric** and **bounded** version of KL divergence:

```
JSD(P || Q) = ½ KL(P || M) + ½ KL(Q || M)
where M = ½(P + Q)     (midpoint distribution)
```

**Properties:**
- Symmetric: JSD(P||Q) = JSD(Q||P)
- Bounded: 0 ≤ JSD(P||Q) ≤ log 2 (in nats) or 1 (in bits)
- √JSD is a valid metric (triangle inequality holds)

JSD is the loss that GANs implicitly minimise in the original formulation. The discriminator at Nash equilibrium achieves D(x) = p_data(x) / (p_data(x) + p_gen(x)), and the generator's loss equals 2 JSD(p_data || p_gen) - log 4.

---

## Bayes' Theorem

```
P(H | E) = P(E | H) P(H) / P(E)
```

- P(H): **prior** — belief about hypothesis before observing evidence
- P(E|H): **likelihood** — probability of evidence given hypothesis
- P(H|E): **posterior** — updated belief after observing evidence
- P(E): **marginal likelihood / evidence** — normalising constant = Σ_H P(E|H)P(H)

**Example**: Medical test. P(disease) = 0.01 (prior). Test sensitivity P(+|disease) = 0.99. P(+|no disease) = 0.05. What is P(disease|+)?

```
P(disease|+) = 0.99 × 0.01 / (0.99×0.01 + 0.05×0.99) ≈ 16.7%
```

Despite a positive test, only 16.7% chance of disease due to low base rate. This is why most ML "positive" predictions on rare events are false positives.

---

## MLE vs MAP

Both are methods for estimating model parameters θ from data D.

### Maximum Likelihood Estimation (MLE)

Find θ that maximises the probability of observing the data:

```
θ_MLE = argmax_θ P(D | θ) = argmax_θ Σᵢ log P(xᵢ | θ)
```

MLE treats parameters as fixed unknowns. No prior over θ.

**Linear regression**: MLE with Gaussian noise → minimise MSE. MLE with Bernoulli → minimise cross-entropy for logistic regression. MLE for maximum likelihood is the default criterion.

**Mechanics, worked through**: given iid data x₁,...,xₙ from a distribution with unknown parameter θ, the likelihood is `L(θ) = Πᵢ p(xᵢ;θ)`, and `θ_MLE = argmax_θ L(θ)`. The **log-likelihood trick** turns the product into a sum (monotonic transform, same argmax): `ℓ(θ) = log L(θ) = Σᵢ log p(xᵢ;θ)`; differentiate w.r.t. θ, set to zero, and solve (checking it's a max via the second derivative or endpoint behaviour). Set up for x₁,...,xₙ iid Bernoulli(p): `ℓ(p) = Σᵢ [xᵢ log p + (1-xᵢ) log(1-p)]` — differentiating this and solving for p̂ recovers the MLE (the well-known result is that the sample mean is the MLE for a Bernoulli's success probability, though the source notes don't carry the differentiation through to that last algebraic step).

### Maximum A Posteriori (MAP)

Use Bayes' theorem; maximise the posterior:

```
θ_MAP = argmax_θ P(θ | D) = argmax_θ [log P(D | θ) + log P(θ)]
```

The log prior log P(θ) acts as a **regulariser**:
- Gaussian prior N(0, σ²) → L2 regularisation (Ridge)
- Laplace prior Laplace(0, b) → L1 regularisation (Lasso)

MAP is MLE + regularisation. When the prior is uninformative (flat), MAP = MLE.

**Full Bayesian inference**: don't just take argmax; integrate over all θ. Much more expensive but gives uncertainty estimates. Approximations: variational inference, MCMC, Laplace approximation.

---

## Bias-Variance Tradeoff

For a prediction problem, expected test MSE decomposes as:

```
E[(y - ŷ)²] = Bias²(ŷ) + Var(ŷ) + Noise
```

- **Bias**: systematic error — how far is the average prediction from the true answer? (underfitting)
- **Variance**: sensitivity to the training data — how much does the model change with different training sets? (overfitting)
- **Noise**: irreducible error from stochasticity in the data

**The tradeoff:**
- Simple models (linear regression): high bias, low variance
- Complex models (deep trees, large NNs): low bias, high variance

Increasing model complexity reduces bias but increases variance. The optimal complexity minimises total error.

**In deep learning**: large neural networks have high capacity (low bias potential) but also, surprisingly, low variance when trained with SGD + regularisation. The bias-variance decomposition doesn't cleanly apply to overparameterised models — they can interpolate training data yet generalise (double descent phenomenon).

---

## No Free Lunch Theorem

The **NFL theorem** (Wolpert, 1996): **no learning algorithm performs better than any other when averaged over all possible problems**.

More precisely: for any two algorithms A and B, averaged over all possible target functions f, the expected generalisation error of A equals that of B.

**Implication**: inductive biases matter. Every learning algorithm makes assumptions about the problem structure. Choosing the right inductive bias for your data domain is the real design problem.

Examples of inductive biases:
- CNNs: translation equivariance, local structure
- RNNs: sequential order matters
- Transformers: pairwise attention (no spatial structure)
- Linear models: linearity
- Decision trees: axis-aligned boundaries

The NFL theorem is often misunderstood as "all algorithms are equally good." What it actually says: there's no free lunch — if your algorithm does better on some problems, it must do worse on others. You can't have it all without committing to assumptions.

---

## Curse of Dimensionality

As the number of dimensions d grows, many intuitions from low-dimensional space break down.

**Volume of a hypersphere**: in d dimensions, most of the volume is near the surface. A d-dimensional sphere of radius r has most mass in a thin shell near radius r.

**Distance concentration**: in high dimensions, all pairwise distances converge to the same value. If x ~ Uniform([0,1]^d): max_dist/min_dist → 1 as d → ∞. This makes nearest-neighbour meaningless.

**Data sparsity**: to have the same density as 10 points in 1D, you need 10^d points in d dimensions. Exponential data requirements.

**Consequences for ML:**
- KNN becomes useless in high dimensions
- Linear separability: in high dimensions, almost any two sets of points are linearly separable (trivially, not usefully)
- Overfitting is easier — models find spurious patterns in the high-dimensional space
- Kernel methods become less effective

**Solutions**: dimensionality reduction (PCA, autoencoders, t-SNE), feature selection, manifold learning (the data often lies on a low-dimensional manifold embedded in high-dimensional space).

---

## Confidence Intervals

A **95% confidence interval** (CI) for a parameter θ is an interval [L, U] such that, if you repeat the experiment many times, 95% of the constructed intervals would contain the true θ.

**Common misconception**: "there is a 95% probability that θ is in [L, U]." Wrong — θ is fixed, not random. The CI procedure is what's random.

**Normal approximation CI for the mean** (central limit theorem):

```
CI = x̄ ± z_{α/2} · (s/√n)
```

where x̄ is sample mean, s is sample standard deviation, n is sample size, z_{α/2} = 1.96 for 95%.

**In ML evaluation**: confidence intervals on test set performance. With a test set of n examples:

```
Standard error of accuracy = √(p̂(1-p̂)/n)
```

For n=1000, accuracy p̂=0.9: SE ≈ 0.95%. A 95% CI is approximately ±1.9%.

**Bootstrap CI**: resample test set with replacement many times, compute statistic each time. The 2.5th and 97.5th percentiles of the bootstrap distribution give the CI. More robust for complex statistics.

**Implications**: model comparisons need CIs, not just point estimates. A 2% accuracy improvement with n=100 test examples is statistically meaningless. This is why proper ML evaluation uses large test sets and statistical significance tests.

---

## Hypothesis Testing Toolkit

A routing guide for the standard battery of statistical tests — which test applies depends on the data type (continuous vs. categorical) and what's being compared (one sample vs. two, paired vs. independent).

**p-value**: the probability, under the null hypothesis, of observing a result at least as extreme as the one actually observed. A small p-value (conventionally <0.05) is evidence against the null — but a p-value is *not* the probability the null is true, and is not a measure of effect size.

**Kolmogorov-Smirnov (KS) test**: tests whether a sample comes from a reference distribution (one-sample), or whether two samples come from the same distribution (two-sample) — by comparing empirical CDFs. The test statistic is the maximum vertical distance between the two CDFs: `D = sup_x |F_1(x) - F_2(x)|`. Distribution-free (no assumption on the underlying distribution's shape), which makes it a general-purpose goodness-of-fit test.

**Chi-squared test**: for categorical data — tests independence between two categorical variables (contingency table) or goodness-of-fit between observed and expected category counts. Test statistic `χ² = Σ (O_i - E_i)²/E_i` summed over categories, compared against a chi-squared reference distribution.

**T-test**: compares means, assuming (approximately) normally-distributed data.
- **One-sample**: is the sample mean different from a hypothesized value `μ₀`? `t = (x̄ - μ₀)/(s/√n)`.
- **Two-sample (independent)**: do two independent groups have different means? Accounts for the variance of *both* samples.
- **Paired**: do paired measurements (e.g. before/after on the same subjects) differ? Reduces to a one-sample t-test on the per-pair differences, which removes subject-to-subject variance and gives more power than treating the two groups as independent.

**ANOVA / F-test**: generalises the two-sample t-test to **more than two groups** — tests whether at least one group mean differs from the others, via the ratio of between-group variance to within-group variance. A significant F-test says *some* group differs, not which one (post-hoc tests like Tukey's HSD are needed to localise the difference).

**McNemar's test**: for paired *categorical* (typically binary) outcomes — e.g. comparing two classifiers' correct/incorrect predictions on the same test set. Tests whether the disagreement between the two is symmetric, using only the discordant pairs (cases where the two methods disagree).

**Pearson correlation test**: tests whether the Pearson correlation coefficient `ρ(X,Y)` (linear correlation, defined above) is significantly different from zero. Assumes a linear relationship and is sensitive to outliers (since it's built from `E[XY]`, a quantity outliers can dominate).

**Spearman correlation test**: like Pearson, but computed on the **ranks** of the data rather than raw values — tests for a monotonic (not necessarily linear) relationship. More robust to outliers and skewed distributions than Pearson, at the cost of discarding information about the actual magnitude of differences.

**Pearson vs. Spearman, choosing between them**: use Pearson when the relationship is expected to be linear and the data is roughly normal/outlier-free — it's more statistically efficient (lower variance) in that regime. Use Spearman when the relationship is monotonic but not necessarily linear, when there are outliers, or when the data is ordinal rather than continuous.

**Mutual information**: a more general dependence measure than correlation — `I(X;Y) = Σ_{x,y} p(x,y) log(p(x,y)/(p(x)p(y)))`, equivalently `I(X;Y) = H(X) - H(X|Y)` (the reduction in uncertainty about X from observing Y). Unlike Pearson/Spearman correlation, mutual information captures *any* statistical dependence, not just linear or monotonic relationships, and `I(X;Y)=0` iff X and Y are truly independent.

---

## Concentration Inequalities: Markov and Chebyshev

Two classical bounds that require knowing only a distribution's mean (Markov) or mean and variance (Chebyshev) — no further assumptions about its shape.

**Markov's inequality**: for non-negative X and a>0, `P(X≥a) ≤ E[X]/a`. Proof: `E[X] ≥ Σ_{x≥a} x·P(X=x) ≥ a·Σ_{x≥a} P(X=x) = a·P(X≥a)`. Intuitively, E[X] can't be smaller than the contribution of the values ≥ a alone.

**Chebyshev's inequality**: for X with mean μ, variance σ², `P(|X-μ| ≥ kσ) ≤ 1/k²` — at least `1 - 1/k²` of the distribution's mass lies within k standard deviations of the mean (e.g. ≥75% within 2 SDs, regardless of the distribution's shape). Proof: apply Markov to `(X-μ)²`: `P(|X-μ|≥kσ) = P((X-μ)²≥k²σ²) ≤ E[(X-μ)²]/(k²σ²) = 1/k²`. Substituting ε=kσ gives the equivalent form `P(|X-μ|≥ε) ≤ σ²/ε²` — this is exactly the form used to prove the Weak Law of Large Numbers below.

**AM-GM inequality**: arithmetic mean ≥ geometric mean, `(a₁+...+aₙ)/n ≤ ⁿ√(a₁⋯aₙ)` — provable by taking logs and applying Jensen's inequality (log is concave; see Convex Functions below) then exponentiating.

**Cauchy-Schwarz inequality**: `E[XY]² ≤ E[X²]E[Y²]`, equivalently `(Σxᵢyᵢ)² ≤ (Σxᵢ²)(Σyᵢ²)`.

**Union bound**: `P(A₁∪...∪Aₙ) ≤ P(A₁)+...+P(Aₙ)`.

---

## Central Limit Theorem and the Law of Large Numbers

**Central Limit Theorem (CLT)**: the sample mean of n iid random variables Xᵢ with mean μ and variance σ² is approximately Gaussian *no matter the shape of the Xᵢ*: `X̄ₙ = (1/n)ΣXᵢ ≈ N(μ, σ²/n)`. Variance derivation: `Var(X̄ₙ) = (1/n²)Var(ΣXᵢ) = (1/n²)·n·σ² = σ²/n`.

**Standard error** = `σ/√n` — error shrinks like √n, so reducing error 10× requires 100× the samples. This is the basis for the confidence-interval formula above.

**Law of Large Numbers (LLN, weak form)**: `P(|X̄ₙ - μ| > ε) → 0` for any ε>0 as n→∞ — the sample mean converges in probability to the true mean. Proved directly from Chebyshev's inequality (substituting ε for kσ): `P(|X̄ₙ-μ|>ε) ≤ Var(X̄ₙ)/ε² = σ²/(nε²) → 0`. This is a clean illustration of why Chebyshev's inequality matters beyond being a standalone bound: it's the textbook proof technique for LLN.

---

## Markov Chains

A **Markov chain** is a sequence of random variables X₀,X₁,X₂,... on a state space S satisfying the **Markov property**: the future depends only on the present, not the past. For finite state spaces, described by a transition matrix P where `P_ij = P(X_{t+1}=j | X_t=i)` (rows sum to 1).

If `π_t` is the distribution at time t: `π_{t+1} = π_t·P`. A **stationary distribution** satisfies `π = π·P`. Under mild conditions (irreducibility, aperiodicity) the chain has a unique stationary distribution and `π_t → π` regardless of the starting state — the theoretical basis for MCMC sampling.

**Expected return time** to state i is the reciprocal of its stationary probability: `E[return to i] = 1/π_i`.

**Absorption problems**, solved via first-step analysis: expected absorption time `h_j` from state j satisfies `h_A=0` for the target absorbing state A, `h_i = 1 + Σ_j P_ij·h_j` otherwise; absorption probability `q_j` satisfies `q_A=1`, `q_B=0` for other absorbing states B, `q_i = Σ_j P_ij·q_j` otherwise. For chains with symmetry, reparametrising by distance-to-target (rather than absolute state) collapses equivalent states and simplifies the recurrence considerably.

In ML, Markov chains underlie MCMC inference (Gibbs sampling, Metropolis-Hastings), and the Markov property itself is the formal justification for treating RL environments as MDPs (see [[rl-fundamentals]]).

---

## Probability Problem-Solving Toolkit

Recurring techniques for probability problems, useful beyond pure theory (e.g. for reasoning about RL environments, sampling algorithms, and evaluation statistics):

**First-step analysis**: express a quantity (expected time, absorption probability) recursively in terms of itself by conditioning on the first step, then solve the resulting (often linear) system. Worked example: expected flips to get two heads in a row. Let E₀ = expected remaining flips having just seen tails (or at the start), E₁ = expected remaining flips having just seen heads: `E₀ = ½(1+E₀) + ½(1+E₁)`, `E₁ = ½(1+E₀) + ½·0`. Solving gives E₀=6.

**Counting in expectation (indicator trick)**: to count how many of n items satisfy a property, define `Xᵢ = 1[item i has the property]`, `N=ΣXᵢ`, compute `E[Xᵢ]=P(Xᵢ)` per item (often via symmetry), then apply linearity: `E[N]=ΣP(Xᵢ)`. Example: expected number of fixed points in a random permutation of n elements is 1, since by symmetry each position has P(fixed)=1/n and there are n positions.

**Order statistics (max/min of n iid RVs)**: for iid X₁,...,Xₙ with CDF F, the max M has `F_M(x) = F(x)ⁿ`; the min m has `P(m>x) = (1-F(x))ⁿ`. Two routes to E[max]/E[min]: the CDF→PDF→integrate recipe, or — for non-negative RVs — the **tail-sum formula** `E[X] = ∫₀^∞ P(X>x)dx` directly. Worked for Xᵢ~Uniform[0,1]: `E[max] = n/(n+1)`, `E[min] = 1/(n+1)` — intuitively, n uniform points split [0,1] into n+1 symmetric segments of mean length 1/(n+1) each.

**Tail formula for expectation** (non-negative RVs): `E[X] = ∫₀^∞ P(X>x)dx = ∫₀^∞ (1-F(x))dx` — often faster than direct integration of x·f(x), and the standard tool for order-statistics problems above.

**Combinatorics quick-reference**:

| Order matters? | Replacement? | Count |
|---|---|---|
| yes | no | n!/(n-k)! (permutations) |
| no | no | C(n,k) (combinations) |
| yes | yes | n^k |
| no | yes | C(n+k-1, k) (stars and bars) |

**Stars and bars** derives the "combination with replacement" count: choosing k items from n types with replacement equals the number of ways to write k as an ordered sum of n non-negative integers, modeled as placing k stars and n-1 bars.

**A few classical puzzles that reuse these tools:**
- **Coupon collector**: n coupon types, one uniform-random coupon per draw; expected draws to collect all n is `n·H_n` (H_n = nth harmonic number ≈ ln n), derived by summing geometric expectations across n phases.
- **Gambler's ruin** (fair game, p=½): starting at wealth a, target N, ruin at 0 — probability of reaching N is `a/N`. Cleanest proof is a **martingale argument**: E[wealth at stopping time] = a (no drift), and since the stopping wealth is in {0,N}, this equals `q_a·N`, giving `q_a=a/N`.
- **Monty Hall** (n doors, generalised): switching to a remaining unopened door is always at least as good as staying, because the probability mass "released" by each opened door (which the host *knows* doesn't have the prize) redistributes across the remaining unopened, unchosen doors.
- **Reservoir sampling**: stream items of unknown length n, maintain a uniformly random sample online by replacing the kept item with the n-th item with probability 1/n at each step — provable by induction that every item seen so far has equal probability 1/n of being retained.
- **Secretary problem**: reject the first k of n randomly-ordered candidates, then hire the next one better than all seen so far; the success probability is maximised at `k ≈ n/e`, derived by approximating the harmonic sum and differentiating `P(k) ≈ -r·log(r)` where r=k/n.

---

## Convex Functions

A function f: Rⁿ → R is **convex** if for all x, y and t ∈ [0,1]:

```
f(tx + (1-t)y) ≤ t f(x) + (1-t) f(y)
```

Geometrically: the chord between any two points lies above the function.

**Equivalent condition** for twice-differentiable f: the Hessian ∇²f(x) is positive semi-definite everywhere.

**Strictly convex**: ≤ replaced by < (unique global minimum).

**Why convexity matters in ML:**
- Convex optimisation problems have no local minima — gradient descent (or any descent method) converges to the global minimum
- Logistic regression, SVMs, linear regression: all convex — provable convergence
- Neural networks: non-convex — SGD finds good solutions in practice but no convergence guarantees

**Useful convex functions:**
- Any linear function f(x) = wᵀx (both convex and concave)
- f(x) = x² (strictly convex)
- f(x) = -log(x) for x > 0 (strictly convex)
- f(x) = e^x
- Cross-entropy loss (as a function of logits)
- L2 norm, L1 norm

**Jensen's inequality** (fundamental consequence of convexity):

```
f(E[X]) ≤ E[f(X)]    (φ convex)
f(E[X]) ≥ E[f(X)]    (φ concave)
```

Used to derive the ELBO in VAEs, the EM algorithm, and many information-theoretic results. Memory aid for the direction: φ(x)=x² (convex), X∈{-1,+1} equally likely — φ(E[X])=φ(0)=0 but E[φ(X)]=1, so φ(E[X]) ≤ E[φ(X)] checks out.

**More convex functions** (φ''≥0 everywhere): x², x⁴ (even powers), |x|, aˣ for a>1, 1/x for x>0, max(x₁,x₂,...).

**Concave functions** (φ''≤0 everywhere): √x, log x, xᵃ for 0<a<1, min(x₁,x₂,...). Linear functions are both convex and concave. Worth memorising as a Jensen consequence: `log E[X] ≥ E[log X]`.

**AM-GM via Jensen**: taking logs of the AM-GM inequality above and applying Jensen's to the concave log function (then exponentiating) gives one of the cleanest proofs of arithmetic-mean ≥ geometric-mean.

---

## Related Topics

- [[deep-learning-fundamentals]] — cross-entropy and KL-divergence loss functions that build directly on the entropy/KL definitions here
- [[rl-fundamentals]] — MDPs formalise the Markov property introduced in the Markov Chains section above
- [[alignment-methods]] — RLHF's KL(π‖π_ref) penalty and GRPO's group-relative advantage normalisation both lean on the KL divergence and variance/standardisation tools on this page
- [[generative-models]] — VAEs' reverse-KL objective and the ELBO derivation via Jensen's inequality
- [[llm-evaluation]] — confidence intervals, bootstrap resampling, and the hypothesis-testing toolkit (t-test, McNemar's, etc.) above for comparing model performance on test sets

## Sources

- Alisa Liu, "Math Notes" (Notion, alisawuffles.notion.site/math-notes, accessed 2026) — discrete/continuous distribution catalogue with derivations (Bernoulli through Gaussian), Markov/Chebyshev/Cauchy-Schwarz inequalities, conditional probability and the tower property, CLT/LLN derivations, combinatorics (permutations, combinations, stars-and-bars), Markov chain absorption analysis, and classical probability puzzles (coupon collector, gambler's ruin, Monty Hall, reservoir sampling, secretary problem). Capture is partial — cuts off mid-derivation in the MLE worked example; Bias & Variance and Math Things (Taylor series, derivative rules) sections were not retrieved. See [`raw/alisa-liu-math-notes.md`](../raw/alisa-liu-math-notes.md).
- Alisa Liu, "Book of LLMs" (Notion, alisawuffles.notion.site/alisa-s-book-of-llms) — the CE(p,q)=KL(p‖q)+H(p) proof, the alternate logits-based entropy formula, and the full Hypothesis Testing Toolkit section (p-value, KS test, chi-squared test, t-tests, ANOVA, McNemar's, Pearson/Spearman correlation tests, mutual information). See [`raw/alisa-liu-book-of-llms.md`](../raw/alisa-liu-book-of-llms.md).
