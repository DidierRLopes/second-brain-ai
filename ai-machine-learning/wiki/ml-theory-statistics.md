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

**Cross-Entropy**: expected information content of distribution p under model q:

```
H(p, q) = -Σ_x p(x) log q(x)
```

This is the loss minimised in classification (cross-entropy loss). It equals H(p) + KL(p||q) — minimising cross-entropy is equivalent to minimising KL divergence when the target distribution p is fixed.

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
f(E[X]) ≤ E[f(X)]
```

Used to derive the ELBO in VAEs, the EM algorithm, and many information-theoretic results.
