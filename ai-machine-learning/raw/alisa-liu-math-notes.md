# Math Notes (Probability & Statistics Reference)

**Source:** https://alisawuffles.notion.site/math-notes
**Author:** Alisa Liu
**Published (per page metadata):** Nov 20, 2025

> **Capture note:** This page was fetched via a Jina Reader proxy (`r.jina.ai`) because the live Notion page is JS-rendered and not directly fetchable. The fetch tool truncates output at a fixed length; the capture below is complete through most of the "Statistics" section (cut off mid-derivation in the Maximum Likelihood Estimation worked example). The following TOC sections were **not retrieved** and are intentionally omitted rather than guessed: the remainder of the MLE worked example, **Bias & variance**, and all of **Math things** (Taylor series, Derivative rules, Handy equations). If this page is re-fetched in the future (e.g. once a JS-rendering browser tool is available), this file should be extended with those sections.

## Cheat-sheet table (concept → which distribution/tool)

| Description | Concept |
|---|---|
| memoryless, waiting | geometric, exponential |
| trials with success & failure | Bernoulli, binomial |
| large sample, average | CLT, approximately Gaussian |
| bound given only mean / variance | Markov / Chebyshev inequalities |
| estimating parameter from data | MLE, MAP, Bayesian update |
| update belief from evidence | Bayes rule |
| function of a RV | Jensen's inequality |
| expected count of things | linearity + indicators |
| expected hitting time / return probability | first-step analysis, solve recurrence |
| Markov property (future depends only on present) | first-step analysis |

Distributions are split along two axes: discrete vs. continuous, and counting vs. waiting. Binomial counts successes in *n* trials; Poisson counts events in a fixed window. Geometric counts trials until first success; exponential measures continuous waiting time until the first event.

## Discrete Distributions

### Bernoulli
$X \sim \text{Bernoulli}(p)$ models a single trial with $p$ probability of success: $P(X=1)=p$, $P(X=0)=1-p$, compactly $P(X=x) = p^x(1-p)^{1-x}$.

$E[X] = p$, $\text{Var}(X) = p(1-p)$.

Proof: $E[X] = 1\cdot p + 0\cdot(1-p) = p$. $E[X^2] = 1^2\cdot p + 0^2 \cdot (1-p) = p$. $\text{Var}(X) = E[X^2] - E[X]^2 = p - p^2 = p(1-p)$.

A **categorical** RV generalizes Bernoulli from 2 to $k$ outcomes: $P(X=i) = p_i$; each item $i$ is $\text{Bernoulli}(p_i)$, so $E[X_i]=p_i$ and $\text{Var}(X_i)=p_i(1-p_i)$.

Handy technique — **indicator squaring**: for any binary $X \in \{0,1\}$, $X^2 = X$, so $E[X^2] = E[X]$.

### Binomial
$X \sim \text{Binomial}(n,p)$ is the sum of $n$ independent $\text{Bernoulli}(p)$ trials: $X = \sum_{i=1}^n X_i$.

$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$ — every sequence with $k$ successes has the same probability $p^k(1-p)^{n-k}$, and there are $\binom{n}{k}$ such sequences.

Binomial theorem: $(a+b)^n = \sum_{k=0}^n \binom{n}{k} a^k b^{n-k}$ — used to sanity-check the PMF sums to 1: $\sum_{k=0}^n \binom{n}{k}p^k(1-p)^{n-k} = (p+(1-p))^n = 1$.

$E[X] = np$, $\text{Var}(X) = np(1-p)$.

Proof: $E[X] = E[\sum X_i] = \sum E[X_i] = np$. $\text{Var}(X) = \text{Var}(\sum X_i) = \sum \text{Var}(X_i) = np(1-p)$ (independence gives the second equality).

Handy technique — decompose $X^2 = (\sum_i X_i)^2 = \sum_i X_i^2 + \sum_{i\neq j} X_i X_j$, and use $X_i^2=X_i$ for indicators, giving $E[X^2] = nE[X_i] + n(n-1)E[X_iX_j]$.

### Poisson
$X \sim \text{Poisson}(\lambda)$ models the number of events in a fixed interval when they occur independently at average rate $\lambda$ over that interval. $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$. The count over an interval of length $t$ is $N(t) \sim \text{Poisson}(\lambda t)$.

Derived as a **binomial limit**: divide the interval into $n$ tiny sub-intervals (at most one event each), so the total count is $\text{Binomial}(n,p)$ with $np=\lambda$. Taking $n\to\infty$:
$$P(X=k) = \binom{n}{k}p^k(1-p)^{n-k} = \frac{\lambda^k}{k!}\cdot\underbrace{\frac{n!}{(n-k)!n^k}}_{\to 1}\cdot\underbrace{\left(1-\frac{\lambda}{n}\right)^n}_{\to e^{-\lambda}}\cdot\underbrace{\left(1-\frac{\lambda}{n}\right)^{-k}}_{\to 1}$$

$E[X] = \text{Var}(X) = \lambda$ ($E[X]=\lambda$ is basically the definition of rate $\lambda$).

Sums of independent Poissons are Poisson with rates adding: $X+Y \sim \text{Poisson}(\lambda_1+\lambda_2)$ (derivable via MGF uniqueness).

### Geometric
$X \sim \text{Geometric}(p)$ models the number of Bernoulli($p$) trials until the first success (inclusive). $P(X=k) = (1-p)^{k-1}p$. Tail: $P(X>k) = (1-p)^k$.

**Memoryless property**: $P(X > m+n \mid X > m) = P(X>n)$ — the fact you've failed $m$ times already is irrelevant. The geometric distribution is the *only* memoryless distribution on the positive integers (any discrete waiting-time distribution that's memoryless must be geometric).

Geometric series: $\sum_{k=0}^\infty r^k$ converges to $\frac{1}{1-r}$ for $|r|<1$ (sanity check: $r=1/2 \Rightarrow \sum 1/2^k \to 2$).

Sanity check PMF sums to 1: $\sum_{k=1}^\infty (1-p)^{k-1}p = p\sum_{i=0}^\infty(1-p)^i = p\cdot\frac{1}{1-(1-p)} = p\cdot\frac1p = 1$.

$E[X] = 1/p$. Two derivations given:
1. Direct: $E[X] = \sum_{k=1}^\infty k\cdot P(X=k)$ (algebra-heavy, gives $1/p$).
2. **First-step analysis**: on the first flip, succeed w.p. $p$ ($X=1$) or fail w.p. $1-p$ (back to start, $X = 1+E[X]$): $E[X] = p\cdot 1 + (1-p)(1+E[X])$, solving gives $E[X]=1/p$. Intuition: every trial has $p$ chance of ending the process, so on average need $1/p$ attempts.

$\text{Var}(X) = \frac{1-p}{p^2}$ — as $p\to 0$, both mean and variance blow up (rare successes have high waiting-time variability).

Handy technique: first-step analysis; re-index with $i=k-1$ to turn $\sum_{k=1}^\infty$ into $\sum_{i=0}^\infty$.

## Continuous Distributions

### Uniform
$X \sim \text{Uniform}(a,b)$ on $[a,b]$: equally likely anywhere on the interval.

$E[X] = \frac{a+b}{2}$, derived via $\int_a^b x\cdot\frac{1}{b-a}dx = \frac{1}{b-a}\cdot\frac{b^2-a^2}{2} = \frac{(b-a)(b+a)}{2(b-a)} = \frac{a+b}{2}$ (fundamental theorem of calculus).

$\text{Var}(X) = \frac{(b-a)^2}{12}$.

PDF: $f_X(x) = \frac{1}{b-a}$ for $a\le x\le b$. CDF: $F_X(x) = 0$ for $x<a$; $\frac{x-a}{b-a}$ for $a\le x\le b$; $1$ for $x>b$.

Notes: the difference between two uniform distributions is triangular; the $n+1$ gaps between $n$ uniform points are identically distributed.

### Exponential
$X \sim \text{Exp}(\lambda)$ with rate $\lambda$ models waiting time until the next event in a Poisson process with rate $\lambda$. $f_X(x) = \lambda e^{-\lambda x}$, $x\ge 0$. Tail/survival function: $P(X>x) = e^{-\lambda x}$.

**Memoryless property**: $P(X>s+t \mid X>s) = P(X>t)$ — the exponential is the *only* continuous distribution on $[0,\infty)$ that is memoryless; it's the continuous analog of the geometric.

$E[X] = 1/\lambda$, $\text{Var}(X) = 1/\lambda^2$ — if events happen at rate $\lambda$/unit time, on average you wait $1/\lambda$ units between events.

**Relation between Poisson and exponential** (two equivalent statements): number of events in a unit interval is $\text{Poisson}(\lambda)$ $\iff$ inter-event time is $\text{Exp}(\lambda)$. ("How many events in the next hour?" → Poisson. "How long until the next event?" → exponential.)

**Minimum of independent exponentials**: if $X_1\sim\text{Exp}(\lambda_1)$, $X_2\sim\text{Exp}(\lambda_2)$ independent, then $\min(X_1,X_2)\sim\text{Exp}(\lambda_1+\lambda_2)$. Intuition: if emails arrive at rate $\lambda_1$ and texts at rate $\lambda_2$, waiting time for "either" is exponential with rate $\lambda_1+\lambda_2$.

Proof: $P(\min(X_1,X_2)>t) = P(X_1>t)P(X_2>t) = e^{-\lambda_1 t}e^{-\lambda_2 t} = e^{-(\lambda_1+\lambda_2)t}$, the tail function of $\text{Exp}(\lambda_1+\lambda_2)$.

### Gaussian
$X \sim N(\mu,\sigma^2)$. Standardization: $Z = \frac{X-\mu}{\sigma}\sim N(0,1)$.

Worked example: heights $\sim N(70,9)$ ($\mu=70,\sigma=3$). Fraction taller than 76 inches: $Z = \frac{X-70}{3}$, so $P(X>76)=P(Z>2)$.

**Percentile** $z_\alpha$: the value such that $P(Z\le z_\alpha)=\alpha \iff P(Z>z_\alpha)=1-\alpha$ — i.e. $z_\alpha$ has probability mass $\alpha$ to its left. Symmetry: $z_{1-\alpha} = -z_\alpha$.

Two-sided intervals: $P(|Z|\le z_{1-\alpha/2}) = 1-\alpha \iff P(|Z|>z_{1-\alpha/2})=\alpha$. A $(1-\alpha)$ confidence interval uses $z_{\alpha/2}$ and $z_{1-\alpha/2}$ as endpoints. As a bound: $P(|Z|\le c)\ge 1-\alpha \iff c\ge z_{1-\alpha/2}$.

For $X\sim N(\mu,\sigma^2)$, the $\alpha$-percentile is $x_\alpha = \mu+\sigma\cdot z_\alpha$ (derived from $P(X\le x_\alpha)=P(Z\le \frac{x_\alpha-\mu}{\sigma})=\alpha$).

**Sum of independent Gaussians is Gaussian**: $X_1\sim N(\mu_1,\sigma_1^2)$, $X_2\sim N(\mu_2,\sigma_2^2)$ independent $\Rightarrow X_1+X_2 \sim N(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$.

## Expectations & Variance

$E[X] = \sum_x x\,P(X=x)$ or $\int_{-\infty}^\infty x\,f(x)\,dx$. For a function of $X$: $E[g(X)] = \sum_x g(x)P(X=x)$ or $\int g(x)f(x)dx$.

**Tail formula for non-negative RVs**: $E[X] = \int_0^\infty P(X>x)\,dx = \int_0^\infty (1-F(x))\,dx$.

$\text{Var}(X) = E[(X-E[X])^2] = E[X^2]-(E[X])^2$.

**Linearity of expectation** (regardless of independence): $E[aX+bY+c] = aE[X]+bE[Y]+c$.

For independent $X,Y$: $E[XY] = E[X]E[Y]$, and $\text{Var}(X+Y) = \text{Var}(X)+\text{Var}(Y)$.

## Zoo of Functions

- **PMF** (discrete): $p(k)=P(X=k)$, with $p(k)\ge 0$ and $\sum_k p(k)=1$.
- **PDF** (continuous): $f(x)$ such that $P(a\le x\le b) = \int_a^b f(x)dx$; differentiate the CDF to get the PDF: $f(x)=F'(x)$.
- **CDF** (both): $F(x)=P(X\le x)$. Discrete: $F(x)=\sum_{k\le x}p(k)$. Continuous: $F(x)=\int_{-\infty}^x f(x)dx$. Note $P(a\le x\le b)=F(b)-F(a)$.

## Inequalities

**Markov's inequality**: for non-negative $X$, $a>0$: $P(X\ge a) \le \frac{E[X]}{a}$. Intuitively $E[X]$ can't be smaller than the contribution of values $\ge a$.

Proof: $E[X] \ge \sum_{x\ge a} xP(X=x) \ge a\sum_{x\ge a}P(X=x) = aP(X\ge a)$.

**Chebyshev's inequality**: for RV $X$ with mean $\mu$, variance $\sigma^2$: $P(|X-\mu|\ge k\sigma) \le \frac{1}{k^2}$ — at least $1-1/k^2$ of the distribution lies within $k$ standard deviations of the mean (e.g. 75% of mass within 2 SDs).

Proof: apply Markov to $(X-\mu)^2$: $P(|X-\mu|\ge k\sigma) = P((X-\mu)^2\ge k^2\sigma^2) \le \frac{E[(X-\mu)^2]}{k^2\sigma^2} = \frac{1}{k^2}$. Alternate form, substituting $\epsilon=k\sigma$: $P(|X-\mu|\ge\epsilon)\le \frac{\sigma^2}{\epsilon^2}$.

**Jensen's inequality**: $\varphi$ convex $\Rightarrow \varphi(E[X])\le E[\varphi(X)]$; $\varphi$ concave $\Rightarrow \varphi(E[X])\ge E[\varphi(X)]$. Memory aid: $\varphi(x)=x^2$ (convex), $X\in\{-1,+1\}$: $\varphi(E[X])=\varphi(0)=0$ but $E[\varphi(X)]=1$.

Convex functions ($\varphi''\ge 0$ everywhere): $x^2, x^4,\dots$ (even powers), $|x|$, $a^x$ ($a>1$), $1/x$ ($x>0$), $\max(x_1,x_2,\dots)$.

Concave functions ($\varphi''\le 0$ everywhere): $\sqrt{x}$, $\log x$, $x^p$ for $0<p<1$, $\min(x_1,x_2,\dots)$. Linear functions are both convex and concave.

Worth memorizing: $\log E[X] \ge E[\log X]$.

**AM-GM inequality**: arithmetic mean $\ge$ geometric mean: $\frac{a_1+\dots+a_n}{n} \le \sqrt[n]{a_1\cdots a_n}$. Proof: take log of both sides, apply Jensen's (log is concave), exponentiate.

**Cauchy-Schwarz inequality**: $E[XY]^2 \le E[X^2]E[Y^2]$, or equivalently $\left(\sum_{i=1}^n x_iy_i\right)^2 \le \left(\sum_i x_i^2\right)\left(\sum_i y_i^2\right)$.

**Union bound**: $P(A_1\cup\dots\cup A_n) \le P(A_1)+\dots+P(A_n)$.

## Conditional Probabilities

For two RVs $X,Y$: joint $P(X=x,Y=y)$; marginal $P(X=x)=\sum_y P(X=x,Y=y)$; conditional $P(X=x\mid Y=y)$.

**Bayes rule**: $P(A\mid B) = \frac{P(B\mid A)P(A)}{P(B)}$. $P(A)$ is the prior (belief before seeing $B$); $P(A\mid B)$ is the posterior (belief after). Compute $P(B)$ via the law of total probability.

**Law of total expectation (tower property)**: $E[X] = \sum_y E[X\mid Y=y]\cdot P(Y=y)$ or $\int E[X\mid Y=y]f(y)dy$; equivalently $E[X] = E_Y[E[X\mid Y]]$, where $E[X\mid Y]$ is itself a random variable in terms of $Y$.

**Law of total probability**: $P(A) = \sum_y P(A\mid Y=y)\cdot P(Y=y)$.

## Limit Theorems

**Central Limit Theorem**: the mean of $n$ iid RVs $X_i$ with mean $\mu$, variance $\sigma^2$ is approximately Gaussian no matter the shape of $X_i$: $\bar X_n = \frac1n\sum_{i=1}^n X_i \approx N(\mu, \sigma^2/n)$.

Variance derivation: $\text{Var}(\bar X_n) = \frac{1}{n^2}\text{Var}(\sum X_i) = \frac{1}{n^2}\sum\text{Var}(X_i) = \frac{n}{n^2}\sigma^2 = \frac{\sigma^2}{n}$.

**Standard error** $= \sigma/\sqrt n$ — error shrinks like $\sqrt n$; need 100× the samples to reduce error 10×. $(1-\alpha)$ confidence interval: $\bar X_n \pm z_{1-\alpha/2}\cdot\frac{\sigma}{\sqrt n}$.

**Law of Large Numbers**: $P(|\bar X_n - \mu| > \epsilon) \to 0$ for any $\epsilon>0$. Proof from Chebyshev (substituting $\epsilon=k\sigma$): $P(|\bar X_n-\mu|>\epsilon) \le \frac{\text{Var}(\bar X_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2}\to 0$.

## Some Problem Types

### First-step analysis
Worked example: expected number of coin flips to get two heads in a row. Let $E_0$ = expected flips after just flipping tails (or start), $E_1$ = expected flips after just flipping heads.
$$E_0 = \tfrac12(1+E_0) + \tfrac12(1+E_1) \qquad E_1 = \tfrac12(1+E_0) + \tfrac12\cdot 0$$
Solving the linear system gives $E_0 = 6$.

**General trick**: focus on the first step of the process, express the desired expectations in terms of themselves, then solve the resulting linear system.

### Counting in expectation
To count how many things satisfy a property: define $X_i = \mathbb{1}[\text{item } i \text{ has property}]$, $N=\sum_i X_i$, compute $E[X_i]=P(X_i)$ for each $i$ (use symmetry), then apply linearity: $E[N]=\sum_i P(X_i)$.

Example: expected number of fixed points in a random permutation — by symmetry $P(X_i=1)=1/n$ for each position.

### Max/min of $n$ RVs
For iid $X_1,\dots,X_n$ with CDF $F_X$: $M=\max_i X_i$ has $F_M(x) = P(M\le x) = [F_X(x)]^n$. The min $m=\min_i X_i$ has $P(m>x) = [1-F_X(x)]^n$, so $F_m(x) = 1-[1-F_X(x)]^n$.

Two approaches to compute $E[\min]$/$E[\max]$: the 3-step recipe (CDF → PDF → integrate), or for non-negative RVs the tail formula $E[X]=\int_0^\infty P(X>x)dx$ directly.

Worked for $X_i \sim \text{Uniform}[0,1]$:
- Max: $F_M(x)=x^n$, $f_M(x)=nx^{n-1}$, $E[M] = \int_0^1 x\cdot nx^{n-1}dx = \frac{n}{n+1}$. Intuition: the $n$ uniform points split $[0,1]$ into $n+1$ equivalently-distributed (by symmetry) segments, each with mean length $1/(n+1)$. Alternative via tail formula: $E[M]=\int_0^1(1-x^n)dx = \frac{n}{n+1}$.
- Min: $P(m>x)=(1-x)^n$, $F_m(x)=1-(1-x)^n$, $f_m(x)=n(1-x)^{n-1}$, $E[m] = \int_0^1 x\cdot n(1-x)^{n-1}dx = \frac{1}{n+1}$.

For $X_i \sim \text{Exponential}(\lambda)$: $P(m>x) = e^{-n\lambda x}$, so $E[m] = \int_0^\infty e^{-n\lambda x}dx = \frac{1}{n\lambda}$. The max is more complicated but grows logarithmically with $n$; for $X_i\sim N(0,1)$, $E[\max_i X_i] \approx \sqrt{2\ln n}$ (difficult to derive).

## Combinatorics

**Combination** (choose $k$ from $n$, order doesn't matter): $\binom{n}{k} = \frac{n!}{k!(n-k)!} = \frac{P(n,k)}{k!}$ — every set of $k$ items can be arranged $k!$ ways; combinations collapse the orderings.

**Permutation** (arrange $k$ from $n$, order matters): $P(n,k) = \frac{n!}{(n-k)!}$.

**Multinomial coefficient** — arranging $n$ objects with $n_1$ of type 1, ..., $n_m$ of type $m$: $\binom{n}{n_1,\dots,n_m} = \frac{n!}{n_1!\cdots n_m!}$ (permute all $n$, divide out within-type rearrangements). A **derangement** is a permutation with no fixed points.

**Combination with replacement** (choose $k$ from $n$ types, with replacement): $\binom{n+k-1}{k}$ — the stars-and-bars argument: represent each choice as $k$ stars and $n-1$ bars, i.e. choosing $k$ star-placements from $n+k-1$ positions. Equivalently: number of ways to write non-negative integer $n$ as a sum of $k$ non-negative integers.

Summary table:

| Order matters? | Replacement? | Count |
|---|---|---|
| yes | no | $\frac{n!}{(n-k)!}$ (permutations) |
| no | no | $\binom{n}{k}$ (combinations) |
| yes | yes | $n^k$ |
| no | yes | $\binom{n+k-1}{k}$ (stars and bars) |

**Inclusion-exclusion principle**: $P(A\cup B) = P(A)+P(B)-P(A\cap B)$; $P(A\cup B\cup C) = P(A)+P(B)+P(C)-P(A\cap B)-P(A\cap C)-P(B\cap C)+P(A\cap B\cap C)$ (same identity holds for set cardinalities).

## Markov Chains

A Markov chain is a sequence of RVs $X_0,X_1,X_2,\dots$ in state space $S$ satisfying the **Markov property**: the future depends only on the present. For finite state spaces, described by transition matrix $P$ where $P_{ij}=P(X_{t+1}=j\mid X_t=i)$ — rows sum to 1.

If $\pi_t$ is the distribution at time $t$: $\pi_{t+1} = \pi_t P$. A **stationary distribution** satisfies $\pi = \pi P$. Under mild conditions the chain has a unique stationary distribution and $\pi_t \to \pi$ regardless of starting point.

**Expected return time** to state $i$ is the reciprocal of its stationary probability: $E[\text{return to }i] = \frac{1}{\pi_i}$.

**General recipe for absorption problems** (via first-step analysis):
- Expected absorption time $h_j$ from state $j$: $h_A=0$ for the target absorbing state $A$; $h_i = 1+\sum_j P_{ij}h_j$ otherwise.
- Absorption probability $q_j$: $q_A=1$ for the desired absorbing state, $q_B=0$ for other absorbing states $B$; $q_i = \sum_j P_{ij}q_j$ otherwise.
- Solve the resulting recurrence: for small chains solve the linear system directly; for chains with structure, guess a closed form (compute small cases, generalize, fit boundary conditions) and verify.

Handy techniques: use symmetry to collapse equivalent states (e.g. for a random walk on a circle, reparameterize $h_i$ by distance rather than by absolute state); use the stationary distribution as a shortcut — on a graph where every node looks the same (cycles, complete graphs, etc.), the expected hitting time $h$ between adjacent vertices is always $n-1$ (since $E[\text{return to self}]=n$ and $E[\text{return to self}] = 1+h$ when each of the $d$ adjacent nodes contributes $\frac1d h$).

## Puzzles

### Birthday problem
How many people $n$ does it take for $P(\text{collision}) > 50\%$? $P(\text{collision}) = 1-P(\text{all unique})$. With $d$ days in a year: $P(\text{all unique}) = \frac{d}{d}\cdot\frac{d-1}{d}\cdots\frac{d-n+1}{d} = \frac{d!}{(d-n)!\cdot d^n}$.

### Gambler's ruin
Start with \$$a$; win \$1 w.p. $p$, lose \$1 w.p. $1-p$; stop at \$$N$ (win) or \$0 (ruin). Let $q_i$ = probability of reaching $N$ from $i$. Boundary: $q_0=0$, $q_N=1$; recurrence $q_i = p\cdot q_{i+1} + (1-p)q_{i-1}$ (second-order linear recurrence).

**Fair game** ($p=1/2$): $q_a = a/N$. Two derivations:
1. Martingale argument: let $T$ be the stop time, $W_T$ the wealth at $T$. $E[W_T]=a$ (no drift), and since $W_T\in\{0,N\}$, $E[W_T] = q_a\cdot N$. Combining: $q_a=a/N$. (In general, a **martingale** is a sequence $M_1,\dots,M_n$ with $E[M_1]=\dots=E[M_n]$.)
2. Solving the recurrence directly: $q_i = \frac12 q_{i-1}+\frac12 q_{i+1} \Rightarrow q_{i+1}-q_i = q_i-q_{i-1}$ — all consecutive differences are equal, so $q_i$ is linear in $i$; with $q_0=0,q_N=1$, get $q_i = i/N$.

This is a random walk with absorbing barriers at $0$ and $N$. The expected absorption time is $h_i = i(N-i)$ (tricky to derive directly, but easy to verify against the first-step-analysis recurrence).

### Random walk
Simple symmetric random walk $S_n = X_1+\dots+X_n$, $X_i=\pm1$ equally likely. $E[S_n]=0$. Each $X_i$ has variance $E[X_i^2]-(E[X_i])^2 = 1-0=1$, so $\text{Var}(S_n) = \sum\text{Var}(X_i) = n$ — typical distance from origin grows like $\sqrt n$.

Biased case ($X_i=1$ w.p. $p$, else $-1$): $E[S_n] = n(p\cdot1+(1-p)(-1)) = n(2p-1)$.

### Coupon collector
$n$ coupon types, one uniformly random coupon per purchase; expected purchases to collect all $n$? Let phase $k$ = having $k$ types; $X_k \sim \text{Geom}((n-k)/n)$, $E[X_k] = \frac{n}{n-k}$.

Total: $E[X] = \sum_{k=0}^{n-1} \frac{n}{n-k} = n\sum_{j=1}^n \frac1j = n\cdot H_n$, where $H_n = 1+\frac12+\dots+\frac1n$ is the $n$-th harmonic number. The final phase alone takes $n$ boxes in expectation.

### Information-theoretic puzzles
**Count the information**: to distinguish $k$ possibilities with tests that each have $m$ outcomes, need at least $\lceil \log_m k \rceil$ tests (a lower bound; whether it's achievable depends on problem constraints).

**9 coins, one heavier, 2 weighings**: divide into 3 groups of 3; weigh two groups against each other. Heavier side (or, if balanced, the left-out group) contains the counterfeit; then weigh 2 of the remaining 3 coins.

### Reservoir sampling
Stream $x_1,x_2,x_3,\dots$ of unknown length $n$; return a uniformly random item at the end. Algorithm: keep $s=x_1$; for $n=2,3,4,\dots$, replace $s$ with $x_n$ with probability $1/n$.

Proof by induction: after processing $n$ items, each $x_1,\dots,x_n$ is in $s$ with probability $1/n$. Base case trivial. Inductive step: $s=x_n$ (replace) happens w.p. $1/n$; $s=x_k$ for $k<n$ (keep) requires $s=x_k$ after step $n-1$ (prob $1/(n-1)$ by induction) **and** rejecting $x_n$ (prob $(n-1)/n$): $\frac{1}{n-1}\cdot\frac{n-1}{n} = \frac1n$. So every $x_k$ is in the reservoir with probability $1/n$ after $n$ steps.

### Secretary problem
$n$ candidates interviewed in random order; decide immediately after each interview to hire/reject; goal: maximize probability of hiring the best candidate. Strategy: reject the first $k$, then hire the next candidate better than all of the first $k$ (or hire the last one if none appears). What's the optimal $k$?

Let $P(k)$ = success probability, $b$ = position of the best candidate. Strategy succeeds iff $b>k$ and the best among positions $1,\dots,b-1$ is among the first $k$:
$$P(k) = \sum_{b=k+1}^n \frac1n\cdot\frac{k}{b-1} = \frac kn\sum_{i=k}^{n-1}\frac1i = \frac kn(H_{n-1}-H_{k-1}) \approx \frac kn\log\frac nk = r\log\frac1r = -r\log r$$
where $r=k/n$ and $H_n\approx \log n$ for large $n$. Differentiating: $-\frac{d}{dr}r\log r = -\log r - 1$; setting to 0 gives $\log r = -1 \Rightarrow r=1/e \Rightarrow k = n/e$.

### Monty Hall
Let $A\in\{1,\dots,n\}$ be the car's door, $B$ the set of doors Monty opens; we picked door $c$ WLOG. Want $P(A=j\mid B)$ for unopened doors $j$. By Bayes, since the prior $P(A=j)=1/n$ and denominator are shared across $j$, comparing posteriors reduces to comparing $P(B\mid A=j)$.

**Case 1 — Monty opens $n-2$ doors**, leaving our door and one other door $r$: $P(B\mid A=t)=1/(n-1)$ (our door has the car: $n-1$ choices for which door to leave closed); $P(B\mid A=r)=1$ (Monty is fully constrained); $P(B\mid A=k)=0$ for opened doors $k$. Normalizing: $P(A=t\mid B)=1/n$, $P(A=r\mid B) = (n-1)/n$.

**Case 2 — Monty opens only 1 door**: for remaining door $r$: $P(B\mid A=t) = 1/(n-1)$; $P(B\mid A=r) = 1/(n-2)$ ($n-2$ choices for which door to open, since door $c$ and door $A$ are excluded). So $P(A=t\mid B)\propto 1/(n-1)$, $P(A=r\mid B)\propto 1/(n-2)$ — remaining doors are each slightly more likely than the originally-picked door.

General pattern: the sum of probability mass "released" by opened doors ($k/n$ for $k$ opened doors) redistributes onto the remaining unopened, unchosen doors (of which there are $n-k-1$).

### Longest run of heads
Flip a fair coin $n$ times; $L_n$ = longest run of consecutive heads. No clean closed form for $E[L_n]$, but a well-known approximation: $E[L_n] \approx \log_2(n)$.

### Throwing balls into bins
Throwing $m$ balls into $m$ bins uniformly: about 37% of bins end up empty, 37% get exactly one ball, and the rest get two or more.

### Path counting
Paths from $(0,0)$ to $(m,n)$ on a grid (steps: right or up only): every path has exactly $m$ right-steps and $n$ up-steps, uniquely specified by which of the $m+n$ steps are right-steps: $\binom{m+n}{m}$ total paths.

Paths passing through $(a,b)$: multiply the path count to $(a,b)$ by the path count from $(a,b)$ to $(m,n)$: $\binom{a+b}{a}\cdot\binom{(m-a)+(n-b)}{m-a}$.

## Statistics

### Maximum Likelihood Estimation
Given data $x_1,\dots,x_n$ drawn from a distribution with unknown parameter $\theta$: MLE picks the $\theta$ that makes the observed data most probable.
$$L(\theta) = \prod_{i=1}^n p(x_i;\theta) \qquad \hat\theta_{\text{MLE}} = \arg\max_\theta L(\theta)$$

**Log-likelihood trick**: take the log to turn the product into a sum: $\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log p(x_i;\theta)$. Differentiate w.r.t. $\theta$, set to zero, solve for $\hat\theta$ (check it's a max via the second derivative / endpoint behavior).

Worked example (capture cuts off mid-derivation here): $x_1,\dots,x_n$ iid $\text{Bernoulli}(p)$, PMF $P(X=x;p)=p^x(1-p)^{1-x}$, log-likelihood
$$\ell(p) = \sum_i \log\left(p^x(1-p)^{1-x}\right) = \sum_i \big[x\log p + (1-x)\log(1-p)\big] \;\dots$$

*(Capture ends here. The remainder of this MLE derivation — and the Bias & variance / Math things sections — were not retrieved; see capture note above.)*
