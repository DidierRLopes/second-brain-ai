# Speculative Decoding - The Bits and the Bytes! (Part 1)

**Source:** https://aakashkumarnain.github.io/posts/ml_dl_concepts/specdec_part1.html
**Author:** Aakash Kumar Nain (@A_K_Nain)
**Published:** June 19, 2026

This is Part 1 of a three-part series ("Speculative Decoding: The bits and
the bytes" → "Eagle" → "Diffusion Based Methods, e.g. DFlash"). Part 1 covers
the fundamentals: the core algorithm, speculative sampling, and a full
mathematical treatment of acceptance rate, expected tokens per iteration,
wall-time improvement, and arithmetic-operation overhead. Notation follows
the original speculative decoding paper (Leviathan et al., [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)),
with \(p\)/\(q\) for full distributions and \(p(x)\)/\(q(x)\) for the scalar
probability of a specific token \(x\).

## Setup and Notation

- \(M_p\): the target model we want to accelerate; \(p(x_t \mid x_{<t})\) is its distribution over the next token given prefix \(x_{<t}\).
- \(M_q\): a smaller, cheaper approximation model; \(q(x_t \mid x_{<t})\) is its distribution for the same prefix.
- \(\gamma\): the number of draft tokens \(M_q\) generates per speculative-decoding iteration.
- \(x \sim q\) means "draw a token from distribution \(q\)"; \(q(x)\) is the probability mass \(q\) assigns to that specific drawn token.

## The Core Algorithm

1. Use \(M_q\) to generate \(\gamma\) tokens autoregressively (the draft).
2. Use \(M_p\) to verify all \(\gamma\) draft tokens in a single parallel forward pass.
3. Accept or reject each token according to an acceptance criterion (speculative sampling, below).
4. Sample one additional token from an *adjusted* distribution — either to replace the first rejected token, or as a bonus token if all \(\gamma\) drafts were accepted.
5. Best case: all \(\gamma\) tokens accepted, plus the bonus token → \(\gamma + 1\) tokens from one parallel \(M_p\) pass. Worst case: the first draft is rejected → still 1 newly generated token from that same single parallel pass. So a single forward pass of \(M_p\) yields anywhere from 1 to \(\gamma+1\) tokens.

Key clarification on "parallel verification": it is structurally identical to
*prefill*. \(M_p\) receives the sequence \([\text{prefix}, x_1, \dots,
x_\gamma]\) as if it were a prompt, and one forward pass over it produces
\(\gamma + 1\) next-token distributions \(p_1(x), \dots, p_{\gamma+1}(x)\) —
the last one being the "bonus" distribution available for free if everything
upstream is accepted.

### Algorithm 1: SpeculativeDecodingStep (pseudocode from the post)

```
Inputs: M_p, M_q, prefix.

# Sample γ guesses x_1..γ from M_q autoregressively.
for i = 1 to γ do
    q_i(x) ← M_q(prefix + [x_1, ..., x_{i-1}])
    x_i ~ q_i
end for

# Run M_p in parallel.
p_1(x), ..., p_{γ+1}(x) ← M_p(prefix), ..., M_p(prefix + [x_1, ..., x_γ])

# Determine the number of accepted guesses n.
r_1 ~ U(0,1), ..., r_γ ~ U(0,1)
n ← min({ i-1 | 1 ≤ i ≤ γ, r_i > p_i(x)/q_i(x) } ∪ {γ})

# Adjust the distribution from M_p if needed.
p'(x) ← p_{n+1}(x)
if n < γ then
    p'(x) ← norm(max(0, p_{n+1}(x) - q_{n+1}(x)))
end if

# Return one token from M_p, and n tokens from M_q.
t ~ p'
return prefix + [x_1, ..., x_n, t]
```

\(n\) is the number of consecutive accepted draft tokens (the first index
where a uniform random draw \(r_i\) exceeds the ratio \(p_i(x)/q_i(x)\),
minus one; or \(\gamma\) if none trigger rejection).

## Speculative Sampling (the acceptance/rejection rule)

1. Sample \(x \sim q\) (draw a candidate token from the draft model).
2. If \(q(x) \le p(x)\): accept unconditionally. If \(q(x) > p(x)\): reject with probability \(1 - \frac{p(x)}{q(x)}\), and discard every token drafted after this position.
3. If rejected, resample the replacement token from the **adjusted distribution**:
   \[ p'(x) = \mathrm{norm}\big(\max(0,\ p(x) - q(x))\big) \]

### Why sample from the adjusted distribution instead of from \(p(x)\) directly?

This is a probability-accounting argument, made fully explicit in the post
(this is the kind of derivation that's typically just asserted in shorter
treatments).

The probability that token \(x\) is produced via the **acceptance path** is:
\[
P(\text{accepted}, x) = \underbrace{q(x)}_{\text{drafted}} \times \underbrace{\min\!\Big(1, \frac{p(x)}{q(x)}\Big)}_{\text{accepted}} = \min(q(x), p(x))
\]

So acceptance alone already delivers \(\min(p(x), q(x))\) of the target
probability mass for token \(x\). For the *total* output probability of
\(x\) to equal \(p(x)\) (the correctness requirement), the resample-on-rejection
path must supply exactly the remainder:
\[ p(x) - \min(p(x), q(x)) = \max(0, p(x) - q(x)) \]
— which, after normalization, is precisely \(p'(x)\). If you resampled from
raw \(p(x)\) instead, you would double-count the mass already paid out by
acceptance, and the output distribution would drift away from \(p(x)\).

**Worked numeric example from the post** (4-token vocabulary \(A,B,C,D\)):

| Token | \(p(x)\) | \(q(x)\) | \(\min(p,q)\) covered by acceptance | Still needed |
|---|---|---|---|---|
| A | 0.50 | 0.20 | 0.20 | 0.30 |
| B | 0.10 | 0.40 | 0.10 | 0.00 |
| C | 0.30 | 0.15 | 0.15 | 0.15 |
| D | 0.10 | 0.25 | 0.10 | 0.00 |

Token B's entire probability (0.10) is already covered by acceptance — it
needs zero contribution from the resample.

- **If you resample from \(p(x)\) directly:** rejection probability \(1-\beta\) where \(\beta = \sum_x \min(p(x),q(x)) = 0.20+0.10+0.15+0.10 = 0.55\), so \(1-\beta=0.45\). Then \(P(B) = 0.10 + 0.45 \times p(B) = 0.10 + 0.45(0.10) = 0.145\) — **too high**, B is over-represented.
- **If you resample from the adjusted \(p'(x)\):** \(p'(B) = \max(0, 0.10-0.40)/0.45 = 0\), so \(P(B) = 0.10 + 0.45 \times 0 = 0.10\) — exactly matches \(p(B)\).

This is the concrete proof-by-example that speculative sampling makes the
output distribution of speculative decoding **exactly identical** to the
target model's distribution \(p\) — not an approximation.

## Acceptance Rate \(\alpha\) and Expected Tokens per Iteration

For a fixed prefix \(x_{<t}\), define the (position-specific) acceptance
rate \(\beta\) as the probability of accepting a draft token sampled from
\(q\):
\[
\beta = \mathbb{E}_{x\sim q}\Big[\min\big(1, \tfrac{p(x)}{q(x)}\big)\Big] = \sum_x q(x)\min\big(1,\tfrac{p(x)}{q(x)}\big) = \sum_x \min(p(x), q(x)) \tag{1}
\]

Assuming the per-position \(\beta\)'s are i.i.d., define the expected
acceptance rate \(\alpha = \mathbb{E}(\beta)\) — a single scalar measuring
how well \(M_q\) approximates \(M_p\) on average.

Let \(N\) be the number of tokens output by one speculative-decoding
iteration (with budget \(\gamma\)). Each draft is independently accepted
with probability \(\alpha\):

- Reject draft 1 → \(N=1\) (just the resampled token)
- Accept 1, reject 2 → \(N=2\)
- ... accept 1..k-1, reject k → \(N=k\)
- Accept all \(\gamma\) → \(N = \gamma+1\) (includes the bonus token)

PMF:
\[ P(N=k) = \alpha^{k-1}(1-\alpha) \text{ for } k=1,\dots,\gamma \tag{2} \]
\[ P(N=\gamma+1) = \alpha^{\gamma} \tag{3} \]

Using \(E(N) = \sum_{k\ge 1} P(N \ge k)\) and \(P(N\ge k)=\alpha^{k-1}\):
\[
E(N) = \sum_{k=1}^{\gamma+1}\alpha^{k-1} = \sum_{j=0}^{\gamma}\alpha^j = \frac{1-\alpha^{\gamma+1}}{1-\alpha} \tag{4}
\]
a finite geometric series with \(\gamma+1\) terms.

## Deriving \(\alpha\) via the Leviathan-Kalman (\(D_{LK}\)) Divergence

The post derives, step by step, how the original paper connects \(\alpha\)
to a divergence measure. Define the midpoint distribution \(M(x) =
\frac{p(x)+q(x)}{2}\) and the divergence:
\[ D_{LK}(p,q) = \sum_x |p(x)-M(x)| = \sum_x |q(x)-M(x)| \tag{5} \]

The post proves the two sums are equal (both reduce to
\(\frac{|p(x)-q(x)|}{2}\)), and that \(D_{LK}\) is exactly the **total
variation distance**:
\[
D_{LK}(p,q) = \sum_x \frac{|p(x)-q(x)|}{2} = 1 - \sum_x \min(p(x),q(x)) \tag{6}
\]
(derived via \(|p-q| = p+q-2\min(p,q)\) and the fact that both distributions sum to 1).

Properties noted: \(D_{LK}\) is symmetric; \(0 \le D_{LK} \le 1\); \(D_{LK}=0
\iff p=q\); \(D_{LK}=1 \iff p,q\) have disjoint support.

Combining (1), (5), (6):
\[
\alpha = 1 - \mathbb{E}[D_{LK}(p,q)] = \mathbb{E}\Big[\sum_x \min(p(x),q(x))\Big] \tag{7}
\]
Practically: run both models over \(N\) prefixes, compute \(\sum_x
\min(p(x),q(x))\) per prefix, and average — this is exactly how the original
paper estimated \(\alpha\) empirically (over 10K generated tokens).

**Edge-case interpretation:**
- \(\alpha = 0\): only the resampled token is produced per run (same throughput as standard decoding) — the drafter provides zero benefit and the drafting overhead is pure loss.
- \(\alpha = 1\): every run produces \(\gamma+1\) tokens (full acceptance + bonus).
- High \(\alpha\) (high distributional overlap between \(M_q\) and \(M_p\)) does **not** by itself guarantee wall-time improvement — drafter cost matters too (see below).

## Wall-Time Improvement Formula

Define the cost coefficient \(c = T_{M_q}/T_{M_p}\) (ratio of one drafter
forward-pass time to one target forward-pass time). In the original paper's
setup, \(M_q\) was a couple of orders of magnitude smaller than \(M_p\), and
\(c\) was always < 0.05, often near 0. This depends on hardware and
implementation.

Let \(T\) = time for one \(M_p\) forward pass.

1. Cost of one speculative-decoding iteration: \(\gamma\) drafter passes at \(cT\) each, plus one parallel \(M_p\) verification pass at \(T\):
   \[ T_{\text{spec}} = \gamma c T + T = (\gamma c + 1) T \tag{9} \]
2. Tokens per iteration: \(E(N) = \frac{1-\alpha^{\gamma+1}}{1-\alpha}\) (eq. 4).
3. Cost per token with speculative decoding:
   \[ \frac{T_{\text{spec}}}{E(N)} = \frac{(\gamma c+1)(1-\alpha)}{1-\alpha^{\gamma+1}} \cdot T \]
4. Cost per token with standard decoding: \(T_{\text{standard}} = T\).
5. **Wall-time improvement factor:**
   \[
   \text{Improvement} = \frac{T_{\text{standard}}}{T_{\text{spec/token}}} = \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)} \tag{10}
   \]

This formalizes the earlier qualitative claim: for an actual speedup you
need **high \(\alpha\) and low \(c\)** simultaneously. If \(\alpha < c\), you
may see no speedup at all even with a "good" drafter, because verification
overhead from drafting outweighs the savings.

For the minimal case \(\gamma=1\), the improvement simplifies cleanly:
\[
\text{Improvement}\big|_{\gamma=1} = \frac{1-\alpha^2}{(1-\alpha)(c+1)} = \frac{1+\alpha}{1+c}
\]
and if \(\alpha > c\), then \(\frac{1+\alpha}{1+c} > 1\) — a guaranteed
speedup at \(\gamma=1\) whenever the drafter's acceptance rate exceeds its
relative cost.

## Arithmetic Operations (Compute) Overhead

Speculative decoding trades **compute for wall time** — it can use *more*
total FLOPs even while running faster, because \(M_p\) is evaluated on
\(\gamma+1\) positions per iteration (though this costs the same single
memory read / wall-clock as one position, since inference is memory-bandwidth
bound, not compute bound).

Let \(F\) = arithmetic ops for one standard \(M_p\) forward pass, and \(\mu =
F_{M_q}/F_{M_p} = F_{M_q}/F\) (ratio of drafter to target ops per token).

1. Ops per iteration: \(\gamma\) drafter runs (\(\gamma\mu F\)) + \((\gamma+1)\) parallel target runs (\((\gamma+1)F\)):
   \[ F_{\text{iter}} = (\gamma\mu + \gamma + 1) F \]
2. Ops per token (divide by \(E(N)\)):
   \[ F_{\text{spec}} = \frac{(\gamma\mu+\gamma+1)(1-\alpha)}{1-\alpha^{\gamma+1}} \cdot F \]
3. Factor of increase in total ops vs. standard decoding:
   \[ F_{\text{factor}} = \frac{F_{\text{spec}}}{F_{\text{standard}}} = \frac{(1-\alpha)(\gamma\mu+\gamma+1)}{1-\alpha^{\gamma+1}} \tag{11} \]

Low \(\alpha\) (more rejections, more wasted draft compute) increases this
factor. In practice the drafter is usually tiny enough (\(\mu \ll 1\)) that
this overhead is negligible — but the post is explicit that the trade is
"memory reads down, raw FLOPs possibly up," which matters because inference
is memory-bandwidth-bound, not FLOP-bound, so the trade is favorable on
typical hardware.

## Choosing \(\gamma\)

The wall-time improvement \(f(\gamma) = \frac{1-\alpha^{\gamma+1}}{1-\alpha}
\cdot \frac{1}{\gamma c+1}\) has two competing factors: the expected-tokens
term grows (with diminishing returns) as \(\gamma\) increases, while the
relative iteration cost \(\gamma c + 1\) grows linearly with \(\gamma\). When
\(c=0\) there's no penalty and larger \(\gamma\) is always at least as good;
for \(c>0\) there is an interior optimum.

**Worked numeric table** for \(\alpha=0.8\), \(c=0.02\):

| \(\gamma\) | \(E(\text{tokens})\) | Cost \(\gamma c+1\) | \(f(\gamma)\) |
|---|---|---|---|
| 1 | 1.80 | 1.02 | 1.76 |
| 2 | 2.44 | 1.04 | 2.35 |
| 3 | 2.95 | 1.06 | 2.78 |
| 5 | 3.69 | 1.10 | 3.35 |
| 7 | 4.16 | 1.14 | 3.65 |
| 10 | 4.57 | 1.20 | **3.81** (peak) |
| 15 | 4.86 | 1.30 | 3.74 |
| 20 | 4.95 | 1.40 | 3.54 |
| 25 | 4.98 | 1.50 | 3.32 |

Improvement peaks around \(\gamma=10\) for this \((\alpha,c)\) pair, then
declines as drafting overhead dominates.

Qualitative regimes (referencing the post's plot of speedup vs. \(\gamma\)
for varying \(c\)):
- \(c=0\) (free drafter): monotonically increasing speedup with diminishing returns — the theoretical ideal, larger \(\gamma\) always at least as good.
- \(c=0.1\) (expensive drafter): speedup peaks then sharply declines — past the peak, more drafting actively hurts wall time.

### Theoretical Ceiling: the Oracle Bound

Since the true position-level acceptance rate \(\beta\) varies (some tokens
easy to draft, some hard), an oracle that knew \(\beta\) at each position
could choose an ideal \(\gamma\) per step — drafting more when acceptance is
likely, skipping drafting when rejection is near-certain.

With fixed \(\gamma\): \(E(N_{\text{fixed}}) = \sum_{j=0}^{\gamma}\alpha^j =
\frac{1-\alpha^{\gamma+1}}{1-\alpha}\).

Removing the \(\gamma\) cap (oracle, \(\gamma\to\infty\)):
\[ E(N_{\text{oracle}}) = \sum_{j=0}^{\infty}\alpha^j = \frac{1}{1-\alpha} \]

For \(\alpha=0.8\): \(E(N_{\text{oracle}}) = 1/0.2 = 5\) tokens/iteration —
matching the table above, where \(E(N)\) saturates near 5 as \(\gamma\)
grows. This is a **hard ceiling**: no matter how large \(\gamma\) is made,
expected tokens per iteration cannot exceed \(1/(1-\alpha)\). The oracle
itself is unrealizable in practice because knowing \(\beta\) at a position
would require running \(M_p\) — the very computation speculative decoding
is trying to avoid — but it serves as a useful theoretical ceiling for
adaptive-\(\gamma\) schemes.

## Series Roadmap (context for Part 1)

The post frames itself as part 1 of 3:
1. **This post** — fundamentals (algorithm, speculative sampling, acceptance rate math, wall-time/compute trade-offs, choosing \(\gamma\)).
2. **Eagle** — described as "the most commonly used algorithm for speculative decoding" (forthcoming part).
3. **Diffusion-based methods (e.g., DFlash)** — described as "the emerging new cool kid on the block" (forthcoming part).

## References (as cited in the post)

- Leviathan, Kalman, Matias — "Fast Inference from Transformers via Speculative Decoding" — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- "Accelerating Large Language Model Decoding with Speculative Sampling" — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
