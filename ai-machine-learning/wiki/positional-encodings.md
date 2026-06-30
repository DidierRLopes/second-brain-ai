# Positional Encodings

Transformers process all tokens in parallel with no inherent sense of order. Positional encodings inject sequence position information so the model knows which token comes first, second, etc. The choice of positional encoding scheme has major implications for context length, extrapolation, and model quality. The 2026 frontier picture: **RoPE is the universal baseline; long-context production recipes layer YaRN (or its predecessor ABF) on top to extend beyond training length, with NoPE / RNoPE / Partial RoPE as more recent variants that handle long-range retrieval differently**.

## Sinusoidal (Original Transformer)

The original "Attention Is All You Need" paper used fixed sinusoidal functions at different frequencies. These encode absolute position and theoretically allow the model to learn relative positions. In practice, they don't extrapolate well beyond training lengths.

## Absolute Position Embeddings

Some models (GPT-2, early BERT) learn position embeddings as trainable lookup tables — index → vector added to token embedding. Simple but **limited to the maximum sequence length seen during training**.

## Rotary Position Embeddings (RoPE)

RoPE ([Su et al., 2021 / 2104.09864](../../papers/02-architecture/attention-variants/RoFormer: Enhanced Transformer with Rotary Position Embedding - 2104.09864.pdf)) is the dominant modern approach, adopted by LLaMA, PaLM, GPT-NeoX, Mistral, and essentially every frontier-2026 model. It encodes position information by **rotating query and key vectors in 2D planes**.

Mechanism: based on the dimensionality of Q/K, RoPE splits each into pairs (they rotate in 2D space) and rotates each pair by an angle proportional to the absolute position and a base frequency `θᵢ = 10000^(-2i/d)`. During attention, the dot product between the rotated Q and K vectors directly encodes relative distance via the phase difference — tokens `n` positions apart always maintain the same angular relationship: `q_m^T k_n = x_m^T W_q^T R_{Θ, n-m}^d W_k x_n`.

The paper derives RoPE constructively from a single constraint — that the inner product of position-encoded query and key should depend only on the *relative* position `m - n` and the original embeddings. In the 2D case, the unique solution is multiplication by `e^{imθ}` (a rotation). This generalizes to `d/2` independent 2D rotations across the head dimension. Three desirable properties fall out for free:

- **Multiplicative, not additive**: positional information enters via a rotation `R_{Θ,m}^d W x` rather than `W(x + p)`, preserving the norm of hidden representations.
- **Long-term decay**: by setting `θᵢ = 10000^{-2i/d}`, the upper bound on `|Σ q_{[2i:2i+1]} k_{[2i:2i+1]}^* e^{i(m-n)θᵢ}|` decays monotonically with `|m - n|` — tokens far apart naturally interact less. The paper shows this decay empirically over relative distances 0 to 250.
- **Compatible with linear attention**: since RoPE is a norm-preserving rotation, it can be applied to the non-negative feature maps `φ(q), φ(k)` of linear attention without breaking the kernel structure.

In practice, the matrix multiplication form is sparse, so an efficient realization avoids materializing `R_Θ^d` and instead applies the rotation as `R x = x ⊙ cos(mθ) + rotate_half(x) ⊙ sin(mθ)`. RoFormer (RoPE + transformer) achieved 27.5 BLEU on WMT14 EN-DE vs 27.3 for the baseline, with **faster convergence** during MLM pre-training.

RoPE's key advantage is extensibility: the rotation framework lets you scale to longer contexts by modifying base frequencies (ABF, YaRN below).

### RoPE, Block-Matrix Form and Implementation

Written out explicitly as block-diagonal 2×2 rotations rather than the complex-exponential form above: at position `m`, dimension pair `i` (indices `(2i, 2i+1)` of the head dimension `H`) is rotated by angle `mθᵢ`, with `θᵢ = Θ^{-2i/H}` (`Θ` the base frequency hyperparameter):

```
R_m = blockdiag(..., R_m^(i), ...) ∈ R^(H×H),   R_m^(i) = [[cos(mθᵢ), -sin(mθᵢ)],
                                                            [sin(mθᵢ),  cos(mθᵢ)]]
q_m ← R_m q_m,   k_m ← R_m k_m
```

This is the same rotation as the `e^{imθ}` form above, just written as a real 2×2 matrix per dimension pair instead of a complex multiplication.

**Caching**: precompute `cos(mθᵢ)`/`sin(mθᵢ)` for every `(position, index)` pair once at initialization, rather than recomputing per forward pass:

```python
positions = torch.arange(max_seq_len, device=device)              # shape (max_seq_len)
thetas = self.theta ** (-torch.arange(0, d_k, 2, device=device) / d_k)  # shape (d_k // 2)
angles = positions.unsqueeze(-1) * thetas.unsqueeze(0)
```

**Avoiding many small 2×2 matmuls**: reshape the head dimension into `(H/2, 2)` to extract even/odd elements, apply the rotation as elementwise multiplies, then re-interleave:

```python
x_pairs = x.reshape(*x.shape[:-1], -1, 2)
x_even = x_pairs[..., 0]
x_odd = x_pairs[..., 1]

x_out_even = x_even * cos - x_odd * sin
x_out_odd  = x_even * sin + x_odd * cos

# torch.stack() adds a new dim; flatten re-interleaves even/odd
rotated = torch.stack([x_out_even, x_out_odd], dim=-1).flatten(start_dim=-2)
```

## Extending RoPE to Long Context

During pre-training, models train on shorter contexts (quadratic attention is expensive; short contexts learn short-range correlations). At inference, rotation angles grow with sequence length:

```
angle(i, dim_k) = i / base^(2k/d)
```

For sequences longer than training, angles exceed what the model has seen. Two main fixes:

- **ABF (Adaptive Base Frequency)**: increase the base frequency as the sequence length increases. Conceptually simple, widely used as a first step. The [Llama Long paper (2309.16039)](../../papers/04-efficiency/context-extension/Effective Long-Context Scaling of Foundation Models - 2309.16039.pdf) showed that bumping Llama 2's RoPE base from 10,000 → 500,000 (32k context) substantially **reduces the decay on attention scores for distant tokens** and is the only variant of {RoPE, PI, ABF, xPos ABF} that maintains performance to the full 32k window on a sentence-retrieval task; it also matches or beats Position Interpolation on standard short-context benchmarks (HumanEval, MMLU, HellaSwag) without degradation.
- **Position Interpolation (PI)**: linearly scale token positions back into the training range. Works to ~8× extension but degrades after fine-tuning because it removes high-frequency components.
- **NTK-aware**: scale the base such that high frequencies are interpolated less and low frequencies more — better for non-finetuned models, but the optimal base must be searched empirically.
- **YaRN**: more granular interpolation of frequencies on different components, plus dynamic attention scaling and temperature adjustment. **Does best for extremely long contexts**. In gpt-oss-120b, YaRN was used to extend dense-layer context to 131k tokens.

### YaRN (Yet another RoPE extensioN)

[YaRN (2309.00071)](../../papers/04-efficiency/context-extension/YARN: Efficient Context Window Extension of Large Language Models - 2309.00071.pdf) is the production-grade RoPE extension recipe. It combines two techniques the authors call **NTK-by-parts interpolation** and **attention temperature scaling**:

1. **NTK-by-parts**: classify each RoPE dimension by the ratio `r(d) = L / λ_d` (training context over wavelength). For dimensions where `r < α` (wavelength ≫ training context, so absolute positions matter), interpolate linearly. For dimensions where `r > β` (wavelength ≪ training context, only relative info), do not interpolate. Between, use a ramp. For Llama, `α = 1, β = 32` work well. This preserves the dimensions that carry useful relative-position information while only stretching the ones that would otherwise extrapolate out of distribution.
2. **Attention temperature scaling**: introduce a temperature `t` on the softmax — `softmax(q^T k / (t · √d))`. Implemented for free by scaling `q` and `k` by `√(1/t)` once at the RoPE step (zero overhead at inference). The empirical fit is `√(1/t) = 0.1 ln(s) + 1` where `s = L'/L` is the scale factor. This pseudo-universal constant fit Llama 7B/13B/33B/65B and Llama 2 (7B/13B/70B) with the same formula.

YaRN requires **~10× fewer tokens and 2.5× fewer training steps** than Position Interpolation to reach the same perplexity. On 128k extension of Llama 2 7B/13B (400 + 200 fine-tune steps total), it preserves short-context benchmarks within 0.5% while reducing perplexity at 128k from PI's ~50+ to ~2.4. With **Dynamic YaRN** (the inference-time scale adapts to the actual sequence length), models can extend ~2× beyond their fine-tuned context without further training.

**SmolLM3's progression**: 4k → 32k → 64k → 128k. 4k → 32k uses RoPE ABF with base 2M; 32k → 64k bumps to 5M. Base 10M improved RULER slightly but hurt GSM8k, so was disregarded. To reach 128k, **YaRN from the 64k checkpoint beat a four-fold extension from 32k** — confirming the hypothesis that training closer to the desired inference length improves performance.

**Kimi K2**: 400B tokens at 4k with LR decay 2e-5 → 7e-6, then 60B at 32k. Used YaRN to extend to 128k.

### Llama Long: training-data findings

The [Llama Long paper (2309.16039)](../../papers/04-efficiency/context-extension/Effective Long-Context Scaling of Foundation Models - 2309.16039.pdf) made two non-obvious empirical findings worth flagging here, because they contradict common intuition:

- **Long-text data is not the key.** Removing long-text data from the pre-train mix and continually pre-training on mostly-short data still recovers most of the long-context performance gain. The benefit primarily comes from data *quality*, not data length distribution. This was confirmed on NarrativeQA, Qasper, QuALITY, QMSum.
- **Continual pre-training beats from-scratch.** Starting from a short-context Llama 2 checkpoint and continually pre-training at 32k for 400B tokens **matches** training from scratch at 32k while saving ~40% FLOPs. Two-stage curricula (4k → 32k @ 20%/40%/80% of training) all converge to similar quality. The implication: there's no need to commit to long-context training from day one.

Architectural change was minimal — just decreasing RoPE's decay effect by bumping the base from 10k to 500k. Validation perplexity drops from 6.548 → 6.323 (Books) and 6.816 → 6.780 (CC) with this single change.

### ProLong / CEPED: long-context training recipe

[ProLong (2410.02660)](<../../papers/04-efficiency/context-extension/How to Train Long-Context Language Models (Effectively) - 2410.02660.pdf>) studies *how* to do continued long-context training effectively, with several actionable findings:

- **Code repositories and books are the best long-data sources.** A 1:1 mix of books and code repos outperforms ArXiv, CommonCrawl, or any single source on HELMET. ProLong's final mix is 30% code repos, 30% books, 3% textbooks, 37% "ShortMix" (FineWeb-Edu/FineWeb/Wikipedia/Tulu/StackExchange/ArXiv/OpenWebMath).
- **Training only on long data hurts.** More long data initially helps long-context tasks, but past ~60% long data, both long and short performance regress. Best long/short ratio: **60/40**.
- **Train longer than your evaluation length.** Training at 512k tokens improves 64k-eval performance over training at 64k — counterintuitive but consistent across recall/RAG/re-rank/ICL.
- **SFT with short-context instruction data is enough.** Adding synthetic long-context SFT (1–50%) did *not* help on the very long-context tasks and even hurt at higher ratios. UltraChat (1.2k token average) alone produced strong long-context behavior post-SFT.
- **Perplexity is a misleading metric.** Long-context PPL keeps improving as more long data is added, but downstream long-context scores *decrease* — measure on real tasks.
- **Disable cross-document attention** during continued training; it improves both short and long-context perf.

Final ProLong-8B (Llama-3-8B-init, 40B token budget, RoPE base 8×10^6 → 1.28×10^8) reaches the best 10B-scale long-context performance using only **5% of Llama-3.1's long-context budget** — and supports up to 512K tokens.

### DeepSeek-LLM / "Longtermism"

[DeepSeek LLM (2401.02954)](../../papers/04-efficiency/context-extension/DeepSeek LLM: Scaling Open-Source Language Models with Longtermism - 2401.02954.pdf) is more about training scaling laws than positional encoding per se, but it confirms the standard recipe: GQA (8 KV heads at 67B), Pre-Norm + RMSNorm, SwiGLU, RoPE, multi-step LR schedule. Its main long-context takeaway is that **multi-step LR schedules let you resume training and extend context cheaply** — important if you're doing 4k → 32k → 128k staged extension.

## NoPE (No Position Embedding)

Uses only causal masking and attention patterns — no rotation, no addition. Doesn't bump into the issue of extrapolating beyond training lengths, but **shows weaker performance on short-context reasoning and knowledge-based tasks**.

[Kazemnejad et al. 2023 (2305.19466)](../../papers/02-architecture/attention-variants/The Impact of Positional Encoding on Length Generalization in Transformers - 2305.19466.pdf) provided the surprising empirical and theoretical case for NoPE in decoder-only Transformers: on a battery of 10 reasoning/algorithmic tasks (copy, reverse, addition, polynomial eval, sorting, summation, parity, LEGO, SCAN, PCFG), NoPE **outperformed all explicit positional encodings** at *length generalization* — i.e., training at length L and testing at lengths > L. Mean reciprocal rank: NoPE 0.69, T5 Relative 0.55, ALiBi 0.50, Rotary 0.33, Absolute 0.22. The theoretical claim is that NoPE can represent both absolute and relative positions (Theorem 1: the first layer can recover absolute positions in its hidden state; Theorem 2: subsequent layers can implement relative encoding), but **in practice it learns to use relative encoding similar to T5's Relative Bias** — confirmed via Jensen–Shannon divergence between NoPE attention patterns and T5/Rotary/APE patterns.

Important caveats: the study trained from scratch at small scale (107M params, scaled up to 1B). At in-distribution lengths all encodings perform similarly; the gap shows up only at extrapolation. Scratchpad/CoT helps but is task-dependent — not a substitute for a good PE choice. ALiBi shows pure recency bias (attention distance peaks near 0); APE/Rotary show no clear preference; **NoPE and T5's Relative are bimodal** — attending to both nearby and far positions, which the authors argue is the more useful pattern.

## RNoPE (Hybrid RoPE + NoPE)

Alternates applying RoPE and NoPE on attention blocks: RoPE handles local context, NoPE helps with longer-range information retrieval. Adopted by **SmolLM3** (NoPE every 4th layer) and **Llama 4** (paired with chunked attention).

The mechanism behind RNoPE was clarified in [Cohere's "Rope to Nope and Back Again" (2501.18795)](../../papers/02-architecture/attention-variants/Rope to Nope and Back Again: A New Hybrid Position Encoding for Efficient Context Scaling - 2501.18795.pdf), which pretrained 8B-parameter models for 750B tokens (then 5T for the final architecture) with three variants — RoPE, QK-Norm, NoPE — and analyzed attention masses on Begin/Needle/Context/End segments of a NIAH task.

Key findings:
- **Pure NoPE has strongest retrieval** (highest attention mass on the needle segment) but weakest base capabilities.
- **Pure RoPE has strong recency bias** (high "End" mass) but weak long-context retrieval.
- **QK-Norm hurts long context** — it flattens the magnitude information from Q·K, producing weak attention sinks and high susceptibility to distractor noise. In their NIAH-65k, QK-Norm scored 7.93 vs RoPE 9.82 and NoPE 9.03.
- **In RNoPE (alternating layers), a spontaneous "division of labor" emerges**: NoPE layers specialize in long-range retrieval (high needle attention mass, low recency bias), while RoPE layers focus on local context aggregation (strong attention sinks at Begin, strong recency bias at End). No explicit training constraint enforces this — it emerges naturally.
- **Surprise about RoPE θ**: in pure-RoPE setups, increasing θ helps long context. In RNoPE, **increasing θ during fine-tuning hurts** — the larger receptive field of RoPE layers introduces noise that disrupts NoPE layers' retrieval. The needle eval drops from 8.04 (θ=10k) to 6.20 (θ=4M).

This motivates **RNoPE-SWA**: NoPE layers keep full attention; RoPE layers get sliding-window attention (window 4,096) — a 1:3 ratio of full to sliding gave the best trade-off. Empirical result: RULER retrieval drops from 96.6 → 57.1 between 8k and 256k for the RoPE baseline, but only 96.1 → 74.8 for RNoPE-SWA. Training is **~50% faster at 64k and 2× faster at 128k** with up to 75% KV-cache reduction. Aligns with concurrent designs in YoCo, Jamba-1.5, and MiniMax-01.

Hugging Face's SmolLM3 ablations compared RoPE, RNoPE, and RNoPE + [[attention-variants|document masking]]:
- All achieve similar performance on short-context tasks.
- RNoPE + document masking is the **foundation for long-context handling**.

## Partial RoPE

Applies RoPE / NoPE within the *same* layer (rather than alternating across layers). A less common variant explored in recent work.

## ALiBi (Attention with Linear Biases)

ALiBi (Press et al., 2022) takes a different approach: instead of adding positional information to embeddings, it applies linear position-dependent penalties directly to attention scores. Trained on 1024 tokens, a model can extrapolate to 2048, and trains 11% faster with 11% less memory than sinusoidal. Less common in frontier-2026 models compared to RoPE variants.

CS224n's self-attention note (Hewitt, Stanford, 2023 draft) frames sinusoidal/learned position embeddings and ALiBi as the only two possible fixes to a concrete problem it proves directly: **self-attention by itself is provably order-invariant.** Worked example: "the oven cooked the bread so" vs. "the bread cooked the oven so" have different meanings, yet for the word "so," `α_{so,0} = exp(q_so^T k_the) / (exp(q_so^T k_the) + ... + exp(q_so^T k_bread))` is computed from a sum whose terms merely get reordered when the sentence is reordered — so every `α_{so,*}` weight is identical regardless of word order. Root cause given as two independent facts: non-contextual token embeddings `x_i = Ew_i` depend only on word identity (not position), and the attention operation itself has no positional dependence built in.

The note frames the two fixes as exhaustive: **"(1) use vectors that are already position-dependent as inputs, or (2) change the self-attention operation itself."** Learned additive position embeddings (option 1, the approach in BERT) add a learned `P_i ∈ R^d` per position to `x_i` before attention runs. ALiBi (option 2) instead modifies the attention scores directly, with the formula given as:

`α_i = softmax(k_{1:n} q_i + [-i, ..., -1, 0, -1, ..., -(n-i)])`

i.e., add a bias vector that linearly penalizes attending to tokens farther away from position `i` (in either direction), applied on top of the raw dot-product scores `k_{1:n}q_i ∈ R^n` — no learned parameters, just a fixed distance penalty. The note's own editorial reaction to this working as well as it does: "it's odd that this works; but interesting!"

## Where Positional Encoding Sits in the Stack

Positional encoding choices are **orthogonal but interacting** with [[attention-variants|attention pattern choices]]:

- **Positional encoding scaling** (ABF, YaRN) changes how position information is encoded.
- **Attention patterns** (chunked, sliding window, DCA, interleaved local/global) change which tokens can attend to which.

For long-context production: choose a positional encoding approach (RoPE + YaRN, or RNoPE + YaRN) and combine with a suitable attention pattern (full, SWA, DCA) based on your context length and compute constraints.

## Related Topics

- [[transformer-architecture]] — The core architecture these encodings augment; CS224n's RNN-motivation and minimal-self-attention sections explain why position representations are needed at all
- [[long-context-training]] — How RoPE/YaRN/RNoPE choices become full long-context training recipes
- [[attention-variants]] — long-context attention patterns (SWA, chunked, DCA, interleaved); document masking
- [[hybrid-architectures]] — linear-attention alternatives that handle long context structurally
- [[inference-optimization]] — Context length extension is critical for inference
- [[frontier-training-playbook]] — where positional encoding sits in the architecture decision tree

## Sources

- [RoFormer: Enhanced Transformer with Rotary Position Embedding (2104.09864)](../../papers/02-architecture/attention-variants/RoFormer: Enhanced Transformer with Rotary Position Embedding - 2104.09864.pdf) — RoPE construction, long-term decay, linear-attention compatibility.
- Attention Is All You Need (arxiv:1706.03762)
- [YaRN: Efficient Context Window Extension of Large Language Models (2309.00071)](../../papers/04-efficiency/context-extension/YARN: Efficient Context Window Extension of Large Language Models - 2309.00071.pdf) — NTK-by-parts + attention temperature, Dynamic YaRN.
- [Effective Long-Context Scaling of Foundation Models / Llama Long (2309.16039)](../../papers/04-efficiency/context-extension/Effective Long-Context Scaling of Foundation Models - 2309.16039.pdf) — ABF base 10k→500k; data quality > data length; continual pre-training matches from-scratch at ~40% fewer FLOPs.
- [How to Train Long-Context Language Models (Effectively) / ProLong (2410.02660)](<../../papers/04-efficiency/context-extension/How to Train Long-Context Language Models (Effectively) - 2410.02660.pdf>) — code repos + books, 60/40 long/short ratio, train longer than eval length, short SFT is enough.
- Alisa Liu, "Book of LLMs" (Notion, alisawuffles.notion.site/alisa-s-book-of-llms) — the explicit block-diagonal 2×2 rotation-matrix formulation of RoPE and the cos/sin-caching + even/odd-interleave PyTorch implementation. See [`raw/alisa-liu-book-of-llms.md`](../raw/alisa-liu-book-of-llms.md).
- [The Impact of Positional Encoding on Length Generalization in Transformers (2305.19466)](../../papers/02-architecture/attention-variants/The Impact of Positional Encoding on Length Generalization in Transformers - 2305.19466.pdf) — NoPE beats all explicit PEs at length generalization in decoder-only LMs.
- [Rope to Nope and Back Again (2501.18795)](../../papers/02-architecture/attention-variants/Rope to Nope and Back Again: A New Hybrid Position Encoding for Efficient Context Scaling - 2501.18795.pdf) — RNoPE-SWA, division-of-labor mechanism, QK-Norm hurts long context.
- [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism (2401.02954)](../../papers/04-efficiency/context-extension/DeepSeek LLM: Scaling Open-Source Language Models with Longtermism - 2401.02954.pdf) — multi-step LR schedule that supports staged context extension.
- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- SmolLM3 report (RNoPE adoption, 4k → 128k stage progression). See `raw/smollm3-hugging-face-report.md`.
- [CS224n: Self-Attention & Transformers (Hewitt, Stanford, 2023 draft)](../raw/cs224n-self-attention-transformers.md) — source: https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf — order-invariance proof for self-attention, the two-options framing (position-dependent inputs vs. modifying attention), and the explicit ALiBi bias-vector formula.
