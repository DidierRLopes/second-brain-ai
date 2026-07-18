# Transformer Architecture

The Transformer is the foundational neural network architecture behind all modern large language models. Introduced in "Attention Is All You Need" (Vaswani et al., 2017), it replaced recurrence and convolutions with a purely attention-based mechanism, enabling massive parallelization and superior performance on sequence tasks.

## Core Design

The original Transformer uses an encoder-decoder structure. The encoder maps an input sequence to a continuous representation, and the decoder generates an output sequence one token at a time, attending to the encoder's output. The key innovation is **multi-head self-attention**, which allows every position in a sequence to attend to every other position simultaneously.

**Scaled dot-product attention** computes: Attention(Q, K, V) = softmax(QK^T / √d_k)V, where Q, K, V are query, key, and value matrices derived from the input. Multi-head attention runs this computation in parallel across multiple "heads," each learning different attention patterns.

Other critical components include layer normalization, residual connections, and position-wise feed-forward networks (two linear transformations with a ReLU activation between them).

## Why Self-Attention Replaced RNNs

CS224n's self-attention note (Hewitt, Stanford, 2023 draft) frames the shift away from RNNs around two specific failure modes of the vanilla RNN recurrence `h_t = σ(W h_{t-1} + U x_t)`. **(1) Parallelization:** GPUs are fast at computing `AB` for independent rows of `A`, but `h_2` cannot be computed before `h_1` is known — the unrolled form `h_2 = σ(W σ(W h_0 + U x_1) + U x_2)` is an inherently serial dependency chain, so an RNN over a length-n sequence requires n sequential steps before the final state is available. **(2) Linear interaction distance:** in a sentence like "The chef who ran out of blackberries and went to the stores is ___", the number of matrix multiplies separating "chef" from the blank scales with the number of intervening words, making it harder for the network to retain "chef" in its hidden state by the time it must agree with it. Self-attention fixes both at once: attention distance between any two positions is O(1) rather than O(n), and `x_{1:n}Q`, `x_{1:n}K`, `x_{1:n}V` are each single matmuls computed for the whole sequence at once, with no serial bottleneck. The note traces this lineage through Bahdanau et al. (2014) attention in machine translation — originally a mechanism for a decoder to look back at a source sequence once per output token — reframed as "an entire replacement for recurrent neural networks just based on attention."

### Building Self-Attention from First Principles

The note defines attention generally as "a method for taking a query, and softly looking up information in a key-value store by picking the value(s) of the key(s) most like the query," averaged over all values and weighted toward values whose keys best match the query. **Self-attention** is the special case where the same elements define the queries, keys, and values: for token `x_i`, query `q_i = Qx_i`, key `k_j = Kx_j`, value `v_j = Vx_j` (each projection `∈ R^{d×d}`), giving contextual representation `h_i = Σ_j α_{ij} v_j` with `α_{ij} = softmax_j(q_i^T k_j)`. The intuition for using three separate projections rather than one is that `Q`, `K`, `V` let the same input vector `x_i` be re-projected into different "views" appropriate to its different roles (asking a question, being matched against, and being retrieved).

A minimal self-attention architecture needs four components beyond the core operation: **position representations** (attention alone is order-invariant — see below), an **elementwise nonlinearity**, **future masking** for autoregressive use, and the attention operation itself. The nonlinearity requirement has a clean algebraic justification: stacking two self-attention layers without a nonlinearity between them collapses to a single linear self-attention layer, since `o_i = Σ_j α_{ij}^{(2)} V^{(2)} (Σ_k α_{jk}^{(1)} V^{(1)} x_k) = Σ_k α*_{ik} V* x_k` for `V* = V^{(2)}V^{(1)}` and some combined weights `α*`. This motivates inserting a feed-forward sublayer between attention layers: `h_FF = W_2 ReLU(W_1 h + b_1) + b_2`, conventionally with `W_1 ∈ R^{5d×d}` (FFN hidden dim ≈ 5x model dim in this note's convention — compare GPT-2's 4x ratio noted above), justified simply because matmuls are cheap to parallelize, making the FFN an efficient place to put extra parameters and compute.

**Future masking implementation detail:** rather than zeroing `α_{ij}` for `j > i` directly, masking is implemented by adding a large negative constant to the pre-softmax score for disallowed positions. The note explicitly warns against using `-∞` for this constant — infinite values can produce NaNs and library behavior on infinite inputs is inconsistently defined across frameworks — recommending instead a finite constant safely within float16 range, e.g. `-10^5`, which still drives the post-softmax weight to exactly zero at finite precision.

### Multi-Head Attention: Why Heads Use Reduced Dimension d/k

Beyond the "multiple views in parallel" intuition, the note gives an implementation-level reason multi-head attention projects each head down to dimension `d/k` (for `k` heads, model dim `d`) rather than keeping each head at full dimension `d`: `x_{1:n}Q`, `x_{1:n}K`, `x_{1:n}V` (each `∈ R^{n×d}`) are computed once for the whole sequence, then reshaped from `R^{n,d}` into `R^{n,k,d/k}` and transposed to `R^{k,n,d/k}` — treating the head axis as an extra batch dimension so the batched softmax runs all heads in parallel. The practical consequence: "multi-head self-attention is no more expensive than single-head due to the low-rankness of the transformations" — total compute matches single-head attention at full dimension `d`, just factored across `k` lower-rank heads, with only the final output projection `O ∈ R^{d×d}` (applied to the concatenation `[v_i^{(1)}; ...; v_i^{(k)}]`) added on top.

### Layer Norm and Pre-Norm/Post-Norm

The note's presentation of LayerNorm (Ba et al., 2016) deliberately **omits the learned elementwise affine (gain/bias) parameters**, citing Xu et al. (2019) that this component "seems not to be crucial, and may even be harmful." It also flags a second, less obvious rationale beyond "reducing uninformative activation variation": per Xu et al. (2019), layer norm's real benefit may lie more in improving **backward-pass gradients** than in its forward-pass smoothing effect. Pre-norm and post-norm are given as explicit competing formulas — pre-norm: `h = f(LN(h)) + h`; post-norm: `h = LN(f(h) + h)` — with Xiong et al. (2020) cited as the resolution: pre-norm gradients are substantially better-behaved at initialization, which is why pre-norm training is faster and is now preferred.

### Cross-Attention, Made Explicit

For the encoder-decoder variant, the note gives cross-attention's query/key/value sourcing explicitly: queries come from the decoder's intermediate representations (`q_i = Q h_i^{(y)}`), while keys and values come from the **encoder's final output** (`k_j = K h_j^{(x)}`, `v_j = V h_j^{(x)}`) — and only the last encoder block's output is attended to, not per-layer encoder representations. The note's own framing: cross-attention "isn't that just what attention always was before we got into this self-attention business? Yeah, pretty much" — i.e., cross-attention is the original Bahdanau-style attention mechanism, with self-attention being the newer, narrower special case. Citing Raffel et al. (2020) (T5), encoder-decoders can outperform decoder-only models "at modest scale" by giving bidirectional context over the source while still generating autoregressively, but this splits parameters between encoder and decoder — which the note gives as the reason "most of the largest Transformers are decoder-only."

## Key Variants

**Encoder-only (BERT):** Bidirectional attention over the full input, pre-trained with Masked Language Modeling. Excels at understanding tasks (classification, NER, QA). Considers both left and right context at all layers, unlike autoregressive models.

**Decoder-only (GPT):** Causal (left-to-right) attention mask prevents attending to future tokens. Pre-trained via next-token prediction. This is the dominant architecture for modern LLMs (GPT-4, Claude, LLaMA, Mistral).

**Encoder-decoder (T5):** Unified text-to-text framework where all NLP tasks are cast as text generation. T5 showed the power of a single architecture across diverse tasks.

## GPT-2: A Concrete Decoder-Only Case Study

[The Illustrated GPT-2](../raw/illustrated-gpt2-alammar.md) (Jay Alammar, Aug 12, 2019) traces the architecture of OpenAI's actual released model, putting concrete numbers on the abstract "decoder-only (GPT)" variant described above. GPT-2 stacks **transformer decoder blocks only** — no encoder, and no second (cross-attention) sublayer either. That lineage traces back through "Generating Wikipedia by Summarizing Long Sequences" (arXiv 1801.10198), which first dropped the encoder-attention sublayer from the original decoder block to create what Alammar calls the "Transformer-Decoder": 6 such blocks with a 4,000-token context, trained to summarize Wikipedia articles. GPT-2 scales this design up; the smallest released configuration uses 12 attention heads, 12 decoder layers, 768-dimensional embeddings, and a 1024-token context window.

Several mechanics are worth recording precisely. **Auto-regression**: each generated token is appended to the running input sequence and fed back in for the next forward pass; the model never re-interprets earlier tokens in light of later ones, unlike BERT's bidirectional encoder. **Masked self-attention** is implemented as an attention mask rather than input corruption — GPT-2 sets the scores for future positions to roughly −1 billion before the softmax, so they contribute ~0 weight, instead of replacing future tokens with a `[MASK]` token the way BERT's MLM objective does. **KV reuse during generation**: GPT-2 "holds on to the key and value vectors" it computes for already-processed tokens at every decoder layer and reuses them on each subsequent step rather than recomputing — an explicit, pre-formal description of what the field now formally calls the [[kv-cache]]. **Top-k sampling**: rather than always emitting the single highest-probability token (top-k=1, prone to repetitive loops), GPT-2 exposes a top-k parameter — commonly set to 40 — that samples from the k highest-scoring vocabulary entries. **FFN sizing**: the feed-forward sublayer's hidden dimension is 4× the model dimension (768→3072→768 for the small model), the same ratio the original Transformer used (512→2048). **Positional encoding**: a trained matrix with one vector per of the 1024 context positions, rather than the original paper's fixed sinusoidal scheme. **Training data**: a 40GB "WebText" corpus OpenAI crawled from the internet; tokens are BPE units, "usually parts of words," not whole words. Alammar also flags an unresolved discrepancy: his own parameter tally for GPT-2-small sums to 124M, not the 117M implied by the model's public name.

The post closes with three transfer applications for the decoder-only stack beyond language modeling: machine translation without an encoder; summarization — the original task the Transformer-Decoder was trained for; transfer learning via pretrain-then-finetune-on-summarization (Sample Efficient Text Summarization Using a Single Pre-Trained Transformer, arXiv 1905.08836); and music generation via the Music Transformer, which one-hot-encodes notes plus velocity (how hard a key is struck) in place of word tokens.

## Tracing Tensor Shapes Through a Decoder

[Mastering Tensor Dimensions in Transformers](../raw/mastering-tensor-dimensions-transformers.md) (Not Lain, Hugging Face, Jan 12, 2025) makes the decoder-only stack concrete by following one tensor's **shape** end-to-end under the convention `[batch, seq_len, embed_dim]`. The running example `Hello world !` (plus `<bos>`/`<eos>`) starts as `[1, 4]` and the embedding layer lifts it to `[1, 4, 768]`; positional encoding then injects order *without changing the shape*, which is necessary precisely because the downstream attention runs in parallel across positions.

The instructive part is the multi-head attention bookkeeping. Three `nn.Linear(768, 768)` projections produce Q, K, V at `[1, 4, 768]`; the embedding dimension factors as `768 = 8 × 96` (8 heads × head size 96) and a transpose lands Q/K/V at `[1, 8, 4, 96]`. Then `QKᵀ = [1, 8, 4, 96] × [1, 8, 96, 4] = [1, 8, 4, 4]`, `softmax(QKᵀ/√d_k)·V = [1, 8, 4, 96]`, and a concat-plus-output-projection restores `[1, 4, 768]`. The recurring theme is **shape invariance through a block**: positional encoding, add-and-normalize, the feed-forward net (which expands `768 → 3×768 → 768`), and the output projection all return to `[1, 4, 768]`, while attention only reshapes temporarily for the matmul. That invariance is exactly what lets decoder layers stack arbitrarily deep and lets the residual/add-and-normalize pattern work. The only deliberate shape *changes* sit at the boundaries — embedding (`[1, 4] → [1, 4, 768]`) and the LM head (`[1, 4, 768] → [1, 4, vocab]`, e.g. `[1, 4, 9735]`). For the encoder-decoder case, the post works a translation example where cross-attention draws **K and V from the encoder and Q from the decoder**, reconciling a source length of 4 with a target length of 6 inside `QKᵀ` (`[1, 8, 6, 96] × [1, 8, 96, 4] = [1, 8, 6, 4]`) and returning to `[1, 6, 768]`. This is the background the [[kv-cache]] note's prerequisites assume — in particular that the attention-weight tensor has shape `[batch, h, seq_len, seq_len]`.

## The Modern Transformer LM, Equation by Equation

This section walks the full decoder-only forward pass at the equation level, following Alisa Liu's "Book of LLMs" notes. **Notation note**: this source uses its own symbol table, which collides with the Austin-et-al. notation already used above (`H` here means *head dimension*, not number of heads as in the FLOPs/Params section below) — treat the two sections as separate, internally-consistent conventions rather than a single merged notation.

| Symbol | Meaning |
|---|---|
| B | batch size (sequences) |
| L | number of layers |
| T | sequence length (tokens to generate) |
| S | sequence length (provided context) |
| V | vocab size |
| D | hidden dimension |
| H | head dimension |
| F | MLP hidden dimension (generally `F = 4D`, or `8D/3` for SwiGLU) |
| N | number of query heads (`N·H = D`) |
| K | number of key/value heads (`K < N` under GQA) |
| G | GQA group size, `G = N // K` |

**Token embedding**: `W_e ∈ R^(V×D)`, initial hidden state `X⁽⁰⁾ ∈ R^(B×S×D) = W_e[tokens]`.

For each layer `ℓ ∈ [0, ..., L-1]`:

*Pre-attention RMSNorm* — divide by the RMS of the hidden state, then apply a learned rescaling `γ`:

```
RMS(X) = sqrt((1/D) Σᵢ xᵢ²)
X̄⁽ˡ⁾ = X⁽ˡ⁾ / (RMS(X⁽ˡ⁾) + ε) ⊙ γ⁽ˡ⁾_attn
```

*Attention projections*, GQA-aware: `W_Q ∈ R^(D×D)`, `W_K, W_V ∈ R^(D×KH)` (note `KH ≤ D` under GQA, vs. the full `D` for queries):

```
Q = X̄ W_Q ∈ R^(B×T×D),   K = X̄ W_K ∈ R^(B×S×D),   V = X̄ W_V ∈ R^(B×S×D)
```

*[optional] QK-norm*: RMSNorm applied to the query/key vectors themselves, to control the magnitude entering the dot product (see [[training-stability]] for why this matters at scale).

Reshape to expose the head axis (`D → N×H` for Q; `K·H → K×H` for K/V), transpose sequence and head-count dims, then expand K/V across each GQA group so every query head has a matching K/V head:

```
Q: (B,T,D) → (B,N,T,H)        K,V: (B,S,K·H) → (B,K,S,H) → (B,N,S,H)  [expanded for GQA]
```

*RoPE*: rotate Q/K by position — see [[positional-encodings]] for the full `R_m^(i)` block-rotation derivation and caching implementation.

*Attention scores and causal mask*: scale by `1/√H` (unscaled dot products grow with `√H`, pushing softmax into a peaky, hard-to-update regime), mask out future positions, then softmax:

```
A = QKᵗ / √H ∈ R^(B×N×T×S)
A[i,j] ← A[i,j] if j ≤ i else -∞
A = softmax(A)   # row-wise over the S dimension
```

Weighted sum of values, reshape back to `(B,T,D)`, and an output projection `W_O ∈ R^(D×D)` mixes information across heads: `O = AV`, `O_proj = reshape(O)·W_O`. Residual: `X⁽ˡ⁾ ← X⁽ˡ⁾ + O_proj`.

*Feed-forward block*: a second RMSNorm, then SwiGLU. Gate/up projections `W_up, W_gate ∈ R^(D×F)`:

```
X̄⁽ˡ⁾ = X⁽ˡ⁾/(RMS(X⁽ˡ⁾)+ε) ⊙ γ⁽ˡ⁾_ffn
U = X̄ W_up,   G = X̄ W_gate
H_ffn = (G ⊙ σ(G)) ⊙ U                  # SwiGLU = Swish(gate) ⊙ up
F_out = H_ffn · W_down                  # W_down ∈ R^(F×D)
X⁽ˡ⁺¹⁾ = X⁽ˡ⁾ + F_out                   # residual
```

**Final norm and unembedding**: `X_final = X⁽ᴸ⁾/(RMS(X⁽ᴸ⁾)+ε) ⊙ γ_final`, then project to vocab logits via `W_u ∈ R^(D×V)`: `Z = X_final · W_u ∈ R^(B×T×V)`.

### Implementation Notes

Causal masking in code uses `scores.masked_fill(~mask, -torch.inf)`, where `mask` is `True` for positions that *can* be attended to. A full attention block, with the reshape/transpose bookkeeping for multi-head attention made explicit:

```python
batch_size, seq_len, _ = x.shape

x_norm = self.norm(x)

qkv = self.qkv_proj(x_norm)  # (batch, seq_len, 3 * d_model)

qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
q, k, v = qkv.unbind(dim=2)  # (batch, seq_len, num_heads, head_dim)
q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
k = k.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
v = v.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
output = scaled_dot_product_attention(q, k, v, mask)  # (batch, num_heads, seq_len, head_dim)

output = output.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
output = output.reshape(batch, seq_len, d_model)  # (batch, seq_len, d_model)
output = self.out_proj(output)  # (batch, seq_len, d_model)

return x + output
```

```python
def scaled_dot_product_attention(q, k, v, mask):
    """
    k, q: (batch_size, ..., seq_len, d_k)
    v: (batch_size, ..., seq_len, d_v)
    returns o (batch_size, ..., seq_len, d_v)
    """
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(~mask, -torch.inf)
    return softmax(scores, dim=-1) @ v
```

Note the masking convention here differs from CS224n's explicit recommendation above (finite `-10^5` to avoid NaN/inf inconsistencies across frameworks) — `-torch.inf` works in practice because `masked_fill` plus a subsequent `softmax` handles `-inf` cleanly in modern PyTorch, but it's worth being aware both conventions exist in real codebases.

### Model Activations (companion to the FLOPs/Params accounting below)

Using this section's notation (`B,S,D,N,K,H` as defined in the table above — *not* the Austin-et-al. `H,K` from the Parameter Count table below), per-layer activation memory is:

```
Attention activations: 6BSD + BNS²   (layernorm in/out, Q/K/V outputs, attention scores (B,N,S,S), attention output)
FFN activations:       2BSD + 2BSF ≈ 8BSD   (for F = 8D/3)
Per-layer total:       14BSD + BNS²
```

This is a more granular breakdown than the FLOPs/Params table below provides on its own, and is the activation-memory term that motivates [[applied-ml-systems|gradient checkpointing]]'s `O(N)→O(√N)` tradeoff at the per-layer level.

This source's own parameter-count formula (`Total ≈ 2VD + L(4D²+2D+3DF) ≈ 2VD + 12LD²` for `F=8D/3`) is consistent with the Austin-et-al. `4D²+3DF` per-layer total below, once GQA's `KH` terms are folded back to `D` for standard (non-GQA) multi-head attention — presented here as a cross-check rather than a competing derivation. Its companion **FLOPs-in-forward-pass** breakdown (`Attention: 8BSD²+4BS²D`; `FFN: 6BSDF≈16BSD²`) is consistent with the per-token FLOPs below once converted out of per-token units; the source's own capture cuts off mid-formula at the FFN down-projection term, so that breakdown isn't reproduced in full here (see `raw/alisa-liu-book-of-llms.md` for exactly what was and wasn't retrieved).

## Modern Architectural Improvements

Since 2017, the transformer has been refined significantly:

- **[[positional-encodings]]**: RoPE (rotary position embeddings) replaced sinusoidal encodings, enabling better length generalization. Downstream variants (NTK-aware, YaRN, RNoPE) extend RoPE-trained models to 128K+ contexts.
- **[[attention-variants]]**: FlashAttention made attention IO-efficient (and FlashAttention-4 retargeted the algorithm to Blackwell's asymmetric hardware, hitting ~1600 TFLOPs/s on B200). GQA reduced KV-cache memory by sharing K/V across query-head groups — with as little as 5% of pre-training compute via the GQA paper's mean-pool uptraining recipe.
- **[[mixture-of-experts]]**: MoE layers enable massive parameter counts with sparse computation
- **Pre-normalization**: Moving LayerNorm before attention (Pre-LN) instead of after improves training stability
- **SwiGLU activations**: Replace ReLU in FFN layers for better performance (used in LLaMA, Mistral)
- **Document masking**: Intra-document attention during packed-batch training avoids cross-document leakage; ProLong (2410.02660) confirms disabling cross-document attention helps both short and long-context performance.

## Hyper-Connections: Generalizing the Residual Stream

The residual connection (`h_l = h_{l-1} + F(h_{l-1})`) has been architecturally frozen since the original Transformer — it's a single stream, and every layer reads and writes the same vector. **Hyper-Connections** (Zhu et al., ByteDance, ICLR 2025, arxiv:2409.19606) generalize this: instead of one residual stream, maintain **n parallel residual streams**, where each layer reads a learned mixture of all n streams, applies its function, and writes the result back into the streams via learned mixing weights (a mix of static, depth-fixed coefficients and dynamic, input-dependent coefficients).

The motivation is a tradeoff the standard residual stream can't escape: a small residual contribution per layer keeps gradients flowing cleanly (good for vanishing-gradient avoidance) but limits how much any single layer can reshape the representation (representation collapse, layers becoming redundant); a large residual contribution does the opposite. Widening the stream into multiple lanes — and letting the network learn how to mix and route between them — gives the optimizer more degrees of freedom to resolve this **seesaw** without trading one failure mode for the other. Pre-training experiments report consistent improvements over plain residual connections, expressible with parameter and compute overhead comparable to widening attention heads.

**Manifold-Constrained Hyper-Connections (mHC)** — DeepSeek's January 2026 follow-up — is the engineering refinement that makes the idea train stably at frontier scale. Unconstrained learned mixing matrices between many streams can drift into ill-conditioned regimes (some streams dominating, others going to zero, breaking the gradient-flow guarantees that made plain residual connections reliable in the first place). mHC constrains the mixing matrices to a manifold (structured, norm-preserving transforms) so the multi-stream mixing stays well-conditioned across depth and training duration — trading some of the original method's flexibility for the stability guarantees large-scale pre-training runs need. This is the same instinct as [[training-stability|logit softcapping and QK-norm]]: a free-form learned mechanism with a desirable property in principle gets a hard geometric constraint once it has to survive a multi-week run at scale.

**Expanded Hyper-Connections (xHC)** separates the width of the residual state from the cost of transforming it. With 16 residual streams, each layer writes into only 4 selected streams but reads a dense mixture of all 16; a temporal-augmentation path also carries information across adjacent layers. This **sparse-write, dense-read** pattern preserves a wide state space without paying for every layer to update every stream. In an 18B-parameter run, xHC lowers validation loss from **1.799** for vanilla residuals and **1.776** for mHC to **1.758**, while raising the reported downstream average from **44.8 to 48.8** for only **4.1% more FLOPs**. A 28B experiment improves the downstream average from **50.5 to 53.6**. Matching xHC's loss required 1.50× as much compute for vanilla residuals and 1.19× for mHC.

The systems detail matters: a fused xHC Flash implementation reduces residual-path memory traffic from 73.5C to 40C units compared with a direct formulation (mHC uses 34C), leaving prefill throughput only **1.3% slower than mHC**. xHC is therefore not merely “more residual streams”; it is a structured routing design, much like [[mixture-of-experts]], applied to the residual state rather than the feed-forward parameters.

## Theoretical Reframing: Transformers as Structured Matrix Operators

[Dao & Gu (2405.21060)](../../papers/02-architecture/alternatives/Transformers are SSMs: Generalized Models and Efficient Algorithms for Sequence Modeling - 2405.21060.pdf) provides a useful re-derivation: any sequence transformation `Y = f(X)` can be expressed as `Y = M X` for some (parameter-dependent) matrix `M`. Self-attention corresponds to `M = softmax(QK^T)`, and the quadratic cost is the cost of multiplying by this dense matrix. **State Space Duality** shows that softmax-free linear-attention variants correspond to a class of *structured (semiseparable) matrices*, which admit `O(TN)` algorithms. The takeaway: attention isn't intrinsically quadratic — it's quadratic because softmax makes the attention matrix dense. Drop the softmax and you can choose a matrix structure with sub-quadratic cost. This framing is what unlocks Mamba-2 and modern hybrid architectures (see [[hybrid-architectures]]).

## Transformer FLOPs and Parameter Math

From "How to Scale Your Model" (Austin et al., 2025). Source: https://jax-ml.github.io/scaling-book/training

### Parameter count

For a single Transformer layer with attention heads H, head dim K, model dim D = H·K, and MLP fan-out F:

| Component | Parameters |
|---|---|
| QKV projections | 3 · D · H · K = 3D² |
| Output projection | H · K · D = D² |
| MLP (up/down/gate) | 3 · D · F |
| **Layer total** | **4D² + 3DF** |

For typical models F ≈ 4D (dense) or F ≈ 8D/3 (SwiGLU-gated), so each layer ≈ 12D² (rough). Embedding layers (vocab × D) and unembedding are often ≈ 0.2N total params.

**Total parameter count:** `N ≈ L · (4D² + 3DF)` where L = number of layers.

### FLOPs per forward pass

A matmul of shape `[B, M] × [M, N]` costs 2·B·M·N FLOPs (multiply + add).

| Component | FLOPs per token |
|---|---|
| QKV projections | 3 · 2 · D · D = 6D² |
| Attention scores | 2 · T · D (= 2 · T · H · K per token) |
| Attention values | 2 · T · D |
| Output projection | 2D² |
| MLP up + gate | 2 · 2 · D · F = 4DF |
| MLP down | 2DF |
| **MLP total per layer** | **6DF** = **18BTDF** (over B·T tokens) |
| **Attention total per layer** | **24BTDHK + 12BT²HK** = **24BTDNH + 12BT²NH** |

**Simplified:** per-token FLOPs ≈ `12D² + 6DF + 12T·D` ≈ `12D(D + F/2 + T)`

### The 6N rule

For large models (T << D, few-shot prompts), attention is negligible vs MLP. Total FLOPs ≈ forward-only = 2N per token (N = num params). Including backward pass (≈ 2× forward), training FLOPs:

> **Training FLOPs ≈ 6 · N · T_tokens** (the standard Chinchilla estimate)

For inference (forward only): **FLOPs ≈ 2 · N · T_tokens**

### When does attention dominate?

Attention term is `O(T²D)` while MLP is `O(TDF)`. Attention becomes larger when:
- `12BT²NH > 18BTDF`  
- Simplifying: **T > 8D** (rough boundary, since NH ≈ D and F ≈ 4D)

For a 7B model (D=4096), attention dominates when T > 32k tokens. Most chat contexts are below this; long-document workloads may exceed it.

### Gradient checkpointing strategies

Training activations grow linearly with depth and sequence length; two main strategies:

| Strategy | Checkpoints saved per layer | Recompute cost | Memory savings |
|---|---|---|---|
| Full remat | 0 | Very high (whole layer) | Maximum |
| Block remat | 1 (layer input) | ~2× forward | ~80% |
| Big matmuls only | 7 per layer | Low (just small ops) | ~3× |
| No remat | All | None | None (OOM at scale) |

In JAX, controlled via `jax.remat`. At scale, **block remat** is the default — saves 80% of activation memory at cost of one extra forward pass per layer, changing training FLOPs from `6NT` to approximately `8NT`.

### Flash Attention

Standard attention requires materializing the T×T attention matrix (O(T²) memory). FlashAttention avoids this by:
1. Computing attention in chunks (tiles of size M)
2. Maintaining running max and running sum for numerical stability (online softmax)
3. Accumulating weighted value outputs without ever writing out the full T×T matrix

Result: **O(T) memory** for attention (down from O(T²)), IO complexity from O(T²) to O(T²/M). Enables long context (128k+) without quadratic memory growth.

See also: [[attention-variants]] for FlashAttention-2/3/4 and hardware-specific optimizations.

## Impact

The Transformer is arguably the most impactful architecture in deep learning history. The original paper has 173,000+ citations and enabled BERT, GPT, T5, ViT, and every modern LLM. Its key insight — that attention alone is sufficient for sequence modeling — unlocked the scaling that led to today's frontier models.

## Related Topics
- [[training-stability]] — mHC's manifold constraint solves the same class of problem as logit softcapping/QK-norm: a flexible learned mechanism made safe for multi-week runs
- [[hybrid-architectures]] — Another axis of "generalizing what's fixed" in the standard Transformer, applied to the attention/recurrence choice rather than the residual stream
- [[positional-encodings]] — CS224n's order-invariance proof and the two-options framing (position-dependent inputs vs. modifying attention itself) motivating sinusoidal/learned embeddings and ALiBi; also the RoPE block-rotation matrix referenced in the equation-by-equation walkthrough above
- [[applied-ml-systems]] — the `14BSD+BNS²` model-activations formula above is the per-layer term that motivates gradient checkpointing's memory/compute tradeoff
- [[optimizers]] — AdamW/Muon consume the gradients that flow back through the layer-by-layer forward pass derived above
- [[mixture-of-experts]] — xHC applies sparse routing to residual-state updates rather than expert parameters

## Sources
- Attention Is All You Need (arxiv:1706.03762)
- BERT: Pre-training of Deep Bidirectional Transformers (aclanthology.org/N19-1423)
- [CS224n: Self-Attention & Transformers (Hewitt, Stanford, 2023 draft)](../raw/cs224n-self-attention-transformers.md) — source: https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf — RNN motivation (parallelization, linear interaction distance), KQV self-attention build-up, the linear-collapse argument for nonlinearities, the -10^5 (not -∞) masking detail, the d/k-per-head implementation rationale for multi-head attention, pre-norm/post-norm formulas, and explicit cross-attention Q/K/V sourcing.
- Hyper-Connections — Zhu et al., ByteDance, ICLR 2025 (arxiv:2409.19606)
- Manifold-Constrained Hyper-Connections (mHC) — DeepSeek, January 2026
- [xHC: Expanded Hyper-Connections (2607.14530)](../../papers/02-architecture/transformers/xHC: Expanded Hyper-Connections - 2607.14530.pdf) — sparse-write/dense-read residual streams, temporal augmentation, scaling results, and the fused xHC Flash implementation.
- [Transformers are SSMs / Mamba-2 (2405.21060)](../../papers/02-architecture/alternatives/Transformers are SSMs: Generalized Models and Efficient Algorithms for Sequence Modeling - 2405.21060.pdf) — structured state-space duality reframing.
- [FlashAttention-4 (2603.05451)](../../papers/02-architecture/attention-variants/FlashAttention-4: Algorithm and Kernel Pipelining Co-Design - 2603.05451.pdf) — Blackwell co-design.
- The Random Transformer — Omar Sanseviero (hands-on NumPy implementation)
- [How to Scale Your Model — Austin et al. (2025)](https://jax-ml.github.io/scaling-book/training) — FLOPs/params tables, 6N rule, gradient checkpointing strategies, Flash Attention derivation
- [The Illustrated Transformer — Jay Alammar](../raw/illustrated-transformer-jay-alammar.md) — definitive visual explainer; covers self-attention step-by-step, multi-head attention, positional encodings, and the encoder-decoder flow.
- [The Illustrated GPT-2 — Jay Alammar](../raw/illustrated-gpt2-alammar.md) — concrete walkthrough of OpenAI's GPT-2: decoder-only block lineage, KV reuse during generation, top-k sampling, FFN/embedding dimensions, the 124M-vs-117M parameter discrepancy, and transfer applications (translation, summarization, music generation).
- [Mastering Tensor Dimensions in Transformers — Not Lain (Hugging Face)](../raw/mastering-tensor-dimensions-transformers.md) — shape-by-shape walkthrough of a decoder-only stack (`[1,4]` → `[1,4,768]` → LM head), multi-head attention `768 = 8×96` reshaping, and a cross-attention example; the background the [[kv-cache]] post's prerequisites assume.
- Alisa Liu, "Book of LLMs" (Notion, alisawuffles.notion.site/alisa-s-book-of-llms) — the full equation-by-equation modern decoder walkthrough (RMSNorm, GQA-aware QKV projections, QK-norm, RoPE pointer, causal-masked softmax, SwiGLU FFN), the attention-block PyTorch implementation, and the `14BSD+BNS²` model-activations formula. See [`raw/alisa-liu-book-of-llms.md`](../raw/alisa-liu-book-of-llms.md) — note its own FLOPs-in-forward-pass derivation is cut off mid-formula at the FFN down-projection term, per that file's capture note.
