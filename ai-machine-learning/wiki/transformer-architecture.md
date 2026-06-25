# Transformer Architecture

The Transformer is the foundational neural network architecture behind all modern large language models. Introduced in "Attention Is All You Need" (Vaswani et al., 2017), it replaced recurrence and convolutions with a purely attention-based mechanism, enabling massive parallelization and superior performance on sequence tasks.

## Core Design

The original Transformer uses an encoder-decoder structure. The encoder maps an input sequence to a continuous representation, and the decoder generates an output sequence one token at a time, attending to the encoder's output. The key innovation is **multi-head self-attention**, which allows every position in a sequence to attend to every other position simultaneously.

**Scaled dot-product attention** computes: Attention(Q, K, V) = softmax(QK^T / √d_k)V, where Q, K, V are query, key, and value matrices derived from the input. Multi-head attention runs this computation in parallel across multiple "heads," each learning different attention patterns.

Other critical components include layer normalization, residual connections, and position-wise feed-forward networks (two linear transformations with a ReLU activation between them).

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

## Sources
- Attention Is All You Need (arxiv:1706.03762)
- BERT: Pre-training of Deep Bidirectional Transformers (aclanthology.org/N19-1423)
- Hyper-Connections — Zhu et al., ByteDance, ICLR 2025 (arxiv:2409.19606)
- Manifold-Constrained Hyper-Connections (mHC) — DeepSeek, January 2026
- [Transformers are SSMs / Mamba-2 (2405.21060)](../../papers/02-architecture/alternatives/Transformers are SSMs: Generalized Models and Efficient Algorithms for Sequence Modeling - 2405.21060.pdf) — structured state-space duality reframing.
- [FlashAttention-4 (2603.05451)](../../papers/02-architecture/attention-variants/FlashAttention-4: Algorithm and Kernel Pipelining Co-Design - 2603.05451.pdf) — Blackwell co-design.
- The Random Transformer — Omar Sanseviero (hands-on NumPy implementation)
- [How to Scale Your Model — Austin et al. (2025)](https://jax-ml.github.io/scaling-book/training) — FLOPs/params tables, 6N rule, gradient checkpointing strategies, Flash Attention derivation
- [The Illustrated Transformer — Jay Alammar](../raw/illustrated-transformer-jay-alammar.md) — definitive visual explainer; covers self-attention step-by-step, multi-head attention, positional encodings, and the encoder-decoder flow.
- [The Illustrated GPT-2 — Jay Alammar](../raw/illustrated-gpt2-alammar.md) — concrete walkthrough of OpenAI's GPT-2: decoder-only block lineage, KV reuse during generation, top-k sampling, FFN/embedding dimensions, the 124M-vs-117M parameter discrepancy, and transfer applications (translation, summarization, music generation).
- [Mastering Tensor Dimensions in Transformers — Not Lain (Hugging Face)](../raw/mastering-tensor-dimensions-transformers.md) — shape-by-shape walkthrough of a decoder-only stack (`[1,4]` → `[1,4,768]` → LM head), multi-head attention `768 = 8×96` reshaping, and a cross-attention example; the background the [[kv-cache]] post's prerequisites assume.
