# Mastering Tensor Dimensions in Transformers

**Source:** https://huggingface.co/blog/not-lain/tensor-dims
**Author:** Not Lain (Hugging Face Community Article)
**Published:** January 12, 2025

## Overview

A Hugging Face community blog post that traces how a tensor's **shape** changes
as it flows through a decoder-only text-generation model, layer by layer. Rather
than re-deriving the math of attention, the post fixes a single running example
and follows its dimensions end-to-end, which makes it a useful companion to the
author's later [[kv-cache]] post (which assumes exactly this background). The
guiding convention throughout is `[batch, seq_len, embed_dim]`, abbreviated to
shape-only notation like `[1, 4, 768]`.

The post explicitly states matrix-multiplication shape rules as a prerequisite
and works the example for a decoder-only architecture (the dominant design for
generative models), with an encoder-decoder/cross-attention coda at the end.

## Running Example and Input Shape

The sentence `Hello world !` tokenizes into three pieces (`Hello`, `world`, `!`),
plus two auxiliary tokens `<bos>` (beginning of sentence) and `<eos>` (end of
sentence), which also ensures the input is shifted correctly. After tokenization
the input is a tensor like `[12, 15496, 2159, 5145]`; batching adds an outer
dimension → `[[12, 15496, 2159, 5145]]`. The post then drops the literal IDs and
tracks only shapes, representing the input as **`[1, 4]`** where `1` is batch
size and `4` is sequence length.

## Layer-by-Layer Shape Walkthrough

**Embedding layer.** Maps each token ID to a learned vector, taking `[1, 4]` →
**`[1, 4, 768]`** where `768` is the embedding dimension. The post stresses two
roles for this layer: the embedding dimension propagates through the whole
network and is heavily used inside attention, and embeddings convert discrete
tokens into high-dimensional vectors that capture semantic relationships (tokens
that look numerically unrelated, e.g. IDs 8848 and 9584 for "king"/"man," become
geometrically related vectors).

**Positional encoding.** Injects order information *without changing the shape*
(`[1, 4, 768]` → `[1, 4, 768]`). This matters because the later computations run
in parallel across positions, so order must be encoded into the values rather
than implied by sequential processing.

**Decoder layer.** A generative model stacks multiple identical decoder layers,
each containing masked multi-head attention, an add-and-normalize step, and a
feed-forward network.

### Masked Multi-Head Attention (the shape-heavy part)

1. **Q/K/V projections.** Three parallel `nn.Linear(768, 768)` layers produce
   Query, Key, Value, each of shape `[1, 4, 768]` (shape preserved).
2. **Split into heads.** The embedding dimension factors as `768 = 8 * 96`
   (8 heads × head size 96), reshaping each to `[1, 4, 8, 96]`. To line up the
   matmul dimensions, seq_len and head_size are transposed → **`[1, 8, 4, 96]`**
   for Q, K, V.
3. **Scores `QKᵀ`.** `Kᵀ` has shape `[1, 8, 96, 4]`, so
   `QKᵀ = [1, 8, 4, 96] × [1, 8, 96, 4] = [1, 8, 4, 4]`.
4. **Masking.** A causal mask sets future-token scores to `-inf` so each token
   attends only to itself and preceding tokens.
5. **Attention weights.** `softmax(QKᵀ / √d_k)` with `d_k = 96`. Scaling by the
   head size prevents large-magnitude disparities; softmax zeros out the `-inf`
   entries and makes each row sum to 1 — values change, shape stays `[1, 8, 4, 4]`.
6. **Apply to values.**
   `softmax(...) · V = [1, 8, 4, 4] × [1, 8, 4, 96] = [1, 8, 4, 96]`.
7. **Concat heads.** Transpose back and merge:
   `[1, 8, 4, 96] → [1, 4, 8, 96] → [1, 4, 768]`.
8. **Output projection.** A final `nn.Linear(768, 768)` keeps it at `[1, 4, 768]`.

The post's key **observation**: the tensor returns to its pre-attention shape
`[1, 4, 768]`, which is what lets the block be stacked and combined with skip
connections.

### Add & Normalize, Feed-Forward, LM Head

- **Add & normalize.** A skip connection adds the pre- and post-attention tensors
  and normalizes; addition *updates* rather than *replaces* the representation,
  normalization prevents value blow-up. Applied after each sublayer.
- **Feed-forward.** Two linear layers that expand then contract, typically with a
  `1 ⇒ 3` expansion and `3 ⇒ 1` contraction: `nn.Linear(768, 3*768)` then
  `nn.Linear(3*768, 768)`, often with dropout. Output returns to `[1, 4, 768]`,
  and because the decoder's output shape matches its input shape, decoder layers
  stack seamlessly.
- **Language-model head.** A final linear layer maps `embed_dim ⇒ vocab_size`,
  giving e.g. `[1, 4, 9735]` for a vocab of 9735 (`1` batch, `4` seq_len, `9735`
  vocab). Softmax + loss against the ground truth yields the error the optimizer
  uses to update weights.

The post reiterates the autoregressive intuition: because input is shifted right
and masked attention conditions each token on itself and its predecessors,
generating a new token requires the vector representations of all previous tokens
(including how each relates to its own predecessors).

## Transformers and Cross-Attention (encoder-decoder coda)

For tasks where context and output differ in length (e.g. translation), the full
encoder-decoder transformer is illustrated, though decoder-only designs are often
preferred today. Encoder layers propagate shapes like the decoder, except the
attention is **unmasked** (each token attends to all others, before and after).

Worked cross-attention example — context `I am at home` (`[1, 4]`) and target
`<bos> je suis à la maison` (`[1, 6]`):

- Encoder output: `[1, 4, 768]`; decoder masked-attention output: `[1, 6, 768]`.
- In cross-attention, **K and V come from the encoder, Q comes from the decoder.**
- Split/transpose: Q → `[1, 8, 6, 96]`, K & V → `[1, 8, 4, 96]`.
- `Kᵀ → [1, 8, 96, 4]`; `QKᵀ = [1, 8, 6, 96] × [1, 8, 96, 4] = [1, 8, 6, 4]`;
  softmax keeps `[1, 8, 6, 4]`.
- `softmax(...) · V = [1, 8, 6, 4] × [1, 8, 4, 96] = [1, 8, 6, 96]`.
- Concat: `[1, 8, 6, 96] → [1, 6, 8, 96] → [1, 6, 768]`.

The differing source/target sequence lengths (4 vs 6) are reconciled inside
cross-attention precisely because `QKᵀ` mixes the decoder's query length with the
encoder's key length, and the output returns to `[1, 6, 768]` — matching the
masked-attention output so subsequent layers stay compatible.

## Why It Matters

The recurring theme is **shape invariance through a block**: most sublayers
either preserve `[1, 4, 768]` outright (positional encoding, add-and-normalize,
output projection, feed-forward) or temporarily reshape for the matmul and then
restore it (multi-head attention). That invariance is what allows arbitrarily
deep stacks of decoder layers and the residual/add-and-normalize pattern. The
only deliberate shape *changes* are at the boundaries: embedding (`[1, 4]` →
`[1, 4, 768]`) and the LM head (`[1, 4, 768]` → `[1, 4, vocab]`).

## Related wiki pages

[[transformer-architecture]], [[attention-variants]], [[positional-encodings]], [[kv-cache]]
