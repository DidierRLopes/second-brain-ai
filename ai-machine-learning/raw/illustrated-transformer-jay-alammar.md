# The Illustrated Transformer

**Source:** https://jalammar.github.io/illustrated-transformer/
**Author:** Jay Alammar
**Published:** June 27, 2018

## Overview

A visual, step-by-step walkthrough of the Transformer architecture from "Attention Is All You Need." One of the most widely referenced explainers — featured in courses at Stanford, Harvard, MIT, Princeton, and CMU.

## High-Level Structure

The Transformer is an encoder-decoder model. The encoder maps an input sequence to a continuous representation; the decoder generates output one token at a time. The paper stacks 6 encoders and 6 decoders.

Each encoder has two sub-layers:
1. **Self-attention** — allows each position to attend to all others in the input
2. **Feed-forward network** — applied independently to each position (parallelizable)

Each decoder has three sub-layers:
1. **Masked self-attention** — only attends to earlier positions (causal)
2. **Encoder-decoder attention** — Q from decoder, K/V from encoder stack output
3. **Feed-forward network**

## Self-Attention (Step by Step)

For each input token, create three vectors: **Query (Q)**, **Key (K)**, **Value (V)** — projections of the embedding via trained weight matrices WQ, WK, WV (dim 64, vs embedding dim 512).

1. Compute scores: dot product of Q with all K vectors
2. Scale by √d_k (√64 = 8) for stable gradients
3. Softmax → attention weights (sum to 1)
4. Multiply each V by its weight, sum → output vector for that position

**Matrix form:** `Attention(Q, K, V) = softmax(QK^T / √d_k) V`

## Multi-Head Attention

Run self-attention 8 times in parallel with different WQ/WK/WV matrices → 8 different Z matrices. Concatenate them, multiply by WO to produce the final output.

Benefits:
- Different heads learn to attend to different relationships (e.g., one head tracks "it" → "animal", another tracks "it" → "tired")
- Multiple representation subspaces

## Positional Encoding

Since attention is order-agnostic, positional encodings are added to input embeddings. The paper uses sinusoidal functions (sine for even dims, cosine for odd dims), producing a pattern that generalizes to unseen sequence lengths.

## Residuals & Layer Norm

Each sub-layer is wrapped with a residual connection and layer normalization:
`LayerNorm(x + Sublayer(x))`

This applies to both encoder and decoder sub-layers.

## Decoder Details

- Self-attention is **masked** (future positions set to −∞ before softmax) to preserve autoregressive property
- Encoder-decoder attention takes Q from decoder, K/V from final encoder output
- Runs sequentially; each step feeds the next as a new input token

## Output Layer

Decoder stack → Linear layer → logits vector (vocab_size) → Softmax → probabilities → argmax = predicted token.

## Training

Loss: cross-entropy between output probability distribution and one-hot target. Backprop updates all weights. Greedy decoding (argmax at each step) vs. beam search (keep top-k hypotheses) are two inference strategies.

## Key Takeaways

- Self-attention lets every token attend to every other token simultaneously — unlike RNNs which process sequentially
- Multi-head attention provides multiple learned "views" of relationships in the sequence
- The architecture is highly parallelizable (unlike RNNs), enabling efficient training on TPUs/GPUs
- Positional encodings are the only mechanism for injecting sequence order
