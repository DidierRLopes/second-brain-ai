# The Illustrated GPT-2 (Visualizing Transformer Language Models)

**Source:** https://jalammar.github.io/illustrated-gpt2/
**Author:** Jay Alammar
**Published:** August 12, 2019

## Overview

A follow-up to [[../raw/illustrated-transformer-jay-alammar.md|The Illustrated Transformer]] that walks through OpenAI's GPT-2 specifically: a large decoder-only transformer language model trained on a 40GB scraped corpus ("WebText"). The post explains why GPT-2 is "architecturally unoriginal" — it's the decoder-only transformer block, scaled up — and then goes deep on the self-attention mechanics that make autoregressive generation work.

## Part 1: GPT-2 and Language Modeling

### What Is a Language Model

A language model predicts the next word/token given preceding context — the same task as a smartphone keyboard's word suggestions, just far larger. GPT-2 was trained on the 40GB WebText corpus; the smallest released variant needs ~500MB of storage for its parameters, the largest ~13× that (6.5GB+).

### One Difference From BERT

GPT-2 is built from **transformer decoder blocks**; BERT is built from **transformer encoder blocks**. GPT-2 outputs one token at a time and is **auto-regressive**: each produced token is appended to the input sequence, and that extended sequence becomes the input for the next step (the same idea that made RNNs effective). BERT is not auto-regressive — it sacrifices that property to attend to context on both sides of a word during pretraining.

### The Evolution of the Transformer Block

The original Transformer paper (arXiv 1706.03762) defined an **encoder block** (full self-attention, no masking, processes up to a fixed max length like 512 tokens) and a **decoder block** (masked self-attention so a position can only attend to itself and earlier positions, plus a second sublayer attending to the encoder's output).

"Generating Wikipedia by Summarizing Long Sequences" (arXiv 1801.10198) introduced the **decoder-only** variant: it drops the encoder entirely and removes the decoder's second (encoder-attention) sublayer, leaving a stack of masked-self-attention-only blocks. That paper's model — call it the "Transformer-Decoder" — used 6 such blocks and could address up to 4,000 tokens, a large jump from the original paper's 512. "Character-Level Language Modeling with Deeper Self-Attention" (arXiv 1808.04444) used a similar architecture for character-level prediction. **GPT-2 uses these decoder-only blocks.**

### Crash Course: Looking Inside GPT-2

GPT-2 processes up to **1024 tokens**, each flowing through every decoder block along its own path. Running the model unconditionally ("rambling") starts from a single start token (`<|endoftext|>`); each step produces a vector that is scored against the model's ~50,000-word vocabulary, and the model either takes the top-scoring word (`top_k = 1`) or samples among the top-k highest-scoring words to avoid getting stuck in repetitive loops. Each layer retains its own interpretation of earlier tokens and does **not** re-interpret them once new tokens arrive — a structural consequence of the masked, auto-regressive design.

### A Deeper Look Inside

**Input encoding:** the model looks up the input token's row in a trained embedding matrix, then adds a positional-encoding vector. Part of the trained model is a matrix holding one positional-encoding vector for each of the **1024 positions**. The smallest GPT-2 uses a **768-dimensional** embedding per token.

**Self-attention recap:** self-attention bakes context into a token's representation before the feed-forward network sees it, by scoring how relevant every other token is and summing their value vectors weighted by those scores. The three components are **Query** (what the current token is looking for), **Key** (labels other tokens present for matching), and **Value** (the actual content retrieved once relevance is scored) — illustrated as a filing-cabinet search: the query is a sticky note, keys are folder labels, values are folder contents, and a blend of multiple folders' contents is retrieved.

**Model output:** the top decoder block's output vector is multiplied by the embedding matrix to produce a score for every vocabulary word; the model then either takes the argmax or samples (commonly `top_k = 40`) to pick the next token. Generation continues until 1024 tokens are produced or an end-of-sequence token appears.

**Oversimplifications the post flags explicitly:** GPT-2 uses **Byte Pair Encoding**, so "tokens" are usually parts of words, not whole words; the worked examples show inference mode (batch size 1, one token at a time), while training uses much larger batches (512) over longer sequences; layer normalization is used heavily throughout but de-emphasized in the visuals in favor of self-attention.

## Part 2: The Illustrated Self-Attention

### Self-Attention (Without Masking)

Three steps, per token path: (1) compute Query/Key/Value vectors; (2) score the current token's query against every other token's key (dot product); (3) multiply each value vector by its score and sum — producing a context-aware representation passed to the feed-forward sublayer.

### The Illustrated Masked Self-Attention

Masked self-attention is identical except at the scoring step: scores for future tokens are forced to ~0 so a position can't "peek" ahead. Mechanically this is done via an **attention mask** — after multiplying the queries matrix by the keys matrix, cells corresponding to future tokens are set to a very large negative number (Alammar cites **−1 billion** for GPT-2) before a row-wise softmax converts the matrix into actual attention scores.

### GPT-2 Masked Self-Attention

**Evaluation-time efficiency (KV reuse):** during generation, GPT-2 only adds one new token per iteration, so recalculating self-attention for already-processed tokens would be wasteful. Instead, "GPT-2 holds on to the key and value vectors" computed for each earlier token at every self-attention layer, and on the next iteration reuses those saved K/V vectors rather than regenerating them — computing only the new token's query, key, and value. This is a pre-formal, plain-language description of what is now formally called the [[../wiki/kv-cache.md|KV cache]].

**Splitting into attention heads:** the small GPT-2 has **12 attention heads**; the concatenated Q/K/V vector is reshaped into a matrix with heads as the first dimension, each head conducting its own scoring/summing independently before all heads' outputs are concatenated and projected back to model dimension via a second large weight matrix.

**Fully-connected sublayer:** two linear layers. The first expands to **4× the model dimension** (768 → 3072 for the small model — the same 4× ratio the original Transformer used, 512 → 2048); the second projects back down to model dimension (3072 → 768).

**Parameter count discrepancy:** Alammar tallies all the weight matrices in GPT-2-small and gets **124M parameters**, not the **117M** the model's published name implies — he flags this as unresolved ("I'm not sure why, but that's how many of them seems to be in the published code").

## Part 3: Beyond Language Modeling

The decoder-only transformer generalizes beyond language modeling:

- **Machine Translation** — an encoder isn't required; a decoder-only transformer can perform translation directly.
- **Summarization** — the original task the first decoder-only "Transformer-Decoder" (arXiv 1801.10198) was trained on: read a Wikipedia article body and summarize it, using the article's own opening section as the training label.
- **Transfer Learning** — "Sample Efficient Text Summarization Using a Single Pre-Trained Transformer" (arXiv 1905.08836) pretrains a decoder-only transformer on language modeling, then finetunes it for summarization, beating a pretrained encoder-decoder transformer in limited-data settings; the GPT-2 paper itself reports summarization results after language-model pretraining.
- **Music Generation** — the Music Transformer treats "music modeling" like language modeling: notes are one-hot encoded together with **velocity** (how hard a piano key is struck), and the model samples performances unsupervised the same way GPT-2 "rambles" text.

## Key Takeaways

- GPT-2 is architecturally just the decoder-only transformer block, scaled up — its impact came from data (40GB WebText) and scale, not a new mechanism.
- Auto-regression plus masked self-attention is what makes one-token-at-a-time generation coherent: each new token is appended to the input and the model never revises its interpretation of earlier tokens.
- The "hold on to key and value vectors" reuse trick described here for GPT-2's evaluation mode is the direct ancestor of the formal KV cache (see [[../wiki/kv-cache.md|KV Cache]]).
- The decoder-only stack is a general sequence-modeling primitive, not just a language-modeling one — translation, summarization, transfer learning, and music generation all reuse the same block.
