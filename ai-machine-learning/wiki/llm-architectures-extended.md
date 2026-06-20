# LLM Architecture Extensions

Architectural variants, decoding strategies, and the pretraining process that bridges transformers, recurrent models, and state space models.

---

## Pretraining

Pretraining is the large-scale unsupervised learning phase that gives LLMs their general knowledge and capabilities.

### Next-Token Prediction (Causal Language Modelling)

The core pre-training objective for decoder-only models (GPT, LLaMA, Qwen, etc.):

```
L = -Σ_{t=1}^T log P(x_t | x_{<t}; θ)
```

Given a sequence of tokens [x_1, x_2, ..., x_T], predict each token from all preceding tokens. This is self-supervised — no human labels needed, and every token in every text sequence is a training signal.

**Why this works**: to predict the next token well, the model must understand syntax, semantics, world facts, reasoning patterns, and discourse structure. Next-token prediction is a surprisingly powerful proxy for general intelligence.

**Masked Language Modelling (MLM)**: used in encoder models (BERT). Mask ~15% of tokens, predict them from context in both directions. Better at understanding tasks, worse at generation.

**Data scale**: modern frontier models train on 10T-100T+ tokens (trillions). Pre-training data quality and diversity dominate final performance. See [[data-curation-mixtures]].

**Compute budget**: the Chinchilla scaling law recommends ~20 tokens per parameter for compute-optimal training. In practice, frontier labs overtrain relative to compute-optimal because inference cost matters — a smaller model trained longer is cheaper to serve. See [[scaling-laws]].

---

## Decoding Techniques

At inference time, the model outputs logits over the vocabulary; decoding converts logits to actual output tokens.

### Greedy Decoding

Select the highest-probability token at each step:

```
x_t = argmax_v P(v | x_{<t})
```

**Problems**: doesn't explore alternatives, tends to produce repetitive or generic output. Not the best choice for creative tasks.

### Beam Search

Maintain the top-k partial sequences (beams) at each step:

1. Start with k=1 sequence
2. At each step, expand each beam by all V vocabulary options
3. Keep only the top-k sequences by cumulative log-probability
4. Return the highest-scoring complete sequence

Beam search finds sequences with high likelihood but can be repetitive and boring. Used in translation and structured generation. k=4 or k=5 is typical.

### Temperature Sampling

Convert logits to probabilities, then sample:

```
P(x_t = v) = softmax(logits / T)_v
```

- T = 1: standard sampling, matches model distribution
- T < 1 (e.g., 0.7): sharper distribution, less random
- T > 1: flatter distribution, more random/diverse
- T → 0: greedy decoding

Temperature is the primary knob for controlling output diversity. Most LLM APIs expose this.

### Top-k Sampling

Restrict sampling to the top-k tokens by probability:

```python
probs = softmax(logits)
top_k_probs, top_k_ids = topk(probs, k)
sampled_id = sample(top_k_ids, weights=top_k_probs)
```

Prevents sampling from very unlikely tokens (which can be incoherent). k=50 is common.

### Top-p (Nucleus) Sampling

Instead of a fixed k, select the smallest set of tokens whose cumulative probability exceeds p:

```python
sorted_probs = sort_descending(softmax(logits))
cumsum = cumulative_sum(sorted_probs)
nucleus = indices where cumsum <= p
```

Adapts the size of the selection based on distribution sharpness. When the model is confident (peaked distribution), nucleus is small; when uncertain, nucleus is large. p=0.9 or p=0.95 is typical.

Top-p is generally preferred over top-k in practice.

### Min-P Sampling

Remove tokens with probability < min_p × max_prob. Scales cutoff relative to the most probable token. Maintains more diversity than top-p when the model is uncertain, and more precision when it's confident.

### Repetition Penalty

Reduce the log-probability of previously generated tokens:

```
logits[v] = logits[v] / penalty if v has been generated
```

Where penalty > 1 reduces probability of repetition. Prevents the common failure mode of LLMs repeating themselves endlessly.

### Speculative Decoding

Speed up inference by using a small draft model to generate K tokens at once, then verifying them in parallel with the large model.

1. Draft model generates [t+1, t+2, ..., t+K] tokens (cheap)
2. Target model evaluates all K positions in a single forward pass (parallel)
3. Accept tokens greedily while draft and target agree; reject at first divergence

No quality change vs standard sampling — rejected tokens are resampled from the target distribution. Speedup: 2-4× for typical K=5-10 drafts. Best when draft model output distribution closely matches target.

---

## Transformer-XL

Transformer-XL (Dai et al., 2019) introduced the **recurrence mechanism** to handle sequences longer than the fixed context window.

**The problem**: standard transformers have a fixed context length. Processing longer sequences requires splitting into segments, losing cross-segment dependencies.

**Segment-level recurrence**: when processing segment τ+1, reuse the hidden states from segment τ as additional "memory":

```
H̃_τ = [SG(H_{τ-1}) ○ H_τ]    # concatenate previous segment (stop-gradient) with current
```

Attention in segment τ+1 can attend to hidden states from segment τ (not gradients, just forward-pass memory). This allows gradient-free infinite context via cached states.

**Relative positional encoding**: required because absolute positions break when segments are concatenated. Transformer-XL's relative PE encodes the distance between query and key positions, rather than their absolute positions. This is a conceptual ancestor of RoPE and ALiBi.

**Limitations**: quadratic attention within each segment, not fully parallelisable across segments. Superseded by:
- Efficient attention (FlashAttention, sparse attention)
- RoPE with long context training
- State space models (S4, Mamba)

Transformer-XL's key contribution was demonstrating that segment-level recurrence works and that relative positional encodings are necessary for it. These ideas fed directly into RoPE (see [[positional-encodings]]).

---

## Griffin

Griffin (De et al., 2024, DeepMind) is a hybrid recurrent-attention architecture designed to combine:
- The efficient inference of RNNs (linear scaling with sequence length)
- The quality of transformers (attention for associative recall)

### Architecture

**Griffin's recurrent block** ("Hawk"): gated linear recurrence:

```
rt = sigmoid(Wᵣ xₜ + bᵣ)          # reset gate
it = sigmoid(Wᵢ xₜ + bᵢ)           # input gate  
ht = rt * ht₋₁ + it * tanh(Wx xₜ + bx)  # state update
```

The key difference from LSTM: the recurrence matrix is **diagonal** (element-wise), not a full matrix multiply. This enables parallelisation via parallel scan (prefix sum) and avoids the O(d²) cost of a full recurrence.

**Parallel scan**: diagonal recurrence y_t = a_t y_{t-1} + b_t can be computed in parallel in O(log T) steps (like prefix sums), rather than O(T) sequentially. This is what makes Griffin trainingable at scale.

**Griffin = Hawk blocks + local sliding window attention (SWA) blocks** interleaved. The local attention captures short-range precise dependencies; the recurrent blocks accumulate long-range information efficiently.

### Significance

Griffin (and its extension **Gemma 2**) showed that recurrence-attention hybrids can:
- Match transformer quality on standard benchmarks
- Require O(1) memory per new token during inference (vs O(T) for KV-cache in full attention)
- Process longer sequences at lower cost

See [[hybrid-architectures]] for the full landscape of Mamba, GLA, DeltaNet, and other hybrid approaches.

---

## Perceiver

Perceiver (Jaegle et al., 2021, DeepMind) decouples architecture size from input/output size, enabling a single model to handle diverse modalities.

### The Bottleneck Architecture

Standard transformers: attention is O(n²) in input length n. Scaling to high-resolution images, long audio, or video is infeasible.

**Perceiver's approach:**
1. Maintain a small set of **latent vectors** (e.g., 512 latent tokens), regardless of input size
2. **Cross-attention**: latent vectors attend to input (queries=latent, keys/values=input). This is O(Mn) where M is the number of latent tokens and n is input size
3. **Self-attention**: latent vectors attend to each other. This is O(M²) — manageable if M << n
4. Alternate between cross-attention and self-attention through multiple layers

**Key property**: the compute cost is O(Mn + LM²) rather than O(n²). For images with n=50K pixels and M=512 latents: compare 2.5B vs 26M operations.

### PerceiverIO

Extends Perceiver to **structured outputs** (not just classification). Add an output cross-attention step: output query vectors attend to final latent state. Different output queries for different output positions — can output text, images, actions, etc.

**Applications**: text + image + audio + video + point clouds in a single architecture. Foundation of some multimodal models.

**Limitations**: the cross-attention compresses all input information through a bottleneck — fine-grained spatial detail is lost. Transformers with efficient attention (Flash Attention) are now often preferred for multimodal work since hardware has improved.

---

## LLM vs RNN vs S4

A comparison of the major sequence modelling paradigms.

| Property | Transformer (LLM) | RNN/LSTM | S4/Mamba (SSM) |
|---------|-------------------|----------|----------------|
| Training parallelism | Full (attention matrix) | None (sequential) | Full (via convolution or parallel scan) |
| Inference memory | O(T) KV-cache | O(1) hidden state | O(state size) |
| Inference compute/step | O(T) (re-attend to all) | O(1) | O(1) |
| Long-range dependencies | Yes (attention) | Poor (vanishing grads) | Yes (state space) |
| In-context learning | Strong | Weak | Improving |
| Associative recall | Strong | Poor | Weak (in pure SSMs) |
| Training cost | O(T² d) | O(Td²) | O(T log T d) or O(Td) |
| Effective context | ~128K tokens (FlashAttn) | Practically limited | Very long (recurrence) |

**Transformers** dominate because: strong associative recall (attention directly matches keys to queries), highly parallelisable training, excellent scaling, and a massive ecosystem.

**RNNs** were dominant 2015-2018 but are obsolete for large-scale language modelling. Still competitive for streaming/edge inference where memory and latency per token matters.

**S4/Mamba (SSMs)** are a strong alternative:
- Linear-time training (no quadratic attention)
- O(1) per-step inference (like RNNs, unlike transformers)
- Selective SSMs (Mamba) can focus on relevant inputs
- Weak at exact associative recall (a fundamental limitation of linear recurrences)

**Hybrid models** (Griffin, Jamba, Zamba) interleave SSM blocks with attention blocks. Get:
- Good associative recall from attention
- Efficient long-context from SSM
- O(n) training (mostly) instead of O(n²)

**The current consensus (2026)**: full transformers with FlashAttention and long-context training remain the frontier approach. Hybrids are increasingly used for inference efficiency. Pure SSMs have not displaced transformers for language.

See [[hybrid-architectures]] for Mamba 2, gated DeltaNet, and production hybrid designs.
