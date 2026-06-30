# Self-Attention & Transformers (CS224n Note 10, 2023 draft)

**Source:** https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf
**Authors:** John Hewitt (author; johnhew@cs.stanford.edu). Course instructors: Christopher Manning, John Hewitt.
**Course:** CS 224n: Natural Language Processing with Deep Learning, Stanford, Winter 2023 (draft Note 10)

## Summary (as stated in the reading)

"This note motivates moving away from recurrent architectures in NLP, introduces self-attention, and builds a minimal self-attention-based neural architecture. Finally, it dives into the details of the Transformer architecture, a self-attention-based architecture that as of 2023 is ubiquitous in NLP research."

## Notation and basics

- A sequence is written `w_{1:n}`, each `w_i ∈ V` (a finite vocabulary). `w_{1:n}` is overloaded to also mean a matrix of one-hot vectors in `R^{n×|V|}`.
- `w_t ∼ softmax(f(w_{1:t-1}))` means `w_t` is drawn from the distribution defined by the right-hand side; `f` maps a sequence to a vector in `R^{|V|}`.
- Softmax is always defined over the last axis of a tensor: for `A ∈ R^{ℓ,d}`, `softmax(A)_{i,j} = exp(A_{i,j}) / Σ_{j'=1}^{d} exp(A_{i,j'})`.
- Embedding matrix `E ∈ R^{d×|V|}` maps vocabulary to hidden dimensionality `d`; `Ex ∈ R^d`.
- Key distinction drawn explicitly: the embedding `E_{w_i}` of a token is a **non-contextual representation** — despite `w_i` appearing in a sequence, `Ew_i` is independent of context. The stated overarching goal of the entire note is to build strong **contextual representations** `h_i` that are a function of the entire sequence `x_{1:n}` (or a prefix `x_{1:i}` for language modeling).

## 1. Why move away from RNNs (the 2017 default)

The note frames the move to self-attention not as "a new idea solving an old problem" alone, but partly as "old techniques become newly relevant as data or computation power becomes newly available" — citing HMMs (Baum & Petrie 1966), CRFs (Lafferty et al. 2001), RNNs (Rumelhart et al. 1985), CNNs (LeCun et al. 1989), and SVMs (Cortes & Vapnik 1995) as the lineage of general-purpose NLP techniques.

Vanilla RNN form given: `h_t = σ(W h_{t-1} + U x_t)`, with `h_t ∈ R^d`, `U, W ∈ R^{d×d}`.

Two specific, named issues with RNNs as of 2017 (both stemming from dependence on the sequence index / "time"):

1. **Parallelization issue.** GPUs are fast at `AB` for `A ∈ R^{n×k}`, `B ∈ R^{k×d}` because the multiplies/sums are mutually independent. But `h_2 = σ(W h_1 + U x_2)` cannot be computed before `h_1` is known, and unrolling shows `h_2 = σ(W σ(W h_0 + U x_1) + U x_2)` — a serial dependency chain. The note's figure annotates an RNN unrolled over "Zuko made his uncle tea," with each rectangle labeled by the number of serial operations required before that state can be computed (0,1,2,3,4,5).
2. **Linear interaction distance.** Using the worked example "The chef who ran out of blackberries and went to the stores is ___", the number of intermediate matrix multiplies/nonlinearities separating "chef" from "is" scales with the number of intervening words. The claim: this makes it hard for the network to "recall" a word's presence after many subsequent operations, hurting modeling of how distant words affect the current word's representation.

The note explicitly connects this lineage to Bahdanau et al. (2014) attention in machine translation (looking back into the source sequence once per output token) and frames self-attention as "an entire replacement for recurrent neural networks just based on attention," solving both problems at once.

## 2. A minimal self-attention architecture

### 2.1 Key-query-value self-attention

Definition given verbatim in substance: "Attention, broadly construed, is a method for taking a query, and softly looking up information in a key-value store by picking the value(s) of the key(s) most like the query" — averaging over all values, weighted more toward values whose keys are most like the query. **Self-attention** specifically means using the same elements to define the queries as define the keys and values.

For token `x_i` in sequence `x_{1:n}`:
- query: `q_i = Q x_i`, `Q ∈ R^{d×d}`
- key: `k_j = K x_j`, `K ∈ R^{d×d}`
- value: `v_j = V x_j`, `V ∈ R^{d×d}`

Contextual representation is a weighted sum of values:
`h_i = Σ_{j=1}^{n} α_{ij} v_j`

Weights from softmax over query-key affinities (dot products):
`α_{ij} = exp(q_i^T k_j) / Σ_{j'=1}^{n} exp(q_i^T k_{j'})`

Stated intuition: `Q, K, V` let the same `x_i` be projected into different "views" for the different roles of key, query, value.

### 2.2 Position representations

Worked example used throughout: **"the oven cooked the bread so"** vs. **"the bread cooked the oven so"** — different meanings (the former implies baking bread; the latter implies the bread breaking the oven), yet self-attention as defined has no built-in notion of order.

Concrete proof sketch given for the word "so": with `α_so = softmax([q_so^T k_the, q_so^T k_oven, q_so^T k_cooked, q_so^T k_the, q_so^T k_bread, q_so^T k_so])`, the individual weight `α_{so,0} = exp(q_so^T k_the) / (exp(q_so^T k_the) + ... + exp(q_so^T k_bread))`. Reordering the sentence rearranges terms in the same sum, so the value of `α_{so,0}` (and every other `α_{so,*}`) is **identical** regardless of word order — proving self-attention is order-invariant by itself. Root cause stated as two facts: (1) non-contextual embeddings `x_i = E w_i` depend only on word identity, not position; (2) the attention operation itself has no positional dependence.

Two fixes are given, framed as the only two options: "(1) use vectors that are already position-dependent as inputs, or (2) change the self-attention operation itself."

- **Learned additive position embeddings** (option 1): introduce `P ∈ R^{N×d}` (N = max sequence length the model handles), then `x̃_i = P_i + x_i`, and run self-attention on `x̃`. Cited as the approach used in BERT (Devlin et al. 2019).
- **Changing α directly** (option 2): example given is **Attention with Linear Biases (ALiBi)** (Press et al. 2022): `α_i = softmax(k_{1:n} q_i + [-i, ..., -1, 0, -1, ..., -(n-i)])`, i.e., add a bias vector that penalizes attention to tokens farther away (in either direction) from position `i`, on top of the raw dot-product scores `k_{1:n} q_i ∈ R^n`. The note's own editorial aside: "it's odd that this works; but interesting!"

### 2.3 Elementwise nonlinearity

Direct algebraic argument that stacking two self-attention layers *without* a nonlinearity collapses to something equivalent to one linear self-attention layer:

`o_i = Σ_j α_{ij} V^(2) (Σ_k α_{jk} V^(1) x_k) = Σ_k α*_{ij} V* x_k`

where `α*_{ij} = Σ_j α_{jk} α_{ij}` and `V* = V^(2) V^(1)`. The note flags this as having a "nuanced answer that's out-of-scope for this note" regarding whether that's actually a problem, but uses it to motivate inserting a feed-forward sublayer.

Standard FFN given: `h_FF = W_2 ReLU(W_1 h_self-attention + b_1) + b_2`, with the explicit dimension note that `W_1 ∈ R^{5d×d}` and `W_2 ∈ R^{d×5d}` — i.e., the FFN hidden dimension is conventionally **5x** the model dimension `d`, justified because matmuls are "an efficiently parallelizable operation, so it's an efficient place to put a lot of computation and parameters."

### 2.4 Future masking

Autoregressive constraint: `w_t ∼ softmax(f(w_{1:t-1}))` — cannot look at the future when predicting it. In an RNN this is automatic from the rollout structure (`h_{t-1} = σ(W h_{t-2} + U x_{t-1})`); in a Transformer there is nothing inherent in `α` preventing attending to `j > i`, so it's enforced explicitly by **adding a large negative constant** to pre-softmax scores for masked positions (equivalently zeroing `α_{ij}` for `j > i`):

`α_{ij,masked} = α_{ij}` if `j ≤ i`, else `0`

**Specific practical detail given:** you do NOT use `-∞` for the mask constant. The reading specifies a constant "within even the float range of the float16 encoding," giving `-10^5` as the example value. Reasoning given: using infinity can produce NaNs, and library behavior on infinite inputs is inconsistently defined; a sufficiently large negative finite constant still drives the softmax weight to exactly zero given finite precision.

Diagram described: a 5x5 attention mask over "Zuko made his uncle tea" where each row only attends to words at or before its own position (e.g., "Zuko" attends only to "Zuko"; "made" attends to "Zuko" and "made"; etc.), with `-∞` symbols marking masked (future) entries in the diagram.

### 2.5 Summary of the minimal architecture

The note explicitly enumerates the four necessary components of a minimal self-attention architecture: **(1) the self-attention operation, (2) position representations, (3) elementwise nonlinearities, and (4) future masking** (the last one specific to language modeling / autoregressive use).

## 3. The Transformer

Framed as: stacked Blocks, each containing self-attention and feed-forward layers, plus additional components beyond the minimal architecture: **multi-head self-attention, layer normalization, residual connections, and attention scaling**.

### 3.1 Multi-head self-attention

Stated intuition: "a single call of self-attention is best at picking out a single value (on average) from the input value set... it requires a balancing game in the key-query dot products in order to carefully average two or more things." Multi-head attention applies self-attention multiple times in parallel with different K/Q/V projections of the same input, then combines outputs.

For `k` heads, per-head matrices `K^(ℓ), Q^(ℓ), V^(ℓ) ∈ R^{d×d/k}` for `ℓ ∈ {1,...,k}` — note the **dimensionality reduction to `d/k` per head** is explicit and motivated later. Per-head attention:

`h_i^(ℓ) = Σ_j α_{ij}^(ℓ) v_j^(ℓ)`, with `α_{ij}^(ℓ) = exp(q_i^(ℓ)T k_j^(ℓ)) / Σ_{j'} exp(q_i^(ℓ)T k_{j'}^(ℓ))`

Output combines heads via an output projection `O ∈ R^{d×d}` applied to the concatenation of all head outputs: `h_i = O [v_i^(1); ...; v_i^(k)]`.

**Sequence-tensor / implementation framing** (a distinctive pedagogical contribution of this note): explains *why* heads use reduced dimension `d/k` rather than full `d` — multi-head attention is implemented by computing `x_{1:n}Q`, `x_{1:n}K`, `x_{1:n}V` (each `∈ R^{n×d}`) once, then **reshaping** into `R^{n,k,d/k}` and transposing to `R^{k,n,d/k}` — i.e., treating the head axis like an extra batch axis so the batched softmax runs in parallel across heads. Net result: "multi-head self-attention is no more expensive than single-head due to the low-rankness of the transformations we apply" — the total compute is the same as single-head, just distributed across lower-rank heads, with only the final linear combination (`O`) added on top. Full single-head matrix form given: `h_{1:n} = softmax(x_{1:n} Q K^T x_{1:n}^T) x_{1:n} V ∈ R^{n×d}`.

### 3.2 Layer Norm

Cites Ba et al. (2016). Two-part intuition given: (1) "reduce uninformative variation in the activations at a layer, providing a more stable input to the next layer," and (2) per Xu et al. (2019), layer norm's real benefit may be improving **backward-pass gradients** more than the forward pass.

Mechanics: statistics (mean/variance) computed **per sequence position** (per token, across the `d` hidden dimensions), independently per batch example — explicitly stated that "the statistics for the token at index `i` won't affect the token at index `j ≠ i`."

`μ̂_i = (1/d) Σ_{j=1}^d h_{ij}`, `σ̂_i = sqrt((1/d) Σ_{j=1}^d (h_{ij} - μ_i)^2)`, `LN(h_i) = (h_i - μ̂_i) / σ̂_i`

The note explicitly **omits** the learned elementwise affine (gain/bias) parameters from its presentation, citing Xu et al. (2019) that this component "seems not to be crucial, and may even be harmful."

### 3.3 Residual connections

`f_residual(h_{1:n}) = f(h_{1:n}) + h_{1:n}`. Two-part rationale given: (1) the identity function's local gradient is 1 everywhere, aiding gradient flow in deep networks; (2) it's easier to learn a function's *difference* from identity than to learn the function outright.

**Pre-norm vs. post-norm**, both formulas given explicitly:
- pre-norm: `h_pre-norm = f(LN(h)) + h`
- post-norm: `h_post-norm = LN(f(h) + h)`

Citing Xiong et al. (2020): pre-norm gradients are much better at initialization, leading to faster training — this is presented as the resolved answer for why pre-norm is preferred.

### 3.4 Attention logit scaling

From Vaswani et al. (2017), "scaled dot-product attention." Stated intuition: as dimensionality `d` of the query/key vectors grows, the dot product of even random (e.g., at-initialization) vectors grows roughly as `√d`, so dividing by `√d` counteracts this scaling:

`α = softmax(x_{1:n} Q K^T x_{1:n}^T / √d) ∈ R^{n×n}`

### 3.5 Transformer Encoder

Embeds sequence via `E`, adds position representation, applies a stack of independently-parameterized Encoder Blocks (each: multi-head attention + Add&Norm, then feed-forward + Add&Norm), **no future masking**. Stated use case: "great in contexts where you aren't trying to generate text autoregressively... and want strong representations for the whole sequence" — every position, including the first token, can see the entire sequence. For getting token-level probabilities (e.g., masked LM as in BERT), a linear layer + softmax is applied to the output.

### 3.6 Transformer Decoder

Differs from the Encoder only by applying future masking at every self-attention call, "to ensure that the informational constraint... holds throughout the architecture." Named examples cited: **GPT-2** (Radford et al. 2019), **GPT-3** (Brown et al. 2020), **BLOOM** (Workshop et al. 2022).

### 3.7 Transformer Encoder-Decoder and Cross-Attention

Takes two sequences: source `x_{1:n}` through a Transformer Encoder, target `y_{1:m}` through a modified Decoder that adds **cross-attention** from the decoder's intermediate representations to the encoder's output.

Cross-attention defined explicitly as: queries come from the decoder side, keys/values come from the encoder side:
`q_i = Q h_i^(y)` for `i ∈ {1,...,m}`; `k_j = K h_j^(x)`, `v_j = V h_j^(x)` for `j ∈ {1,...,n}`, where `h^(x)_{1:n} = TransformerEncoder(w_{1:n})`.

The note's own framing of cross-attention: "isn't that just what attention always was before we got into this self-attention business? Yeah, pretty much." Cross-attention always attaches to the **final** encoder block's output (not per-layer encoder outputs).

Stated tradeoff: encoder-decoders give bidirectional context for the source (useful e.g. for summarization) while still generating autoregressively, citing Raffel et al. (2020) (T5) that this architecture can outperform decoder-only models "at modest scale" — but it splits parameters between encoder and decoder, and "most of the largest Transformers are decoder-only."

## Architecture diagrams described in the reading

- **Transformer Decoder** (no cross-attention): Embeddings → Add Position Embeddings → [Block: Masked Multi-Head Attention → Add&Norm → Feed-Forward → Add&Norm] × repeat → Linear → Softmax → Probabilities.
- **Transformer Encoder**: Embeddings → Add Position Embeddings → [Block: Multi-Head Attention → Add&Norm → Feed-Forward → Add&Norm] × repeat → Linear → Softmax → Probabilities.
- **Transformer Encoder-Decoder**: Encoder Inputs → Encoder stack (Multi-Head Attention → Add&Norm → Feed-Forward → Add&Norm, repeated) producing representations attended to only by the last Encoder Block's output; Decoder Inputs → Decoder stack (Masked Multi-Head Attention → Add&Norm → Multi-Head [cross-]Attention → Add&Norm → Feed-Forward → Add&Norm, repeated) → Linear → Softmax → Probabilities.

## Full reference list (as cited in the reading)

Ba, Kiros & Hinton (2016) Layer Normalization; Bahdanau, Cho & Bengio (2014) Neural MT by jointly learning to align and translate; Baum & Petrie (1966) HMMs; Bengio et al. (2000, 2003) neural probabilistic language model; Brown et al. (2020) GPT-3; Collobert et al. (2011) NLP almost from scratch; Cortes & Vapnik (1995) SVMs; Devlin et al. (2019) BERT; Elman (1990) Finding structure in time; Fukushima & Miyake (1982) Neocognitron; Lafferty, McCallum & Pereira (2001) CRFs; LeCun et al. (1989) Backprop for zip code recognition; Manning (2022) Human Language Understanding & Reasoning; Mikolov et al. (2013) word2vec; Press, Smith & Lewis (2022) ALiBi; Radford et al. (2019) GPT-2; Raffel et al. (2020) T5; Rong (2014) word2vec parameter learning explained; Rumelhart, Hinton & Williams (1985, 1988) backprop; Schütze (1992) Dimensions of meaning; Vaswani et al. (2017) Attention Is All You Need; Workshop et al. (BigScience, 2022) BLOOM; Xiong et al. (2020) On layer normalization in the Transformer architecture; Xu et al. (2019) Understanding and improving layer normalization.
