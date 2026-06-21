# Hybrid Architectures

Transformers don't handle long context efficiently; RNNs / state-space models do, but lose the expressivity advantages of softmax attention. **Hybrid architectures** combine both — typically interleaving attention layers with linear-attention or state-space layers — to keep transformer-class performance while drastically reducing long-context cost. Frontier-2026: Mamba-2 powers production hybrids (Nemotron-H, Falcon H1); Qwen3-Next uses a gated DeltaNet update; Kimi's next model is expected to use "kimi delta attention". Hybrids are promising but still **harder to reason about and operationalize** than pure transformer baselines.

## The Core Idea

Drop the softmax from attention to get a recurrent structure. Standard attention output at token `t`:

```
o_t = sum_{i ≤ t} softmax(q_t · k_i / sqrt(d)) * v_i
```

Drop the softmax:

```
o_t = sum_{i ≤ t} (q_t · k_i) * v_i = q_t · (sum_{i ≤ t} k_i v_i^T)
```

Defining `S_t = sum_{i ≤ t} k_i v_i^T`, we get a recurrent relation:

```
S_t = S_{t-1} + k_t v_t^T
o_t = q_t · S_t
```

`S_t` summarizes all past key-value pairs in a fixed-size state, making per-token cost O(1) in sequence length.

## Why It Needs More Than Just Dropping Softmax

Softmax stabilizes training; the bare linear form is unstable without normalization. Adding a learned **forget gate** for the previous state helps:

```
S_t = γ_t ⊙ S_{t-1} + k_t v_t^T
```

where `⊙` is elementwise multiplication and `γ_t` is the gate. This is the RNN-style "forget the past selectively" pattern that makes linear attention competitive.

## Production Hybrids

- **Mamba-2**: the most popular linear / state-space variant in 2026. Used in:
  - **Nemotron-H** (NVIDIA)
  - **Falcon H1** (TII)
- **Qwen3-Next**: uses a **gated DeltaNet** update — a variation on the delta-rule update that gives finer control over state evolution.
- **Kimi's next model**: likely uses "kimi delta attention" (announced but not fully detailed at time of writing).

## Mamba-2 and State Space Duality (SSD)

[Dao & Gu's "Transformers are SSMs" (2405.21060)](../../papers/02-architecture/alternatives/Transformers are SSMs: Generalized Models and Efficient Algorithms for Sequence Modeling - 2405.21060.pdf) is the foundational paper for Mamba-2 and is worth understanding even if you never train an SSM. The core theoretical result — **State Space Duality (SSD)** — is that structured SSMs and certain attention variants are **dual forms of the same operation** on structured (semiseparable) matrices. The duality has practical consequences:

- **Recurrent (linear) form**: linear in sequence length, fixed-size state during generation — the "SSM view".
- **Dual quadratic form**: `(L ∘ QK^T) · V` where `L_{ij} = a_i × ⋯ × a_{j+1}` for `i ≥ j` and 0 otherwise. This is masked attention with a *data-dependent positional mask* — softmax is dropped, and the `L` matrix replaces RoPE / sinusoidal positional encoding with input-dependent decay.

The big insight: this duality lets you transfer the entire transformer systems-engineering toolbox (tensor parallelism, FlashAttention-style tiling, sequence parallelism, variable-length batching) to SSMs. Concretely, the **SSD algorithm** uses block decompositions of semiseparable matrices and runs in time `O(TN)` for sequence length T and state size N — **2–8× faster than the optimized Mamba-1 selective scan** while supporting state sizes 8× larger (or even bigger with minimal slowdown). It crosses over with FlashAttention-2 at sequence length 2K and is 6× faster at 16K.

Architecture-side changes from Mamba to Mamba-2:
- **A matrix simplified from diagonal to scalar-times-identity** — each `A_t` is a single scalar. Slight loss of expressivity, big speedup.
- **Larger head dimension** P = {64, 128} (Mamba used P = 1), matching modern transformer conventions.
- **All data-dependent projections moved to the start of the block**, allowing tensor parallelism in Megatron style with half the per-block synchronization points.
- **Multi-input SSM (MIS) head structure**, analogous to **multi-value attention (MVA)** — Mamba's heads are analogous to MHA heads, and you can mix in grouped-value attention (GVA) for finer control.

Empirical: Mamba-2 with 2.7B params trained on 300B Pile tokens outperforms Mamba-2.8B, Pythia-2.8B, and even Pythia-6.9B on the same dataset. Pareto-dominates both Mamba-1 and Transformer++ in Chinchilla scaling. Tensor-parallel friendly, sequence-parallel for very long sequences (recurrent state passed between devices), and trains on variable-length sequences with no padding tokens at all — useful when fine-tuning on mixed-length data.

However: [RULER (2404.06654)](../../papers/04-efficiency/context-extension/RULER: What's the Real Context Size of Your Long-Context Language Models - 2404.06654.pdf) showed that **non-Transformer architectures (RWKV, Mamba-2.8B-slimpj) still lag the Llama2-7B Transformer baseline by large margins** on real long-context tasks beyond simple NIAH. Both degrade significantly at 8K and underperform up to 4K. Pure-SSM long-context performance is not yet competitive with transformers — which is why hybrids dominate in production.

## Why Hybrid (Not Pure Linear)

Pure linear attention loses some compositional and exact-recall capacities of softmax attention — including the multi-query associative recall task. Hybrids typically interleave **linear-attention layers with full-attention layers** — full attention layers handle precision-critical operations, linear layers carry the bulk of the long-context computation cheaply.

The interleaving ratio is itself a design choice, similar to the global/local interleaving in attention-pattern-modified transformers (see [[attention-variants]]).

## Tradeoffs

- **Compute**: linear-time per token, fixed-size state. Major win for long-context.
- **Training stability**: state-space models can be harder to train; many recipes use careful initialization or gating to stabilize.
- **Tooling**: less mature than transformer infra. Fewer reference kernels, fewer benchmarks for ablation comparison.
- **Reasoning about behavior**: harder to interpret what a state-space layer does compared to attention with its explicit token-level interactions.

## Sleep: Consolidating Evicted Context into Fast Weights

A 2026 CMU/University of Maryland paper, ["Do Language Models Need Sleep?"](../../papers/04-efficiency/context-extension/Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference - 2605.26099.pdf), probes a specific weakness of the SSM-attention hybrid recipe above: when the attention KV cache is evicted (window full, tokens dropped), the SSM blocks' fixed-size fast weights are supposed to carry forward whatever the attention cache can no longer hold. The paper shows this works for *storage* but not for *deep computation* — a 4-layer GDN-attention hybrid (attention → GDN → attention → GDN) with hard eviction every 24 tokens degrades to near-random guessing on a Rule-110 cellular-automaton task as the required reasoning depth increases, even though the amount of information to store is held constant. The bottleneck isn't fast-weight capacity, it's that a single forward pass isn't enough computation to transform raw evicted tokens into a fast-weight state that supports later multi-step reasoning.

The fix is a **sleep phase**: right before the KV cache for a window is cleared, the model runs N *additional* offline forward passes over that about-to-be-evicted context, using each pass to further refine the SSM fast-weight update (`S_t = α_t·S_{t-1} + β_t·v_t k_t^T`, a gated Hebbian/delta-rule rule, the same family as the linear-attention recurrence in this page's "Core Idea" section). Wake-time prediction still costs one forward pass; the extra reasoning compute is paid offline. On the k-hop Depo graph-retrieval task and on GSM-Infinite math reasoning (fine-tuning Jet-Nemotron 2B and Ouro 1.4B), increasing N specifically improves the hardest, deepest-reasoning examples — e.g., Jet-Nemotron's eight-operation GSM-Infinite accuracy rose from 0.351 to 0.388 with 6 sleep loops — while easy examples saturate regardless of loop count. The same mechanism extends to sliding-window eviction (rather than hard-clearing the whole context window): fine-tuning Ouro 1.4B with a small window (L=512) and sleep loops raised GSM-Infinite two-operation accuracy from 0.596 to 0.905, a 52% relative gain, indicating sleep helps even basic retrieval once the active window is much smaller than the problem. The tradeoff is training cost: sleep makes training sequential across context windows (window j+1 can't start until window j's N sleep passes finish) and cost scales roughly linearly with N — the authors note this only hurts wall-clock time when the window is too small to keep the GPU saturated. See [[kv-cache|the KV-cache wiki page]] for the full mechanism writeup alongside related cache-replacement work (Cartridges, IndexCache).

## When to Reach for a Hybrid

- You need **very long context** (>128k tokens) and softmax attention's quadratic cost is prohibitive.
- You can absorb the engineering cost of less-mature infra.
- Your team has the bandwidth to do hybrid-specific ablations rather than relying on transformer-tuned defaults.

If any of these aren't true, a transformer with [[positional-encodings|RNoPE/YaRN]] + document masking + a long-context attention pattern (sliding window, DCA — see [[attention-variants]]) remains the safer default.

## Related Topics

- [[attention-variants]] — long-context attention patterns that achieve similar goals within pure transformers
- [[transformer-architecture]] — the baseline being hybridized
- [[positional-encodings]] — RNoPE/YaRN are the alternative path to long context within transformers
- [[kv-cache]] — the Sleep mechanism's offline recurrence is a direct alternative to KV-cache eviction/compression; see also Cartridges and IndexCache there
- [[frontier-training-playbook]] — where hybrid architectures sit in the decision tree
- [[diffusionblocks-blockwise-training]] — a different recurrence axis: weight-shared recurrent-depth transformers, trained via block-wise local losses instead of full BPTT

## Sources

- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality / Mamba-2 (2405.21060)](../../papers/02-architecture/alternatives/Transformers are SSMs: Generalized Models and Efficient Algorithms for Sequence Modeling - 2405.21060.pdf) — SSD framework, 2–8× over Mamba-1, TP-friendly architecture.
- [RULER (2404.06654)](../../papers/04-efficiency/context-extension/RULER: What's the Real Context Size of Your Long-Context Language Models - 2404.06654.pdf) — non-Transformer architectures (RWKV, Mamba) lag Transformers at long context.
- [Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference (2605.26099)](../../papers/04-efficiency/context-extension/Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference - 2605.26099.pdf) — sleep-time consolidation of evicted context into SSM fast weights in attention-SSM hybrids.
- Nemotron-H, Falcon H1, Qwen3-Next technical reports.
