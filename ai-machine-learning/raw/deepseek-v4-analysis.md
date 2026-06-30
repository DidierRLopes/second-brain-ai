# DeepSeek-V4: Architecture and Training Breakdown

**Source:** https://www.k-a.in/DeepSeek-V4.html
**Author:** arjun (https://x.com/arjunkocher) — a third-party breakdown of DeepSeek's own technical report
**Underlying report:** DeepSeek-AI, "DeepSeek_V4.pdf" (huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) — referred to in this writeup as a "preview" release
**Companion summary:** elie bakouch's thread (x.com/eliebakouch/status/2047519300399837677)

## Framing / Motivation

Reasoning models' test-time scaling is bottlenecked by the quadratic cost of vanilla attention over ultra-long contexts. DeepSeek-V4 targets million-token contexts and agentic/long-horizon workloads by replacing attention wholesale rather than incrementally optimizing MLA.

**Headline efficiency numbers (vs. DeepSeek-V3.2, at 1M-token context):**
- DeepSeek-V4-Pro: 27% of the single-token inference FLOPs (FP8-equivalent) and 10% of the KV cache size of V3.2.
- DeepSeek-V4-Flash: 10% of the single-token FLOPs and 7% of the KV cache size of V3.2.
- Against a BF16 GQA-8, head-dim-128 baseline, KV cache size drops to ~2% in the 1M-context setting.

**Headline capability claim:** DeepSeek-V4-Pro-Max (max reasoning-effort mode of V4-Pro) is described as matching GPT-5.4 on coding competitions — "the first time an open model has matched a closed model on this task."

## Architectural Changes vs. DeepSeek-V3

Retained from V3: DeepSeekMoE framework (fine-grained routed experts + shared experts), Multi-Token Prediction (MTP) strategy/loss.

New in V3→V4:
- **Hash-routed MoE in early layers.** The first several Transformer blocks replace dense FFN with MoE layers using **Hash routing** — expert assignment via a predefined hash of the input token ID (not a learned router) for those layers.
- Load balancing still auxiliary-loss-free, augmented with a **sequence-wise balance loss** to prevent extreme intra-sequence imbalance.
- **MLA is gone.** Replaced by a hybrid of two new attention mechanisms: **Compressed Sparse Attention (CSA)** and **Heavily Compressed Attention (HCA)**.
- **Manifold-constrained Hyper-Connections (mHC)** strengthen residual connections — expands the residual stream width by factor `n_hc = 4`. Update rule: `X_{l+1} = B_l X_l + C_l F_l(A_l X_l)`. The residual mapping matrix `B_l` is constrained to the Birkhoff polytope (doubly stochastic matrices: rows/columns sum to 1, non-negative), bounding its spectral norm ≤ 1 for non-expansive, numerically stable residual transforms even in deep stacks. `A_l` and `C_l` are bounded non-negative via Sigmoid. All three mappings are dynamically generated (split into input-dependent + static components).
- **Muon optimizer** adopted for training (replacing/supplementing AdamW for most modules), for faster convergence and stability.

## CSA (Compressed Sparse Attention) — centerpiece mechanism

1. Compresses KV cache every `m` tokens into one entry (m=4 in both V4 variants), via two parallel KV projections `C_a, C_b` with learned compression weights `Z_a, Z_b` and learnable positional biases `B_a, B_b`, combined through a row-softmax gating equation. Compression is **overlapped** between adjacent blocks (each `C_i^Comp` derived from 2m KV entries), netting an effective 1/m sequence-length reduction.
2. **Lightning Indexer**: after compression, applies DeepSeek Sparse Attention (DSA) to select top-k compressed entries. Index score for query token t against compressed block s: `I_{t,s} = Σ_h w_{t,h} · ReLU(q_{t,h} · K_s^{IComp})`. The indexer's attention computation runs in **FP4 precision** for speed at long context.
3. A **sliding window** branch (`n_win = 128` uncompressed KV entries) is concatenated with selected compressed entries for local fine-grained dependencies; causality is strictly preserved (queries only see preceding compressed blocks).
4. Output goes through **Shared Key-Value MQA** (compressed entries serve as both key and value).

## HCA (Heavily Compressed Attention) — second mechanism

Similar compression strategy to CSA but with a much larger, non-overlapped compression rate `m' = 128` (vs CSA's m=4) and **no sparse selection** — keeps dense attention over the (now much shorter) compressed sequence. Also uses shared-KV MQA and grouped output projection.

**Layer interleaving:**
- DeepSeek-V4-Flash: first 2 layers = pure sliding-window attention; remaining layers interleave CSA/HCA.
- DeepSeek-V4-Pro: first 2 layers = HCA; remaining layers interleave CSA/HCA.

## Shared attention-mechanism details (apply to both CSA and HCA)

- **Grouped Output Projection**: splits `n_h` outputs into `g` groups, each projected to a smaller `d_g`-dim intermediate (where `d_g < c·n_h/g`) before final projection — avoids the cost of projecting a very large `c·n_h` dimension directly.
- **Partial RoPE**: applied only to the last 64 dimensions of query/KV vectors; RoPE with position `-i` reapplied to core-attention outputs so they carry relative (not absolute) position info, since compressed KV entries serve as both keys and values.
- **QK Normalization**: RMSNorm applied to each query head and the (single) compressed KV head before core attention, to prevent exploding logits. Because of this, **QK-Clip is not needed** in the Muon optimizer.
- **Attention Sink**: learnable per-head sink logits `z'_h` added to the attention-score denominator, letting a head's total attention mass be less than 1 (even near 0).
- **Mixed-precision KV storage**: BF16 for RoPE dimensions, FP8 for the rest.

## Infrastructure / Systems Engineering

- **MoE Expert Parallelism overlap**: experts split into "waves"; computation for a wave starts as soon as its communication finishes, without waiting on other experts — computation, next-wave token transfer, and result-sending of finished experts proceed concurrently. Claimed **1.92x speedup** vs. **1.42x for Comet** (a prior overlap baseline). Implementation open-sourced as **MegaMoE**, a CUDA mega-kernel, part of DeepGEMM.
- Hardware-vendor proposals: target compute/communication balance rather than raw bandwidth scaling; address power throttling under extreme kernel fusion; move from pull-based to push-based cross-GPU communication primitives; replace SwiGLU with a cheaper element-wise activation (no exp/division).
- **TileLang**: DSL for GPU kernels. "Host Codegen" cuts CPU-side validation overhead from tens/hundreds of microseconds to <1 microsecond per invocation. Integrates the **Z3 SMT solver** for formal integer analysis of tensor index arithmetic (translated to QF_NIA), improving vectorization/barrier-insertion/code-simplification passes at the cost of a few seconds of compile time.
- **Batch invariance / determinism**: avoids split-KV (which spreads one sequence's attention across multiple SMs) via a dual-kernel strategy — one kernel for fully-occupied waves (single SM per sequence), one for partially-filled waves (multi-SM, carefully ordered accumulation to stay bitwise-identical to the first kernel), using distributed shared memory across SM thread-block clusters.
- **FP4 Quantization-Aware Training (QAT)** applied to (1) MoE expert weights and (2) the CSA indexer's QK path. FP4→FP8 dequantization is lossless (FP8 E4M3 has 2 more exponent bits than FP4 E2M1, absorbing the FP4 sub-block scale range). Index scores quantized FP32→BF16 for a 2x top-k-selector speedup while preserving 99.7% KV-entry recall. During RL rollout/inference (no backward pass), real FP4 weights are used rather than simulated quantization.
- **Muon + ZeRO**: AdamW retained only for embeddings, prediction head, mHC static biases/gates, and RMSNorm weights; everything else uses Muon. Dense params assigned to ZeRO ranks via a knapsack algorithm; MoE expert matrices (down/up/gate projections) flattened and padded across ranks. Newton-Schulz iterations batched across same-shape parameters; remain stable in BF16, so MoE gradients are quantized to BF16 with stochastic rounding for data-parallel sync (halves comm volume), using a two-phase all-to-all + local FP32 sum instead of tree/ring reduce-scatter to avoid low-precision accumulation error.
- **Hybrid Newton-Schulz iteration** (Muon's matrix orthogonalization): 10 iterations in two stages — first 8 with coefficients (a,b,c) = (3.4445, -4.7750, 2.0315) for rapid convergence toward singular values of 1; final 2 with (a,b,c) = (2, -1.5, 0.5) to stabilize precisely at 1.
- **Contextual Parallelism**: two-stage communication for CP ranks — rank i sends its last m uncompressed KV entries to rank i+1 before local compression, then an all-gather across CP ranks collects compressed KV entries, reorganized via a fused select-and-pad operator.
- **KV cache management**: two-part cache (classical KV cache for CSA/HCA + a state cache for SWA and not-yet-compressed tokens); each classical cache block covers `lcm(m, m')` original tokens. Three on-disk SWA caching strategies: Full SWA Caching, Periodic Checkpointing (every p tokens), Zero SWA Caching (recompute last `n_win · L` tokens from CSA/HCA cache instead of storing SWA KV at all).
- **Activation checkpointing**: tensor-level, automatic-differentiation-aware, built on TorchFX tracing; developers annotate tensors and the framework finds the minimal recomputation subgraph via backward traversal.

## Pre-Training

- Corpus: **more than 32T tokens**, built on top of V3's pretraining data — higher-quality web data (filtered to remove templated/auto-generated content, mitigating model collapse), strong math/code corpora, agentic data added during mid-training for coding, larger multilingual corpus, emphasis on long-document data (scientific papers, technical reports).
- Tokenizer: same vocabulary size as V3, **128K**, with a few new special tokens added for context construction ("Quick Instruction" tokens).

**DeepSeek-V4-Flash hyperparameters:**
- 43 Transformer layers, hidden dim d = 4096.
- First 2 layers: pure sliding-window attention; rest interleave CSA/HCA.
- CSA compression rate m=4, attention top-k=512; HCA compression rate m'=128.
- MoE: 1 shared expert + 256 routed experts, 6 activated per token.
- Trained on 32T tokens.
- AdamW (where used): β1=0.9, β2=0.95, ε=1e-20, weight decay=0.1.
- Muon: momentum=0.95, weight decay=0.1, update-matrix RMS rescaled to 0.18.
- LR: warmup over 2000 steps to peak 2.7e-4, cosine decay to 2.7e-5.
- Sequence length curriculum: 4K → 16K → 64K → 1M.
- Attention curriculum: dense attention for first 1T tokens → indexer warmup → sparse attention for the remainder.
- MTP loss weight: 0.3, reduced to 0.1 at the start of LR decay.

**DeepSeek-V4-Pro hyperparameters:**
- 61 Transformer layers, hidden dim d = 7168.
- First 2 layers: HCA; rest interleave CSA/HCA.
- CSA compression rate m=4, attention top-k=1024; HCA compression rate m'=128.
- MoE: 1 shared expert + 384 routed experts, 6 activated per token.
- Trained on 33T tokens.
- LR: peak 2.0e-4, end 2.0e-5. Longer dense-attention warmup stage than Flash.
- Auxiliary-loss-free bias update speed: 0.001; balance loss weight: 0.0001.
- MTP loss weight: 0.3 (most of training), 0.1 after LR decay starts.

## Training Stability Fixes

Both models hit instability tied to **outliers in the MoE layers**, exacerbated by the routing mechanism; naive rollbacks didn't prevent recurrence.

- **Anticipatory Routing**: decouples backbone and router update timing. At step t, current params θ_t compute features, but routing indices use historical params θ_{t-Δt} (computed/cached in advance at step t-Δt to avoid double-loading weights). An automatic detector triggers a short rollback and activates this mode only when a loss spike occurs, then reverts to normal training after a period.
- **SwiGLU Clamping**: the SwiGLU linear component is clamped to [-10, 10], with the gate component's upper bound also capped at 10. Eliminates outliers without hurting performance.
- The authors note Anticipatory Routing and SwiGLU Clamping work empirically but their underlying principles are "insufficiently understood."

## Product-Level Features

- **Tool-call schema**: new `|DSML|` special token with an XML-based tool-invocation format, reducing escaping failures/tool-call errors vs. prior formats.
- **Interleaved Thinking**: in tool-calling scenarios, full reasoning history is preserved across the entire conversation (including across user-turn boundaries) — a change from DeepSeek-V3.2, which discarded thinking traces on every new user turn. In general conversational scenarios, the V3.2 behavior (discard reasoning on new user message) is retained. Note: agent frameworks that simulate tool calls via user messages won't trigger the persistent-reasoning path; non-think models are recommended for those.
- **Quick Instruction**: dedicated special tokens appended to the input sequence for auxiliary tasks (e.g., web-search triggering, intent recognition) that reuse the existing KV cache directly instead of requiring a separate small model with redundant prefilling — reduces user-perceived time-to-first-token.

## Post-Training

Largely mirrors V3.2's pipeline with one major substitution: **the mixed RL stage was entirely replaced by On-Policy Distillation (OPD).**

- **Specialist training**: each model goes through fine-tuning then RL via **GRPO** with domain-specific prompts/rewards, producing distinct specialists across reasoning-effort tiers: **Non-think**, **Think High**, **Think Max**. Think Max prepends an explicit "Reasoning Effort: Absolute maximum, no shortcuts permitted" instruction to the system prompt, directing exhaustive decomposition, adversarial stress-testing, and documentation of every considered/rejected alternative.
- **Generative Reward Model (GRM)**: for hard-to-verify tasks, rubric-guided RL data is curated and a GRM judges policy trajectories; RL optimization is applied directly to the GRM itself, with the actor network natively serving as the GRM — jointly optimizing generation and judgment, achieving strong performance from a minimal set of human annotations.
- **On-Policy Distillation (OPD)**: the primary technique for merging N domain specialists `{π_E1...π_EN}` into one final model. Objective: `L_OPD(θ) = Σ_i w_i · D_KL(π_θ || π_Ei)` (reverse KL), sampled on-policy from the student. **More than ten teacher models** across domains distill into a single student — full-vocabulary logit distillation (not top-k), which the authors say gives more stable gradients and more faithful distillation than weight-merging or mixed-RL approaches. Efficiency trick: teacher weights offloaded to centralized distributed storage and loaded on demand; only last-layer teacher hidden states are cached, full logits reconstructed on the fly — avoids materializing full teacher logits in memory.

## Benchmark / Evaluation Claims

- **Long context**: DeepSeek-V4-Pro beats Gemini-3.1-Pro on MRCR (in-context retrieval) but trails Claude Opus 4.6. Retrieval stable through 128K context; visible degradation beyond 128K but still "remarkably strong" at 1M tokens vs. both proprietary and open-source peers. Also beats Gemini-3.1-Pro on CorpusQA.
- **Reasoning**: DeepSeek-V4-Pro-Max leads all prior open models, matches closed SOTA on many metrics. Ranks **23rd among human candidates on Codeforces**, but falls "marginally short" of GPT-5.4 and Gemini-3.1-Pro — estimated 3-6 months behind frontier closed models.
- **Agent**: on par with leading open models (**Kimi-K2.6**, **GLM-5.1**), slightly behind frontier closed models on public benchmarks. Internally, outperforms **Claude Sonnet 4.5** and approaches **Opus 4.5**.

## Stated Future Directions

Distill the hybrid CSA/HCA architecture toward something more "elegant" without losing performance; explore further model sparsity (e.g., sparse embedding modules); continue work on low-latency long-context serving; deepen agentic/long-horizon task support; multimodal capabilities are in progress (not yet in this release).

Model weights and inference code: huggingface.co/collections/deepseek-ai/deepseek-v4 (per this writeup; not independently verified).
