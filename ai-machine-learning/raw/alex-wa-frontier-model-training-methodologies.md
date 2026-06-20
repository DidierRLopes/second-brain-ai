# Frontier Model Training Methodologies

**Source:** Alex Wa's Blog
**URL:** https://djdumpling.github.io/2026/01/31/frontier_training.html
**Author:** Alex Wa
**Date:** January 31, 2026
**Fetched:** 2026-05-25

A synthesis of how frontier labs train multi-billion parameter open-weight models, drawing primarily from Hugging Face's SmolLM3 report and supplemented by Prime Intellect's Intellect-3, Nous Research's Hermes 4, OpenAI's gpt-oss-120b, Moonshot's Kimi K2, DeepSeek's DeepSeek-R1, and Arcee's Trinity series. The emphasis is on training methodology over infrastructure.

## TL;DR

- Frontier training is a systems problem: data mixture, architecture, and stability choices dominate most algorithmic tweaks.
- Start from a strong baseline and ablate fast and reliably; derisk changes and avoid multi-variable edits.
- For long context, document masking + RNoPE/YaRN-style scaling is a robust default; attention variants trade compute for reach.
- GQA with small groups (2/4/8 groups) typically outperforms MHA and MQA in ablations at similar model scales; MLA cuts KV cache but raises implementation complexity.
- MoE is efficient when load-balanced; routing, auxiliary or bias balancing, and global stats are non-negotiable.
- Tokenizer design should mirror target data; vocab size trades embedding cost against token compression and KV cache.
- AdamW is still the default; Muon can help but needs careful infra (all-to-all, padding, scaling quirks).
- Scaling laws guide, but many frontier models overtrain; inference cost and sparsity tradeoffs often drive final choices.
- Data scheduling matters: multi-stage mixtures and late-stage high-quality injection shape final behavior.
- Mid-training and post-training (SFT + preference/RL/distillation) often determine reasoning and tool-use behavior.
- Training ops are frequent failure points: dataloader design, throughput, seeds in TP, and checkpointing.
- Most training failures stem from common causes: high learning rates, problematic data batches, load imbalance in MoE models, or storage/infrastructure issues.

## Models Covered (selected)

| | Kimi-K2 | Trinity Large | gpt-oss-120b | OLMo 3 | SmolLM |
|---|---|---|---|---|---|
| Parameter Count | 1.06T | 400B | 116.83B | 32B | 3B |
| Attention | MLA | GQA (8 groups) | GQA (8 groups) | GQA (?) | GQA (4 groups) |
| Positional Embedding | RoPE (?) + YARN | RoPE + YARN | RoPE + YARN | RoPE + YARN | RNoPE + YARN |
| Architecture | MoE | MoE | MoE | dense | dense |
| Tokenizer | tokenization_kimi | custom | o200k_harmony | cl_100k | Llama3 |

## A Minimal Training Playbook

1. Define the product goal and lock evals early across knowledge, math, code, long-context, and instruction following.
2. Pick a baseline architecture with known failure modes; default to dense + GQA + RoPE/RNoPE unless MoE is essential.
3. Choose a tokenizer matched to your target languages and domains; freeze vocab and special tokens early.
4. Build the data pipeline with deduplication, filtering, and contamination checks; measure data quality explicitly.
5. Run small ablations for attention, positional encoding, optimizer, and learning rate schedule; change one variable at a time.
6. Plan a multi-stage data mixture; delay the best data and reasoning-heavy data toward the end.
7. Add stability guardrails: logit softcapping (preferred, per Gemma) or z-loss/QK-norm, gradient clipping, precision policy, loss spike alerts.
8. Validate throughput on long runs and confirm dataloader behavior (packing, shuffling, random access).
9. Run the main training with interval evals and consistent seeds, especially for tensor parallelism.
10. Mid-train for domain gaps if SFT reveals them; extend context length gradually (4k → 32k → 64k → 128k).
11. Post-train with SFT, then choose preference/RL/distillation based on verifiable rewards and tool-use goals.
12. Re-evaluate, run safety checks, and lock a release checkpoint with full logs and configs.

## General Practices

> "Learn to identify what's worth testing, not just how to run tests. Perfect ablations on irrelevant choices waste as much compute as sloppy ablations on important ones."

- Ablations need to be fast (faster iteration -> more hypotheses tested) and reliable (need strong discriminative power, otherwise it may be noise).
- Choose an established baseline with good architecture and training setup design. These take years of iteration.
- Follow the principle of *derisking*: "never change anything unless you've tested that it helps."
- In evals, look for monotonicity (score improvement), low noise (e.g. resistance to random seeds), above-random performance, and ranking consistency (ranking of approaches should remain stable throughout training).
- Prioritize evals! Core evals should be preserved between pre-training and post-training, and their implementation should be finished long before the base model is finished training.
- Balance exploration and execution. For methods, choose flexibility and stability over peak performance; set a deadline for exploration.

## Architecture and Set-up

### Architecture Decision Heuristics

- If memory- or infra-constrained, default to a dense model with GQA and RoPE/RNoPE.
- If you need inference efficiency at scale and can manage routing complexity, consider MoE with strong load balancing.
- If long context is a core requirement, plan for document masking plus RoPE scaling (ABF/YaRN) or RNoPE variants.
- If you need simpler kernels and faster iteration, avoid novel attention variants unless you can ablate them cleanly.

Hugging Face's dense-vs-MoE decision tree: pick dense if memory-constrained (MoEs must keep all experts loaded), new to LLM training, or working under a tighter timeline.

### Attention

- **MHA** uses separate Q/K/V projections per head, with a large KV-cache that bottlenecks inference and hoards GPU memory.
- **MQA** shares KV across all heads, saving KV cache but leaking attention capacity because heads can't specialize their stored info.
- **GQA** shares KV across small head groups (e.g. 4) — a middle ground.
- **MLA** stores a compressed latent variable that decompresses/projects into KV at runtime, often achieving 4-8x compression with stronger performance than MQA at comparable parameter counts.

Hugging Face found GQA with 2/4/8 groups beat MHA in ablations, while MHA beat MQA and GQA with 16 groups across HellaSwag, MMLU, ARC.

### Gated Attention

Apply an elementwise gating mechanism to the scaled dot-product attention output before the output projection. A gate vector is computed from the input via a learned projection through a sigmoid, then split across heads. Each head's attention output is elementwise-multiplied by its gate segment, the gated outputs are concatenated, and projected through the output matrix.

Benefits: reduces attention sinks (tokens receiving disproportionately high attention), reduces large activations that destabilize training, improves evals and long-sequence generalization. Critically, it stabilizes training and reduces loss spikes.

### Document Masking

Pre-training uses fixed sequence-length tensors [batch, seqlen, hidden] for GPU efficiency. To avoid padding waste, packing concatenates documents within a sequence. With standard causal masking, tokens from unrelated document A can attend to document B, degrading performance. Intra-document masking restricts attention to tokens within the same document.

Hugging Face saw small PIQA improvements but no notable impact on short-context tasks; it becomes crucial when scaling from 4k to 64k tokens. For smaller models, the overhead may not be worth it.

### Embedding Sharing

Input and output embeddings together are `2 * vocab_size * d_model`. In small LMs, this can be up to 20% of params (Llama 3.2 1B); in larger models it's small (3% in Llama 3.1 70B). Tying them saves params but mixes geometries; frequent tokens like "the" can dominate representation learning by getting gradients from both streams.

On a 1.2B Hugging Face model, tied embeddings did comparably with 18% fewer params (down from 1.46B). An untied model with matched 1.2B params (fewer layers) showed higher loss and lower downstream scores.

### Positional Encodings

- Without positional encoding, transformers are bag-of-words.
- Absolute embeddings: learned lookup; limited to training-time max length.
- Relative encodings: capture distance between tokens.
- **RoPE**: encodes position by rotating Q/K vectors in 2D planes. Splits Q/K into pairs and rotates by an angle proportional to absolute position and a base frequency. Attention's dot product encodes relative distance via the phase difference.

When extending context, rotation angles grow; fix via base-frequency increase (**ABF**) or **YaRN**, which does granular interpolation per component plus dynamic attention scaling and temperature adjustment. YaRN does best for extremely long contexts; gpt-oss-120b used it to extend dense-layer context to 131k.

- **NoPE**: causal masking only; no rotation, so no extrapolation issues, but weaker short-context performance.
- **RNoPE**: alternates RoPE and NoPE attention blocks. RoPE for local; NoPE for long-range retrieval.
- **Partial RoPE**: applies RoPE/NoPE within the same layer.

Hugging Face ran ablations on RoPE vs RNoPE (removing PE every 4th layer) vs RNoPE + doc masking. All similar on short-context; they adopted RNoPE + document masking as the long-context foundation.

### Attention Patterns for Long Contexts

These modify *which tokens attend to which*, distinct from positional-encoding scaling:

- **Chunked Attention**: fixed-size chunks; tokens attend only within chunk. Llama 4 paired with RNoPE; long-context performance degraded.
- **Sliding Window Attention (SWA)**: every token sees `w` positions back. Gemma 3 alternated SWA and full attention.
- **Dual Chunk Attention (DCA)**: chunks with intra-chunk normal attention, plus a local cross-chunk window and capped inter-chunk attention. Qwen-2.5 used DCA to reach 1M tokens.
- **Interleaved local/global**: alternates local (efficient) and global (full reach) per layer. Adjusting the global ratio during instability can speed loss recovery.

### MoE

MoEs replace feed-forward with multiple MLP experts plus a learned router that picks top-k experts per token (e.g. 8 out of 384). Increasing total experts (sparsity) improves loss; recent models have >100 experts with ~10 active.

**Granularity** = (intermediate dim per expert) / (dense MLP intermediate dim), basically how many experts you'd need to match dense MLP width. Recent: 2 (gpt-oss-120b) to 8 (qwen3-next-80b-a3b). Ant Group: granularity doesn't significantly change loss but does drive *efficiency leverage* (FLOPs needed for MoE to match dense at same loss).

**Shared experts**: always-on, absorb basic recurring patterns; usually one is enough (DeepSeek-V2 uses two, with extra complexity).

**Load balancing** is non-negotiable. Routing uses top-k gating: router computes affinity scores (linear + softmax), selects top-k, routes tokens. To avoid collapse to a few experts:

- **Loss-based load balancer (LBL)**:  `L_balance = α * E * sum_e (f_e * p_e)` where `f_e` is fraction of tokens routed to expert `e`, `p_e` is average routing prob, `α` controls strength. Must use global stats (across batches), not local (narrow local batches mislead).
- **DeepSeek-V3 loss-free**: bias term added to affinity pre-softmax, updated based on load imbalance.
- **Sequence-wise auxiliary loss**: promotes balance within a sequence; encourages products `f_e * P_e` to be similar across experts.
- **Auxiliary-loss-free**: maintains a bias vector `b_e` updated decoupled from gradients. `b_e <- b_e + u * sign(c_e - c̄)` with bias update speed `u`, recentered.
- **SMEBU (Sequence-wise MoE Balancing with Uniformity)**: sequence-level violation `v_e = (load_e - mean_load)/std_load`, momentum buffer with factor `β`, soft-clamping `tanh` with tunable scale `α` for saturation control. `tanh` over `sign` preserves continuity; momentum dampens noise.

### Hybrid Models

Drop softmax from output and rewrite attention so it becomes a recurrent relation summarizing past KV pairs into a state `S_t`. Softmax stabilizes training, so the linear form needs normalization, plus a learned forget gate. Mamba-2 is popular (Nemotron-H, Falcon H1). Recent: Qwen3-Next (gated DeltaNet update), Kimi's next model (likely "kimi delta attention").

### Architecture Takeaways

- Use a proven dense baseline unless you have strong reasons and infra for MoE.
- GQA with small groups is a robust default; MQA is cheapest but underperforms.
- For long context, plan RNoPE/YaRN + document masking early.
- Hybrid architectures are promising but harder to operationalize.

## Stability

### z-loss

Regularization term added to standard cross-entropy that keeps logits from drifting large. Softmax denominator is `Z = sum_v exp(z_v)`; penalize `(log Z)^2`. On their 1B Hugging Face model, didn't affect training loss or downstream performance; they skipped it for overhead. **Logit softcapping is generally preferred** in modern recipes (per Gemma 2/3).

### Logit Softcapping

Maps logits into a bounded range via a smooth differentiable transform: `softcap(z) = cap * tanh(z / cap)`. Smooth (unlike hard clipping), gradient nonzero at boundaries. Gemma 2 applies softcapping to attention logits (pre-softmax) and the final LM head; `cap=50` for attention, `cap=30` for the final layer. Caveat: incompatible with Flash Attention / SDPA during training (fused kernels assume standard attention); must use `attn_implementation="eager"` for stable fine-tuning. Inference can still use SDPA with minimal quality difference.

### Weight Decay and Embeddings

Removing weight decay from embeddings can improve stability. Weight decay shrinks embedding norm, which causes larger gradients in earlier layers (LayerNorm Jacobian has a `1/||x||` term). Hugging Face found no significant difference between baselines, so they removed weight decay from embeddings.

### QK-norm

Apply LayerNorm to Q and K before attention to keep attention logits bounded. But the RNoPE paper found it hurts long-context tasks: normalization de-emphasizes relevant tokens by stripping query-key dot-product magnitude.

### RMSNorm

Comparable to LayerNorm, computationally simpler (no mean-centering). **Depth-scaled sandwich norm** applies normalization before and after attention/MLP blocks, with scale adjusted by layer depth. Arcee initializes `γ ~ N(1, 0.02^2)` and `γ_depth ~ 1/sqrt(2L)`. The sandwich pattern is stabilizing for deep networks. Arcee also applies RMSNorm before the LM head.

### Other Stability Notes

- **Init**: TruncDNormal with `σ=0.006` and clipping at ±3σ; or μP (maximal update parametrization). Heuristic `σ ~ 1/sqrt(d_model)`.
- **Embedding scaling**: scale embedding activations by `sqrt(d_model)` on forward pass (Grok-1, Grok-2, Trinity Large, Gemma 1/2).
- **Activation**: SwiGLU (most modern LLMs); GeGLU (Gemma 2); squared-ReLU (some NVIDIA models).
- **Width vs depth**: deeper outperforms wider at small scales; larger models trend wider for parallelism/inference.

### Stability Takeaways

- Stabilization is mostly sane defaults, not exotic tricks.
- Logit softcapping (Gemma-style) is preferred for attention/LM-head; z-loss and QK-norm are alternatives.
- QK-norm can hurt long-context; don't assume "always good."
- Init/normalization details matter more as depth grows.
- Track loss spikes early; many "mystery failures" are config or data issues.

## Tokenizer

- **Domains**: digits and special chars need care. Single-digit splitting helps arithmetic; Llama3 encodes 1–999 as unique tokens.
- **Languages**: English tokenizer is inefficient on Mandarin/Farsi.
- **Mixture**: when training from scratch, mirror the final training mix.

Larger vocab → better compression but bigger embedding matrix. English-only: ~50k often enough. Multilingual: >100k. Compression gains decrease exponentially with size. Large models benefit more from large vocabs (more savings in forward pass + smaller KV cache from fewer tokens).

**BPE** is still default. Metrics: *fertility* (avg tokens per word), *proportion of continued words* (% words split). Smaller = more efficient.

Often you can use existing tokenizers (GPT-4, Gemma 3). Train your own only for low-resource languages or unusual data mixtures.

## Optimizers and Training Hyperparameters

### AdamW

Still the default. Per-parameter adaptive learning rates from EMA of gradients (`m_t`) and squared gradients (`v_t`). Standard hyperparameters: weight decay `0.1` or `0.01`, `β1=0.9`, `β2=0.95–0.999`, `ε=1e-8`.

### Muon

Treats weight matrix as a singular object. Reduces axis-aligned bias and encourages exploration of suppressed directions. Update:

```
M_t = μ * M_{t-1} + G_t            # momentum
G_t' = NewtonSchulz5(M_t)          # approximates matrix sign
θ_{t+1} = θ_t - η * G_t'
```

Newton-Schulz approximates the matrix-sign function via repeated `f(X) = aX + b(XX^T)X + c(XX^T)^2 X`, normalizing singular values. More sample-efficient than AdamW, especially at large batch sizes.

Trinity Large hybrid: Muon on hidden layers, AdamW on embeddings and output. Embeddings/output benefit from per-param adaptive lr; hidden layers gain from matrix-level awareness.

Infra: Muon needs the full gradient tensor for Newton-Schulz. One approach: overlapping round-robin where each rank gathers gradient matrices for its index. Breaks at scale due to many overlapping collectives. Prime's alternative: all-to-all bulk permutation — each rank temporarily owns full gradients for its matrices, runs Muon, permutes back. Fewer collectives, may need padding due to packing.

### MuonClip (Kimi K2)

Prevents exploding attention logits during large-scale training. For each head h, define max logit `S_max^h = max over batch and positions of (Q_i K_j / sqrt(d))/τ`. Hyperparameter threshold `τ`. When `S_max = max_h S_max^h > τ`, rescale `Q_h` and `W_K_h` (or `W_Q_h` and `W_K_h`) multiplicatively by `(τ/S_max)^α` (commonly `α=0.5` for equal scaling). Per-head clipping based on `S_max^h` is straightforward for MHA, harder for MLA: clip latent-to-key projection and the latent variable; apply scaling separately to head-specific Q, K, and rotary components, and shared rotary.

Main Muon algorithm modified to match Adam RMS: scaling factor adapts update magnitude to matrix size; weight decay applied multiplicatively pre-update.

Visual: in a 9B-active 53B-total MoE, attention logits diverge without MuonClip. With MuonClip and `τ=100`, max logits decay to a stable range after ~30% of training steps.

### Learning Rates

- Warmup 1%–5% of steps (large labs fix warmup steps), then anneal.
- **Cosine annealing**: classic but inflexible (cosine period must match total duration).
- **Warmup-Stable-Decay (WSD)**: 10–20% decay matches cosine; lets you reuse a stable checkpoint for varied total-token ablations.
- **Multi-step**: discrete drops; 80/10/10 matches cosine, 70/15/15 or 60/20/20 can beat it.
- DeepSeek-V3: cosine between drops + constant phase before final sharp step.

Hugging Face's 1B settled on 2e-4; higher → instability. WSD underperforms before decay but catches up by the end. Kimi K2 used WSD: 10T tokens at 2e-4 after 500-step warmup, then 5.5T tokens cosine 2e-4 → 2e-5.

### Batch Size

Critical batch size: too small → underuse compute; too large → need more tokens. Larger gives more efficient gradient estimates.

Rule of thumb: scale batch by `k` → scale lr by `sqrt(k)`. Cov shrinks by `1/k`; SGD update variance proportional to `η^2/B`, so `η ∝ sqrt(B)`.

Critical batch grows during training: early model makes large updates (large `∇L`), small critical batch; later → larger batches work. Motivates batch-size warmup.

**Random Sequential Document Buffer (RSDB)** (Arcee): reduce intra-batch correlation. Load tokens into a buffer with read heads; sample random doc + read-head position, copy tokens to sequence buffer, repeat. Internal buffer 2x user-specified; refilled at user threshold. Significantly improved Trinity Large dataloader performance.

### Scaling Laws

`C ≈ 6 * N * D` where C is FLOPs, N is params, D is tokens.

Original GPT-3: 175B params, 300B tokens. Chinchilla re-derivation: compute-optimal would have been 3.7T tokens. Labs now "overtrain" (Qwen 3: 36T tokens; Kimi K2 1T model: 15.5T tokens; SmolLM3 3B: 11T tokens) — partly because compute-optimal scaling ignores inference cost.

Kimi K2: sparsity (total experts / active experts) yields substantial gains for fixed FLOPs. They use sparsity 48 (8 of 384 experts active, vs 256 in DeepSeek-V3), reduced attention heads 128 → 64 — sacrificed 0.5–1.2% validation loss for 45% inference FLOP reduction.

## Data Curation and Pre-training

For fixed compute, increasing one data domain's share decreases another. Existing pre-training corpora: FineWeb2, The Pile. Add specialized math/code datasets to fill gaps.

**Quality vs repetition**: only filtering for highest quality → too few tokens → repetition → harm. Ideal mix balances quality and diversity.

**Safety**: OpenAI's CBRN (chemical, biological, radiological, nuclear) pre-training filters applied to gpt-oss-120b, originally developed for GPT-4o.

### Multi-stage Training

Evolve mixture during training. Save best data for the end (final-stage data shapes final behavior). Performance-driven intervention: if a benchmark plateaus, inject high-quality data for that domain.

### Ablation

Architectural ablations on small models; data mixture ablations *at scale*. Annealing ablations on main-run checkpoints (e.g. 7T/11T) to choose what datasets to introduce.

Validation/holdout loss approaches for optimal proportions tend to converge to dataset-size distributions and underperform careful manual ablations.

### Token Utility

Token efficiency = performance per token. Improve via token utility (effective signal per token). Balance high-quality tokens (max leverage) vs overfitting risk.

Kimi K2 rephrases knowledge/math data: style/perspective-diverse prompting, chunk-wise autoregressive generation for long docs, fidelity verification for semantic alignment. Each corpus rephrased at most twice. For math: "learning-note style" + translation to other languages.

### SmolLM3 Pre-training Mix

Stage 1: 75/12/10/3 across English web / multilingual web / code / math.

- English web: FineWeb-Edu + DCLM, 60/40 or 50/50 best; add Pes2o, Wikipedia/Wikibooks, StackExchange later.
- Multilingual: 5 European languages from FineWeb2-HQ, smaller portions of Chinese/Arabic for downstream continual pretraining; 12% multilingual best.
- Code: Stack v2 + StarCoder2; 16 languages, PRs, notebooks, issues, SE threads. Recommended code mixture *degraded* English. Delayed Stack-Edu (educationally filtered) to end.
- Math: FineMath3+, InfiWebMath3+, MegaMath, OpenMathInstruct, OpenMathReasoning.

Three stages: 8T @4k (base), 2T @4k (high-quality injection), 1.1T @4k (reasoning/Q&A). New stages: 40/60 baseline:new.

### Hermes 4 Pre-training

Semantic dedup at 0.7 cosine similarity using embeddings; LLM-as-judge filters incomplete/malformed messages. **DataForge**: graph-based synthetic data generator. Nodes implement struct→struct mappings; edges require postconditions of A to satisfy preconditions of B. Random walks generate QA pairs with intermediate medium transformations (wiki article → rap song). LLM-as-judge grades. Recursive DFS taxonomy generation for data-scarce subdomains.

### Data Takeaways

- Data quality/mixture dominates architecture tweaks at fixed compute.
- Multi-stage: save best data for late training.
- Dedup and contamination checks non-optional for honest evals.
- Ablate mixtures at scale; small-model ablations mislead.

## Mid-Training

Intermediary between pre-training and post-training: train base further on a large amount of domain-specific tokens (coding, reasoning). Decision often made after initial SFT reveals gaps. For shallow capabilities (style, conversation), spend compute on post-training instead.

Qwen3: 30T @4k → 5T reasoning STEM/code → long-context @32k.

SmolLM3: scale sequentially 4k → 32k → 64k → 128k. Upsampling long docs didn't help (baseline already has long docs via RNoPE). RoPE ABF: 4k → 32k bumps base freq to 2M, then to 5M for 64k. 10M further improved RULER but hurt GSM8k. To 128k: YaRN from 64k checkpoint beat 4x from 32k. Train closer to inference length helps.

Kimi K2: lr decay 2e-5 → 7e-6, 400B tokens @4k, 60B @32k, YaRN to 128k.

**Distilled mid-training**: powerful approach. Phi-4-Mini-Reasoning distilled from DeepSeek-R1 → AIM24 3x, MATH-500 +11, GPQA-D +6. SmolLM3 also did distilled mid-training; considered DeepSeek-R1 reasoning (4M samples), QwQ-32B (1.2M samples) but delayed Mixture of Thoughts to final SFT mix. Insight: almost always worth mid-training if base hasn't seen lots of reasoning during pre-training; even `/no_think` mode benefits on reasoning benchmarks.

## Post-Training

### Evals

Four broad classes:
- **Knowledge**: GPQA Diamond (small-model signal beats MMLU); SimpleQA for factuality (hard for small models).
- **Math**: AIME leading; MATH-500 as sanity check.
- **Code**: LiveCodeBench (competitive); SWE-bench Verified (sophisticated, hard for small).
- **Multilinguality**: Global MMLU.

Additional axes:
- **Long context**: RULER, HELMET, MRCR, GraphWalks.
- **Instruction following**: IFEval, IFBench (more constraints), Multi-IF / MultiChallenge for multi-turn.
- **Alignment**: LMArena (human + leaderboards); LLM-as-judge alternatives AlpacaEval, MixEval.
- **Tool calling**: TAU-Bench (retail, airline customer service).

Prevent overfitting: GSMPlus (perturbs GSM8k), vibe evals, interval evals. Other tips: use small correlated subsets, fix LLM-as-judge model, treat anything used in ablations as validation, use avg@k, don't benchmax.

### Intellect-3 Post-training Data

106B-param MoE (12B active), post-trained on GLM-4.5-Air base. Stack: prime-rl, verifiers library, sandbox code execution, compute orchestration, Environments Hub.

- Math: 21.2K problems from Skywork-OR1, Acereason-Math, DAPO, ORZ-Hard (derived from AIME, NuminaMath, Tulu3 math). LLM judge: CompassVerifier-7B.
- Science: 29.3K filtered from MegaScience + LLM-judge + standard math verifiers.
- Logic (Sudoku, Minesweeper, etc.): 11.6K problems from SynLogic.
- Code: Synthetic-2 + Prime Sandboxes. Two SWE envs: R2E-Gym, SWE-smith, Multi-SWE-bench; agents fix GitHub issues with bash + edit tooling. Max 200 turns.
- Deep research: web search env with search tools, DeepDive dataset (1K SFT, 2.2K RL). Reward 0/1. Qwen3-4B-Instruct: 26 SFT steps (bs 34) + 120 RL steps (group 16, bs 512) → mean reward 0.7.

### Hermes 4 Post-training Data

300k prompts mostly STEM/code from WebInstruct-Verified, rSTAR-Coder, DeepMath-103k. Dedup + filter prompts >2k chars. Rejection sampling against ~1k task-specific verifiers via Atropos. Environments:
- Answer Format Training: rewards `\boxed{}` etc., 150+ output formats, enforces `<think>` delimiters.
- Instruction Following: RLVR-IFEval verifiable constraints ("every Nth word in French").
- Schema Adherence: JSON generation + repair.
- Tool Use: agentic via `<tool_call>` token.

### Kimi K2 Post-training Data

Tool use focus. Three-step synthetic pipeline at scale:
1. Tool spec generation from real + LLM-synthetic tools.
2. Agent + task generation per toolset.
3. Trajectory generation per agent/task.

3k+ real MCP tools from GitHub, 20k synthetic in domains like financial trading, software apps, robot control. Diversification via system-prompt × tool-set combinations. Rubric + LLM judge.

RL: math/STEM/logic similar to other models. Coding from competitions + GitHub PRs/issues. Instruction following: deterministic via code interpreters + LLM-as-judge; prompts/rubrics from expert + AutoIF-style augmentation + failure-mode-targeted model.

### Chat Template Considerations

System role customizability, tool calling, reasoning, inference-engine compatibility (vLLM, SGLang). Qwen3 and GPT-OSS satisfy all; Qwen3 supports hybrid reasoning.

SmolLM3: discards reasoning content for all but final turn at inference (avoid context blow-up), but retains during training to condition properly. Initial bug: custom instructions not passed into the template, patched fast.

Intellect-3: always reasons (not hybrid), trained dominantly on reasoning-only SFT traces; uses qwen3_coder tool parser, deepseek_r1 reasoning parser.

gpt-oss-120b: harmony chat template with "channels" — `final` for user-visible answers, `commentary` for tool calls, `analysis` for CoT. Allows interleaving tool calls with CoT.

Hermes 4: changes Llama 3's assistant token to a first-person identifier → markedly different first-person behaviors. DeepSeek-R1-Zero: similar to others but uses `\boxed{}` for final answers.

### SFT

Cheap, stable, gives a strong base. Often distillation from stronger models. Models without stronger teachers skip SFT (DeepSeek-R1-Zero).

Dataset curation matters — datasets can overindex domains (e.g., science). Hugging Face SmolLM3 SFT: ~100k examples, 76.1M tokens, mostly instruction following + reasoning + steerability for think + no-think. **Pair data across modes** so model learns when to be concise vs reason extensively.

Training details: full FT vs LoRA/QLoRA; FlashAttention or SonicMoE; mask loss to assistant tokens only; parallelism; lr tuning; sequence length matched to data.

**Cute Cross-Entropy (CCE) kernel**: memory-efficient CUDA. Computes only correct-token logit, evaluates log-sum-exp over vocab on-the-fly using faster memory tiers. Skips gradient for negligible elements. Valuable for large-vocab models.

Intellect-3 SFT: two stages. (1) General reasoning: Nemotron post-training + AM-DeepSeek-R1-0528-Distilled, 9.9B tokens. (2) Agentic: SWE-Swiss + Environments Hub synthetic from DeepSeek-R1. Side effect: pushed context 65K → 98K via context parallelism.

Hermes 4 SFT: two stages, both reasoning. Sequences capped at 16k but reasoning often exceeds 41k. Stage 2 teaches `</think>` insertion at 30k tokens (budget). Fixed-token-count insertion teaches counting behavior ("when you reach N tokens, stop") while preserving the model's own distribution and avoiding collapse from recursive self-generated training.

### Capabilities

Hugging Face IFThink dataset: Qwen3-32B augments single-turn to multi-turn with verifiable instructions + reasoning traces. Dramatically improved multi-turn reasoning by helping with `/think` vs `/no_think` disambiguation across turns.

**Masking user turns**: small but real improvement (a few points). Otherwise loss on user queries trades off assistant quality.

### Sequence Packing

Pack sequences in batches to avoid padding waste, with min cross-batch truncation. **Best-fit decreasing** (TRL): place sequence in batch that minimizes remaining space. **First-fit decreasing** (Hermes-4): place in first batch with enough space; ~93% batch efficiency.

Up to 33x tokens/batch/step, but at fixed token budget, more data → fewer gradient updates. Hurts small datasets. Effective bs 128 hurt IFEval ~10%; bs > 32 averaged drops in SmolLM3. Large datasets: almost always beneficial.

### Learning Rate and Epochs

SFT lr ~10x smaller than pre-training (model already has rich representations; aggressive updates → catastrophic forgetting). SmolLM3: 3e-6 or 1e-5 best. Lower further with packing.

More than one epoch helps a few % once mix is right; on LiveCodeBench v4, performance nearly doubled epoch 2 → 3.

Pre/post optimizer choice: AdamW default for both; even with Muon, same optimizer for both stages was best.

### Preference Optimization (PO)

Beyond SFT's imitation ceiling. Less data needed (strong starting point).

Generating preference data:
- **Strong vs weak**: fixed prompt, strong-model output preferred over weak-model output. Easy, reliable.
- **On-policy with grading**: same model generates candidates, external grader (LLM-judge/rubric/verifier) provides labels. Allows ongoing bootstrapping.

Hyperparameters:
- **lr**: 10x–20x smaller than SFT (SmolLM3: 1e-6).
- **β**: 0 → preference-strong, 1 → reference-strong. Too large erases SFT capabilities. ~0.1 typical.
- **Dataset size**: stable from 2k to 340k pairs; SmolLM3 saw drops in extended thinking past 100k pairs. Cheap inference → create your own.

### PO Algorithms

- **DPO**: vanilla.
- **KTO** (Kahneman-Tversky Optimization): per-sample desirable/undesirable labels (no pairs) + reference point + log-ratio reward.
- **ORPO** (odds ratio): combines PO + SFT via odds-ratio term added to cross-entropy. No separate reference model needed → cheaper.
- **APO** (anchored preference optimization): APO-zero pushes `y_w` up and `y_l` down; APO-down pushes both down (useful when `y_w` quality is below current model).

Hugging Face found APO-zero had best overall OOD performance.

### RL

Verifiers + reward signals through environment interaction. RLHF: human comparisons → reward model → policy fine-tuning. RLVR: verifiers (math correctness, unit tests) directly. Particularly valuable when reward drift is a concern, when KL control is needed to prevent policy collapse, when addressing stale-policy artifacts in multi-step reasoning.

Most algorithms are on-policy in theory but slightly off-policy in practice (e.g., GRPO with sequential rollout batches + optimizer updates makes only the first batch on-policy). These are **in-flight updates**. Tradeoff: throughput vs policy consistency; mitigated via importance-sampling clipping (IcePop).

Intellect-3: CPU orchestrator between training and inference clusters. Orchestrator polls trainer; on new policy, inference temporarily halts, updates weights, continues rollouts. Long rollouts can span multiple policies, bounded by `max_off_policy_steps`. IcePop:

```
A_ic = mask(s_ic) * A_i
s_ic = π_train(o_ic | q_i, o_i,<c) / π_inf(o_ic | q_i, o_i,<c)
mask = 1 if l ≤ s_ic ≤ u, else 0
```

Asymmetric `l`, `u` is fine. Tight `u` clips high-entropy tokens when `π_inf` is small. Prime defaults `l=0.5, u=1.2`.

Kimi K2 policy optimization (from K1.5):

```
L = -E[(r_i - r_mean) * log π(o_i | q)] + τ * KL(π || π_ref)
```

**PTX loss**: pre-training cross-entropy added during joint RL to prevent catastrophic forgetting on high-quality data + mitigate RL overfitting.

**Temperature decay**: high temp early for creative writing / complex reasoning exploration; lower temp later for reliability.

DeepSeek-R1-Zero: GRPO cold-start RL without SFT. 10.4k steps, bs 512, reference policy replacement every 400 steps, lr 3e-6, KL coefficient 0.001. Two rewards: accuracy + format (thinking tags). Reflection/exploration emerged; "wait"/"mistake" usage 5-7x baseline; sudden onset between steps 4k–7k, regular by 8k.

DeepSeek-R1: long-CoT SFT first → RL → fixes (language consistency reward for mixed languages, readable pattern with summary) → second RL stage for helpfulness/harmlessness. Helpfulness preferences via DeepSeek-V3 4-query averaging + large-margin filtering, pairwise loss. Harmlessness: safe/unsafe labels + pointwise reward.

Distilling DeepSeek-R1 outputs into smaller models (Qwen-32B) beats large-scale RL on those smaller models for reasoning while requiring much less compute.

DeepSeek extras:
- **GRPO over PPO**: per-token KL in PPO penalizes long reasoning; GRPO doesn't. GRPO beat PPO on MATH (β=0.04 > β=0.0).
- **First-person voice**: post-FT on small long-CoT SFT data → "I" usage over "we" in R1 vs R1-Zero. Annotator-rewrite for human-interpretable CoT.
- **Temperature**: greedy decoding for long-output reasoners gives higher repetition / more variability. Risk aversion + inductive bias for temporally correlated errors → loops.

### RLVR and Rubrics

RLVR on hybrid reasoners: avoid extending tokens too much. Naive GRPO on `/no_think` rewards drift to longer CoT (reward hacking). SmolLM3 saw "/no_think" traces sprouting "Wait..." patterns.

**Overlong completion penalty**: function of soft + hard punishment thresholds. SmolLM3 `/no_think`: 2.5k–3k. Joint hybrid RL hard because separate length penalties interact unstably — why labs ship instruct + reasoning variants separately.

**Kimi K2 self-critique rubric reward**: actor generates N rollouts; critic ranks via pairwise rubric evaluation. Rubrics: core (fundamental values), prescriptive (anti-reward-hacking), human-annotated (specific instructional contexts). Critic itself refined using verifiable signals — transfer learning grounds subjective judgments in verifiable data.

### Online Data Filtering

Curriculum learning: sort problems by difficulty (observed solve rate). Intellect-3 math/code: Qwen3-4B-Thinking-2507 × 8 generations; science/logic: × 16. Per stage: balanced curriculum, avoid trivially easy/hard (preserves GRPO gradients). Kimi K2: SFT model pass@k.

### Alternatives to RL

**Online DPO** (see strong/weak grading above).

**On-policy distillation**: student samples per step, KL between student/teacher logits is signal. Cheap (one sample per prompt, single forward-backward) vs GRPO (multiple rollouts). Qwen3 tech report: bigger across-the-board boost. Limitation: tokenizers must match. **Hugging Face GOLD** (General On-Policy Logit Distillation): any teacher → any student.

Thinking Machines: on-policy distillation mitigates catastrophic forgetting. Mid-training 70% + on-policy distillation ≈ best of model + mid-trained version, restoring behavior cheaply.

| Algorithm | When | Tradeoffs | Best size |
|---|---|---|---|
| Online DPO | Preference labels cheap | Iterative, stable, label-dependent | Any |
| On-policy distillation | Stronger teacher available | Simple, cheap, inherits teacher bias | <30B |
| RL | Verifiable rewards or multi-step reasoning | Flexible, expensive, reward-hacking risk | 20B+ |

Semi-online DPO (sync every 100 steps) was generally best vs sync-10, online DPO, GRPO.

### Limitations

DeepSeek failed experiments:
- **MCTS**: token search space too large; max extension limit → local optima. Fine-grained value model hard.
- **Process Reward Models**: defining a fine-grained step is hard; LLM-as-judge inconsistent; reward hacking on "appearance of good reasoning."

### Post-training Takeaways

- SFT is stable baseline; PO/RL needs verifiable rewards or clear gains.
- Hybrid reasoners need careful length control.
- Tool-use / agentic datasets are first-class post-training targets.
- Many "fancy" methods fail in practice; track what doesn't work.

## Behaviors and Safety

### Safety Testing and Mitigation

OpenAI's gpt-oss-120b post-training: RL stage rewards policy compliance on unsafe prompts.

Adversarial fine-tuning evaluation: collegiate CTF, emulated network ops, in-domain biorisk fine-tuning, SWE-bench Verified / OpenAI PRs / PaperBench. Safety advisory: even with robust FT, gpt-oss-120b didn't reach high-capability thresholds in biological/chemical, cyber, or AI self-improvement domains.

Frontier impact assessment: other open-weight models at/near gpt-oss-120b capability → low frontier impact from release.

Other indicators:
- **Disallowed content**: ProductionBenchmarks (PII, sexual, harassment, hate, self-harm); LLM-judge `not_unsafe`. On par with o4-mini.
- **Jailbreaks**: StrongReject. On par with o4-mini.
- **Instruction Hierarchy**: system > developer > user > assistant > tool. Post-trained with conflicts; chooses higher-tier. PII protection on par with o4-mini; message conflict ~15% behind o4-mini.
- **Hallucinations and CoT**: reasoning CoT is useful for misbehavior detection, but pressure against "bad thoughts" can teach hiding. Hallucination rate ≠ 1 − accuracy (can answer "I don't know"). Slightly worse than o4-mini (size expected).
- **Bias**: BBQ across 9 social dimensions. On par with o4-mini.

### Behaviors and Latent Capabilities

Hermes 4's assistant-token change → first-person persona, fewer meta-disclaimers, higher behavioral plasticity. Greater contextual fidelity over policy rigidity: interprets fictional prompts as role-play, lower refusal rate (RefusalBench: Hermes 4 reasoning lowest; gpt-oss-120b/20b highest). Political analysis shows balanced framing over policy-driven hedging.

Anti-sycophancy in Hermes 4 → deeper CoT shift (steering user away from inference), sometimes introducing embodied/emphatic language.

## The Training Marathon

Before main run: Slurm reservations, GPU stress-test (GPU Fryer, DCGM), avoid storage bloat (upload to third-party, delete local), checkpoint + auto-resume.

Evals are time-consuming (Allen Institute ~20% of compute on evals); automate logging — scores, throughput, loss, grad norm, node health.

### Vanishing Throughput

Hugging Face: ~40% throughput drop (14k → 8k tokens/sec/GPU) after hours. Cause: network-attached storage with "keep-hot" caching, 24TB training data, cold shards evicted to S3, fetching back stalled training. Fix: spare node preloaded with dataset, `fpsync` copy (2x faster than `s5cmd`). Repurposed spare for evals/dev jobs.

Then smaller residual drops. Smaller step counts → smaller drops → nanotron dataloader was growing lookup table per step instead of keeping it bounded. Global-memory growth → allocation failures, page faults, worse cache locality. Switched to Tokenizedbytes dataloader.

### Noisy Loss

SmolLM3 loss curve noisy. Cause: dataloader reads sequences sequentially per document; without shuffling, batches non-representative, grad variance spikes. Long files (code) supply consecutive sequences → loss spikes. Fix: offline re-shuffle of tokenized sequences (alternative random-access dataloader has higher memory and slower runtime).

### Tensor Parallelism

After 1T tokens, evals showed SmolLM3 underperforming SmolLM2 at same stage despite similar recipe. SmolLM2 weights fit on one GPU; SmolLM3 needed TP across 2 GPUs. Two TP ranks initialized with same random seed → similar activations/gradients → feature-diversity loss → slower convergence. Fix: distinct seeds per TP rank.

### Multi-client Orchestrator

Prime: vLLM standard multi-node DP didn't scale linearly with nodes. They abstracted a multi-client orchestrator — each inference node runs its own vLLM engine, scheduler, KV cache, batches; orchestrator has one client per node (no single-shared queue bottleneck), rollout requests distributed round-robin.

### The Usual Suspects

Common training instability causes: high lr, bad data, data-parameter state interactions, poor init (OLMo2: `σ=0.02` more stable than scaled init), precision (avoid fp16).

Mitigations: logit softcapping, z-loss, QK-norm, data filtering (OLMo2 removed docs with 32+ repetitions of 1–13 token spans). If spikes persist: retrain skipping problematic batches, tighten gradient clipping.

### Training Ops Takeaways

- Throughput failures are usually data pipeline / storage, not model code.
- Dataloader behavior (shuffling, packing, access patterns) silently changes training dynamics.
- Seed handling in parallelism is high-leverage; verify early.
- Evals and logging are first-class citizens — how you notice regressions.
