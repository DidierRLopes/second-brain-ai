# Frontier Training Playbook

How modern labs actually train multi-billion-parameter open-weight models, distilled from seven 2025–2026 reports (SmolLM3, Intellect-3, Hermes 4, gpt-oss-120b, Kimi K2, DeepSeek-R1, Arcee's Trinity). The central claim from Alex Wa's synthesis: **frontier training is a systems problem**, where data mixture, architecture, and stability choices dominate most algorithmic tweaks. Most decisions reduce to picking sane defaults and then ablating the few that actually matter for your use case.

## The 12-step minimal playbook

1. **Lock evals early** across knowledge, math, code, long-context, and instruction following. Eval implementations must be finished long before the base model is.
2. **Pick a baseline architecture with known failure modes.** Default to dense + GQA + RoPE/RNoPE unless MoE is essential.
3. **Choose a tokenizer matched to your target languages and domains.** Freeze vocab and special tokens early.
4. **Build the data pipeline with dedup, filtering, and contamination checks.** Measure data quality explicitly.
5. **Run small ablations** for attention, positional encoding, optimizer, and learning rate schedule. Change one variable at a time.
6. **Plan a multi-stage data mixture.** Delay the best data and reasoning-heavy data toward the end (the final stage shapes the final behavior most).
7. **Add stability guardrails:** logit softcapping (preferred), z-loss/QK-norm, gradient clipping, precision policy (avoid fp16), loss spike alerts.
8. **Validate throughput on long runs** and confirm dataloader behavior (packing, shuffling, random access).
9. **Run the main training with interval evals and consistent seeds**, especially across tensor parallelism ranks.
10. **Mid-train for domain gaps** if SFT reveals them; extend context length gradually (4k → 32k → 64k → 128k).
11. **Post-train with SFT**, then choose preference/RL/distillation based on verifiable rewards and tool-use goals.
12. **Re-evaluate, run safety checks, lock a release checkpoint** with full logs and configs.

## The principle: derisking

> "Never change anything unless you've tested that it helps."

Ablations need to be (a) **fast** so more hypotheses can be tested, and (b) **reliable** with strong discriminative power so what you measure isn't noise. The real value of a solid ablation setup goes beyond the model itself — when something inevitably goes wrong in the main run, you want confidence in every prior decision so you can quickly identify which components weren't properly tested.

In evals, look for:
- **Monotonicity** (scores actually improve over training),
- **Low noise** (resistance to random seeds),
- **Above-random performance** (random-level for extended periods isn't useful), and
- **Ranking consistency** (the ranking of approaches should be stable throughout training).

Balance exploration and execution: choose flexibility and stability over peak performance for methods, and set a deadline for exploration.

## What dominates, what doesn't

Across all seven reports, the same things drive results:
- **Data quality and mixture** dominate architecture tweaks at fixed compute. See [[data-curation-mixtures]].
- **Multi-stage schedules** matter: save the best data for late training to shape final behavior.
- **Mid-training and post-training** (SFT + preference/RL/distillation) determine reasoning and tool-use behavior more than the base model architecture does.
- **Training ops** (dataloader design, throughput, seeds in TP, checkpointing) are the most frequent failure points. See [[training-ops]].
- **Most training failures stem from common causes**: high learning rates, problematic data batches, load imbalance in MoE models, or storage/infra issues.
- **Rejection sampling is a near-universal post-training staple** across Qwen 2, AFM, and Llama 3.1 alike, even where the labs otherwise disagree on DPO vs. RLHF-with-PPO-style methods — see the Qwen 2/AFM/Llama 3.1 sections above (Raschka, `raw/raschka-llm-pretraining-posttraining-paradigms.md`).
- **Model/weight averaging recurs as a post-training stabilization technique** distinct from RL-time KL regularization: Gemma 2's WARP/WARM and Llama 3.1's SFT/DPO/RM checkpoint averaging arrive at the same idea independently.
- **Knowledge distillation at pretraining time** is now a standard alternative to scaling raw token count for smaller models in a family — AFM's 3B (distilled+pruned from 6.4B) and Gemma 2's 9B/2B (distilled from 27B/7B teachers) both choose this over training from scratch.

What rarely matters as much as people think: exotic attention variants, novel optimizers without strong infra support, "fancy" post-training methods.

For the small-model version of the same decisions, see [[small-efficient-models]]: 1-3B models often spend extra tokens, cleaner data, and WSD-style decay phases to buy cheaper inference and on-device deployment, which is a different objective than frontier-scale capability maximization.

## Reference table: covered models

| | Kimi-K2 | Trinity Large | gpt-oss-120b | OLMo 3 | SmolLM3 | Llama 3.1 405B | DeepSeek-V2 | DeepSeek-V3 | Qwen3-235B | Gemma 2 27B |
|---|---|---|---|---|---|---|---|---|---|---|
| Parameter count | 1.06T | 400B | 116.83B | 32B | 3B | 405B | 236B (21B active) | 671B (37B active) | 235B (22B active) | 27B |
| Attention | MLA | GQA (8) | GQA (8) | GQA | GQA (4) | GQA (8 KV heads) | MLA | MLA | GQA (8 KV) | GQA (2 groups) |
| Pos. embedding | RoPE + YaRN | RoPE + YaRN | RoPE + YaRN | RoPE + YaRN | RNoPE + YaRN | RoPE (θ=500k) | RoPE (decoupled) | RoPE | RoPE (θ=1M) | RoPE |
| Architecture | MoE | MoE | MoE | Dense | Dense | Dense | MoE (DeepSeekMoE) | MoE (DeepSeekMoE) | MoE | Dense + SWA |
| Training tokens | — | — | — | 6.6T | — | 15.6T | 8.1T | 14.8T | 36T | 13T |

The dense-vs-MoE decision tree (per Hugging Face): pick dense if memory-constrained (MoEs must keep all experts loaded), new to LLM training, or working under a tight timeline.

## GPT-4: predictable scaling as a first-class goal

[GPT-4 Technical Report (2303.08774)](../../papers/01-models/gpt-deepseek-v2-v3/GPT-4 Technical Report - 2303.08774.pdf) is deliberately sparse on architecture and training details ("contains no further details about the architecture, hardware, training compute, dataset construction, training method, or similar") due to competitive sensitivity. Its primary contribution to the training playbook is the **predictable scaling methodology**: a deep learning infrastructure and optimization stack designed to behave predictably across multiple compute scales. This allowed OpenAI to predict GPT-4's final loss from models trained with 1,000–10,000x less compute before the main run started. The fitted scaling law `L(C) = aC^b + c` predicted GPT-4's final loss on an internal codebase benchmark with high accuracy.

A second contribution is the capability-prediction methodology. Rather than predicting loss alone, they predicted pass rates on HumanEval subsets by fitting an approximate power law: `-E_P[log(pass_rate(C))] = α·C^-k`. This was registered before training completed, using only information available prior to training. The practical lesson: **locking capability predictions before the main run is a tractable goal**, not just theoretical, and doing so forces early investment in discriminative eval infrastructure.

GPT-4 benchmark highlights (post-training, RLHF model): Uniform Bar Exam ~90th percentile, MMLU 86.4%, HellaSwag 95.3%, HumanEval 67.0%, GSM-8K 92.0% (chain-of-thought). Importantly, exam capabilities "appear to stem primarily from the pre-training process and are not significantly affected by RLHF" — base and RLHF models perform equally on multiple-choice evals. Post-training matters most for open-ended instruction following, where GPT-4 responses were preferred over GPT-3.5 on 70.2% of prompts.

## DeepSeek-V2: MLA origin and economical MoE

[DeepSeek-V2 (2405.04434)](../../papers/01-models/gpt-deepseek-v2-v3/DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model - 2405.04434.pdf) introduced two architectural innovations that became the backbone of the DeepSeek series: **Multi-head Latent Attention (MLA)** and **DeepSeekMoE**. For attention mechanics see [[attention-variants]]; the training-level lessons are here.

MLA compresses KV pairs into a low-rank latent vector `c_t^KV = W^DKV · h_t` of dimension `d_c << d_h·n_h`. Keys and values are reconstructed at runtime via up-projections. For inference, `W^UV` can be absorbed into `W^O`, meaning DeepSeek-V2 never materializes the full KV cache — it caches the latent instead. With `d_c = 4d_h` and `d_h^R = d_h/2` for a decoupled RoPE key, the total KV cache equals GQA with only 2.25 groups, while empirically outperforming MHA. The practical result: **93.3% KV cache reduction vs DeepSeek-67B**, 5.76x max generation throughput, 42.5% training cost reduction.

The RoPE-MLA incompatibility is worth noting for the playbook: standard RoPE cannot be applied to the compressed latent because `W^UK` can no longer be absorbed into `W^O` once a position-sensitive matrix sits between them. DeepSeek-V2 resolves this with decoupled RoPE — separate multi-head queries `q^R` and a shared key `k^R` that carry positional information, leaving the latent KV free of positional encoding. This requires caching both the KV latent and the decoupled RoPE key, for a total of `(d_c + d_h^R)·l` elements.

DeepSeekMoE extends the standard MoE architecture with two ideas: **fine-grained expert segmentation** (more, smaller experts per layer) and **shared experts** that are always activated to absorb general knowledge, freeing the routed experts to specialize. 236B total parameters, 21B activated per token, 8.1T training tokens. Load balance is maintained through three auxiliary losses: expert-level (`L_ExpBal`), device-level (`L_DevBal`), and communication balance (`L_CommBal`). Device-limited routing caps each token to at most M=3 devices, bounding MoE communication cost regardless of expert count.

## DeepSeek-V3: auxiliary-loss-free MoE and multi-token prediction

DeepSeek-V3 (671B total / 37B active per token, 14.8T training tokens) extended the DeepSeekMoE architecture from V2 with two key training-level innovations.

**Auxiliary-loss-free load balancing.** Rather than adding explicit balance losses that distort the primary training signal, DeepSeek-V3 uses a **bias term per expert** added to the routing scores: `s'_i = s_i + b_i`, where the bias is updated at each step by `b_i += γ` if expert i is overloaded and `b_i -= γ` if underloaded. This keeps routing logits on the main training graph clean of auxiliary objectives. The result is better model quality at the same load balance target compared to V2's auxiliary loss approach — the tradeoff between capacity utilization and training fidelity is eliminated.

**Multi-token prediction (MTP).** At each position, DeepSeek-V3 predicts the next D tokens (not just the next one) using D sequential MTP modules, each with its own embedding layer, transformer block, and output head. The additional next-token prediction heads share the main model's embedding and are trained jointly, contributing additional gradients to the main model's representation. At inference, the MTP heads can be repurposed for **speculative decoding** — they generate candidate continuations that the main model verifies, boosting throughput without changing the output distribution.

Training cost for DeepSeek-V3: approximately 2.788 million H800 GPU hours for 14.8T tokens — substantially lower than comparable models due to the MLA inference efficiency reducing activation memory and the FP8 mixed-precision training framework. See [[mixture-of-experts]] for additional DeepSeek routing details.

## DeepSeek-R1: brief mention

DeepSeek-R1 is an RL-trained reasoning model built on top of DeepSeek-V3. The key pipeline: (1) cold-start SFT on a small set of long-CoT examples, (2) GRPO-based RL with verifiable rewards (math, code), (3) rejection sampling + additional SFT, (4) final RL pass. The notable finding is that the base model develops chain-of-thought reasoning behaviors (self-verification, backtracking) emergently during RL rather than by supervised imitation. Detailed treatment belongs in [[reasoning-models]].

## Llama 3: dense simplicity at scale

[The Llama 3 Herd of Models (2501.12948)](../../papers/01-models/llama-qwen-gemma/The Llama 3 Herd of Models - 2501.12948.pdf) is the clearest public articulation of the "managing complexity" philosophy: **deliberately choose boring architecture and simple post-training to maximize ability to scale and debug**. The architectural choices reflect this directly: standard dense Transformer, GQA with 8 KV heads, SwiGLU activations, RoPE θ=500,000, 128K vocabulary (100K tiktoken + 28K for non-English). No MoE, no novel attention variants.

The data pipeline is unusually thorough for a published report. Deduplication at three levels: URL-level (keep most recent), document-level (global MinhHash), line-level (remove lines appearing >6 times per 30M-document bucket). Quality filtering: heuristics (n-gram coverage ratio, dirty-word count, KL divergence on token distribution), then model-based classifiers — fasttext for Wikipedia-referenceability, DistilRoberta for quality, domain-specific DistilRoberta for code and math. The final Llama 3.1 pre-training data mix is approximately **50% general knowledge, 25% math/reasoning, 17% code, 8% multilingual**, on 15.6T tokens.

Scaling law methodology: IsoFLOPs curves fit at compute budgets from `6×10^18` to `10^22` FLOPs, using model sizes 40M–16B. Parabolas identify the compute-optimal point at each budget. Power-law extrapolation to `3.8×10^25` FLOPs suggests ~402B parameters on 16.55T tokens; they trained 405B for practical reasons since IsoFLOPs curves flatten near the minimum. Then a two-stage downstream prediction: (1) correlate NLL on downstream tasks with FLOPs, (2) correlate NLL with task accuracy using a sigmoidal fit. This accurately predicted ARC Challenge accuracy, extrapolating four orders of magnitude.

Infrastructure for the 405B run: up to 16K H100 GPUs (80GB HBM3, 700W TDP), 4D parallelism (TP×CP×PP×DP), BF16 MFU of 38–43%. A RoCE-based 24K GPU cluster with three-layer Clos topology; E-ECMP for load balancing (16 network flows per pair instead of 1). Annealing on high-quality code and math data boosted a pre-trained 8B model's GSM8K by 24.0% and MATH by 6.4%; the 405B saw negligible improvement, suggesting the flagship already generalizes well without in-domain annealing.

Post-training used rounds of SFT, rejection sampling (RS), and DPO — deliberately avoiding complex RL methods. The rationale: simpler post-training is more stable and scales better across the model family. This is the "managing complexity" principle applied to alignment. Llama 3.1 still trains a reward model (seeded from a pretraining checkpoint plus human-annotated preference data) purely to drive rejection sampling, not for PPO-style RL. It also applies **model/weight averaging** at every post-training stage — not just to the reward model, but across recent checkpoints of the SFT and DPO models too — the same averaging-as-stabilization pattern Gemma 2 implements via WARP/WARM, arrived at independently.

## Qwen3: 36T tokens and unified thinking/non-thinking

[Qwen3 Technical Report (2505.09388)](../../papers/01-models/llama-qwen-gemma/Qwen3 Technical Report - 2505.09388.pdf) extends the Qwen series with both dense (0.6B–32B) and MoE (30B-A3B, 235B-A22B) models, all trained on **36 trillion tokens** across 119 languages. Pre-training follows a three-stage strategy: (S1) ~30T tokens at sequence length 4096 for general foundation, (S2) ~5T tokens upsampling STEM/coding/reasoning, (S3) hundreds of billions of tokens at 32,768 sequence length for long-context extension using YaRN and DCA.

Architecture is similar to Qwen2.5 with two notable changes: **QK-Norm** (RMSNorm on query and key projections, following Dehghani et al. 2023) for training stability, and removal of QKV bias. All dense models use GQA with 8 KV heads; MoE models use 4 KV heads. MoE architecture: 128 total experts, 8 activated per token, excluding shared experts (diverging from Qwen2.5-MoE which had shared experts), and a global-batch load balancing loss.

The key Qwen3 post-training innovation is the **unified thinking/non-thinking mode**. Flagship models go through four stages: long-CoT cold start → reasoning RL (GRPO, 3,995 query-verifier pairs) → thinking mode fusion → general RL. Lightweight models (up to 14B) instead use **strong-to-weak distillation** from the flagship — distilling output logits from teacher to student, which achieves better Pass@1 and Pass@64 than running the four-stage pipeline independently on every small model, at 1/10 the GPU hours. The thinking budget mechanism lets users allocate inference compute dynamically by capping the number of reasoning tokens.

Scaling results: Qwen3-32B-Base outperforms Qwen2.5-72B-Base on 10/15 benchmarks despite having less than half the parameters, particularly strong on STEM and coding. Qwen3-235B-A22B-Base outperforms DeepSeek-V3-Base on 14/15 benchmarks with 1/3 the total parameters and 2/3 the activated parameters.

**Qwen 2 predecessor pipeline** (Raschka, see `raw/raschka-llm-pretraining-posttraining-paradigms.md`): Qwen 2 (0.5B/1.5B/7B/72B dense + a 57B-total/14B-active MoE) used previous-generation Qwen checkpoints to synthesize additional pretraining data, then a two-stage post-training DPO recipe — **offline DPO** on an existing static preference dataset, followed by **online DPO/rejection sampling**, where a reward model picks the preferred response among multiple candidates generated live during training. The 0.5B model was trained on 12T tokens while the larger models stopped at 7T because more tokens showed no improvement at that scale — an early, explicit per-size token-budget ablation.

## Apple Intelligence Foundation Models (AFM): distillation-first pretraining and iTeC

Apple's AFM report (covered in Raschka's pre/post-training survey, `raw/raschka-llm-pretraining-posttraining-paradigms.md`) is notable less for scale than for committing to distillation and committee-based rejection sampling at every stage where most labs would just scale tokens.

**Three-stage pretraining**, all server + 3B on-device models: (1) **core pretraining** — the server model on 6.3T tokens at 4,096 context; the on-device 3B model is **distilled and pruned from a 6.4B model**, with a distillation loss that replaces hard target labels with a convex combination of true labels and the teacher's top-1 prediction (weight 0.9 on the teacher) — heavier reliance on the teacher signal than typical logit-distillation setups; (2) **continued pretraining** — 1T tokens, context to 8,192, math/code up-weighted and web-crawl down-weighted; distillation loss was tested here too but found *not* beneficial at this stage, an explicit negative result worth flagging; (3) **context lengthening** — 100B tokens, context to 32,768, synthetic long-context Q&A data.

**Post-training** combines SFT with multiple RLHF rounds using two algorithms not seen in the other three reports: **iTeC** (Rejection Sampling Fine-tuning with Teacher Committee) trains independent SFT/DPO/IPO/online-RL models, has the committee generate candidate responses, collects human preference labels over them, trains a reward model on those labels, then uses the committee + reward model together to drive rejection sampling for the next round — a multi-model ensemble version of the rejection-sampling pattern every other 2024-era report also converged on. The RLHF step itself uses **Mirror Descent Policy Optimization** in place of PPO, chosen for effectiveness at Apple's scale. AFM's data mixture was tuned empirically rather than fixed by predetermined ratios — consistent with the playbook's broader "ablate, don't assume" principle above.

## Gemma 2: SWA + global attention interleaving and knowledge distillation

The "Gemma 3 Technical Report" PDF is actually Gemma 2. [Gemma 2 (Google DeepMind, 2024)](../../papers/01-models/llama-qwen-gemma/Gemma 3 Technical Report.pdf) introduced two architectural choices that distinguish it from contemporaries.

**Interleaved local/global attention.** Every other layer uses a sliding window (local) attention with window size 4096, while the remaining layers use full global attention with span 8192. The sliding window size can be changed at inference time with small perplexity impact (4096→1024 degrades perplexity by only 0.01), enabling inference speed tuning. This is directly relevant to [[attention-variants#Sliding Window Attention]].

**Logit soft-capping.** Logits are capped via `logits ← soft_cap × tanh(logits/soft_cap)` with soft_cap=50.0 for self-attention and 30.0 for the final layer. This bounds activations and improves training stability, though it is incompatible with FlashAttention fused kernels unless specifically handled (see [[training-stability]]).

**Knowledge distillation for small models.** The 2B and 9B Gemma 2 models are trained with distillation from a 7B teacher (for 2B) and 27B teacher (for 9B), training on tokens substantially above the compute-optimal count for their size. A 2B model trained on 500B tokens (10x compute-optimal for 2B) with distillation averages 67.7 vs 60.3 for the same model trained from scratch — a 7.5 point gap that persists as model size increases. The practical prescription for small models: **if you have a larger model available, always distill rather than train from scratch, especially when overtraining beyond Chinchilla-optimal**.

Architecture: GQA with 2 groups (ablation showed GQA≈MHA on quality while being faster), GeGLU activations, pre-norm and post-norm both applied with RMSNorm, 256K vocabulary (Gemini's SentencePiece tokenizer). Training data: 27B trained on 13T tokens, 9B on 8T, 2B on 2T — all primarily English. Post-training: SFT on synthetic + human prompts (with knowledge-distillation-generated responses folded into the SFT mix itself, not just pretraining), RLHF with a reward model **10× larger** than the policy model, then **WARP** (Weight Averaged Rewarded Policies, successor to WARM/Weight-Averaged Reward Models) to average the resulting policy models — see the model-averaging takeaway below, which recurs independently in Llama 3.1's post-training.

- **Memory- or infra-constrained?** Default dense + GQA + RoPE/RNoPE.
- **Need inference efficiency at scale + can manage routing complexity?** MoE with strong load balancing.
- **Long context is a core requirement?** Document masking + RoPE scaling (ABF/YaRN) or RNoPE variants, planned early.
- **Need simpler kernels + fast iteration?** Avoid novel attention variants unless you can ablate them cleanly.

## How the wiki pieces fit together

This article is the spine; the deep dives live in:

- [[attention-variants]] — GQA, MQA, MHA, MLA, gated attention, document masking, attention patterns for long context (chunked, SWA, DCA, interleaved).
- [[positional-encodings]] — RoPE, NoPE, RNoPE, Partial RoPE, ABF/YaRN scaling.
- [[long-context-training]] — bridge map for RoPE/YaRN/RNoPE, long-data mixtures, document masking, KV-cache cost, and effective-context evals.
- [[mixture-of-experts]] — routing, load balancing strategies (LBL, auxiliary-loss-free, sequence-wise, SMEBU), shared experts, granularity, sparsity.
- [[hybrid-architectures]] — linear-attention/RNN + transformer hybrids (Mamba-2, gated DeltaNet, kimi delta attention).
- [[gpu-kernel-engineering]] — hardware-aware attention, MoE, inference, and state-space kernels.
- [[training-stability]] — z-loss, logit softcapping, QK-norm, RMSNorm, weight decay on embeddings, depth-scaled sandwich norm, init.
- [[optimizers]] — AdamW, Muon, MuonClip; learning rate schedules (cosine, WSD, multi-step); batch size and the critical batch.
- [[tokenizers]] — vocab sizing, fertility, BPE, domain considerations.
- [[scaling-laws]] — compute-optimal vs inference-optimal, overtraining, model-specific sparsity tradeoffs.
- [[data-curation-mixtures]] — multi-stage training, ablation, token utility, rephrasing.
- [[data-quality-vs-diversity]] — stage-dependent data tradeoffs: broad pretraining, high-quality SFT, and filtering caveats.
- [[supervised-fine-tuning]] — sequence packing, masking, chat templates, learning rate, epochs, CCE kernel.
- [[alignment-methods]] — DPO, KTO, ORPO, APO, RLHF, GRPO, RLVR, in-flight updates, IcePop, rubric rewards.
- [[rl-training-systems]] — rollout/trainer systems for verifier RL, stale-policy control, and in-flight updates.
- [[on-policy-distillation]] — OPD, OPSD, SDFT, GOLD, MOPD.
- [[reasoning-models]] — RLVR, length penalties, DeepSeek-R1 pipeline, MCTS limits.
- [[reasoning-data-generation]] — teacher traces, synthetic verifiable tasks, cognitive behaviors, and reasoning data placement.
- [[training-ops]] — vanishing throughput, noisy loss, tensor parallelism seeds, multi-client orchestrator, the usual suspects.
- [[small-efficient-models]] — what changes when the target is a 1-3B deployable model rather than a frontier-scale run.
- [[model-report-case-studies]] — named-model map from technical reports to reusable training lessons.

## Sources

- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- Primary references the post draws from: SmolLM3 report (`raw/smollm3-hugging-face-report.md`), gpt-oss-120b system card, Kimi K2 technical report, Hermes 4 technical report, Intellect-3 (Prime Intellect), DeepSeek-R1 paper (`raw/deepseek-r1-reasoning-via-rl.md`), Arcee Trinity series, Thinking Machines on-policy distillation (`raw/on-policy-distillation-thinking-machines.md`).
- [GPT-4 Technical Report (2303.08774)](../../papers/01-models/gpt-deepseek-v2-v3/GPT-4 Technical Report - 2303.08774.pdf) — predictable scaling methodology; pre-training drives capability, RLHF drives instruction following.
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (2405.04434)](../../papers/01-models/gpt-deepseek-v2-v3/DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model - 2405.04434.pdf) — MLA origin, DeepSeekMoE, device-limited routing, three-part auxiliary loss.
- [DeepSeek-V3 Technical Report (2407.21783)](../../papers/01-models/gpt-deepseek-v2-v3/DeepSeek-V3 Technical Report - 2407.21783.pdf) — auxiliary-loss-free MoE via bias terms, multi-token prediction for speculative decoding, FP8 training.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.00656)](../../papers/01-models/gpt-deepseek-v2-v3/DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning - 2501.00656.pdf) — RL-trained reasoning on top of V3 base; cold-start SFT + GRPO pipeline. See [[reasoning-models]].
- [The Llama 3 Herd of Models (2501.12948)](../../papers/01-models/llama-qwen-gemma/The Llama 3 Herd of Models - 2501.12948.pdf) — three-level dedup, model-based quality filtering, IsoFLOPs + two-stage downstream scaling law, 4D parallelism on 16K H100s, 15.6T tokens.
- [Qwen3 Technical Report (2505.09388)](../../papers/01-models/llama-qwen-gemma/Qwen3 Technical Report - 2505.09388.pdf) — 36T token training, three-stage pre-training, unified thinking/non-thinking post-training, strong-to-weak distillation for small models.
- [Gemma 2 Technical Report (Google DeepMind, 2024)](../../papers/01-models/llama-qwen-gemma/Gemma 3 Technical Report.pdf) — SWA + global attention interleaving, logit soft-capping, knowledge distillation for small models (67.7 vs 60.3 average for 2B distilled vs from-scratch).
- Sebastian Raschka, ["New LLM Pre-training and Post-training Paradigms"](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training) (Aug 2024) — cross-report comparison of Qwen 2, Apple AFM, Gemma 2, and Llama 3.1; source for the Qwen 2 two-stage DPO recipe, the full AFM pretraining/iTeC/Mirror-Descent-RLHF pipeline, and the WARP/WARM-vs-Llama-3.1-averaging cross-model takeaway. See `raw/raschka-llm-pretraining-posttraining-paradigms.md`.
