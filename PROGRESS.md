# ML Knowledge Base — Topic Coverage Progress

Last updated: 2026-06-21 (depth audit pass)

Legend: ✅ Covered | 🔄 Partial | ❌ Missing

---

## Reinforcement Learning

| Topic | Status | File |
|-------|--------|------|
| Q-Learning / TD Learning | ✅ | [[rl-fundamentals]] |
| Bellman Equations | ✅ | [[rl-fundamentals]] |
| PPO | ✅ | [[policy-gradient-actor-critic]], [[alignment-methods]] |
| GRPO | ✅ | [[alignment-methods]], [[rl-training-systems]] |
| GAE | ✅ | [[policy-gradient-actor-critic]] |
| Variance Reduction in RL | ✅ | [[policy-gradient-actor-critic]] |
| DPO (Direct Preference Optimisation) | ✅ | [[alignment-methods]] |
| Policy Gradient Theorem | ✅ | [[policy-gradient-actor-critic]] |
| On-Policy vs Off-Policy | ✅ | [[rl-fundamentals]] |
| Exploration vs Exploitation Dilemma | ✅ | [[rl-fundamentals]] |
| Credit Assignment Problem | ✅ | [[rl-fundamentals]] |
| MuZero | ✅ | [[model-based-rl-advanced]] |
| World Models / Dreamer | ✅ | [[model-based-rl-advanced]] |
| AlphaGo | ✅ | [[model-based-rl-advanced]], [[alignment-methods]] |
| Soft Actor-Critic | ✅ | [[model-based-rl-advanced]] |
| Model-Based vs Model-Free | ✅ | [[model-based-rl-advanced]] |
| Markov Property | ✅ | [[rl-fundamentals]] |
| Monte Carlo vs TD | ✅ | [[rl-fundamentals]] |
| Actor Critic | ✅ | [[policy-gradient-actor-critic]] |
| SARSA | ✅ | [[rl-fundamentals]] |
| Importance Sampling | ✅ | [[rl-fundamentals]] |
| Markov Decision Process | ✅ | [[rl-fundamentals]] |
| Curriculum Learning | ✅ | [[rl-fundamentals]] |

---

## LLMs

| Topic | Status | File |
|-------|--------|------|
| Flash Attention | ✅ | [[gpu-kernel-engineering]] |
| LoRA | ✅ | [[parameter-efficient-fine-tuning]] |
| TransformerXL | ✅ | [[llm-architectures-extended]] |
| Griffin | ✅ | [[llm-architectures-extended]] |
| Perceiver | ✅ | [[llm-architectures-extended]] |
| Scaling Laws | ✅ | [[scaling-laws]] |
| Mixture of Experts | ✅ | [[mixture-of-experts]] |
| LLM scaling factor | ✅ | [[scaling-laws]] |
| RoPE | ✅ | [[positional-encodings]] |
| Sinusoidal embeddings | ✅ | [[positional-encodings]] |
| Relative positional embeddings | ✅ | [[positional-encodings]] |
| LLM vs RNN vs S4 | ✅ | [[llm-architectures-extended]] |
| Tokenisation | ✅ | [[tokenizers]] |
| Pretraining | ✅ | [[llm-architectures-extended]], [[data-curation-mixtures]] |
| Finetuning | ✅ | [[supervised-fine-tuning]], [[practical-fine-tuning]] |
| RLHF | ✅ | [[alignment-methods]] |
| Decoding techniques | ✅ | [[llm-architectures-extended]] |
| Causal Attention | ✅ | [[transformer-architecture]] |
| Cross Attention | ✅ | [[transformer-architecture]] |
| GaLore (Gradient Low-Rank Projection) | ✅ | [[parameter-efficient-fine-tuning]] |
| SimPO | ✅ | [[alignment-methods]] |
| Hyper-Connections / mHC | ✅ | [[transformer-architecture]] |
| MXFP4 / NVFP4 microscaling formats | ✅ | [[quantization-fundamentals]] |
| TurboQuant / PolarQuant (KV-cache vector quantization) | ✅ | [[kv-cache]] |
| Agent memory layers (Mem0, Zep/Graphiti) | ✅ | [[llm-agents]] |

---

## Generative Modelling

| Topic | Status | File |
|-------|--------|------|
| GANs | ✅ | [[generative-models]] |
| VAEs and VAE ELBO | ✅ | [[generative-models]] |
| Score Function | ✅ | [[generative-models]] |
| Diffusion Forward Process | ✅ | [[generative-models]] |
| Diffusion Reverse Process (DDIM / DDPM) | ✅ | [[generative-models]] |
| Diffusion Forward / Reverse SDE | ✅ | [[generative-models]] |
| Flow Matching ODE | ✅ | [[generative-models]] |
| Classifier Free Guidance | ✅ | [[generative-models]] |

---

## Applied ML

| Topic | Status | File |
|-------|--------|------|
| Tensor Parallelism | ✅ | [[how-to-scale-your-model]] |
| FSDP | ✅ | [[how-to-scale-your-model]] |
| DDP | ✅ | [[applied-ml-systems]] |
| Pipeline Parallelism | ✅ | [[how-to-scale-your-model]] |
| Communication Primitives | ✅ | [[applied-ml-systems]] |
| Mixed precision training | ✅ | [[applied-ml-systems]] |
| Gradient checkpointing | ✅ | [[applied-ml-systems]] |
| Gradient accumulation | ✅ | [[applied-ml-systems]] |
| Profiling | ✅ | [[applied-ml-systems]] |
| Gradient clipping | ✅ | [[applied-ml-systems]], [[training-stability]] |
| Numerical precision tricks | ✅ | [[applied-ml-systems]] |
| Exploding / vanishing gradients | ✅ | [[applied-ml-systems]] |
| Floating point representation | ✅ | [[applied-ml-systems]] |
| JIT compiling | ✅ | [[applied-ml-systems]] |
| JAX, PyTorch, TensorFlow | ✅ | [[applied-ml-systems]] |

---

## General ML

| Topic | Status | File |
|-------|--------|------|
| Curse of dimensionality | ✅ | [[ml-theory-statistics]] |
| S4 | ✅ | [[deep-learning-fundamentals]] |
| CNNs | ✅ | [[deep-learning-fundamentals]] |
| RNNs / LSTMs | ✅ | [[deep-learning-fundamentals]] |
| Autoencoders | ✅ | [[deep-learning-fundamentals]] |
| Gumbel-Softmax | ✅ | [[deep-learning-fundamentals]] |
| MLE vs MAP | ✅ | [[ml-theory-statistics]] |
| Newton's Method | ✅ | [[optimization-regularization]] |
| Linear Regression | ✅ | [[classical-ml]] |
| Activation Functions | ✅ | [[deep-learning-fundamentals]] |
| Loss Functions | ✅ | [[deep-learning-fundamentals]] |
| No Free Lunch Theorem | ✅ | [[ml-theory-statistics]] |
| BatchNorm / LayerNorm / RMSNorm | ✅ | [[deep-learning-fundamentals]] |
| Variance and Covariance | ✅ | [[ml-theory-statistics]] |
| Adam / AdamW / Adagrad | ✅ | [[optimizers]] |
| Bias-Variance Tradeoff | ✅ | [[ml-theory-statistics]] |
| Backprop | ✅ | [[deep-learning-fundamentals]] |
| Regularisation Methods | ✅ | [[optimization-regularization]] |
| Unsupervised vs Supervised | ✅ | [[classical-ml]] |
| Clustering Algorithms (e.g. k-means) | ✅ | [[classical-ml]] |
| K-Nearest Neighbours | ✅ | [[classical-ml]] |
| SVMs | ✅ | [[classical-ml]] |
| Boosting | ✅ | [[classical-ml]] |
| Bagging | ✅ | [[classical-ml]] |
| Decision Trees | ✅ | [[classical-ml]] |
| Ensembles | ✅ | [[classical-ml]] |
| Bayes Theorem | ✅ | [[ml-theory-statistics]] |
| Precision / Recall / F1 / AUC-ROC | ✅ | [[ml-theory-statistics]] |
| KL Divergence | ✅ | [[ml-theory-statistics]] |
| Jensen-Shannon Divergence | ✅ | [[ml-theory-statistics]] |
| Weight initialisation | ✅ | [[deep-learning-fundamentals]] |
| Gradient Descent / SGD | ✅ | [[optimization-regularization]] |
| Overfitting / Underfitting | ✅ | [[optimization-regularization]] |
| Cross validation | ✅ | [[optimization-regularization]] |
| Data Whitening | ✅ | [[optimization-regularization]] |
| Convex functions | ✅ | [[ml-theory-statistics]] |
| Early Stopping | ✅ | [[optimization-regularization]] |
| Domain Adaptation | ✅ | [[optimization-regularization]] |
| Dimensionality Reduction | ✅ | [[optimization-regularization]] |
| Transfer Learning | ✅ | [[optimization-regularization]] |
| Few shot / Zero shot learning | ✅ | [[optimization-regularization]] |
| Second Order Methods | ✅ | [[optimization-regularization]] |
| Expectation | ✅ | [[ml-theory-statistics]] |
| Entropy | ✅ | [[ml-theory-statistics]] |
| PDF / PMF | ✅ | [[ml-theory-statistics]] |
| Confidence Intervals | ✅ | [[ml-theory-statistics]] |

---

## Summary

- **Total topics:** 136
- **Covered:** 136
- **Partial:** 0
- **Missing:** 0

## 2026-06-20 Gap Audit

Ran a full audit of raw/ (130 files) and papers/ (73 PDFs) against wiki/ coverage, plus web research targeted at the stated interests (LLMs & Agents, Quantization, Transformer Architecture, Fine Tuning). Structural coverage was already complete — every raw source and paper is reflected somewhere in the wiki, with one known, intentionally-documented exception: `raw/k-a-in-rl-algorithm-questions.md` is a dead-end source (the source page is a content-free "chapter opener" stub; re-verified live via fetch on this pass, still no real content to ingest).

Six genuinely missing concepts were found via web research (none present anywhere in the wiki, verified by grep before writing) and added:
- **GaLore** — gradient low-rank projection, full-parameter training under LoRA-like memory budgets → [[parameter-efficient-fine-tuning]]
- **SimPO** — reference-free, length-normalized preference optimization → [[alignment-methods]]
- **Hyper-Connections / mHC** — generalizing the residual stream to multiple learned streams; DeepSeek's manifold-constrained stabilization → [[transformer-architecture]]
- **MXFP4 / NVFP4** — hardware-native 4-bit microscaling float formats (OCP vs NVIDIA Blackwell) → [[quantization-fundamentals]]
- **TurboQuant / PolarQuant** — near-optimal vector quantization for KV cache (Google, ICLR/AISTATS 2026) → [[kv-cache]]
- **Mem0 / Zep** — production agent memory layers (extract/update vs temporal knowledge graph) → [[llm-agents]]

## 2026-06-21 Gap Audit (Depth Pass)

This was a depth audit, not a coverage audit: no new topics were added, and the Summary counts above (136/136 covered) are unchanged. The previous pass (2026-06-20) only checked that every raw source and paper *resolved* to a wiki section; it didn't check whether that section captured the paper in full.

Between the two audits, the 15 PDFs sitting in `papers/TO-BE-ORGANIZED/` were filed (`papers/` went from 73 → 88 PDFs). This pass re-read the full text of all 15 newly-filed papers and diffed it against the wiki paragraphs written for them. 14 of 15 had captured the headline mechanism/numbers but missed secondary results, ablations, or stated limitations. Extended (not replaced) the existing sections with that missing depth for: u-μP, CompleteP, Pre-training under infinite compute, Cartridges, Do LMs Need Sleep?, IndexCache, AutoHarness, Meta-Harness, ECHO, Long-Horizon Q-Learning, Self-Distilled Policy Gradient, Memorization Dynamics in KD, Qwen-VLA, and Self-Revising Discovery Systems for Science — touching [[agent-harness-engineering]], [[agentic-rl]], [[ai-rd-automation]], [[attention-variants]], [[hybrid-architectures]], and [[knowledge-distillation]]. POSTTRAINBENCH was verified already thorough; no changes needed.

All new citations were verified to resolve to existing local PDFs and all wiki-links to resolve to existing pages.
