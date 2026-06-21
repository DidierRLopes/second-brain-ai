# Research Papers Index

> 73 papers (PDFs on disk) plus 1 URL-only reference (DeepSeek-V3), organized into 8 thematic categories (with subfolders) under `papers/`. This index is generated from the actual folder structure on disk — if you move or rename a file, update its entry here.

## Quick Reference

| Category | Subfolders | Papers |
|---|---|---|
| [01-models](#01-models) | gpt-deepseek-v2-v3, llama-qwen-gemma, small-efficient | 10 (+1 URL-only) |
| [02-architecture](#02-architecture) | transformers, attention-variants, alternatives | 7 |
| [03-scaling](#03-scaling) | scaling-laws, compute-optimal, sparse-moe, training-optimization | 15 |
| [04-efficiency](#04-efficiency) | quantization, context-extension, inference-kernels | 8 |
| [05-learning](#05-learning) | alignment-preferences, reinforcement-learning, reasoning, fine-tuning | 19 |
| [06-data](#06-data) | curation-filtering, datasets, tokenization | 6 |
| [07-applications](#07-applications) | agents-swe | 4 |
| [08-evaluation](#08-evaluation) | benchmarking, analysis, safety | 4 |

---

## 01-models

Named model releases and technical reports.

**gpt-deepseek-v2-v3/**
- [GPT-4 Technical Report](<papers/01-models/gpt-deepseek-v2-v3/GPT-4 Technical Report - 2303.08774.pdf>) — arXiv:2303.08774
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](<papers/01-models/gpt-deepseek-v2-v3/DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model - 2405.04434.pdf>) — arXiv:2405.04434
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — arXiv:2412.19437 — *not currently in the repo as a PDF; arXiv link only*
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](<papers/01-models/gpt-deepseek-v2-v3/DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning - 2501.12948.pdf>) — arXiv:2501.12948

**llama-qwen-gemma/**
- [Gemma 2: Improving Open Language Models at a Practical Size](<papers/01-models/llama-qwen-gemma/Gemma 2: Improving Open Language Models at a Practical Size - 2408.00118.pdf>) — arXiv:2408.00118
- [Gemma 3 Technical Report](<papers/01-models/llama-qwen-gemma/Gemma 3 Technical Report - 2503.19786.pdf>) — arXiv:2503.19786
- [OLMo 2 Furious](<papers/01-models/llama-qwen-gemma/OLMo 2 Furious - 2501.00656.pdf>) — arXiv:2501.00656
- [Qwen3 Technical Report](<papers/01-models/llama-qwen-gemma/Qwen3 Technical Report - 2505.09388.pdf>) — arXiv:2505.09388
- [The Llama 3 Herd of Models](<papers/01-models/llama-qwen-gemma/The Llama 3 Herd of Models - 2407.21783.pdf>) — arXiv:2407.21783

**small-efficient/**
- [MiniCPM: Unveiling the Potential of Small Language Models](<papers/01-models/small-efficient/MiniCPM: Unveiling the Potential of Small Language Models - 2404.06395.pdf>) — arXiv:2404.06395
- [SmolLM2: When Smol Goes Big](<papers/01-models/small-efficient/SmolLM2: When Smol Goes Big - 2502.02737.pdf>) — arXiv:2502.02737

## 02-architecture

Core transformer mechanics, positional encoding, and architectural alternatives.

**transformers/**
- [Fast Transformer Decoding: One Write-Head is All You Need](<papers/02-architecture/transformers/Fast Transformer Decoding: One Write-Head is All You Need - 1911.02150.pdf>) — arXiv:1911.02150 (MQA)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](<papers/02-architecture/transformers/GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints - 2305.13245.pdf>) — arXiv:2305.13245

**attention-variants/**
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](<papers/02-architecture/attention-variants/RoFormer: Enhanced Transformer with Rotary Position Embedding - 2104.09864.pdf>) — arXiv:2104.09864
- [The Impact of Positional Encoding on Length Generalization in Transformers](<papers/02-architecture/attention-variants/The Impact of Positional Encoding on Length Generalization in Transformers - 2305.19466.pdf>) — arXiv:2305.19466
- [Rope to Nope and Back Again: A New Hybrid Position Encoding for Efficient Context Scaling](<papers/02-architecture/attention-variants/Rope to Nope and Back Again: A New Hybrid Position Encoding for Efficient Context Scaling - 2501.18795.pdf>) — arXiv:2501.18795
- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design](<papers/02-architecture/attention-variants/FlashAttention-4: Algorithm and Kernel Pipelining Co-Design - 2603.05451.pdf>) — arXiv:2603.05451

**alternatives/**
- [Transformers are SSMs: Generalized Models and Efficient Algorithms for Sequence Modeling](<papers/02-architecture/alternatives/Transformers are SSMs: Generalized Models and Efficient Algorithms for Sequence Modeling - 2405.21060.pdf>) — arXiv:2405.21060

## 03-scaling

Scaling laws, compute-optimal training, MoE scaling, and training-optimization methods.

**scaling-laws/**
- [PaLM: Scaling Language Modeling with Pathways](<papers/03-scaling/scaling-laws/PaLM: Scaling Language Modeling with Pathways - 2204.02311.pdf>) — arXiv:2204.02311
- [Training Compute-Optimal Large Language Models](<papers/03-scaling/scaling-laws/Training Compute-Optimal Large Language Models - 2203.15556.pdf>) — arXiv:2203.15556 (Chinchilla)
- [Scaling Laws for Predicting Downstream Performance in LLMs](<papers/03-scaling/scaling-laws/Scaling Laws for Predicting Downstream Performance in LLMs - 2410.08527.pdf>) — arXiv:2410.08527
- [Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models](<papers/03-scaling/scaling-laws/Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models - 2507.17702.pdf>) — arXiv:2507.17702

**compute-optimal/**
- [Scaling Data-Constrained Language Models](<papers/03-scaling/compute-optimal/Scaling Data-Constrained Language Models - 2305.16264.pdf>) — arXiv:2305.16264

**sparse-moe/**
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](<papers/03-scaling/sparse-moe/Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer - 1701.06538.pdf>) — arXiv:1701.06538
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](<papers/03-scaling/sparse-moe/GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding - 2006.16668.pdf>) — arXiv:2006.16668
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](<papers/03-scaling/sparse-moe/Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity - 2101.03961.pdf>) — arXiv:2101.03961
- [Unified Scaling Laws for Routed Language Models](<papers/03-scaling/sparse-moe/Unified Scaling Laws for Routed Language Models - 2202.01169.pdf>) — arXiv:2202.01169

**training-optimization/**
- [SGDR: Stochastic Gradient Descent with Warm Restarts](<papers/03-scaling/training-optimization/SGDR: Stochastic Gradient Descent with Warm Restarts - 1608.03983.pdf>) — arXiv:1608.03983
- [An Empirical Model of Large-Batch Training](<papers/03-scaling/training-optimization/An Empirical Model of Large-Batch Training - 1812.06162.pdf>) — arXiv:1812.06162
- [Scaling Scaling Laws with Board Games](<papers/03-scaling/training-optimization/Scaling Scaling Laws with Board Games - 2104.03113.pdf>) — arXiv:2104.03113
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](<papers/03-scaling/training-optimization/ThunderKittens: Simple, Fast, and Adorable AI Kernels - 2410.20399.pdf>) — arXiv:2410.20399
- [PipelineRL: Faster On-policy Reinforcement Learning by Leveraging Pipeline Parallelism](<papers/03-scaling/training-optimization/PipelineRL: Faster On-policy Reinforcement Learning by Leveraging Pipeline Parallelism - 2509.19128.pdf>) — arXiv:2509.19128
- [Scalable Training of Mixture-of-Experts Models with Efficient Sparsity](<papers/03-scaling/training-optimization/Scalable Training of Mixture-of-Experts Models with Efficient Sparsity - 2603.07685.pdf>) — arXiv:2603.07685

## 04-efficiency

Quantization, long-context extension, and inference kernels.

**quantization/**
- [QuIP: 2-Bit Quantization of Large Language Models With Guarantees](<papers/04-efficiency/quantization/QuIP: 2-Bit Quantization of Large Language Models With Guarantees - 2307.13304.pdf>) — arXiv:2307.13304
- [Extreme Compression of Large Language Models via Additive Quantization](<papers/04-efficiency/quantization/Extreme Compression of Large Language Models via Additive Quantization - 2401.06118.pdf>) — arXiv:2401.06118

**context-extension/**
- [YARN: Efficient Context Window Extension of Large Language Models](<papers/04-efficiency/context-extension/YARN: Efficient Context Window Extension of Large Language Models - 2309.00071.pdf>) — arXiv:2309.00071
- [Effective Long-Context Scaling of Foundation Models](<papers/04-efficiency/context-extension/Effective Long-Context Scaling of Foundation Models - 2309.16039.pdf>) — arXiv:2309.16039
- [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](<papers/04-efficiency/context-extension/DeepSeek LLM: Scaling Open-Source Language Models with Longtermism - 2401.02954.pdf>) — arXiv:2401.02954
- [How to Train Long-Context Language Models (Effectively)](<papers/04-efficiency/context-extension/How to Train Long-Context Language Models (Effectively) - 2410.02660.pdf>) — arXiv:2410.02660 (ProLong)
- [RULER: What's the Real Context Size of Your Long-Context Language Models](<papers/04-efficiency/context-extension/RULER: What's the Real Context Size of Your Long-Context Language Models - 2404.06654.pdf>) — arXiv:2404.06654

**inference-kernels/**
- [Interfaze: The Future of AI is Built on Task-Specific Small Models](<papers/04-efficiency/inference-kernels/Interfaze: The Future of AI is Built on Task-Specific Small Models - 2602.04101.pdf>) — arXiv:2602.04101

## 05-learning

Preference optimization, RL, reasoning, and fine-tuning/adaptation.

**alignment-preferences/**
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](<papers/05-learning/alignment-preferences/Direct Preference Optimization: Your Language Model is Secretly a Reward Model - 2305.18290.pdf>) — arXiv:2305.18290
- [KTO: Model Alignment as Prospect Theoretic Optimization](<papers/05-learning/alignment-preferences/KTO: Model Alignment as Prospect Theoretic Optimization - 2402.01306.pdf>) — arXiv:2402.01306
- [ORPO: Monolithic Preference Optimization without Reference Model](<papers/05-learning/alignment-preferences/ORPO: Monolithic Preference Optimization without Reference Model - 2403.07691.pdf>) — arXiv:2403.07691
- [Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Direct Preference Optimization](<papers/05-learning/alignment-preferences/Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Direct Preference Optimization - 2408.06266.pdf>) — arXiv:2408.06266
- [On the Design of KL-Regularized Policy Gradient](<papers/05-learning/alignment-preferences/ON THE DESIGN OF KL-REGULARIZED POLICY GRADIENT - 2505.17508.pdf>) — arXiv:2505.17508

**reinforcement-learning/**
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](<papers/05-learning/reinforcement-learning/High-Dimensional Continuous Control Using Generalized Advantage Estimation - 1506.02438.pdf>) — arXiv:1506.02438 (GAE)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](<papers/05-learning/reinforcement-learning/Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm - 1712.01815.pdf>) — arXiv:1712.01815
- [DAPO: An Open-Source LLM Reinforcement Learning Framework](<papers/05-learning/reinforcement-learning/DAPO: An Open-Source LLM Reinforcement Learning Framework - 2503.14476.pdf>) — arXiv:2503.14476
- [Reuse your FLOPs: Scaling RL on Hard Problems by Recycling Computation](<papers/05-learning/reinforcement-learning/Reuse your FLOPs: Scaling RL on Hard Problems by Recycling Computation - 2601.18795.pdf>) — arXiv:2601.18795

**reasoning/**
- [Cognitive Behaviors that Enable Self-Improving Reasoners](<papers/05-learning/reasoning/Cognitive Behaviors that Enable Self-Improving Reasoners - 2503.01307.pdf>) — arXiv:2503.01307
- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](<papers/05-learning/reasoning/GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning - 2507.19457.pdf>) — arXiv:2507.19457
- [OpenThoughts: Data Recipes for Reasoning Models](<papers/05-learning/reasoning/OpenThoughts: Data Recipes for Reasoning Models - 2506.04178.pdf>) — arXiv:2506.04178
- [QED-Nano: Teaching a Tiny Model to Prove Hard Theorems](<papers/05-learning/reasoning/QED-Nano: Teaching a Tiny Model to Prove Hard Theorems - 2604.04898.pdf>) — arXiv:2604.04898
- [SynLogic: A Data Synthesis Framework for Logical Reasoning](<papers/05-learning/reasoning/SynLogic: A Data Synthesis Framework for Logical Reasoning - 2505.19641.pdf>) — arXiv:2505.19641
- [Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data](<papers/05-learning/reasoning/Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data - 2510.03264.pdf>) — arXiv:2510.03264
- [Generative Recursive Reasoning (GRAM)](<papers/05-learning/reasoning/Generative Recursive Reasoning (GRAM) - 2605.19376.pdf>) — arXiv:2605.19376
- [Wait, Wait, Wait... Why Do Reasoning Models Loop](<papers/05-learning/reasoning/Wait, Wait, Wait... Why Do Reasoning Models Loop - 2512.12895.pdf>) — arXiv:2512.12895

**fine-tuning/**
- [Revisiting DAgger in the Era of LLM-Agents](<papers/05-learning/fine-tuning/Revisiting DAgger in the Era of LLM-Agents - 2605.12913.pdf>) — arXiv:2605.12913
- [Teaching Pretrained Language Models To...](<papers/05-learning/fine-tuning/TEACHING PRETRAINED LANGUAGE MODELS TO - 2511.07384.pdf>) — arXiv:2511.07384

## 06-data

Data curation/filtering, datasets, and tokenization.

**curation-filtering/**
- [FineWeb2: One Pipeline to Scale Them All](<papers/06-data/curation-filtering/FineWeb2: One Pipeline to Scale Them All - 2506.20920.pdf>) — arXiv:2506.20920
- [MEGASCIENCE: Pushing the Frontiers of Large-Scale Data Collection](<papers/06-data/curation-filtering/MEGASCIENCE: Pushing the Frontiers of Large-Scale Data Collection - 2507.16812.pdf>) — arXiv:2507.16812
- [A Bitter Lesson for Data Filtering](<papers/06-data/curation-filtering/A Bitter Lesson for Data Filtering - 2605.19407.pdf>) — arXiv:2605.19407

**datasets/**
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](<papers/06-data/datasets/The Pile: An 800GB Dataset of Diverse Text for Language Modeling - 2101.00027.pdf>) — arXiv:2101.00027

**tokenization/**
- [Getting the Most Out of Your Tokenizer for Pre-training and Domain Adaptation](<papers/06-data/tokenization/Getting the most out of your tokenizer for pre-training and domain adaptation - 2402.01035.pdf>) — arXiv:2402.01035
- [Decoupling the Benefits of Subword Tokenization for Cross-Lingual Transfer](<papers/06-data/tokenization/Decoupling the Benefits of Subword Tokenization for Cross-Lingual Transfer - 2604.27263.pdf>) — arXiv:2604.27263

## 07-applications

**agents-swe/**
- [KIMIK2: Open Agentic Intelligence](<papers/07-applications/agents-swe/KIMIK2: Open Agentic Intelligence - 2507.20534.pdf>) — arXiv:2507.20534
- [Multi-SWE-bench: A Multilingual Benchmark for Software Engineering](<papers/07-applications/agents-swe/Multi-SWE-bench: A Multilingual Benchmark for Software Engineering - 2504.02605.pdf>) — arXiv:2504.02605
- [R2E-Gym: Procedural Environments and Hybrid Verifiers for Reinforcement Learning Agents](<papers/07-applications/agents-swe/R2E-Gym: Procedural Environments and Hybrid Verifiers for Reinforcement Learning Agents - 2504.07164.pdf>) — arXiv:2504.07164
- [SWE-smith: Scaling Data for Software Engineering Agents](<papers/07-applications/agents-swe/SWE-smith: Scaling Data for Software Engineering Agents - 2504.21798.pdf>) — arXiv:2504.21798

## 08-evaluation

Benchmarking, training-dynamics analysis, and safety/misalignment.

**benchmarking/**
- [Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models](<papers/08-evaluation/benchmarking/Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models - 2504.03624.pdf>) — arXiv:2504.03624
- [Open-World Evaluations for Measuring Frontier AI Capabilities](<papers/08-evaluation/benchmarking/Open-World Evaluations for Measuring Frontier AI Capabilities - 2605.20520.pdf>) — arXiv:2605.20520

**analysis/**
- [Analysing the Impact of Sequence Composition on Language Model Pre-Training](<papers/08-evaluation/analysis/Analysing The Impact of Sequence Composition on Language Model Pre-Training - 2402.13991.pdf>) — arXiv:2402.13991

**safety/**
- [Natural Emergent Misalignment from Reward Hacking](<papers/08-evaluation/safety/Natural Emergent Misalignment from Reward Hacking.pdf>) — no arXiv ID on file

---

## Sources note

Titles and arXiv IDs above are read directly from filenames in `papers/`. If you add a paper, follow the existing convention: `<Title> - <arXiv ID>.pdf`, filed under the most specific matching subfolder.

A 2026-06-20 audit found that titles and arXiv IDs had been cyclically swapped across several filenames in `01-models`, `03-scaling`, `04-efficiency`, `05-learning`, and `08-evaluation` (the arXiv ID embedded in each filename was always correct for its PDF's actual content; only the title-text portion, and downstream wiki citations referencing these papers, had been swapped between files). All affected filenames, this index, and cross-references in `ai-machine-learning/wiki/` have been corrected and verified against arxiv.org and the PDFs' own content.

Note: `papers/TO-BE-ORGANIZED/` contains 17 raw, untitled PDFs awaiting triage and is intentionally excluded from this index.
