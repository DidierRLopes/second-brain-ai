# AI / Machine Learning Knowledge Base — Wiki Index

## Frontier Training (synthesis)
- [[frontier-training-playbook]] — The 12-step minimal playbook; how SmolLM3, gpt-oss-120b, Kimi K2, Hermes 4, Intellect-3, DeepSeek-R1, and Trinity actually train; plus Raschka's Qwen 2/Apple AFM/Gemma 2/Llama 3.1 pre/post-training comparison (iTeC, Mirror Descent RLHF, WARP/WARM, model averaging); spine for the rest of the wiki
- [[model-report-case-studies]] — Named-model map from GPT-4, DeepSeek, Llama, Qwen, Gemma, Nemotron-H, SmolLM2, MiniCPM, and Kimi K2 to the design lessons they teach; plus an RL-era addendum covering Kimi K1.5/Kimi-Researcher, Composer 2/2.5, Olmo 3, MiniMax-M1/M2, and the Nemotron 3 family
- [[small-efficient-models]] — How 1-3B deployable models trade extra tokens, cleaner data, WSD schedules, and distillation for cheap inference and local deployment

## Transformer Architecture
- [[transformer-architecture]] — The foundational architecture behind all modern LLMs, from attention mechanisms to encoder-decoder design; Hyper-Connections and DeepSeek's manifold-constrained mHC as a generalization of the residual stream; full equation-by-equation modern decoder walkthrough (RMSNorm, GQA-aware QKV, RoPE, SwiGLU) with attention-block PyTorch implementation and the model-activations formula
- [[positional-encodings]] — How transformers encode sequence order: sinusoidal, RoPE, ALiBi, NoPE, RNoPE, ABF, YaRN
- [[attention-variants]] — Per-token computation (MHA, MQA, GQA, MLA, gated attention) and long-context patterns (SWA, chunked, DCA, interleaved local/global); document masking
- [[long-context-training]] — Bridge note for RoPE/YaRN/RNoPE, document masking, long/short data mixtures, KV-cache limits, and effective-context evaluation
- [[vision-transformers]] — Applying transformers to computer vision: ViT, DeiT, and cross-modal unification
- [[mixture-of-experts]] — Routing, load balancing (LBL, auxiliary-loss-free, sequence-wise, SMEBU), sparsity, granularity, shared experts
- [[hybrid-architectures]] — Linear-attention / state-space + transformer hybrids: Mamba-2, gated DeltaNet, kimi delta attention

## Stability and Optimization
- [[training-stability]] — Logit softcapping, z-loss, QK-norm, RMSNorm, depth-scaled sandwich, weight-decay-on-embeddings, init choices, MuonClip
- [[optimizers]] — AdamW, Muon, MuonClip; learning-rate schedules (cosine, WSD, multi-step); critical batch size, batch-size warmup, RSDB
- [[gpu-kernel-engineering]] — Hardware-aware kernels: FlashAttention-4, ThunderKittens, MoE dispatch, memory hierarchy, and inference kernel tradeoffs
- [[tokenizers]] — Vocab sizing, fertility, BPE, when to train vs reuse
- [[how-to-scale-your-model]] — JAX Scaling Book (Austin et al., Google DeepMind, 2025): roofline analysis, TPU architecture, sharded matmuls, training and inference parallelism (data/tensor/pipeline/expert), ZeRO/FSDP, applied to LLaMA 3; 12-part free online reference
- [[diffusionblocks-blockwise-training]] — Sakana AI (ICLR 2026, arXiv 2506.14202): block-wise training reframed as a diffusion model's reverse/denoising process; B× activation-memory reduction; validated on ViT/DiT/masked-diffusion/AR/recurrent-depth; single-forward-pass alternative to BPTT for looped transformers

## Embeddings & Retrieval
- [[embeddings]] — From Word2Vec to Matryoshka: dense vector representations for semantic similarity and search
- [[retrieval-augmented-generation]] — RAG: grounding LLM outputs in retrieved evidence, chunking strategies, and advanced variants
- [[kv-cache]] — The key-value cache: mechanics, memory bottlenecks, PagedAttention, compression (Scissorhands, TurboQuant/PolarQuant near-optimal vector quantization), and scaling context

## Quantization
- [[quantization-fundamentals]] — Core concepts: what quantization is, why it matters, how it works, and hardware-native microscaling formats (MXFP4, NVFP4)
- [[quantization-methods]] — Key algorithms: GPTQ, AWQ, QuIP, LLM.int8(), and their tradeoffs
- [[activation-outliers]] — Why transformers are hard to quantize: super weights, emergent features, and attention head behavior
- [[practical-quantization]] — Running quantized models locally: formats, tooling, and deployment guides

## Data and Pre-training
- [[data-curation-mixtures]] — Multi-stage training, ablation at scale, token utility, rephrasing; SmolLM3 and Hermes 4 / DataForge examples
- [[data-quality-vs-diversity]] — Bridge note on when diversity beats filtering, when quality matters, and how this changes across pretraining, SFT, RL, and evaluation
- [[scaling-laws]] — Chinchilla and compute-optimal training; overtraining as the frontier-2026 norm; Kimi K2 sparsity-driven design

## Fine-Tuning and Alignment
- [[supervised-fine-tuning]] — SFT specifics: sequence packing, masking, chat templates, learning rate, epochs, CCE kernel, IFThink, multi-stage SFT
- [[parameter-efficient-fine-tuning]] — LoRA, QLoRA, DoRA, and GaLore (gradient low-rank projection): adapting or fully training LLMs under tight memory budgets; PorTAL: porting a frozen task latent + hypernetwork core across base models by refitting only a thin per-base converter
- [[alignment-methods]] — From RLHF and DPO to KTO, ORPO, APO, SimPO, GRPO, RLVR, IcePop, rubric rewards, in-flight updates, PTX loss
- [[grpo]] — Group Relative Policy Optimization from first principles: group-relative advantages, the atomic loop objective, the staleness problem, importance-sampling + clipping (ε=0.2), β=0 KL removal, forking tokens, and the DeepSeek-R1-Zero recipe
- [[icepop-stabilizing-rl-moe]] — IcePop: stabilizing RL training in MoE architectures via token discrepancy masking and per-expert-path gradient clipping
- [[rl-training-systems]] — Rollout/trainer systems for GRPO, DAPO, PipelineRL, PrefixRL, KL control, stale policies, and verifier-backed RL; SemiAnalysis three-actor producer/consumer throughput-matching model, environment-state-level staleness, OpenRLHF→slime/verl framework lineage
- [[rl-environments-frameworks]] — Adithya S K & Sergio Paniego (May 2026): practical Rosetta-stone comparison of OpenEnv, ORS, NeMo Gym, Verifiers, SkyRL Gym, and GEM across Jupyter agent / Wordle / Desktop computer-use reference envs; 8-component framework-agnostic design methodology; HTTP-vs-in-process decision; agent skills for code generation
- [[frontier-async-rl]] — Luke Huang (May 2026): survey of async RL across all frontier labs; policy lag K; TIS/MIS/DeepSeek masking; MoE routing replay, TITO, batch-invariant kernels, FP32 LM head, RDMA weight sync; Sequence IS vs Token IS; low-bias compute scaling hypothesis; environment-state-level staleness as a third granularity (with SemiAnalysis); open questions
- [[rl-scaling-laws]] — RL-stage compute scaling laws: ScaleRL's sigmoidal fit and recipe, Scaling Behaviors' power-law fit across Qwen2.5, IsoCompute Playbook's compute-optimal sampling allocation, NVIDIA's ProRL V1/V2 prolonged-training lineage, and Polaris's calibrated-difficulty recipe (whose dataset ScaleRL reuses)
- [[reward-hacking-dynamics]] — Prime Intellect (May 2026): reward hacking as gradient dynamics, not specification; backdoor-ifeval experiments; rarity floor (no safe threshold); Goldilocks difficulty zone; prompt injection backfire; 3-phase liftoff; Prime Sprints
- [[agent-rl-instability-tool-conditioned]] — Microsoft/Aditya Challapally (Jan 2026): tool-conditioned variance amplification in production agent RL; invisible to aggregate metrics; Post-Training Toolkit (OSS, TRL-integrated); slice-aware diagnostics
- [[on-policy-distillation]] — OPD, OPSD, SDFT/GATES/CRISP/RLSD, MOPD, cross-tokenizer OPD; gradient geometry, multi-teacher recipes, the algorithm-selection table; foundational methods (MiniLLM, GKD, DistiLLM, G-OPD/ExOPD, AOPD); why on-policy methods forget less (RL's Razor, forward/reverse-KL, SFT-memorizes-RL-generalizes); failure modes (Rock Tokens, prefix drift, CaOPD)
- [[coding-agent-over-editing]] — nrehiew (June 2026): over-editing in coding LLMs as a measurable failure mode (Levenshtein patch score, Added Cognitive Complexity); BigCodeBench-derived benchmark; frontier-model leaderboard (GPT-5.4 worst, Opus 4.6 best); SFT/rSFT/DPO/RL training-method comparison on Qwen3; LoRA-rank-64 ablation; LoRA-scale reward-hacking incident
- [[practical-fine-tuning]] — Hands-on guidance: hyperparameters, dataset prep, and common pitfalls

## LLM Agents & Reasoning
- [[llm-agents]] — Agent architectures: ReAct, Toolformer, RLMs, and the LLM + Memory + Planning + Tools framework; production memory layers (Mem0 extract/update, Zep/Graphiti temporal knowledge graphs); the Mismanaged Geniuses Hypothesis
- [[chain-of-thought-reasoning]] — Eliciting reasoning: CoT, Tree of Thoughts, Self-Consistency
- [[reasoning-models]] — Test-time compute scaling; DeepSeek-R1 multi-stage pipeline, R1-Zero cold-start RL; MCTS and process-reward limits; RLVR length control
- [[reasoning-data-generation]] — Bridge note for OpenThoughts, SynLogic, QED-Nano, cognitive behaviors, reasoning pretraining, and teacher/verifier data
- [[prompt-optimization]] — Improving compound LLM systems without weight updates: GEPA, MIPROv2, TextGrad, Reflexion
- [[ai-rd-automation]] — Agents running the modelcrafting loop end-to-end: FrogsGame-Posttraining, where research intuition is the bottleneck, not method knowledge
- [[agent-harness-engineering]] — The engineering scaffolding around production agents: context strategy, CursorBench, Keep Rate, error taxonomies, planner/worker/judge orchestration, training vs. production asymmetry, thin-harness/fat-skills; Southbridge's GLM-5.2 vs. Claude Opus 4.8 single-shot build comparison (silent-vs-loud failure modes)
- [[cloud-agent-infrastructure]] — VM isolation, sandboxing primitives (Firecracker, gVisor), snapshot/resume for async gaps, Stripe Minions, sandbox-as-a-service landscape, runtime shift and dev/prod parity
- [[swe-agent-benchmarks]] — SWE-agent training and evaluation environments: R2E-Gym, SWE-smith, Multi-SWE-bench, synthetic issues, and hybrid verifiers
- [[agentic-rl]] — Training agents with RL: the PBRFT-to-POMDP reframing (Agentic RL survey), DeepSWE's GRPO++/Compact Filtering, AgentRL's Cross-Policy Sampling/Task Advantage Normalization, AutoForge's environment synthesis, Agent-R1's masking, and long-context multi-turn SWE-agent RL
- [[dont-build-multi-agents]] — Walden Yan (Cognition): context engineering is the #1 job; why parallel multi-agent architectures fail in 2025; single-threaded linear agents as the default
- [[context-engineering]] — Tobi Lütke's definition and the shift from prompt engineering to dynamic context assembly; the 7-part context taxonomy (instructions, user prompt, history, long-term memory, RAG, tools, structured output); cheap-demo-vs-magical-agent framing
- [[learning-the-bitter-lesson]] — Lance Martin (LangChain): applying Sutton's Bitter Lesson to AI engineering; add structure for current model capabilities, remove it as models improve; open-deep-research case study
- [[letta-code]] — Memory-first coding agent; persistent agents that learn across sessions via memory blocks and skill learning; #1 model-agnostic OSS harness on Terminal-Bench
- [[claude-prompting-best-practices]] — Anthropic's official prompting reference for Claude 4.x: clarity, XML tags, examples, adaptive thinking, effort levels, agentic system design

## Evaluation and Safety
- [[llm-evaluation]] — Closed benchmarks, executable task tests, trace analysis, and open-world evaluations for frontier agents
- [[safety-misalignment]] — Reward hacking as a seed for emergent and context-dependent misalignment in tool-using agents

## Infrastructure and Operations
- [[inference-optimization]] — Making models fast: KV-cache, speculative decoding, batching, parallelism, and serving frameworks
- [[knowledge-distillation]] — Compressing large models into smaller ones: from Hinton's soft targets to reasoning distillation
- [[training-ops]] — Vanishing throughput, noisy loss, tensor-parallelism seeds, multi-client orchestrator, the usual suspects

## Interpretability
- [[neural-geometry]] — Concepts inside neural networks live on curved manifolds, not straight lines; the Goodfire research program on representation geometry and manifold-respecting steering

## Model Releases
- [[qwen3]] — Qwen3 (April 2025): 235B-A22B and 30B-A3B MoE + 6 dense models; hybrid thinking/non-thinking modes; 36T tokens / 119 languages; 4-stage post-training (CoT cold start → reasoning RL → mode fusion → general RL); MCP support; Apache 2.0

## Reinforcement Learning
- [[rl-fundamentals]] — MDP, Markov property, Bellman equations, Q-learning/TD, SARSA, Monte Carlo vs TD, on-policy vs off-policy, exploration vs exploitation, credit assignment, importance sampling, curriculum learning
- [[policy-gradient-actor-critic]] — Policy gradient theorem, REINFORCE, variance reduction (baselines, GAE), actor-critic, PPO with clip objective; connections to GRPO and LLM alignment
- [[model-based-rl-advanced]] — AlphaGo/AlphaZero/MuZero: MCTS + neural networks; World Models and Dreamer V3; Soft Actor-Critic (SAC) with entropy maximisation; model-based vs model-free comparison

## Generative Models
- [[generative-models]] — GANs (adversarial training, WGAN, StyleGAN); VAEs and the ELBO derivation; score function and score matching; diffusion forward process, DDPM/DDIM reverse, SDE formulation; flow matching ODE; classifier-free guidance

## Classical Machine Learning
- [[classical-ml]] — Supervised vs unsupervised; linear regression (OLS, Ridge, Lasso); KNN and curse of dimensionality; SVMs (kernel trick, soft-margin); decision trees (Gini/entropy splits); bagging and Random Forest; AdaBoost and gradient boosting (XGBoost/LightGBM); k-means and other clustering; precision/recall/F1/AUC-ROC

## Deep Learning Fundamentals
- [[deep-learning-fundamentals]] — Backpropagation and computational graphs; activation functions (ReLU, GELU, SwiGLU); loss functions (cross-entropy, MSE, contrastive); weight initialisation (Xavier, He); BatchNorm/LayerNorm/RMSNorm; CNNs (conv, ResNet); RNNs/LSTMs (gating, vanishing gradients); S4 and state space models; autoencoders (denoising, sparse, VQ-VAE); Gumbel-Softmax trick

## ML Theory and Statistics
- [[ml-theory-statistics]] — PDF/PMF; expectation; variance and covariance; entropy and cross-entropy (with the CE=KL+H(p) proof); KL divergence (forward vs reverse); Jensen-Shannon divergence; Bayes' theorem; MLE vs MAP; bias-variance tradeoff; No Free Lunch theorem; curse of dimensionality; confidence intervals; hypothesis testing toolkit (p-values, KS/chi-squared/t-tests, ANOVA, McNemar's, Pearson/Spearman, mutual information); convex functions and Jensen's inequality

## Optimisation and Regularisation
- [[optimization-regularization]] — Gradient descent (SGD, mini-batch, momentum); Newton's method and second-order methods (L-BFGS, K-FAC); overfitting/underfitting; regularisation (L1/L2, dropout, data augmentation); cross-validation; early stopping; transfer learning; domain adaptation; few-shot/zero-shot; dimensionality reduction (PCA, t-SNE, UMAP); data whitening

## Applied ML Systems
- [[applied-ml-systems]] — Floating point representation (FP32/BF16/FP16/FP8); mixed precision training (AMP, loss scaling); gradient checkpointing; gradient accumulation; gradient clipping; exploding/vanishing gradients; DDP (data parallel); communication primitives (all-reduce, all-gather, reduce-scatter); profiling; JIT compilation (torch.compile, JAX XLA); JAX vs PyTorch vs TensorFlow

## LLM Architecture Extensions
- [[llm-architectures-extended]] — Pretraining (next-token prediction, scale, compute budget); decoding techniques (greedy, beam, temperature, top-k, top-p, nucleus, speculative); Transformer-XL (segment recurrence, relative PE); Griffin (diagonal linear recurrence, parallel scan); Perceiver / PerceiverIO; LLM vs RNN vs S4 comparison

## Frontier Debate & Source Material
- [[agi-timelines]] — Synthesis of timeline views (Aschenbrenner, Karpathy, Sutskever, Ege/Tamay, Sutton, Hassabis, Chollet, Schulman, Amodei) and the five recurring cruxes
- [[dwarkesh-podcast]] — Indexed AI/ML coverage of the Dwarkesh Podcast corpus, organized by theme; pointer to ~53 primary-source files in `raw/dwarkesh-*`
