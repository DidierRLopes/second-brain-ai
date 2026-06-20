# Reasoning Models

Reasoning models represent a paradigm shift from standard LLMs: instead of generating answers in a single forward pass, they spend variable compute "thinking" — generating internal chains of reasoning before producing a final answer. More thinking time generally produces better answers, a property called **test-time compute scaling**.

## DeepSeek-R1

DeepSeek-R1 (January 2025) demonstrated perhaps the most surprising finding in reasoning: sophisticated reasoning behaviors — verification, reflection, exploring alternatives, backtracking — can **emerge purely from reinforcement learning** without any human-annotated reasoning demonstrations.

Using Group Relative Policy Optimization (GRPO) on DeepSeek-V3 Base, the model naturally learned to verify its own work, consider alternative approaches, and catch its own mistakes. No chain-of-thought examples were provided during training — the model discovered these strategies on its own through RL on outcome rewards.

DeepSeek-R1 is competitive with OpenAI's o1 on math, code, and science benchmarks, and is fully open-source. Distilled versions (1.5B to 70B) retain strong reasoning, showing that reasoning capabilities transfer well through distillation.

### DeepSeek-R1-Zero (cold-start RL)

The variant trained from base with no SFT first. Configuration: 10.4k GRPO steps, batch size 512, reference policy replacement every 400 steps, learning rate 3e-6, KL coefficient 0.001. **Two reward types**: accuracy (correctness) and format (enforced thinking tags). Reflection/exploration emerged spontaneously — the count of words like "wait" or "mistake" rose 5–7× compared to the start of training. Onset was sudden: between steps 4k–7k there was only occasional usage; after step 8k significant spikes appeared.

### Full DeepSeek-R1 multi-stage pipeline

R1-Zero readability issues (mixed languages, lacking markdown formatting) motivated the full R1 pipeline:

1. **Cold-start SFT** on thousands of long-CoT data from DeepSeek-V3-Base.
2. **Reasoning RL** with two fixes: a **language consistency reward** (proportion of target-language words in CoT) and a **readable pattern** including a summary at the end of each response.
3. **Second RL stage** for helpfulness and harmlessness while retaining reasoning.

For helpfulness: query DeepSeek-V3 four times, randomly assign as Response A/B, average judgments, retain pairs with large score differences, train with pairwise loss focused on the relevance of the final summary. For harmlessness: dataset with model-generated responses annotated "safe" or "unsafe" per safety guidelines, point-wise reward training.

### Distillation beats RL for smaller models

Distilling DeepSeek-R1 outputs into smaller models (Qwen-32B and below) **significantly improved reasoning** — even compared to running large-scale RL on those smaller models directly, which would require much more compute. The lesson: distillation strategies are effective and economical for smaller models, but pushing the frontier still requires more powerful base models and larger-scale RL.

### What didn't work (DeepSeek's failed experiments)

- **Monte Carlo Tree Search (MCTS)**: inspired by AlphaGo/AlphaZero, tested for enhancing test-time compute scalability. Token generation has an *exponentially larger* search space than chess. A max extension limit per node caused the model to get stuck in local optima. Fine-grained value model training was difficult too.
- **Process Reward Models** (rewarding intermediate thoughts): three failures. (a) Defining a fine-grained step in general reasoning is difficult. (b) Determining whether the current intermediate step is correct is hard — LLM-as-judge doesn't yield satisfactory results. (c) Leads to reward hacking — the model optimizes for the appearance of good reasoning without doing the underlying work.

### Other DeepSeek findings

- **First-person voice**: after fine-tuning on small amount of long-CoT data, DeepSeek-R1 uses "I" more whereas R1-Zero uses "we" more — users find first-person reasoning more intuitive.
- **Temperature**: greedy decoding for long-output reasoners gives higher repetition rates and more variability. Explained by risk aversion (hardness of learning) and inductive bias for temporally-correlated errors — at decision points, the model tends to reselect previously favored actions, causing looping.

## RLVR Length Control for Hybrid Reasoners

The goal of [[alignment-methods|RLVR]] on hybrid reasoning models is to improve reasoning without extending token count radically. For `/no_think`, naively applying GRPO leads to **reward hacking**: the model emits longer CoT (drifting toward `/think`), so both reward and token length increase. SmolLM3 observed `/no_think` traces sprouting cognitive markers like "Wait, …" associated with reasoning modes.

**Mitigation**: an overlong-completion penalty parameterized by a soft punishment threshold and a hard punishment threshold (max completion length). Penalty increases from soft to hard threshold; past the hard threshold, punishment is -1 (effective reward = 0).

SmolLM3 chose a length penalty in the 2.5k–3k range for `/no_think`. But **doing RL jointly on hybrid reasoning models is difficult** because the interplay between separate length penalties for the two modes can cause instability — which is why labs typically release instruct and reasoning variants separately.

## OpenAI o1 / o3

OpenAI's reasoning models use a proprietary approach where the model generates internal "thinking tokens" that are hidden from the user. o3 achieves 3× the accuracy of o1 on the ARC-AGI benchmark. The key idea: instead of scaling model parameters (training-time compute), scale the amount of inference-time reasoning (test-time compute).

## OpenThoughts: Open-Source Reasoning Data

The OpenThoughts project (2025) addresses the lack of public training data for reasoning models. Through 1,000+ controlled experiments on data generation pipelines, they produced OpenThoughts3 — a dataset that trains a 7B model to achieve 53% on AIME 2025 and 51% on LiveCodeBench, demonstrating that public data can match proprietary distilled models.

The full paper — [Data Recipes for Reasoning Models (2506.04178)](../../papers/05-learning/reasoning/Front-Loading Reasoning: The Synergy between Forward and Backward Passes - 2506.04178.pdf) — details the pipeline and its systematic ablations. OpenThinker3-7B (trained on OpenThoughts3-1.2M with QwQ-32B as teacher) achieves: **69% AIME24**, 93.5% AMC23, 53.7% GPQA Diamond, 51.7% LiveCodeBench 06/24-01/25 — outperforming DeepSeek-R1-Distill-7B on average across 12 tasks by 12.4 points.

### OpenThoughts data pipeline — five empirical takeaways

The pipeline has six stages: source questions, mix questions, filter questions, generate answers, filter answers, select teacher model. Over 1,000 ablations (each at 31,600 examples fine-tuning Qwen2.5-7B-Instruct) produced five durable findings:

1. **Sampling 16× answers per question** is the single highest-leverage step — cheap scale that consistently improves performance.
2. **QwQ-32B outperforms DeepSeek-R1 as a teacher**, despite scoring lower on the same benchmarks. A stronger model on the training benchmark does not make it a better teacher for distillation purposes.
3. **Answer filtering provides no reliable gain** — majority-consensus filtering, response-length filtering, GPT verification — none beat the no-filtering baseline. The field's common practice of filtering answers is largely wasted compute.
4. **Fewer high-quality question sources beats diversity** — mixing 2 top-ranked sources outperformed mixing 16. The top sources for math were OpenMath-2-Math and NuminaMath-1.5; for code, StackExchange-CodeGolf and OpenCodeReasoning.
5. **LLM-based question filtering (difficulty or response-length) beats fastText and embedding filters** by 4–6 percentage points.

The final OpenThoughts3-1.2M recipe: 850K math / 250K code / 100K science, exact dedup for math and science, no dedup for code, 16× sampling per question, QwQ-32B teacher.

## Cognitive Behaviors that Enable Self-Improvement

[Cognitive Behaviors that Enable Self-Improving Reasoners (2503.01307)](../../papers/05-learning/reasoning/Cognitive Behaviors that Enable Self-Improving Reasoners - 2503.01307.pdf) (Stanford / SynthLabs, COLM 2025) asks a precise question: why does Qwen-2.5-3B improve dramatically under RL while Llama-3.2-3B plateaus on identical training? The answer is not model size or architecture — it is the presence or absence of four specific **cognitive behaviors** in the base model's outputs.

### The four behaviors

| Behavior | Example phrase | What it does |
|---|---|---|
| Verification | "Let me check my answer..." | Systematic error-checking of intermediate results |
| Backtracking | "This approach won't work because..." | Abandoning failing paths and revising |
| Subgoal setting | "To solve this, we first need to..." | Decomposing complex problems into manageable steps |
| Backward chaining | "To reach target 75, we need a number divisible by..." | Goal-directed reasoning from desired outcomes |

These four behaviors represent non-linear, search-like reasoning strategies that go beyond the "linear monotonic" patterns typical of most LLM outputs. They were identified and measured using a GPT-4o-mini classifier on reasoning traces, with high agreement to human annotations.

### Key findings

**Qwen naturally exhibits all four; Llama exhibits almost none.** On the Countdown task, Qwen reaches ~60% accuracy under RL while Llama reaches ~30%. The behavioral frequency data explains the gap: Qwen's verification count starts at ~0.62 and backtracking at ~0.31; Llama's are near zero throughout.

**Behaviors matter more than correctness.** Models primed with incorrect solutions that exhibit the right reasoning patterns achieve the same RL improvement as models primed with correct solutions. This is the paper's most important finding: RL can only amplify behaviors that already appear in successful trajectories — so if the behaviors are absent from the base model, RL has nothing to amplify. Reasoning behavior patterns are the enabling factor; correct answers are not.

**Empty chain-of-thought does nothing.** Priming with an empty `<think></think>` (with or without length-matched filler tokens) yields no improvement — confirming that longer computation time alone is insufficient. The cognitive behaviors are specifically necessary, not just extended token budgets.

**RL selectively amplifies and suppresses.** When primed with all four strategies, RL strengthens backtracking and verification while suppressing backward chaining and subgoal setting. When only backtracking is present, it survives — the behaviors that survive RL are those that happen to be most useful on the specific task being optimized.

**Behavioral augmentation of pretraining data can close the gap.** The team curated a subset of OpenWebMath where cognitive behaviors appear at elevated frequency (verified using Qwen-2.5-32B as a classifier on 200K documents). Continued pretraining Llama on this behavior-enriched dataset, then applying RL, allowed Llama to match Qwen's self-improvement trajectory. This establishes a practical recipe: if your base model lacks these behaviors, targeted pretraining data selection (not SFT on task-specific priming data) can instill them.

The broader implication: the choice of base model for reasoning RL is a function of its pre-existing cognitive behaviors, not just its benchmark scores. This framework gives a diagnostic tool — check the frequency of backtracking, verification, subgoal setting, and backward chaining in the base model's outputs before committing to an RL training run.

## SynLogic: Synthetic Logical Reasoning Data at Scale

[SynLogic (2505.19641)](../../papers/05-learning/reasoning/SynLogic: A Data Synthesis Framework for Logical Reasoning - 2505.19641.pdf) (MiniMax, June 2025) is a data synthesis framework and dataset covering **35 diverse logical reasoning tasks** including Sudoku, ciphers, Game of 24, Arrow Maze, math path, and ARC-AGI-style tasks. The key insight: logical reasoning is a fundamental building block of general reasoning, and all logical tasks have rule-based verifiers making them ideal for RLVR training.

### Framework design

Each task in SynLogic has six components: task selection, parameter identification (e.g., grid size controls difficulty), logic instance generation via rule-based code, difficulty control (difficulty bounds set so R1/o3-mini have pass@10 > 0 at upper end; chat models have 0–0.5 pass rate at lower end), prompt formalization, and a task-specific verification suite. This dual-bound difficulty control prevents including instances too easy or too hard to train on.

Two dataset versions: **SynLogic-Hard** (33K samples for 32B models) and **SynLogic-Easy** (16K samples for 7B models, with 8 tasks removed as beyond 7B capability).

RL training uses DAPO (a variant of GRPO) with binary rewards: reward = 1 iff format correct AND answer verified correct by task verifier.

### Results and cross-domain transfer

SynLogic-32B surpasses DeepSeek-R1-Distill-Qwen-32B by **6 points on BBEH** (Kazemi et al., 2025) despite being trained purely on logical tasks. SynLogic-7B achieves 48.1% on KOR-Bench, outperforming Qwen2.5-7B-Instruct by nearly 10 points.

The more surprising finding is **cross-domain transfer to mathematics**. SynLogic-7B achieves 10.0% on AIME 2024 (vs. 0.3% for Qwen2.5-7B-Base), 71.8% on MATH 500, and 55.0% on AMC 2023 — all without any math training data. SynLogic-32B achieves 19.6% on AIME 2024 (4.4× over base). This strongly supports the hypothesis that logical reasoning and mathematical reasoning share abstract reasoning mechanisms.

**Mixed training (SynLogic + Math or Coding) further improves both domains.** Mixing SynLogic-Easy with 17K math samples trains math at the same accuracy as math-only training while consuming fewer math samples — and simultaneously achieves +10 points on KOR-Bench vs math-only. The Zero-Mix-3 configuration (SynLogic + Math + Coding at 32B scale) outperforms DeepSeek-R1-Zero-Qwen-32B across multiple benchmarks including BBEH (28.6 vs 18.5) and GPQA Diamond (57.5 vs 55.2).

## Front-Loading Reasoning: When to Inject Reasoning Data

[Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data (2510.03264)](../../papers/05-learning/reasoning/WHY DO REASONING MODELS LOOP - 2510.03264.pdf) (NVIDIA / CMU / Boston University / Stanford, Sep 2025) provides the first systematic study of how reasoning data placement across the training pipeline (pretraining vs. SFT) affects final performance. The central finding is an **asymmetric allocation principle**: diversity drives pretraining effectiveness, while quality governs SFT effectiveness.

### Experimental setup

An 8B hybrid model (Mamba 2 + self-attention + FFN) pretrained from scratch on 1T tokens. Four pretraining variants: M_base (no reasoning data), M_LDQ (large-scale diverse data: 336B tokens, 56% math, 17% code, 27% science/general), M_SHQ (small high-quality: 1.2M carefully curated long-CoT examples), M_LMQ (combined). Each is then SFT'd on 4.8M reasoning samples and RL'd with GRPO.

### Key findings with numbers

**Front-loading creates durable compounding advantage (+19% average gain after full pipeline).** Models pretrained with reasoning data vs. no reasoning data diverge through SFT and RL — the advantage widens at each stage rather than closing.

**The "catch-up" hypothesis is false.** Doubling SFT data for M_base (+7.39% average) still fails to match the weakest reasoning-pretrained model M_SHQ (+SFT, +3.32% more). SFT cannot compensate for a weak pretraining foundation.

**Diversity beats quality in pretraining.** M_LDQ (+28.4% math, +9% code over M_base post-pretraining) dramatically outperforms M_SHQ despite the latter having higher-quality data. Scale and diversity are the signal at pretraining time.

**Quality beats diversity in SFT.** Training all models on D_SHQ (small high-quality) yields the best SFT results. Training on D_LDQ (large diverse) during SFT actively degrades math by 4.92%. Naively doubling mixed-quality SFT data is detrimental.

**High-quality pretraining data has latent effects (+4% additional gain unlocked by SFT).** M_LMQ (mixed quality pretraining) appears similar to M_LDQ during pretraining, but reveals an additional +4.25% gain over M_LDQ after SFT. The high-quality component acts as a "latent amplifier" whose benefits only emerge during alignment.

The practical blueprint: fill 20% of pretraining tokens with large-scale diverse reasoning data; SFT exclusively on small high-quality long-CoT data; apply RL. Do not scale SFT with mixed-quality data.

## QED-Nano: Small Models for Olympiad-Level Theorem Proving

[QED-Nano: Teaching a Tiny Model to Prove Hard Theorems (2604.04898)](../../papers/05-learning/reasoning/QED-Nano: Teaching a Tiny Model to Prove Hard Theorems - 2604.04898.pdf) (CMU / Hugging Face / ETH Zurich / Numina, April 2026) demonstrates that a 4B parameter model can achieve competitive theorem-proving performance through a carefully designed three-stage post-training recipe. This is an important existence proof for [[knowledge-distillation|small-model specialization]] — Olympiad-level reasoning is not reserved for 100B+ models.

QED-Nano operates entirely in natural language (no Lean or formal tools) and targets IMO-style proof problems. The three training stages:

1. **SFT initialization** on ~7,500 proofs generated by DeepSeek-Math-V2 (685B, fine-tuned specifically for Olympiad math with meta-verifiers) from a curated 5,000-problem corpus (AoPS + AI-MO/olympiads, filtered by GPT-5-Nano, deduplicated against benchmarks). The SFT stage introduces a known pathology: **length explosion** — the model imitates the superficial form of long proofs rather than learning structured reasoning, producing outputs in the hundreds of thousands of tokens on incorrect attempts.

2. **RL with rubric-based rewards** using GRPO on PipelineRL (off-policy, max 5 gradient steps lag from reference). Reward is a 0–7 grading scheme generated by Gemini 3 Pro per problem (not a simple binary). RL corrects the SFT length explosion pathology and improves training stability. Training uses n=16 rollouts per problem (batch of 64 problems = 1024 samples total), max response length 50K tokens.

3. **Reasoning Cache (RC) scaffold** — decomposition of long proofs into iterative summarize-and-refine cycles: the model generates a partial trace, summarizes its progress into a compact textual "state representation", and conditions the next rollout on both the original problem and this summary. RC is trained into the model (not just an inference scaffold), making QED-Nano naturally benefit from multi-turn test-time compute.

### Results

Without any test-time scaffold: **40% on IMO-ProofBench, 44.9% on ProofBench, 67.5% on IMO-AnswerBench** — outperforming Nomos-1 (30B) and Qwen3-235B-A22B-Thinking. With RSA scaffold (summarizes multiple parallel rollouts): 56.9% IMO-ProofBench, 62.6% ProofBench, 76.5% IMO-AnswerBench — approaching Gemini 3 Pro at **at least 3× lower inference cost**.

The RC scaffold comparison (Table 2): single turn = 93,690 tokens; RC = 237,379 (2.53×); DSM = 1,605,879 (17.14×); RSA = 2,045,764 (20.84×). RSA achieves the best final performance but at high cost; DSM offers the best performance-per-token tradeoff.

Key lesson: training explicitly for test-time adaptation (via RC) generalizes beyond the specific scaffold used in training — QED-Nano benefits from RSA and DSM scaffolds it was not trained on. See also [[knowledge-distillation]] for the SFT-from-large-model initialization pattern.

## Generative Recursive Reasoning (GRAM)

[Generative Recursive Reasoning (2605.19376)](../../papers/05-learning/reasoning/DATA RECIPES FOR REASONING MODELS - 2605.19376.pdf) (KAIST / Mila / NYU / Université de Montréal, May 2026 — **note: preprint, treat claims with care**) proposes a fundamentally different architecture for reasoning: instead of extending autoregressive sequences, it makes recursive latent-state models **probabilistic**, enabling multi-trajectory exploration.

Prior Recursive Reasoning Models (RRMs) like HRM and TRM perform iterative latent-state refinement with *deterministic* transitions — given the same input, they converge to the same trajectory. GRAM introduces **stochastic latent transitions**: at each recursion step, the model samples a transition from a learned conditional Gaussian rather than deterministically updating. This induces a distribution over reasoning trajectories rather than a single path.

### Architecture

GRAM uses a hierarchical latent state z = (h, l):
- High-level component h: updated once per latent transition, carries abstract reasoning state
- Low-level component l: refined K times within each transition, carries fine-grained intermediate state

The stochastic guidance is: ε_t ~ p_θ(ε_t | u_t) := N(μ_θ(u_t), σ²_θ(u_t)·I), then h_t = u_t + ε_t, where u_t is the deterministic high-level update. The variance σ²_θ(u_t) controls the amount of exploration, learned rather than fixed.

Training uses amortized variational inference (ELBO), with a variational posterior q_φ(ε_t | u_t, y) that has access to ground truth during training but not at inference. Gradients are truncated to each supervision step for memory efficiency.

### Inference-time scaling

GRAM supports two complementary scaling axes: **depth** (more recursive transitions) and **width** (sample N parallel trajectories). Width scaling is GRAM's unique contribution — deterministic RRMs cannot explore multiple solution paths and collapse to a single attractor. A Latent Process Reward Model (LPRM) selects among parallel trajectories by predicting correctness from latent state alone.

### Results (on structured reasoning tasks)

- **Sudoku-Extreme**: GRAM 97.0% vs TRM 87.4%, Looped TF 61.3%, HRM 55.0%
- **ARC-AGI-1**: GRAM (high compute) 66.7% vs o3-mini (low) 55.7%, HRM 44.6%, TRM 52.0%
- **ARC-AGI-2**: GRAM 16.0% vs Grok-4 thinking 9.7%, HRM 7.8%, TRM 11.1%
- **N-Queens (8×8)**: 99.7% accuracy, 90.3% coverage vs deterministic baselines that degrade sharply as solution count grows

Ablation confirms stochastic guidance is the core gain — it consistently improves every architecture (Looped TF, HRM, TRM) it is added to. Naive stochasticity (random initialization, stochastic decoder) does not help; the gain requires the variational framework.

**Limitation**: sequential deep supervision in GRAM limits training throughput compared to Transformers, posing a barrier to scaling toward frontier-scale models. The results so far are on small controlled benchmarks, not general-purpose LLM tasks.

## Why Do Reasoning Models Loop?

[Why Do Reasoning Models Loop? (2512.12895)](../../papers/05-learning/reasoning/Generative Recursive Reasoning - 2512.12895.pdf) — see the **Failure Modes** section below.

## Failure Modes of Reasoning Models

### Looping

The paper "Why Do Reasoning Models Loop" investigates the pathological looping behavior observed in extended reasoning sequences — where models repeat the same or similar reasoning steps without progress. The key mechanistic finding connects to the temperature discussion in the DeepSeek-R1 section above. At decision branch points in autoregressive generation, models with **temporally correlated errors** tend to reselect previously favored tokens — particularly when risk aversion (hardness of the task) makes exploration costly. This creates self-reinforcing loops.

Two contributing factors identified:

1. **Risk aversion at hard decision points**: the model has learned that exploring a new branch risks reward = 0 on a difficult problem, so it preferentially continues an established path even when that path is not converging. This is a consequence of RL training with sparse rewards.

2. **Inductive bias for temporal correlation**: at each position, the model conditions on its own prior outputs. If a prior output established a particular trajectory frame (e.g., "Let me try approach X"), subsequent tokens are biased toward continuing that frame — even when approach X is failing.

**Mitigations**: the paper finds that greedy decoding (temperature = 0) dramatically worsens looping because it eliminates the small stochastic perturbations that would occasionally escape a loop. Some entropy in decoding is beneficial. The DeepSeek-R1 team made the same observation empirically. Process-level interventions that detect repeated reasoning content and inject explicit "let me try a different approach" prompts can break loops but introduce their own overhead.

Reasoning models suggest a new scaling axis. Traditional scaling increases parameters and training data (see [[scaling-laws]]). Reasoning models show you can also scale **inference compute** — letting the model think longer on harder problems. This is more efficient because you allocate compute where it's needed rather than uniformly.

The progression: [[chain-of-thought-reasoning]] (prompting trick) → learned reasoning via RL (fundamental capability). Reasoning is not just a formatting choice but a trainable behavior.

## Is RL on LLMs actually the path? — The Sutton / Karpathy critique

Reasoning models are the most prominent application of RL on top of LLMs. But two of the most respected voices in the field argue the entire approach is structurally limited:

**Richard Sutton** (Turing Award 2024; author of The Bitter Lesson) argues LLMs are a dead end because real intelligence requires continual experience-based learning, which gradient-free reward shaping on a frozen base model cannot deliver. See `raw/dwarkesh-richard-sutton-rl-father-llms-dead-end.md`.

**Andrej Karpathy** calls RL "terrible" — but agrees everything else is worse. He frames the current paradigm as a workable hack rather than a fundamental method, and predicts AGI is "still a decade away" precisely because of these limits. See `raw/dwarkesh-andrej-karpathy-agi-decade-away.md`.

**Dwarkesh's "RL is even more information inefficient than you thought"** essay makes the quantitative case: RL gives one scalar per trajectory vs. supervised learning's full distribution per token. See `raw/dwarkesh-bits-per-sample-rl-info-inefficient.md`.

The opposing view, articulated by **Sholto Douglas & Trenton Bricken** (Anthropic) in 2025, is that scaling RL on verifiable tasks plus mechanistic interpretability findings are quietly addressing these critiques. See `raw/dwarkesh-sholto-trenton-rl-llms-agi-2025.md`.

**John Schulman** (OpenAI co-founder, "PPO author") and **Demis Hassabis** (AlphaZero atop LLMs) both ground their AGI plans in the reasoning-via-RL paradigm — and explain in detail how they think the limits get overcome. See `raw/dwarkesh-john-schulman-openai-reasoning-rlhf-agi.md` and `raw/dwarkesh-demis-hassabis-scaling-alphazero-llm.md`.

This debate is largely the same debate as [[agi-timelines]] — they are not separable.

## Related Topics
- [[chain-of-thought-reasoning]] — The prompting techniques that preceded learned reasoning
- [[reasoning-data-generation]] — teacher traces, synthetic verifier-backed tasks, cognitive behaviors, and reasoning-data placement
- [[alignment-methods]] — GRPO, RLVR, IcePop, rubric rewards
- [[rl-training-systems]] — the rollout/trainer infrastructure behind verifier RL
- [[supervised-fine-tuning]] — Hermes 4's fixed-position `</think>` for reasoning budgets
- [[on-policy-distillation]] — distillation as an efficient alternative to RL for reasoning
- [[llm-agents]] — Agents that leverage reasoning models for planning
- [[scaling-laws]] — Reasoning offers a new scaling dimension beyond parameters
- [[agi-timelines]] — Whether reasoning-via-RL gets us there is the central timeline question
- [[dwarkesh-podcast]] — Primary source material for the RL-on-LLMs debate
- [[frontier-training-playbook]] — where reasoning training sits in the broader recipe

## Sources
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.00656)](../../papers/01-models/gpt-deepseek-v2-v3/DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning - 2501.00656.pdf); see `raw/deepseek-r1-reasoning-via-rl.md`
- [Data Recipes for Reasoning Models (2506.04178)](../../papers/05-learning/reasoning/Front-Loading Reasoning: The Synergy between Forward and Backward Passes - 2506.04178.pdf) — OpenThoughts3 data recipe and ablations.
- [Cognitive Behaviors that Enable Self-Improving Reasoners (2503.01307)](../../papers/05-learning/reasoning/Cognitive Behaviors that Enable Self-Improving Reasoners - 2503.01307.pdf) — verification, backtracking, subgoal setting, backward chaining.
- [SynLogic: A Data Synthesis Framework for Logical Reasoning (2505.19641)](../../papers/05-learning/reasoning/SynLogic: A Data Synthesis Framework for Logical Reasoning - 2505.19641.pdf) — synthetic logical reasoning data for RLVR.
- [Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data (2510.03264)](../../papers/05-learning/reasoning/WHY DO REASONING MODELS LOOP - 2510.03264.pdf) — reasoning data placement across pretraining, SFT, and RL.
- [QED-Nano: Teaching a Tiny Model to Prove Hard Theorems (2604.04898)](../../papers/05-learning/reasoning/QED-Nano: Teaching a Tiny Model to Prove Hard Theorems - 2604.04898.pdf) — small-model theorem proving with SFT distillation, rubric RL, and reasoning cache.
- [Generative Recursive Reasoning (2605.19376)](../../papers/05-learning/reasoning/DATA RECIPES FOR REASONING MODELS - 2605.19376.pdf) — probabilistic recursive latent-state reasoning.
- [Why Do Reasoning Models Loop? (2512.12895)](../../papers/05-learning/reasoning/Generative Recursive Reasoning - 2512.12895.pdf) — looping mechanisms and temperature effects in reasoning models.
- Demystifying Reasoning Models — Cameron R. Wolfe
- Dwarkesh Podcast — Sutton, Karpathy, Schulman, Hassabis, Sholto/Trenton interviews (see `raw/dwarkesh-*`)
- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`. Source for DeepSeek-R1 multi-stage pipeline, MCTS limitations, RLVR length control.
