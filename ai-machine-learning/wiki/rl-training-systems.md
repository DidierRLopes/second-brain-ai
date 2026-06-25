# RL Training Systems

RL training systems are the machinery that makes post-training with rollouts, verifiers, stale policies, long traces, and distributed inference actually work. This is a bridge between [[alignment-methods]], [[reasoning-models]], [[training-ops]], [[on-policy-distillation]], [[safety-misalignment]], and [[llm-evaluation]]: the algorithm is only half the story; the other half is keeping generation, training, reward computation, and policy freshness aligned.

For the practical layer below this — how to actually build and deploy an RL environment across the six major frameworks (OpenEnv, ORS, NeMo Gym, Verifiers, SkyRL Gym, GEM) — see [[rl-environments-frameworks]].

## The Core Loop

Modern LLM RL usually looks like this:

1. Sample multiple rollouts from the current policy.
2. Score them with verifiers, reward models, rubric judges, or human preference signals.
3. Convert scores into advantages.
4. Update the policy with a trust-region objective.
5. Refresh rollout workers so the next batch is not too stale.

DeepSeek-R1 popularized GRPO for reasoning, but the surrounding system matters as much as the loss. Long-CoT rollouts are expensive, sparse rewards are noisy, and the model can reward hack by emitting longer traces, exploiting verifier bugs, or drifting into a different mode.

## GRPO, DAPO, And KL Design

[[alignment-methods]] covers the algorithmic details, but the systems-level split is useful:

- GRPO removes the value model and uses group-relative advantages.
- DAPO makes GRPO more stable with Clip-Higher, dynamic sampling, token-level policy-gradient loss, and overlong reward shaping.
- KL-regularized policy gradient methods define how hard the policy is allowed to move away from the reference.

[DAPO (2503.14476)](../../papers/05-learning/reinforcement-learning/DAPO: An Open-Source LLM Reinforcement Learning Framework - 2503.14476.pdf) is the practical open-source reference for stabilizing long-CoT RL. [On the Design of KL-Regularized Policy Gradient Algorithms (2505.17508)](../../papers/05-learning/alignment-preferences/ON THE DESIGN OF KL-REGULARIZED POLICY GRADIENT - 2505.17508.pdf) is the reference for not confusing an implementation convenience with the actual optimized KL objective. If rollouts are off-policy, the KL term needs to match that sampling story.

## Throughput Versus Freshness

RL systems fight a constant tradeoff: large generation batches keep vLLM/SGLang-style inference engines efficient, but many optimizer steps per generated batch make the data stale. [PipelineRL (2509.19128)](../../papers/03-scaling/training-optimization/PipelineRL: Faster On-policy Reinforcement Learning by Leveraging Pipeline Parallelism - 2509.19128.pdf) addresses this by running generation and training concurrently with in-flight weight updates. The model can receive new weights during ongoing generation, keeping throughput high while sample efficiency stays close to on-policy baselines.

[Reuse Your FLOPs / PrefixRL (2601.18795)](../../papers/05-learning/reinforcement-learning/Reuse your FLOPs: Scaling RL on Hard Problems by Recycling Computation - 2601.18795.pdf) tackles the opposite inefficiency: hard tasks produce few correct trajectories, so most rollout FLOPs return no useful signal. Prefix-conditioned RL reuses prefixes of previously successful traces so the model can train closer to the part of the search tree where reward is reachable.

## Throughput Matching as a Producer/Consumer Problem

A 2026 SemiAnalysis case-study piece (Chen, Wen & Patel — see `raw/semianalysis-rl-systems-mind-the-gap.md`) reframes the throughput-versus-freshness tension above as a literal producer/consumer queue: the **generator** (inference) produces rollouts into a queue; the **trainer** (backprop) consumes from it; a third actor, the **RL environment/sandbox**, scores rollouts in between (Firecracker micro-VMs up to full QEMU VMs; sandbox-as-a-service vendors like Modal optimize cold-start latency via content-addressed caching).

- **Generation-bound** systems (generator slower than trainer): the queue empties, the trainer idles between steps. This was the dominant regime across all four production case studies the piece documents.
- **Trainer-bound** systems (generator faster): the queue grows, samples age, and staleness becomes the binding constraint instead — the regime [[frontier-async-rl|frontier async RL]]'s policy-lag analysis targets.

**Trainer consumption rate** depends on group size, advantage-filtering (dropping zero-advantage samples when a group's solve rate collapses to 0% or 100%, per the Verifier section above), and step time (model size/precision, TP/PP/DP/EP, FSDP, offloading). **Generator production rate** depends on concurrent rollouts ÷ end-to-end latency, further reduced by an **acceptance rate**: not every generated sample survives to be consumed. Two levers reduce wasted generation: **early pruning** (abandon a rollout mid-generation based on length caps, value functions, or intermediate verifier checks) and **adaptive sampling** (e.g. online advantage filtering, as in Prime Intellect's Intellect-2). Concurrency itself is capped by aggregate KV-cache memory ÷ average sequence length, and the longest rollout in a group (the **straggler**) sets that group's completion time — motivating both oversampling and partial rollout (below).

**Four production case studies** (all agentic-coding, MoE models, 10+ nodes) ground the framework in real numbers:

1. **Qwen3-235B-A22B-Thinking-2507** (Prime RL + Mini SWE Agent Plus): long-response variance drives severe tail latency, forcing 60% oversampling-discard; generation-bound, trainer at 10.5% MFU using 3× less compute than the generator.
2. **GLM-5** (Prime RL, PD-disaggregated generator): response length and tool calls triple over the run (20→51 calls), shifting the workload prefill-heavy; curriculum too easy (55% of problems hit 100% solve rate → flat reward); trainer idles 74% of wall-clock.
3. **Qwen3-235B-A22B-Instruct** (verl + uni-agent, Modal sandboxes, GRPO): generation-bound; tail latency hit 7500s at 960 concurrent rollouts from sandbox init-dead errors and hour-long spin-up stragglers, forcing a scale-back to 96 concurrent rollouts — direct evidence that sandbox-infra reliability, not just algorithm design, gates achievable concurrency.
4. **Qwen3-235B-A22B-Thinking-2507-FP8** (slime + GSPO, SWE-Bench, Modal, partial rollout): 60% trainer wait, 12K-token responses, 12 tool calls/sequence — the case study that motivates partial rollout, below.

## Partial Rollout and Environment-State-Level Staleness

[[frontier-async-rl]] covers policy staleness at the **trajectory level** (a rollout started under an old policy, consumed later) and **token level** (in-flight weight updates mid-rollout). The SemiAnalysis case studies surface a third granularity specific to agentic/sandboxed RL: **environment-state-level staleness**.

**Partial rollout** aborts straggler rollouts instead of discarding them outright, saves them to a replay buffer, and resumes them in a later batch — trading stateful-sandbox complexity for reduced tail-latency waste. This requires the sandbox to persist across batches (e.g. half-applied file edits in a SWE-Bench repo), be lazily created (don't spin one up for a sample that never calls a tool), and be robust to failure during the abort→resume window. slime triggers abortion when a target batch fills or the trainer pushes a weight update; aborted rollouts evict KV cache (unlike PipelineRL's in-flight updates), so resumption becomes a large prefill request — exactly the workload PD disaggregation is built to absorb.

The staleness this introduces is structural, not just statistical: when a resumed rollout's sandbox holds edits/files from an *older* policy, the *current* policy continues from a state it didn't create, and the resulting advantage gets attributed to a trajectory the current policy only partially owns. This is a training-signal corruption distinct from the IS-ratio mismatch that trajectory- and token-level staleness produce, and isn't addressed by any of the IS-reshaping methods (TIS, IcePop/MIS, M2PO) surveyed in [[frontier-async-rl]] — it lives in the environment layer, not the policy-gradient estimator.

## prime-rl 0.6.0: Inference and Training Optimizations at Trillion-Parameter Scale

Prime Intellect's prime-rl 0.6.0 release (Prime Intellect Team & Matej Sirovatka, June 2026 — see `raw/primeintellect-rl-at-1t-scale.md`) gives the GLM-5 case study above (case study #2) a concrete systems explanation: **GLM-5.1 trains on SWE tasks at up to 131k sequence length, sub-5-minute step times, batch size 256 rollouts, on only 28 H200 nodes**, with step time staying flat as scale grows toward 1T parameters.

On the **async RL mechanics**, prime-rl pushes new weights to inference as soon as an optimizer step finishes; in-flight rollouts keep their active prefix cache (so a single rollout's KV cache can mix tokens from multiple policy versions), while a **KV-cache salt** forces new rollouts to populate a fresh cache rather than reuse an older one. `max_off_policy_steps` drops requests from a policy that has fallen too far behind — the concrete instantiation of the "policy-lag cutoff" mechanism [[frontier-async-rl]] already attributes to GLM-5.

On **inference**, prime-rl combines FP8 (via DeepEP/DeepGEMM kernels) with **Wide Expert Parallelism** — EP spanning ≥32 GPUs plus a large DP rank (e.g. 32), tuned for throughput rather than latency. **P/D disaggregation** matters because some model↔environment pairs hit a 4:1 prefill:decode token ratio (directly explaining why case study #2's prefill-heavy shift, from response/tool-call growth, justified a PD-disaggregated generator); disaggregating keeps decode latency predictable across the hundreds of turns a long agentic rollout can take. **KV cache offloading** is tiered to CPU/disk via either native vLLM offloading (one pool per DP-rank worker) or **Mooncake Store** (one centralized pool reachable from any worker — the better fit once routing gets sophisticated, e.g. the NVIDIA Dynamo router scoring on live KV-reuse/queue-depth/load metrics).

The most direct extension of this wiki's framing is **router replay (R3)**: prime-rl reports it reduces trainer/inference KL mismatch **by roughly an order of magnitude** — a concrete production number for the mechanism [[frontier-async-rl]] already documents in its Systems-Level Fixes section. **FP8 training** (DeepGEMM block-scaled FP8, the DeepSeek V3 recipe) attacks the same trainer/inference mismatch problem from the numerics side instead of the routing side: it does not meaningfully raise throughput, but aligning trainer and inference precision (and sometimes kernels) further stabilizes training.

On **training-side parallelism**, the trainer is built on [torchtitan](https://github.com/pytorch/torchtitan) and composes FSDP2 (`fully_shard`), Expert Parallelism, and Context Parallelism. The EP rationale is concrete: at 800B params / 78 layers / FP32 master weights, all-gathering one full layer costs roughly `(800B × 4) / 78 ≈ 40GB`; EP=8 avoids that by dispatching/combining tokens via all2all instead (torch-native all2all wins within a node, [[gpu-kernel-engineering|DeepEP]] wins once EP spans nodes). At 131k+ sequence length, activations rather than parameters dominate memory, which is what Context Parallelism shards away — see [[kv-cache]] for the Ring-Attention-vs-Ulysses comparison and GLM-5's custom DSA context-parallel scheme.

## Open-Source RL Framework Lineage

DeepSeek R1's release triggered the open-source RL-infrastructure wave documented in the Open-Source Frameworks table in [[frontier-async-rl]]. **OpenRLHF** (PPO, REINFORCE++, then GRPO) was an early, influential framework; several of its maintainers went on to build **slime** and **verl** — seeding much of the open RL-training ecosystem and giving academic researchers practical access to RL-systems research that previously required frontier-lab-scale infrastructure.

## RL Framework Papers: verl/HybridFlow, AReaL, AsyncFlow, FP8-RL

The "Open-Source RL Framework Lineage" above names the frameworks; the underlying papers explain the actual systems choices.

[HybridFlow (2409.19256)](https://arxiv.org/abs/2409.19256) — not currently in the repo as a PDF; arXiv link only — is the foundational verl paper (HKU + ByteDance, EuroSys '25). Its core idea is a hierarchical hybrid-controller architecture: a single-controller layer (Ray-based RPC) provides the flexibility to express arbitrary dataflows between actor/critic/reward/reference models, while a multi-controller layer (SPMD) keeps each individual model's execution efficient. The **3D-HybridEngine** is the piece that makes this practical for RLHF specifically: it reshards actor-model weights between training and generation parallelism layouts with zero redundant memory, cutting transition time by 55.2% on average (up to 89.1% in the best case). The payoff is large end-to-end speedups over DeepSpeed-Chat, OpenRLHF, and NeMo-Aligner — 1.53×–20.57×, with up to 12.52× average over NeMo-Aligner at 70B scale — and an API (`ResourcePool`, `@register(transfer_mode=...)`, `TensorDict`) simple enough that a full PPO loop fits in roughly 8 lines. This paper is the reason verl became the default synchronous RL framework that AReaL and AsyncFlow (below) both benchmark themselves against.

[FP8-RL (2601.18150)](https://arxiv.org/abs/2601.18150) — not currently in the repo as a PDF; arXiv link only — is the paper behind the "more up-to-date verl info" link some readers expect to be a verl v2 paper. It isn't — it's a practical, separately-published low-precision training stack (NVIDIA, v1 Jan 2026, v2 Apr 2026) built inside the verl ecosystem. It combines FP8 W8A8 linear layers, FP8 KV-cache, and a truncated-importance-sampling (TIS) correction (clip threshold C=2, "strongly recommended" by the authors) to control the precision-induced train/inference mismatch — the same family of fix as the IS-reshaping methods in [[frontier-async-rl]]. Reported speedups: ~10–20% for dense Qwen3-8B, ~30–50% for MoE Qwen3-30B-A3B, with a combined headline of 44% (verl-side) up to 48% (NeMo-RL trainer-side). One important negative result: an FP8-rollout-only MoE run **crashed around step 700** even with TIS active, and needed the heavier MIS/Rollout-Router-Replay fixes (see [[frontier-async-rl]] § MoE Routing Replay) to stay stable — keeping the router itself in BF16 was sufficient, and going further to FP32 gave negligible extra benefit. The paper explicitly maps the 2026 framework landscape as verl, OpenRLHF, NeMo-RL, ROLL, slime, and AReaL.

[AReaL (2505.24298)](https://arxiv.org/abs/2505.24298) — not currently in the repo as a PDF; arXiv link only — is the full paper behind the AReaL GitHub link already cited in [[frontier-async-rl]]'s frameworks table (Tsinghua IIIS, Ant Group, HKUST; NeurIPS 2025). It fully decouples generation from training using **Interruptible Rollout Workers** plus a **staleness-aware decoupled PPO objective**: instead of clipping against the stale behavior policy that actually produced a trajectory, it clips against a recent "proximal" policy π_prox, which the paper proves (Proposition 1) keeps the gradient estimator well-behaved under significant staleness. The staleness bound is `⌊(N_r−1)/B⌋ ≤ i+η`, with η=4 for code tasks and η=8 (the default) for math. The ablation is stark: naive PPO at η=4 collapses AIME24 accuracy from 42.0 to 23.3, while the decoupled objective holds at 42.2. End-to-end, a 1.5B model matches DeepScaleR's AIME24 score (42.2 vs. 43.1) in 14.8 hours versus 41.0 hours for a synchronous run — the paper's headline 2.77× speedup — and AReaL keeps scaling near-linearly out to 512 GPUs, a regime where the paper shows verl (used throughout as the synchronous SOTA baseline) runs out of memory. PipelineRL is never discussed in this paper — the two systems solve the staleness problem differently (in-flight weight updates vs. decoupled-objective tolerance) without engaging with each other directly.

[AsyncFlow (2507.01663)](https://arxiv.org/abs/2507.01663) — not currently in the repo as a PDF; arXiv link only — is Huawei's asynchronous streaming RL framework, built for Ascend NPUs on top of MindSpeed-RL. Its centerpiece is **TransferQueue**, a 2D columnar data store (inspired by software-defined networking's data-plane/control-plane split) that decouples *what* data is needed from *how* it's transferred, plus a **delayed parameter update** scheme that tolerates one step of staleness without measurable degradation. AsyncFlow taxonomizes the framework landscape into **task-colocated** systems (TRL, DeepSpeed-Chat, NeMo-Aligner, and verl, which run all roles in the same process group) versus **task-separated** systems (OpenRLHF, Kimi k1.5, StreamRL, and AReaL, which run rollout and training as independent services) — and self-identifies as task-separated. Reported throughput: 1.59× average and 2.03× peak over verl on a 7B model across 256 NPUs. Like AReaL, it never mentions PipelineRL.

Taken together, these four papers plus PipelineRL (above) show the same engineering problem — keep the generator and trainer both busy without corrupting the policy gradient — solved five different ways: hybrid-controller resharding (verl/HybridFlow), low-precision compute with IS correction (FP8-RL), staleness-tolerant objectives (AReaL), a decoupled data-plane (AsyncFlow), and in-flight weight syncing (PipelineRL). See [[rl-scaling-laws]] for how ScaleRL's recipe selects PipelineRL specifically as its execution layer, and [[frontier-async-rl]] for the policy-lag formalism that explains why all five need *some* staleness fix in the first place.

## Foundations And Variance Reduction

[AlphaZero (1712.01815)](../../papers/05-learning/reinforcement-learning/Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm - 1712.01815.pdf) is the clean conceptual ancestor: self-play, search-improved policies, and outcome feedback. The analogy breaks because language does not have perfect simulators or compact action spaces, but the loop of "sample, evaluate, improve, repeat" is the same.

[GAE (1506.02438)](../../papers/05-learning/reinforcement-learning/High-Dimensional Continuous Control Using Generalized Advantage Estimation - 1506.02438.pdf) is the variance-control ancestor. In LLM post-training, the exact estimator may differ, but the same pressure remains: reduce variance without introducing too much bias, especially when rewards are sparse and traces are long.

## RL Environment Taxonomy: E = {T, H, V, S, C}

Han Lee's "A Taxonomy of RL Environments for LLM Agents" (Mar 2026) provides a formal decomposition that makes explicit what every training system implicitly contains. A complete RL environment is:

**E = {T, H, V, S, C}** — Tasks, Harness, Verifier, State, Configuration.

### T: Tasks

Tasks vary structurally in ways that demand different capabilities — not just in difficulty but in number of actions, tools, tokens, and time required. Key task types:

| Type | What the agent must do | Examples |
|------|----------------------|---------|
| **Single-turn Q&A** | One prompt → one response | Math benchmarks, SimpleQA |
| **Code generation** | Write code, run it, check outputs | SWE-Bench, LiveCodeBench |
| **Repository-level coding** | Navigate large codebases, multi-file edits | SWE-Bench Verified, RepoBench |
| **Stateful enterprise** | Modify persistent DB state, work within access controls | EnterpriseOps-Gym |
| **Open-ended research** | No single correct answer; report quality matters | ADR-Bench |
| **Productivity workflows** | Draft emails, manage calendars, triage | WorkArena, OSWorld |

The task distribution is an important data design decision. Training only in clean deterministic environments produces agents that fail in stochastic production environments. Curriculum design — ordering tasks by difficulty from simple to complex — mirrors how human education works.

**Synthetic data** for tasks is a first-class problem. With real-world tasks, you rarely have large labeled datasets. Strategies: back-translation (start from a desired output, reconstruct the input), graph-based synthesis (build a knowledge graph, generate multi-hop queries), and automated environment generation (~$4/env cost per AutoEnv, 2025).

### H: Agent Harness (in the RL context)

The harness is the scaffolding enabling the model to interact with the environment. See [[agent-harness-engineering]] for deep coverage. In the RL context, key harness properties:

- **Rollout protocol:** single-turn, multi-turn, tool-use, stateful tool-use, full OODA agentic loop
- **Context manager:** recency-based retention, Markovian reconstruction, reference-preserving summarization/folding
- **Tool mix determinism:** non-deterministic tools (web search, browser) mean two runs of the same trajectory can produce different outcomes, complicating verifier design

Modern harness design reduces tools to atomic basics: `read`, `write`, `edit`, `bash`, `tasks` (spawns subagents), `mcp` (MCP resources), `skill`, and `askUserQuestions`. This contrasts with early agent days of manually wiring individual API calls.

### V: Verifier

The verifier maps a completion to a reward: V(task_prompt, completion, info) → [0, 1]. Key types:

| Type | Reward signal | When to use |
|------|-------------|-------------|
| **Exact match** | Binary (0/1) | Ground truth available |
| **Code execution** | Binary or partial | Output testable programmatically |
| **LLM-as-judge** | Continuous [0,1] | Open-ended quality, no other option |
| **Checklist-style** | Continuous | Multi-criteria tasks |
| **Evolving rubric (RLER)** | Continuous | Resistant to reward hacking |
| **Process reward model (PRM)** | Per-step continuous | Long-horizon credit assignment |

Key principles:
- **Verifiable beats judgeable.** Programmatic checks are faster, cheaper, and more consistent. Use LLM-as-judge only when there's no other option.
- **Static rubrics get gamed.** Models learn to write answers that score well on the rubric, not answers that solve the problem. DR Tulu's RLER (Rubric-Level Evolving Reward) co-evolves the rubric with the policy.
- **Noise injection.** Step-DeepResearch deliberately injects 5–10% tool errors during training, producing agents significantly better at handling flaky APIs in production.
- **Scorer independence.** Using the same model family to generate completions and judge them creates a feedback loop that actively teaches wrong behavior.

### S: State and C: Configuration

**State** ranges from stateless (each episode starts fresh — LeetCode problems) to deeply stateful (EnterpriseOps-Gym maintains 164 database tables and 512 tools across episodes, where actions in one task affect state seen by subsequent tasks).

**Configuration** covers turn limits, context budgets, sampling temperature, and curriculum scheduling. A turn limit of 5 vs. 600 changes what skills the agent can develop. Step-DeepResearch progressively scales context windows from 32K to 128K during mid-training.

### Benchmarks as Frozen Environments

A benchmark is just a frozen RL environment: B = (Request, Environment, Stopping Criteria, Scorer) = (T, subset of H+S, C, V). The design principles that make benchmarks good apply directly to training environments — with one key difference: training environments can evolve their parameters over the course of a run.

Key benchmark quality principles that double as training environment principles:
- **Task naturalness:** SWE-bench works because tasks are real GitHub issues filed by real developers. Tasks no human would actually encounter may ace evals without teaching usefulness.
- **Automatic verifiable scoring.** Training may need millions of reward signals; human judges don't scale.
- **Difficulty calibration.** Too easy: agent ceilings quickly. Too hard: reward signal too sparse to learn from. Training environments benefit from curriculum scheduling that benchmarks can't do.

### Emerging Infrastructure

- **Environment-as-package model:** Prime Intellect Environments Hub creates a shared ecosystem around RL environments (like PyPI for model weights, HuggingFace for RL envs).
- **OpenReward** (General Reasoning, 2026): 330+ RL environments as managed API endpoints, backed by 4.5M+ tasks.
- **Open Reward Standard (ORS):** Extends MCP with RL primitives (episodes, reward signals, task splits, curriculum management). ORS is to RL environments what MCP is to tool integration. The six frameworks active in 2026 — OpenEnv (Meta), ORS (General Reasoning), NeMo Gym (NVIDIA), Verifiers (PrimeIntellect), SkyRL Gym (NovaSky-AI), and GEM (Axon-RL) — span two architectural clusters: HTTP server (deployable, language-agnostic) and in-process (fast iteration, same Python process as the trainer). See [[rl-environments-frameworks]] for a Rosetta-stone comparison across all six using the same reference environments.

## Safety And Evaluation

RL training systems are where [[safety-misalignment]] can enter. Reward hacking is not just an eval bug; it is a training signal. Verifiers, rubric rewards, and harnesses must be treated as part of the model's environment. For agentic tasks, [[llm-evaluation]] needs trace review because the final reward can hide fabricated inputs, brittle tool calls, or unsafe strategies.

## Related Topics

- [[alignment-methods]] - DPO, GRPO, DAPO, KL-regularized policy gradients
- [[training-ops]] - multi-client orchestrators, in-flight updates, throughput logging
- [[reasoning-models]] - long-CoT behavior, looping, RLVR length control
- [[on-policy-distillation]] - a cheaper same-family alternative when a teacher exists
- [[frontier-async-rl]] - policy-lag/staleness mechanics and IS-estimator fixes that complement the throughput-matching view here
- [[rl-scaling-laws]] - compute scaling laws for the RL recipes these systems execute
- [[safety-misalignment]] - reward hacking and context-dependent misalignment
- [[llm-evaluation]] - trace analysis and open-world evals
- [[agentic-rl]] - ECHO trains an auxiliary cross-entropy objective on environment-observation tokens alongside GRPO, turning terminal feedback already present in every rollout into dense supervision
- [[kv-cache]] - Ring Attention vs. Ulysses context parallelism, and GLM-5's custom DSA context-parallel scheme, that prime-rl uses to train at 131k+ sequence length
- [[mixture-of-experts]] - Wide EP and the FSDP+EP memory math behind prime-rl's expert-parallel training
- [[gpu-kernel-engineering]] - DeepEP/DeepGEMM kernels and FP8 precision underlying prime-rl's inference and training stack

## Sources

- [DAPO: An Open-Source LLM Reinforcement Learning Framework (2503.14476)](../../papers/05-learning/reinforcement-learning/DAPO: An Open-Source LLM Reinforcement Learning Framework - 2503.14476.pdf)
- [On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning (2505.17508)](../../papers/05-learning/alignment-preferences/ON THE DESIGN OF KL-REGULARIZED POLICY GRADIENT - 2505.17508.pdf)
- [PipelineRL: Faster On-policy Reinforcement Learning by Leveraging Pipeline Parallelism (2509.19128)](../../papers/03-scaling/training-optimization/PipelineRL: Faster On-policy Reinforcement Learning by Leveraging Pipeline Parallelism - 2509.19128.pdf)
- [HybridFlow: A Flexible and Efficient RLHF Framework (2409.19256)](https://arxiv.org/abs/2409.19256) — not currently in the repo as a PDF; arXiv link only. Foundational verl paper; hybrid-controller architecture, 3D-HybridEngine.
- [FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning (2601.18150)](https://arxiv.org/abs/2601.18150) — not currently in the repo as a PDF; arXiv link only. FP8 W8A8 + KV-cache, TIS correction, built in the verl ecosystem.
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning (2505.24298)](https://arxiv.org/abs/2505.24298) — not currently in the repo as a PDF; arXiv link only. Interruptible Rollout Workers, staleness-aware decoupled PPO.
- [AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training (2507.01663)](https://arxiv.org/abs/2507.01663) — not currently in the repo as a PDF; arXiv link only. TransferQueue, delayed parameter update, MindSpeed-RL/Ascend.
- [Reuse your FLOPs: Scaling RL on Hard Problems by Recycling Computation (2601.18795)](../../papers/05-learning/reinforcement-learning/Reuse your FLOPs: Scaling RL on Hard Problems by Recycling Computation - 2601.18795.pdf)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (1506.02438)](../../papers/05-learning/reinforcement-learning/High-Dimensional Continuous Control Using Generalized Advantage Estimation - 1506.02438.pdf)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (1712.01815)](../../papers/05-learning/reinforcement-learning/Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm - 1712.01815.pdf)
- "A Taxonomy of RL Environments for LLM Agents" — Han Lee (Mar 21, 2026): https://leehanchung.github.io/blogs/2026/03/21/rl-environments-for-llm-agents/
- "The Ultimate Guide to RL Environments: Building and Scaling Them in the LLM Era" — Adithya S K & Sergio Paniego (May 5, 2026): https://huggingface.co/spaces/AdithyaSK/rl-environments-guide — companion code: https://github.com/adithya-s-k/RL_Envs_101
- "RL Systems Mind the Gap: Matching Trainer and Generator Throughput" — Kimbo Chen, Cheang Kang Wen, Dylan Patel, SemiAnalysis (June 16, 2026). See `raw/semianalysis-rl-systems-mind-the-gap.md`. Source for the throughput-matching framework, the four production case studies, partial rollout/environment-state staleness, and the OpenRLHF→slime/verl lineage.
- "RL at 1T Scale: prime-rl Performance Deep Dive" — Prime Intellect Team, Matej Sirovatka (June 21, 2026). See `raw/primeintellect-rl-at-1t-scale.md`. Source for the GLM-5 131k-sequence-length/28-H200-node headline result, async RL mechanics (KV-cache salt, `max_off_policy_steps`), the inference stack (FP8, Wide EP, P/D disaggregation, KV cache offloading, request routing, R3 router replay's order-of-magnitude KL reduction), and the training stack (torchtitan, FSDP/EP/CP 3-D parallelism, FP8 training).
