# Agentic RL

Most RL post-training for reasoning (see [[rl-training-systems]], [[rl-scaling-laws]]) treats the task as what *The Landscape of Agentic Reinforcement Learning for LLMs* calls **Preference-Based RLHF / PBRFT**: a degenerate single-step MDP where the model emits one response and gets one reward. **Agentic RL** widens this to a full POMDP — multi-turn, tool-using, environment-interacting — where the action space is `A_agent = A_text ∪ A_action` (text generation plus environment actions) and the episode unfolds over many turns before a terminal reward arrives. This note covers six resources that train agents this way: a survey that formalizes the shift, three coding/tool-use agent papers that each contribute a distinct stabilization trick, and two multi-task/environment-synthesis papers that push agentic RL beyond single-domain training. See [[llm-agents]] for the agent architectures being trained and [[swe-agent-benchmarks]] for the executable environments several of these papers train and evaluate in.

## The Landscape of Agentic RL: A Survey

[The Landscape of Agentic Reinforcement Learning for LLMs: A Survey (2509.02547)](https://arxiv.org/abs/2509.02547) — not currently in the repo as a PDF; arXiv link only — is the framing document for this whole cluster. Its central formal move is the PBRFT-to-agentic-RL reframing above: PBRFT's single-step MDP collapses planning, memory, and tool use into a single emitted response, while agentic RL's POMDP formulation makes each of those a first-class part of the action space and the credit-assignment problem.

The survey organizes the field along two taxonomies. A **six-dimension capability taxonomy** — Planning, Tool Use, Memory, Self-Improvement, Reasoning, Perception (plus an "Other" catch-all) — covers what an agentic-RL-trained model needs to do. An **eight-domain application taxonomy** — Search/Research, Code, Math, GUI, Vision, Embodied, Multi-Agent, Other — covers where it's deployed. It names the GRPO/PPO-variant family that recurs across nearly every paper in this cluster: DAPO, GSPO, GMPO, VAPO, Dr.GRPO, ASPO, PAPO, LUFFY, ARPO, KTAE — each a different fix to GRPO's known instabilities (length bias, reward variance collapse, off-policy drift). Its environment compendium (ALFWorld, WebShop, WebArena, SWE-bench, τ-bench, and others) and framework compendium (verl/HybridFlow, OpenRLHF, TRL, AReaL, SkyRL, Verifiers — all covered in [[rl-training-systems]] and [[rl-environments-frameworks]]) make it a useful index even where its own text runs thin: several sections (4.5 Vision, 4.6 Embodied, 4.8 Other, and all of Sections 6–7) could not be retrieved even from the survey's own companion GitHub repository — a gap in the available source material itself, not just a fetch failure, and worth flagging if you go looking for those sections.

## DeepSWE: No-Warm-Start Coding Agent and GRPO++

[DeepSWE](https://www.together.ai/blog/deepswe) (Together AI + Agentica / UC Berkeley Sky Computing Lab) is a 32B coding agent (Qwen3-32B base) trained with **no SFT warm-start** — RL from the base model directly — reaching 59% on SWE-Bench-Verified (42.2% Pass@1) for roughly $50K of compute on a 64×H100 Kubernetes cluster. It is built on the **rLLM** training framework and trains against **R2E-Gym** environments (the same procedurally-generated executable gym covered in [[swe-agent-benchmarks]]).

Its main contribution is **GRPO++**, a bundle of fixes layered onto vanilla GRPO: Clip-High (from DAPO), removing the KL loss term entirely, removing reward-standard-deviation normalization, length normalization, a Leave-One-Out baseline, and removing the entropy loss term. The one genuinely novel piece is **Compact Filtering**, DeepSWE's own contribution: it masks the loss for trajectories that hit a timeout, exceed the max context length, or exceed the max step count — these trajectories carry no clean credit-assignment signal (the agent didn't fail at reasoning, it ran out of budget), so training on them as if they were ordinary negative examples introduces noise rather than signal.

## AgentRL: Multi-Task, Multi-Turn Scaling

[AgentRL (2510.04206)](https://arxiv.org/abs/2510.04206) — not currently in the repo as a PDF; arXiv link only — trains a single policy jointly across five distinct multi-turn environments (ALFWorld, a database task, a knowledge-graph task, an OS task, and WebShop) rather than specializing on one domain. Two mechanisms make joint multi-task training work where naive joint GRPO would collapse:

- **Cross-Policy Sampling**: rollouts are drawn from a *pool* of recent checkpoints, not just the current policy — directly analogous to the "proximal policy" trick in AReaL (see [[rl-training-systems]]), but applied for diversity rather than staleness tolerance. The ablation is large: removing it drops average success rate from 65.0% to 60.7%.
- **Task Advantage Normalization**: advantages are z-score normalized *per task*, not globally, so that a task with a different reward scale or difficulty doesn't dominate or starve the gradient signal for the others. Removing it drops average success rate from 65.0% to 59.4%.

With a Qwen2.5-32B backbone and no SFT warm-start, AgentRL reaches 70.4% average success on AgentBench-FC — beating GPT-5 (52.2%), Claude-Sonnet-4-Thinking (58.2%), and DeepSeek-R1 (49.3%) — and generalizes to the held-out BFCL-v3 benchmark (+3.0pp / +1.5pp over baselines), evidence that the multi-task training itself, not just per-domain overfitting, is doing the work.

## AutoForge: Synthesizing Environments, Not Just Policies

[AutoForge (2512.22857)](https://arxiv.org/abs/2512.22857) — not currently in the repo as a PDF; arXiv link only — addresses a different bottleneck: agentic RL needs environments to train in, and hand-building them doesn't scale. AutoForge automates environment synthesis via tool-graph construction and DAG-structured task generation, then trains on the result with two new mechanisms:

- **ERPO (Environment-level Relative Policy Optimization)**: normalizes advantage at the level of the *environment*, not the rollout group — a finer-grained analogue of AgentRL's task-level normalization, one level down.
- **MEU (Masking Erroneous Users)**: in simulated multi-turn dialogues, a synthetic "user" persona can itself behave incoherently or contradict earlier turns; MEU detects and masks the loss contribution from turns following such errors, so the agent isn't trained to anticipate or paper over a broken user simulator.

AutoForge also introduces **Interleaved Thinking** and **User-Centered Rollout** as harness-level changes, and evaluates on τ-bench, τ²-Bench, VitaBench, and ACEBench-zh — benchmarks specifically built around multi-turn tool-use dialogues with a simulated user, distinct from the single-user executable-environment setting DeepSWE and the SWE-agent benchmarks use.

## Agent-R1: A General Synchronous Framework

[Agent-R1 (2511.14460)](https://arxiv.org/abs/2511.14460) — not currently in the repo as a PDF; arXiv link only — is a general-purpose, open framework rather than a single trained model, built around `BaseTool` / `BaseToolEnv` abstractions that let a new tool-using environment be defined without rewriting the RL loop. Its main technical contribution is **Action / Loss / Advantage Masking** for multi-turn credit assignment: in a multi-turn trajectory, only the tokens the policy actually generated (not tool outputs or environment observations echoed back into context) should receive policy-gradient credit, and Agent-R1 masks the loss and advantage computation accordingly at each of three levels. It runs a **synchronous** rollout pipeline — a direct contrast to DeepSWE and AgentRL, which both run async — and uses GRPO as its base algorithm, evaluated on multi-hop QA benchmarks (HotpotQA, 2WikiMultihopQA, Musique) using exact-match.

## Training Long-Context, Multi-Turn SWE Agents with RL

[Training Long-Context, Multi-Turn Software Engineering Agents with Reinforcement Learning (2508.03501)](https://arxiv.org/abs/2508.03501) — not currently in the repo as a PDF; arXiv link only — tackles a systems problem specific to long-horizon coding agents: a multi-turn SWE trajectory can run far past typical context windows. The paper extends context to 131K tokens via YaRN (the same RoPE-extension technique covered in [[positional-encodings]] and reused by Polaris for inference-time scaling, see [[rl-scaling-laws]]) combined with context parallelism, and trains full-parameter (not LoRA) on a Kubernetes-based rollout infrastructure using vLLM.

Two details distinguish it from DeepSWE and AgentRL. First, it runs a **synchronous** pipeline (like Agent-R1, unlike DeepSWE/AgentRL's async setups) and includes an explicit **RFT (Rejection Fine-Tuning) warm-start phase** before RL — the opposite design choice from DeepSWE and AgentRL's "no warm-start" approach, making this paper a useful counterexample to "skip SFT" as a universal agentic-RL recipe. Second, it identifies an **importance-sampling-ratio validity issue** caused by decoding-parameter mismatches between the rollout engine and the training engine — structurally the same train/inference mismatch problem covered in depth in [[frontier-async-rl]], surfacing here in a synchronous rather than asynchronous setting. The paper evaluates on SWE-bench Verified and on SWE-rebench using a temporal train/test split (May vs. June data) to check for benchmark contamination rather than relying on a fixed holdout.

## Cross-Cutting Design Choices

Three axes recur across this cluster and are useful for orienting a new paper into the landscape: **sync vs. async rollouts** (Agent-R1 and the long-context SWE paper run synchronous; DeepSWE and AgentRL run async); **warm-start vs. no warm-start** (DeepSWE and AgentRL skip SFT entirely and start RL from a base model; the long-context SWE paper uses an RFT phase first); and **single-task vs. multi-task** (DeepSWE specializes on coding; AgentRL and AutoForge both push toward joint training across many environments, via task-level and environment-level advantage normalization respectively). None of these axes is settled — the survey's breadth and the disagreement between these five concrete training recipes is itself evidence that agentic RL is still in a phase of competing recipes, not a converged one.

## Related Topics

- [[llm-agents]] — ReAct, tool use, memory architectures that agentic RL trains into a policy
- [[swe-agent-benchmarks]] — R2E-Gym, SWE-smith, Multi-SWE-bench: the executable environments DeepSWE and the long-context SWE paper train against
- [[rl-training-systems]] — GRPO/DAPO mechanics, AReaL's proximal-policy clipping (echoed by AgentRL's Cross-Policy Sampling), staleness and throughput systems
- [[rl-scaling-laws]] — Compute scaling for the RL stage these agents are trained with
- [[frontier-async-rl]] — The train/inference mismatch problem that resurfaces as an IS-validity issue in the long-context SWE paper
- [[agent-harness-engineering]] — Production harness design for the same class of long-running coding agents
- [[positional-encodings]] — YaRN context extension, used here and in Polaris

## Sources

- [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey (2509.02547)](https://arxiv.org/abs/2509.02547) — not currently in the repo as a PDF; arXiv link only. PBRFT-to-POMDP reframing, six-dimension capability taxonomy, eight-domain application taxonomy.
- [DeepSWE](https://www.together.ai/blog/deepswe) — Together AI / Agentica (UC Berkeley Sky Computing Lab) blog. GRPO++, Compact Filtering, rLLM + R2E-Gym, no-SFT-warm-start.
- [AgentRL: Scaling RL for Multi-Turn, Multi-Task Agents (2510.04206)](https://arxiv.org/abs/2510.04206) — not currently in the repo as a PDF; arXiv link only. Cross-Policy Sampling, Task Advantage Normalization, joint multi-environment training.
- [AutoForge: Environment Synthesis for Agentic RL (2512.22857)](https://arxiv.org/abs/2512.22857) — not currently in the repo as a PDF; arXiv link only. Automated environment synthesis, ERPO, MEU.
- [Agent-R1: Training Agents with End-to-End Reinforcement Learning (2511.14460)](https://arxiv.org/abs/2511.14460) — not currently in the repo as a PDF; arXiv link only. BaseTool/BaseToolEnv framework, Action/Loss/Advantage Masking, synchronous pipeline.
- [Training Long-Context, Multi-Turn Software Engineering Agents with Reinforcement Learning (2508.03501)](https://arxiv.org/abs/2508.03501) — not currently in the repo as a PDF; arXiv link only. YaRN to 131K context, RFT warm-start, IS-validity issue from decoding mismatches.
