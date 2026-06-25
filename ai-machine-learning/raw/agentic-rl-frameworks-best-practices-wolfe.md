# Agentic RL: Frameworks and Best Practices

**Source:** https://cameronrwolfe.substack.com/p/agentic-rl
**Author:** Cameron R. Wolfe, Ph.D.
**Published:** June 22, 2026
**Filed under:** Agentic RL / LLM agent training systems

## Overview

A long-form survey-style post (Deep (Learning) Focus newsletter) on how LLMs are trained via RL to handle long-horizon, multi-turn, tool-using tasks rather than single-turn reasoning. The post first builds a foundational vocabulary for agents and agentic RL (components of an agent, the MDP formulation, environment scaling), then walks through six concrete papers/frameworks — ToRL, AgentGym-RL (+ ScalingInter-RL), Agent-R1, AgentRL, AutoForge, and RAGEN/StarPO — extracting practical design patterns common across them.

## The Basics of Agents and RL

An agent is "just an LLM that runs in an agentic loop, using both tools and its own reasoning capabilities to solve complex problems." The agent system has four components — **LLM backbone**, **instructions**, **tools**, and **environment** — orchestrated by an **agent harness**. At each loop iteration the LLM backbone generates output, executes tool calls, ingests environment feedback, and a termination check decides whether to continue.

The LLM backbone needs strong instruction-following, tool-calling, and reasoning ability; reasoning models (see [[reasoning-models]]) are commonly preferred because long-horizon tasks require decomposing problems into smaller parts and self-reflecting/recovering from mistakes. Instructions should "balance simplicity and specificity" — detailed enough to reliably guide behavior, not so detailed that they become brittle. One quoted engineering finding on this point: agents "tended to try to do too much at once — essentially to attempt to one-shot the app," and the fix was to set the agent up to "work step-by-step and feature-by-feature," prompting incremental progress while leaving the environment in a clean state at the end of each session.

Tool calls are represented directly in the LLM's token stream via special delimiter tokens. Qwen3 models use XML-style delimiters as a concrete example: `<tools>`/`</tools>` wrap tool definitions in the instructions, `<tool_call>`/`</tool_call>` wrap an invocation (tool name + arguments) and the closing tag triggers generation to stop so the call can be parsed/executed, and `<tool_response>`/`</tool_response>` wraps the returned observation.

## MDP Formulations: Single-Turn vs. Multi-Turn RL

The post derives an explicit Markov Decision Process (MDP) formulation for both setups, which is a useful precise complement to the more abstract PBRFT-vs-POMDP framing already used elsewhere in this wiki (see [[agentic-rl]]'s "Landscape" survey section).

**Single-turn RL MDP:**
- *State* — the current token context (prompt + generated tokens so far).
- *Action* — the token selected at each next-token-prediction step (each token is its own action).
- *Transition function* — deterministically appends the selected token to the context.
- *Reward* — usually a terminal/outcome reward over the full completed rollout.
- *Trajectory* — the full generated token sequence.

**Multi-turn (agentic) RL MDP:** the state becomes a *joint* state of (1) the textual context — prompt, generated tokens, tool calls, and tool observations — and (2) the external environment state that the agent's actions can modify. Actions are still token-level at the lowest granularity, but sequences of tokens form higher-level actions (tool calls). Crucially, the transition function is no longer purely deterministic: text-only actions still just append tokens, but tool-call actions update environment state and return observations that "may be stochastic" depending on tool/environment behavior. Rewards include both terminal/outcome rewards and intermediate step-level process rewards. A batch of multi-turn trajectories is generated per training iteration, same as the single-turn case, but each trajectory now spans multiple rounds of agent action and environment interaction.

## Environment Execution and Scaling

Each agentic rollout needs an *isolated* environment instance (its own filesystem/database/codebase state) so that concurrent rollouts in the same RL training step — e.g., the group of rollouts sampled for GRPO — don't corrupt each other's state. This isolation is typically implemented via Docker containers or similar sandboxes, and at RL training scale (thousands of concurrent rollouts per update) environment startup/execution/teardown latency becomes the rollout-generation bottleneck.

A concrete scaling anecdote, quoting an R2E-Gym-related source:

> "A challenge we encountered was scaling up SWE-Bench environments… each RL iteration spawned 512 Docker containers in parallel… overloading Docker's API server and eventually crashing the Docker daemon… To remove that bottleneck, we integrated Kubernetes support into `R2E-Gym`, letting the orchestrator schedule containers across a pool of nodes."

The general lesson: a naive per-worker local Docker daemon becomes a bottleneck once many rollout workers create/destroy containers concurrently, motivating a cluster-orchestration layer (Kubernetes) that schedules environment instances across a resource pool, manages lifecycles, and avoids single points of failure. (R2E-Gym is the same procedurally-generated executable gym that DeepSWE trains against — see [[agentic-rl]]'s DeepSWE section and [[swe-agent-benchmarks]].)

## ToRL: Scaling Tool-Integrated RL (arXiv 2503.23383)

ToRL (Tool-Integrated Reinforcement Learning) trains tool-integrated-reasoning (TIR) agents — LLMs that interleave natural-language reasoning with executable code — using RL rather than the more common SFT-on-teacher-trajectories approach. The motivating critique: "existing tool-integrated reasoning approaches face critical limitations. Most studies distill trajectories from stronger models and perform SFT, restricting models to predetermined tool usage patterns and limiting exploration of optimal strategies."

**Setup.** ToRL uses an RL-Zero setup — RL training starts directly from a pretrained base model with no post-training warm-start. A code interpreter is added as a tool in the RL environment, and a seed prompt nudges the model toward fused natural-language+code reasoning, which RL training then refines via reward-driven exploration. Code blocks are wrapped in `` ```python ... ``` ``; when the model emits a closing ` ```output ` tag, generation pauses, the code executes, the output is appended to context, and generation continues. To bound the slowdown from repeated execution pauses, a max of `C` tool calls per problem is enforced — most experiments use `C=1`; `C=2` gives a moderate accuracy gain at a significant efficiency cost.

**Error handling.** Execution errors are returned to the model as observations, but truncated to only the last line of the traceback to avoid overloading context: "We deliberately return error messages to the LLM when code execution fails… these error diagnostics enhance the model's capacity to generate syntactically and semantically correct code in subsequent iterations." Code output is masked from the RL loss (the model isn't trained to predict text it didn't generate), and all code executes in sandboxed environments isolated from the main training process.

**Reward mechanism** is a simple three-way outcome reward: `+1` correct response, `−1` incorrect response, `−0.5` non-executable code. Notably, the `−0.5` error penalty does *not* help — pure outcome reward (no extra penalty for non-executable code) matches or exceeds the penalized variant, with the hypothesis that the penalty makes the model overly conservative about generating code at all.

**Data.** GRPO over ~29K math competition questions (MATH, Numina-MATH), with proof-style/hard-to-verify questions removed and final examples selected via the **LIMR** technique (arXiv 2502.11886): LIMR tracks each training sample's reward trajectory across epochs and scores questions by alignment with the model's own learning trajectory, prioritizing questions that are learnable *at the right time* over difficult or randomly selected ones.

**Results.** A 14.7% absolute accuracy improvement over an SFT baseline for Qwen2.5-Math-7B. Behaviorally, the fraction of problems solved via generated code rises from ~40% to ~80% over the first 100 training steps, and the fraction of successfully-executed code also rises throughout training — evidence that meaningful tool-use strategies, not just code-generation frequency, are being learned. Authors report concrete examples of agents using error-message feedback to reflect on and correct their own code, and of agents cross-verifying answers with a mixture of code and natural-language reasoning.

## AgentGym-RL and ScalingInter-RL (arXiv 2509.08755)

AgentGym-RL is a generalized, more efficient/modular extension of the prior AgentGym framework for training LLM agents on long-horizon, multi-turn decision-making tasks via pure outcome-reward RL (typically starting from an instruct checkpoint like Qwen2.5-3B-Instruct, not a base model). It has three components: **Environment** (a realistic task, e.g. web navigation, embodied tasks, scientific experiments — each an independent service exposing a unified HTTP interface), **Agent** (the LLM running in an agentic loop), and **Training** (a modular RL pipeline).

**Update loop.** Each training step initializes a batch of environment instances in parallel (one per agent, to avoid cross-task interference), then for each interaction turn: `response = actor.generate(prompt)` samples a response; `state, reward = env.step(response)` sends it to the environment and gets back an observation/reward; the trajectory is updated with `add_assistant_message(response)` and `add_user_message(state)`. The loop terminates on task completion or a fixed turn budget; completed rollouts across all concurrently-running agents form the training batch.

**Engineering optimizations.** AgentGym-RL extends the `verl` RL library (originally single-turn) for multi-turn use via a `RolloutHandler` that tracks full trajectory/reward history and builds attention/loss masks distinguishing agent- vs. environment-generated tokens. Per-environment engineering fixes: "we replaced WebArena's default single-browser-per-process design with a subprocess-based architecture, enabling a single server to manage multiple Chromium instances concurrently… in SciWorld environment, we redesigned the environment's initialization and reset routines to support robust parallel creation and resetting of multiple instances… we support longer training horizons through a full-reset interface in WebArena, which restores each web server to its initial state after every episode and mitigates state inconsistencies over time."

**ScalingInter-RL** is a curriculum-learning fix for a specific failure mode: early in training, agents struggle to produce *meaningful* long interactions, collapsing into redundant reasoning loops or pointless actions — but capping all training at short horizons would prevent the agent from ever learning long-horizon, non-trivial reasoning patterns. The fix: split training into `N` phases of `Δ = T/N` steps each, with a monotonically increasing interaction-turn budget `h_1 < h_2 < … < h_N` across phases (concretely, `N=3` phases of `Δ=80` iterations each, with budgets of 8, 12, and 15 turns). The interaction loop is hard-terminated (forcing a final answer) once the current phase's turn budget is hit. Rationale quoted directly: "As the horizon increases, the agent is incentivized to explore longer decision paths, facilitating the emergence of higher-order cognitive behaviors such as planning, reflection, and strategic backtracking… This phased scaling allows ScalingInter-RL to align the depth of interaction with the agent's evolving policy capabilities, bridging efficient early-stage exploitation and long-horizon generalization." A fixed large budget (e.g. 10 turns from the start) instead produces an initial improvement followed by training collapse; the gradually-increasing budget is measurably more stable.

**Results.** RL training brings models as small as 7B to performance comparable with much larger closed models on some tasks — the best model achieves 26% (web search) and 38.25% (deep research) success, surpassing GPT-4o and even Llama-3.1-70B (10× the parameters). RL benefits are uneven across domains: most pronounced on structured, rule-based environments like TextCraft and SciWorld, less pronounced on the noisier, more realistic WebArena. RL-trained agents also show strong test-time scaling along two axes — more sequential interaction turns, and more parallel sampled trajectories per task — outperforming baselines as both are scaled up. Optimizer choice matters: a 3B model trained with GRPO outperforms a 7B model trained with REINFORCE.

## Agent-R1: Step-Level Trajectories and Context Management (arXiv 2511.14460)

Agent-R1 is a unified, modular framework whose central design decision is representing each multi-turn trajectory as a sequence of **step-level** records (current/next state, action, environmental feedback, step-level and terminal rewards, and a termination signal) rather than as either a flat token sequence or a list of chat-style messages.

**Why not flat tokens or messages.** A pure flat-token trajectory preserves exact tokens but leaves step boundaries implicit and forces an append-only context strategy. The message-list representation (store as chat messages, reconstruct into a prompt via a chat template at training time) introduces **retokenization drift**: "In [agentic RL], the usual view of a trajectory as one ever-growing token sequence becomes increasingly inadequate: it makes context evolution rigid and creates representation mismatches between rollout and training." Because tokenization is not perfectly reversible, parsing rollout tokens into messages and retokenizing them for training can produce a trajectory that doesn't exactly match what was generated.

**Flexible context management.** Because Agent-R1 stores the full step-level trace but doesn't require the agent to see all of it, an environment-specific **context rule** decides what subset/transformation of prior steps becomes the agent's next observation — append-only, sliding-window (most recent steps only), or LLM-summarization are all tested. This matters because unconstrained append-only context risks **context rot** from verbose tool outputs or irrelevant reasoning steps overloading the window.

**Two standardized interfaces:** a **Tool** interface (OpenAI function-calling-schema-based: execution logic + a JSON-schema metadata spec) for atomic actions, and a **ToolEnv** interface whose `step()` function parses tool calls from agent output, executes them, updates environment state, computes rewards, and returns the next observation — distinguishing deterministic text generation from non-deterministic, environment-altering tool-call transitions.

**Masking and credit assignment.** An **Action Mask** (1 for agent-generated tokens, 0 otherwise) ensures the policy-gradient loss only touches tokens the model actually generated — prompts, tool outputs, and environment observations are excluded. Because trajectories are step-structured, rewards can be flexibly assigned: broadcast uniformly across all tokens in a step, attached only to specific tokens, or assigned only at the terminal step (outcome reward) — and Agent-R1 deliberately decouples this trajectory representation from the choice of RL optimizer (PPO, GRPO, or REINFORCE can all consume the same interaction data).

**Empirical results.** Training Qwen3-4B across four environments (GSM8K, HotpotQA, ALFWorld, WebShop): RL improves performance on all four, but the best optimizer is task-dependent — GRPO wins on most tasks, PPO wins on WebShop specifically, and REINFORCE variants are competitive though not best. Context-management strategy also matters: on GSM8K, the sliding-window strategy beats both append-only and LLM-summarization, supporting the general finding that retaining *all* context is not always optimal — "less is more in some cases."

## AgentRL: Async Training and Multi-Task Stability (arXiv 2510.04206)

AgentRL targets the systems and algorithmic problems specific to scaling agentic RL across many heterogeneous multi-turn environments at once, rather than one domain. The core complexity: multi-turn rollouts are long-running, environment-interaction-dependent, and highly variable in length/wall-clock time — "The shift from single-turn to multi-turn defines the problem of agentic RL, where the LLM acts as an autonomous agent that performs multi-turn reasoning, interacts with tools or environments, and adapts its behavior over extended trajectories." A synchronous train/generate loop (fine for single-turn RL) wastes GPU time idling on short trajectories while long ones finish.

**Three components:**

1. **Fully asynchronous training pipeline** — separate inference and training engines with dedicated resource pools, running concurrently. The inference engine continuously schedules new rollout jobs as old ones finish; the training engine pulls *all* completed trajectories (rather than waiting for a fixed rollout batch) to perform each update, so batch size varies (bounded by a configured min/max) rather than being fixed. To bound off-policy staleness in this async setup, completed rollouts sit in a size-capped data queue that gets fully drained into the trainer at each step: "To avoid off-policy bias of the rollout engine, we set a maximum size of the data queue and enforce all trajectories to be moved to the training engine at each step. In doing so, all trajectories are kept as up-to-date as possible with the latest policy."

2. **Unified environment deployment infrastructure** — a standardized, function-call-based API interface replaces each environment's bespoke action format, and every environment worker is containerized as an isolated unit under a central controller that manages thousands of concurrent instances. Quoted design rationale: "we unify the worker API across all tasks, such that each task can be instantiated and managed using an identical set of lifecycle operations... the controller provides a single gateway API to the RL engine, abstracting away task heterogeneity."

3. **Algorithmic changes** for two specific failure modes — declining exploration in large multi-turn action spaces, and instability from mixing multiple task domains in one training run.

   - **Cross-policy sampling**: rather than sampling every action in a trajectory from one model, actions are drawn from a *pool* of models (recent checkpoints from earlier training steps, kept available via dedicated rollout engines updated at staggered frequencies) — different steps within a single trajectory can come from different policy versions, generating exploration patterns no single model would produce alone. This empirically beats a simpler alternative (mixing whole trajectories from different models in a batch, rather than mixing within a trajectory).
   - **Task-level advantage normalization**: after computing standard GRPO trajectory-level advantages (group-relative within same-prompt rollouts) and broadcasting them to tokens via the action mask, AgentRL adds a second normalization pass — grouping all agent-token advantages *by task/domain* within the batch and re-normalizing to zero-mean/unit-variance per domain, so no single domain's reward scale dominates the policy update. The exact mechanic, as a PyTorch snippet from the post:

```python
import torch

eps = 1e-8

# terminal reward for each sampled trajectory
rewards = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0, 1.0])

# trajectories with same prompt ID were sampled for the same input
# we have three trajectory groups for GRPO here
prompt_ids = torch.tensor([0, 0, 1, 1, 2, 2])

# first four trajectories belong to task / domain 0
# final two trajectories belong to task / domain 1
task_ids = torch.tensor([0, 0, 0, 0, 1, 1])

# 1 marks tokens generated by the agent
# 0 marks prompt, padding, or environment tokens.
action_mask = torch.tensor([
    [0, 1, 1, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 1, 1],
    [0, 1, 1, 0],
], dtype=torch.float32)


# -------------------------------------------------
# compute GRPO-style advantage for each trajectory
# -------------------------------------------------
trajectory_advantages = torch.zeros_like(rewards)

for prompt_id in prompt_ids.unique():
    group_mask = prompt_ids == prompt_id
    group_rewards = rewards[group_mask]

    group_mean = group_rewards.mean()
    group_std = group_rewards.std(unbiased=False)

    trajectory_advantages[group_mask] = (
        group_rewards - group_mean
    ) / (group_std + eps)


# ----------------------------------------------------
# assign trajectory advantage to each generated token
# ----------------------------------------------------
token_advantages = trajectory_advantages[:, None] * action_mask


# --------------------------------------------
# normalize token advantages within each task
# --------------------------------------------
task_normalized_advantages = torch.zeros_like(token_advantages)

for task_id in task_ids.unique():
    task_rows = task_ids == task_id
    task_action_mask = action_mask[task_rows].bool()

    # gather all agent-token advantages for the entire task
    task_values = token_advantages[task_rows][task_action_mask]

    task_mean = task_values.mean()
    task_std = task_values.std(unbiased=False)

    normalized_values = (
        token_advantages[task_rows] - task_mean
    ) / (task_std + eps)

    # zero out non-agent-generated token positions
    task_normalized_advantages[task_rows] = (
        normalized_values * action_mask[task_rows]
    )
```

**Results.** Five tasks (ALFWorld, WebShop, plus new SQL/OS/knowledge-graph tasks), RL-Zero setup (no SFT warm-start), tested on GLM-4-9B and Qwen2.5-Instruct models of varying size. AgentRL's average pass rate surpasses GPT-5 and Claude-Sonnet-4 on these tasks — notably even a Qwen2.5-3B-Instruct model outperforms most proprietary models after AgentRL training. Generalization is genuine and not pure overfitting to training domains: modest gains on the held-out BFCL-v3 benchmark (not included in training), and multi-task-trained agents roughly match agents trained individually per domain — evidence of true generalist multi-task capability rather than catastrophic interference.

## AutoForge: Automated Environment Synthesis and ERPO (arXiv 2512.22857)

AutoForge addresses a different bottleneck: curating real-world environments with accurate ground truth for agentic RL is expensive and doesn't scale, but semi-automated synthetic-environment methods tend to produce tasks lacking difficulty/diversity. AutoForge is an LLM-in-the-loop framework for synthesizing realistic, difficult, verifiable RL environments, paired with an RL-algorithm variant (ERPO) tuned for the resulting simulated-user setting.

**Three-stage task synthesis pipeline:**

1. **Environment Generation** — starting only from provided tool documentation, an LLM constructs a key-value state space `S = [(K_1,V_1), …, (K_n,V_n)]` and a Python implementation of each tool, yielding an environment `E = (S, F)`. Implementing tools as Python functions over dict state supports concurrent, efficient execution needed for RL.
2. **Task Construction** — an LLM organizes the available tools into a directed graph (edge = output of one tool feeds another, e.g. `get_project_id_by_name(name) → delete_project(project_id)`), then samples tool-call sequences via random walks over this graph, merging/deduplicating sequences (via LLM) to raise complexity. Reasoning nodes (points where the agent should reason over tool-call results) are inserted into the sequence, and an LLM predicts edges between tool- and reasoning-nodes to form a directed acyclic graph (DAG) representing the task's solution trajectory.
3. **Task Instantiation** — an LLM assigns concrete values to state variables, generates both an explicit task *intent* and a less-explicit, user-phrased task *question* (e.g. intent "delete project Auro, create project Lumina" vs. question "Hi, I no longer need the Auro project. Could you please delete it? Also, please create a new project named Lumina."), fills in valid tool-call arguments, executes the tool sequence against the initial environment to get a golden final state, and iteratively refines the task to remove unneeded steps. Quoted pipeline summary: "Our synthesis pipeline starts from tool description documentation, enabling the automated construction of a database to store environment states and the generation of tool implementations in Python. A dependency graph of the tools is then constructed, upon which random walks yield diverse tool sequences. These sequences are merged and augmented with reasoning nodes and edges to form a complex directed acyclic graph (DAG), which in turn serves as the blueprint for producing tasks."

**Simulated user + outcome reward only.** AutoForge introduces a simulated user agent into RL training: it gives the agent the task *question*, and at each step the agent either makes a tool call or asks the user for more information (the user-agent's reply or the tool's result becomes the next observation). The rollout terminates when the simulated user judges all requirements satisfied; reward is binary — 1 if the final environment state exactly matches the golden state, 0 otherwise. AutoForge uses *only* outcome-based rewards, deliberately, since multiple valid tool-call sequences can reach the same correct final state.

**ERPO (Environment Relative Policy Optimization)** — a GRPO variant with three changes on top of the simulated-user rollout setup:

1. *Interleaved Thinking* — prior-step thinking traces are retained across steps (rather than discarded before each new generation, the typical default), preserving task analysis/planning continuity.
2. *Masking Erroneous User Behavior* — an LLM judges whether the simulated user's responses contained an error that caused task failure; if so, that trajectory is excluded from the advantage/loss computation.
3. *Environment-Level Advantage Estimation* — rather than GRPO's per-question group-relative advantage, ERPO keeps the same question-level reward mean in the numerator but computes the standard deviation over *all valid trajectories across all questions sharing the same environment* — pooling the denominator at the environment level for more outlier-robust scaling, one level coarser than AgentRL's per-task normalization and one level finer-grained relative to AutoForge's own environment scope.

**Empirical results.** 10 synthesized environments, 1,078 tasks total, generated using Qwen3-235B-A22B-Thinking; used to finetune Qwen3-30B-A3B-Thinking (cold-start SFT on valid self-collected trajectories, then ERPO). AutoForge surpasses larger open base models (e.g. Qwen3-235B-A22B-Thinking itself) on in-domain τ-Bench/VitaBench benchmarks and nears proprietary models like Gemini-2.5-Pro in some cases. Out-of-domain generalization holds up even under a deliberately harsh test (AceBench-zh: different prompt/tool-call format, unseen tools, and Chinese instead of English at eval time despite English-only training) — RL training still improves performance there. Ablations confirm each component matters: interleaved thinking helps despite the extra context cost; a stronger simulated-user model (GPT-5 vs. GPT-4.1) at eval time improves measured performance; erroneous-user masking helps (performance degrades without it); and environment-level advantage estimation is "crucial for keeping agentic RL training stable."

## RAGEN and StarPO: Self-Evolution via Trajectory-Level Optimization (arXiv 2504.20073)

RAGEN is a modular training framework for agentic RL using rule-based, verifiable rewards, framing the agent's learning process as **"self-evolution"** — the agent learns from its own generated outputs and the corresponding environment rewards. Its core algorithmic contribution is **StarPO** (State-Thinking-Actions-Reward Policy Optimization), a trajectory-level learning algorithm built specifically for interactive multi-turn agents:

> "Unlike previous methods for static tasks that treat each action independently, StarPO treats the entire trajectory — including observations, reasoning traces, actions, and feedback — as a coherent unit for rollout and model optimization."

In a StarPO rollout, the agent runs in an agentic loop where, at each step, it generates a structured output containing both a textual reasoning trace *and* one or more environment-executable actions (tool calls) — i.e., reasoning-guided structured outputs, not bare actions. After each step the environment updates and may return an optional intermediate reward (RAGEN's agents specifically receive a negative reward for incorrectly formatted output). The rollout ends at a stopping condition or a max-turn cap, at which point a final verifiable reward is computed over the full trajectory of states and actions. The training objective is the standard maximize-cumulative-reward objective, but — consistent with AutoForge's interleaved thinking — optimized over the *entire* multi-turn trajectory at once, and StarPO likewise retains intermediate reasoning traces in the trajectory rather than discarding them between steps.

(Note: the source fetch for this post was truncated mid-sentence partway through the StarPO discussion, so this section reflects only the framing and mechanism described up to that point — it does not cover whatever empirical results, ablations, or RAGEN-specific findings, such as reward-hacking or "echo trap" behaviors, may appear later in the full post.)

## Key Takeaways

- An agent is an LLM-backbone-plus-harness running in a loop over instructions, tools, and an environment; agentic RL's MDP differs from single-turn RL's mainly in a *joint* (text + environment) state and a transition function that can be non-deterministic once tool calls are involved.
- Environment isolation (one sandbox per rollout) and cluster-level orchestration (Kubernetes over a raw per-worker Docker daemon) are the standard fix for the systems bottleneck of running thousands of concurrent multi-turn rollouts.
- Nearly every framework in this set converges on some form of *finer-grained advantage normalization* as the fix for multi-task/multi-domain training instability — AgentRL normalizes per task, AutoForge's ERPO normalizes per environment (same idea, different granularity) — and on *retaining intermediate reasoning traces* across turns rather than discarding them (AutoForge's Interleaved Thinking, StarPO's full-trajectory framing).
- Trajectory *representation* (flat tokens vs. chat messages vs. Agent-R1's step-level records) is a real design axis with real failure modes — message-based representations risk retokenization drift; step-level records enable flexible context-management strategies (sliding-window beat append-only and LLM-summarization on GSM8K in Agent-R1's tests).
- A curriculum over interaction-turn budgets (ScalingInter-RL's monotonically increasing `h_1 < h_2 < … < h_N`) stabilizes long-horizon training that collapses under a fixed large turn budget from the start.
- Async rollout/train pipelines (AgentRL) with a staleness-bounded data queue are the systems answer to highly variable multi-turn rollout completion times, where a synchronous loop would otherwise leave hardware idle.

## Related wiki pages

[[agentic-rl]], [[llm-agents]], [[swe-agent-benchmarks]], [[rl-training-systems]], [[frontier-async-rl]], [[reasoning-models]]
