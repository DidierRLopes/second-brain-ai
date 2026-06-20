# LLM Agents

An LLM agent is a system that uses a language model as its core reasoning engine, augmented with the ability to take actions in the world — calling tools, searching the web, executing code, and interacting with APIs. The canonical framework (Lilian Weng, 2023) defines:

**Agent = LLM + Memory + Planning + Tool Use**

## Core Agent Patterns

### ReAct: Reasoning + Acting

ReAct (Yao et al., 2022) is the foundational agent pattern. It interleaves **reasoning traces** (thinking about what to do) with **actions** (actually doing it). For example: "I need to find when Einstein was born → Action: Search Wikipedia for Einstein → Observation: Born 1879 → Thought: Now I can answer."

The key insight: reasoning without action leads to hallucination (the model makes things up). Action without reasoning leads to inefficient exploration. Together, they outperform either alone. ReAct directly influenced LangChain, AutoGPT, and every modern agent framework.

### Toolformer: Self-Supervised Tool Learning

Toolformer (Schick et al., 2023, Meta) showed that LLMs can **teach themselves** to use tools without human annotation. The model learns when to call APIs (calculator, search, translation) and critically, when NOT to — it only uses tools when they improve the output. This is the intellectual foundation for function calling in GPT-4, Claude, and other production LLMs.

## Agent Components

**Planning & Decomposition:** Breaking complex tasks into subtasks. Approaches include chain-of-thought decomposition, Tree of Thoughts (explore and backtrack), and least-to-most prompting (solve simpler sub-problems first).

**Memory:** Short-term working memory lives in the context window. Long-term memory uses external storage (typically vector databases with embedding-based retrieval). The [[retrieval-augmented-generation|RAG]] pattern is the standard approach for long-term memory.

**Tool Use:** APIs, code execution, web search, file manipulation. Modern LLMs are trained with tool-use capabilities, typically via special function-calling tokens and structured output formats.

## Production Memory Architectures: Mem0 and Zep

"Long-term memory = a vector database" was a reasonable 2023 default, but production agent-memory layers now do more than nearest-neighbor lookup over raw chat logs — they decide *what's worth remembering* and keep it consistent over time, which a plain vector store doesn't do on its own.

- **Mem0** sits between the LLM and a vector store and runs a two-stage **extract → update** loop: an LLM call extracts candidate facts from new conversation turns, then a second decision step issues **ADD / UPDATE / DELETE / NOOP** against existing memory entries (indexed by user/session/agent) rather than blindly appending. This keeps memory from growing unboundedly with redundant or contradicted facts — the system actively reconciles new information against what it already believes. Mem0 is vector-first, with an optional graph layer for relationship-heavy memory.
- **Zep** takes a different structural bet: instead of a flat fact store, it builds **Graphiti**, a temporal knowledge graph where every fact carries valid/invalid timestamps. New information doesn't overwrite old facts — it closes the old fact's validity interval and opens a new one, so the agent can answer "what did I believe at time T" as well as "what's true now." This matters most for agents that need to track state that changes over multi-session, multi-week interactions (a user's preferences, a project's status) without losing the history of how it changed.

The practical split: reach for a Mem0-style extract/update store when you need broad framework compatibility and simple fact persistence; reach for a Zep-style temporal graph when the agent's domain has meaningfully time-varying state and "when did this become true" is itself a query you need to answer. Both are complements to, not replacements for, the [[retrieval-augmented-generation|RAG]] pattern — RAG retrieves from a static-ish corpus, these systems manage a corpus that the agent itself is constantly writing to.

## Training Agents to Use Tools

The Kimi K2 report treats agentic behavior as something that must be learned from interactive trajectories, not just unlocked with a ReAct prompt. [KIMIK2: Open Agentic Intelligence (2507.20534)](../../papers/07-applications/agents-swe/KIMIK2: Open Agentic Intelligence - 2507.20534.pdf) builds a large-scale agentic data pipeline around four objects:

- **Tool specs**: 3,000+ real MCP tools plus 20,000+ synthetic tools evolved across domains such as developer tooling, finance, browser automation, file systems, search, and robotics.
- **Agents**: generated system prompts and tool bundles, so the model sees many agent roles rather than one fixed assistant persona.
- **Tasks and rubrics**: generated tasks include explicit success criteria, expected tool-use patterns, and checkpoints.
- **Trajectories**: user simulators drive multi-turn conversations; tool simulators maintain state and return successes, partial failures, and edge cases; LLM judges filter trajectories against the rubrics.

The important design choice is hybrid grounding. Simulation gives scale and domain coverage, but K2 also uses real execution sandboxes for coding and software-engineering tasks where simulated tool feedback is too easy to fool. This is the same reason [[swe-agent-benchmarks]] put so much emphasis on executable environments: for agents, the difference between "looks plausible" and "works" often only appears after the tool or test suite runs.

For agent design, this shifts the recipe from "give the model tools" to "train and evaluate the model on unfamiliar tool schemas, persistent environment state, recovery from tool errors, and rubric-checked task completion." The harness still matters, but the model's prior over how tools behave is now a training target.

## Challenges

The main limitations of current agents are: finite context windows constraining working memory, difficulty with long-horizon planning (agents struggle with tasks requiring 50+ steps), reliability of natural language interfaces (LLM outputs are probabilistic, not deterministic), and error propagation (mistakes compound across steps).

## Optimizing Compound Agent Systems

Most production agents are compound systems: pipelines of multiple LLM modules (query writer, retriever, summarizer, planner, answerer) glued together by control flow. Each module has its own prompt. Improving such a system module-by-module — without retraining weights — is the domain of [[prompt-optimization|prompt optimization]] (GEPA, MIPROv2, TextGrad). For agents in particular, the rich rollouts they already produce (reasoning, tool calls, errors, judge rationales) are exactly the signal these optimizers exploit.

## Recursive Language Models (RLMs) and the Mismanaged Geniuses Hypothesis

A **Recursive Language Model (RLM)** is an agent harness where the model can launch sub-agents over sub-problems and combine their answers — a programmatic alternative to keeping the entire reasoning trace in one CoT. The Zhang & Khattab paper (Dec 2025) and Prime Intellect's `verifiers` implementation are reference examples.

The **Mismanaged Geniuses Hypothesis (MGH)**, articulated by Alex Zhang in an April 2026 case study on the LongCoT benchmark, claims that *we underestimate frontier models because the harnesses we use inhibit them*. New benchmark releases that claim "frontier models cannot solve X" often turn out, on inspection, to reflect prompting and harness choices more than capability limits.

LongCoT-mini case study:
- Reported GPT-5.2 base: 38.7%. Raymond Weitekamp's tuned DSPy.RLM lifted Claude Sonnet 4.5 from 13.0% → 45.4% in a day.
- Zhang's RLM(GPT-5.2): 50.6% but with terrible MATH (5.6%) and CS (11.0%) splits.
- Manual trace inspection: failures weren't graph-reasoning failures — they were *decomposition decision-making* failures (brute-force REPL crashes, sub-agent answers not being verified).
- Fix: ask Claude Code to inspect traces and write tips for the RLM. Same prompt across all tasks. Result: **38.7% → 65.6% overall; over 70% with partial credit.**

Takeaways for agent harness design:
- "General method cannot do XYZ" is a very strong claim that usually doesn't survive a focused prompt-engineering pass.
- The more expressive an agent's mechanism (RLM-style decomposition + sub-agents), the more sensitive it is to prompting choices, not less. Better steering pays compound returns through capability scaffolding.
- LMs themselves can generate the prompts that other LMs (in RLM harnesses) need. The fix here was authored by Claude Code from trace logs — a small instance of an [[ai-rd-automation|automated research loop]].
- Naively bootstrapping RLM-like behavior from pure RL may be sub-optimal; a workable interim path is to steer with prompts during trajectory generation, then gradually remove the priors during RL.

The MGH is the agent-side analogue of the [[on-policy-distillation|"best standalone teacher ≠ best distillation teacher"]] insight in OPD: raw benchmark score doesn't predict harness leverage.

## Related Topics
- [[chain-of-thought-reasoning]] — The reasoning techniques agents use for planning
- [[reasoning-models]] — Models that internalize multi-step reasoning
- [[retrieval-augmented-generation]] — The standard memory/knowledge pattern for agents
- [[prompt-optimization]] — Improving compound agent pipelines without weight updates
- [[ai-rd-automation]] — Agents running the modelcrafting loop end-to-end; research intuition is the bottleneck, not method knowledge
- [[agent-harness-engineering]] — Production-side engineering around agents: context windows, evals (CursorBench, Keep Rate), error taxonomies, planner/worker/judge orchestration for long-running coding work
- [[swe-agent-benchmarks]] — Executable coding-agent environments, synthetic issue generation, and hybrid verifiers
- [[llm-evaluation]] — Why closed benchmarks and open-world evaluations measure different parts of agent capability

## Sources
- Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory (arxiv:2504.19413)
- Zep / Graphiti — temporal knowledge graph documentation (getzep.com)
- ReAct: Synergizing Reasoning and Acting (arxiv:2210.03629)
- Toolformer: Language Models Can Teach Themselves to Use Tools (arxiv:2302.04761)
- LLM Powered Autonomous Agents — Lilian Weng
- AI Agents from First Principles — Cameron R. Wolfe
- [KIMIK2: Open Agentic Intelligence (2507.20534)](../../papers/07-applications/agents-swe/KIMIK2: Open Agentic Intelligence - 2507.20534.pdf) — large-scale agentic SFT data synthesis, hybrid simulated/real execution environments, and joint RL for tool-use agents.
- A Mini Exercise on the Mismanaged Geniuses Hypothesis (RLMs on LongCoT) — Alex Zhang & Omar Khattab (April 26, 2026)
- Recursive Language Models — arxiv:2512.24601
- LongCoT — arxiv:2604.14140; dataset on HuggingFace (LongHorizonReasoning/longcot)
