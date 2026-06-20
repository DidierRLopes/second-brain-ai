# Don't Build Multi-Agents

A post by Walden Yan (Cognition, June 2025) making the case that multi-agent architectures — where multiple LLMs run in parallel and collaborate — are fundamentally fragile in 2025, and that the root cause is a context engineering failure.

## Core Argument: Context Engineering Over Architecture

The central claim is that **context engineering** is the #1 job of engineers building AI agents. "Prompt engineering" was about formatting a task for a chatbot. Context engineering is about automatically providing the right context in a dynamic, multi-turn system.

Two principles follow directly from this framing:

**Principle 1: Share context — share full agent traces, not just individual messages.** When you spawn subagents, they need the full history of decisions made by the orchestrator and other agents, not a summarized subtask description. Without this, subagents misinterpret their tasks.

**Principle 2: Actions carry implicit decisions, and conflicting decisions lead to bad results.** When subagents work in parallel without visibility into each other's actions, they make conflicting assumptions. Example: two subagents building different parts of a game end up with incompatible visual styles because neither could see what the other was building.

## Why Parallel Multi-Agent Architectures Fail

The Flappy Bird example: an orchestrator splits "build a Flappy Bird clone" into two parallel subtasks (background, bird). Subagent 1 misinterprets the background; subagent 2 builds a non-game-looking bird. Even if you share the original task, real-world multi-turn conversations carry implicit context that can't be summarized without loss.

The failure mode scales badly: real tasks have many layers of nuance, all of which can be miscommunicated. The more parallel subagents, the more implicit conflicting decisions pile up.

## What to Do Instead

**The simplest solution: a single-threaded linear agent.** Context is continuous by construction. This gets you very far. The main limit is context window overflow for very long tasks.

**For truly long tasks:** Introduce a compression model whose job is to distill action history, conversation, and key decisions into a compact representation. This is hard to get right — it requires domain-specific judgment about what's information-dense. Cognition fine-tunes a smaller model for this.

**The architecture spectrum:**
1. Single-threaded linear agent → full context, no parallelism
2. Hierarchical agent with trace sharing → partial parallelism, shared context
3. Multi-agent with inter-agent communication → fragile in 2025

## Real-World Applications of These Principles

**Claude Code Subagents (as of June 2025):** Never runs subagents in parallel with the main agent. Subagents are limited to answering questions, not writing code. The benefit is that subagent investigative work doesn't clutter the main agent's context window, extending the trace before context overflow. A deliberately simple approach.

**Edit Apply Models (2024):** Used to have a large model output markdown descriptions of code edits, then a small model rewrite the file. Failure mode: the small model would misinterpret the large model's instructions due to ambiguity — a context-passing failure between "agents." Today, edit decision-making and application are done by a single model in one action.

## On Multi-Agent Futures

Yan is optimistic about long-term multi-agent collaboration but argues it requires a solved cross-agent context-passing problem that doesn't exist yet in 2025. The efficient human-to-human communication that makes team collaboration work is nontrivial intelligence. The expectation is that this will come "for free" as single-threaded agents get better at communicating with humans — and when it does, it will unlock genuine parallelism.

## Related Topics
- [[llm-agents]] — Foundational agent patterns; the ReAct / tool-use substrate these principles apply to
- [[agent-harness-engineering]] — Production harness engineering
- [[learning-the-bitter-lesson]] — A companion piece by Lance Martin on how this principle played out in open-deep-research development
- [[context-engineering]] — The discipline Yan is defining here

## Sources
- [Don't Build Multi-Agents — Walden Yan, Cognition (June 12, 2025)](https://cognition.ai/blog/dont-build-multi-agents)
