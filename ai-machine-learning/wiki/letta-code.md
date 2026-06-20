# Letta Code: Memory-First Coding Agent

Letta Code (December 2025) is a coding agent built around persistent, cross-session memory. The core thesis: coding agents accumulate valuable experience (user preferences, codebase knowledge, tool outcomes) that current session-isolated agents throw away. Letta Code is designed to retain and compound that experience.

Letta Code is also the #1 model-agnostic, open-source harness on [Terminal-Bench](https://www.tbench.ai/), achieving performance comparable to provider-specific harnesses (Claude Code, Gemini CLI, Codex CLI) across model providers.

Install: `npm install -g @letta-ai/letta-code`

## Core Differentiator: Persistent Agents

Unlike Claude Code, Cursor, or Copilot — which start fresh each session — Letta Code ties each session to a persisted agent that accumulates memory over time. The agent is model-agnostic and portable: you can switch the underlying LLM without losing the accumulated context.

Memory is managed through three mechanisms:
- **Memory blocks:** structured sections of the system prompt that the agent can rewrite as it learns (project context, user preferences, codebase patterns)
- **Agentic context engineering:** the agent manages its own context window composition, deciding what to surface from long-term storage
- **Skill learning:** learned procedures stored as `.md` files, usable by the same agent or others

## Key Features

**`/init` command.** Triggers deep research on the local codebase. The agent reads code, forms memories, and rewrites its system prompt to reflect what it learned. Run once at project start or when switching to a new codebase.

**`/remember` command.** Explicitly triggers reflection and memory consolidation. Also fires automatically as the agent works.

**Skill Learning.** After completing a complex or repeated task, the agent can be triggered to encode its experience as a reusable skill. Examples from Letta's own team: generating DB migrations on schema changes, creating PostHog dashboards via CLI, API change best practices. Skills are `.md` files — versioned in git, shareable across agents.

**`/search` command.** Search through past conversations (vector, full-text, or hybrid) via the Letta API. Find which session you discussed a problem or what another agent decided.

**Persisted state across models.** Unlike harnesses built for a specific provider, Letta Code's memory persists across model upgrades. An agent that learned your codebase on Claude Sonnet 4.5 continues learning when you switch to Opus 4.8.

## Why This Matters

Current coding agents are stateless by design. Each session, you re-explain the project, re-establish preferences, re-discover the same patterns. This is a compounding loss — the agent's "experienced" state resets to zero every time. Letta Code's bet is that statefulness is the next major axis of coding agent improvement after raw capability.

The Terminal-Bench results support this as a harness story independent of memory: Letta Code's baseline (without memory) outperforms all other model-agnostic OSS harnesses, and is comparable to provider-optimized harnesses. Memory is the additional layer on top.

## Context Repositories (February 2026 update)

Letta introduced Context Repositories: git-backed memory for coding agents. Memory is stored in the git repository itself, versioned alongside the code. This makes memory introspectable, diffable, and recoverable — and aligns memory management with the version control workflow developers already use.

## Connection to Broader Letta Research

Letta Code is the product surface of a research program focused on:
- **Continual learning in token space:** learning that persists across model generations, stored as text rather than weights
- **Sleep-time compute:** agents using idle time to consolidate memories and form new connections
- **Context-bench:** benchmarking how well LLMs manage agentic context (chaining file ops, tracing entity relationships, multi-step retrieval)
- **Recovery-bench:** measuring agents' ability to recover from errors and corrupted states

## Related Topics
- [[agent-harness-engineering]] — The broader engineering discipline; Letta Code is an existence proof of a memory-first harness design
- [[llm-agents]] — Foundational agent patterns
- [[dont-build-multi-agents]] — A contrasting perspective on agent architecture tradeoffs
- [[context-engineering]] — The core capability Letta Code is built around

## Sources
- [Letta Code: A Memory-First Coding Agent — Letta Blog (December 16, 2025)](https://www.letta.com/blog/letta-code)
- [Terminal-Bench leaderboard](https://www.tbench.ai/)
- [Letta documentation](https://docs.letta.com/letta-code)
