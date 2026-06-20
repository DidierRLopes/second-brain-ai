# Context Engineering

Context engineering is the discipline of building dynamic systems that assemble the right information, tools, and instructions into an LLM's context window so the task is *plausibly solvable* — as opposed to prompt engineering, which optimizes the wording of a single static string. Shopify CEO Tobi Lütke's framing (tweet, June 2025) is the most-cited definition: "the art of providing all the context for the task to be plausibly solvable by the LLM." Andrej Karpathy amplified the same shift around the same time, describing industrial-strength LLM apps as systems that do "the right thing" with context at each step of an agentic loop — filling the window with just the right information for the next step, not stuffing it with everything available.

## Why "Prompting" Stopped Being the Right Word

Prompt engineering treats the model's input as a single hand-tuned string: get the wording, the examples, and the instructions right, and you have a working system. That framing matches single-turn Q&A, but it breaks down for agents, because most of what an agent sees on any given turn isn't something a person typed — it's assembled by code: retrieved documents, prior tool outputs, conversation history, memory lookups. Context engineering names the bigger discipline: prompt engineering is one component of it (how you phrase the instructions slice), not a synonym for it.

## The Context Taxonomy

Philipp Schmid's synthesis (drawing on Karpathy, the LangChain team, Simon Willison, and Lance Martin) breaks "context" into the pieces a production system has to manage, each with its own engineering problem:

- **Instructions / System Prompt** — the baseline rules and persona. Static, but needs maintenance as the system's scope grows.
- **User Prompt** — the immediate request. The one slice classic prompt engineering actually optimizes.
- **State / History (short-term memory)** — the running conversation and the actions taken so far in the current session.
- **Long-Term Memory** — facts, preferences, and prior-session summaries that persist across conversations (see [[llm-agents]] for Mem0/Zep-style production memory layers).
- **Retrieved Information (RAG)** — knowledge pulled in from outside the model's parameters, fetched on demand rather than baked into the system prompt (see [[retrieval-augmented-generation]]).
- **Available Tools** — the tool/function definitions the model can call, and the descriptions that tell it when and how to call them.
- **Structured Output** — the schema the model's response is constrained to, which itself shapes how much of the context budget downstream code can parse reliably.

The engineering problem is deciding, on every single turn, which subset of these seven sources actually belongs in the window — and in what order, format, and amount — rather than assuming "more context" is strictly better.

## Cheap Demo vs. Magical Agent

Schmid's illustrative contrast: a "cheap demo" sends the user's raw message straight to the model with a generic system prompt. A "magical" agent, given the literal same user message and the literal same model, instead assembles: conversation history, retrieved documents relevant to the question, a user-profile/preferences memory lookup, the right tool definitions for the task at hand, and a calendar or environment check if relevant — then formats all of it cleanly before the call. The model is identical in both cases; the difference in perceived "magic" is entirely a context-assembly difference. This is the practical argument for treating context construction, not model choice, as the primary lever available to most application teams.

## Relationship to Agent Design

This is the same problem [[dont-build-multi-agents]] names as the agent engineer's actual job ("context engineering is the #1 job" for builders of coding/research agents), and the same one [[agent-harness-engineering]] documents at production scale — Cursor's "dynamic context" era is context engineering applied to a coding agent, and [[letta-code]]'s memory-block architecture is context engineering applied to long-term memory specifically. [[learning-the-bitter-lesson]]'s "add structure now, remove it as models improve" rule applies directly here too: how much manual context curation a system needs shrinks as models get better at deciding what they need on their own, but the underlying taxonomy above doesn't go away — it just gets pushed from explicit harness code into the model's own judgment.

## Related Topics

- [[dont-build-multi-agents]] — Names context engineering as the central job of agent builders; single-threaded agent design
- [[llm-agents]] — Memory architectures (Mem0, Zep) that fill the long-term-memory slice of the taxonomy
- [[retrieval-augmented-generation]] — The retrieved-information slice in depth
- [[agent-harness-engineering]] — Production-scale dynamic context strategy (Cursor's static-to-dynamic context arc)
- [[learning-the-bitter-lesson]] — Why manual context curation is a temporary scaffold, not a permanent architecture
- [[claude-prompting-best-practices]] — The instructions/user-prompt slice of the taxonomy, in detail
- [[letta-code]] — Memory-block architecture as applied long-term-memory context engineering

## Sources

- Tobi Lütke, tweet defining context engineering (June 2025): https://x.com/tobi/status/1935533422589399127
- Andrej Karpathy, tweet on context engineering in agentic loops (June 2025): https://x.com/karpathy/status/1937902205765607626
- Philipp Schmid, "The New Skill in AI is Not Prompting, It's Context Engineering" (June 30, 2025): https://www.philschmid.de/context-engineering
- LangChain blog, "The Rise of Context Engineering": https://blog.langchain.com/the-rise-of-context-engineering/
- humanlayer, 12-factor-agents, Factor 3 ("Own your context window"): https://github.com/humanlayer/12-factor-agents
- Simon Willison, "Context engineering" (June 27, 2025): https://simonwillison.net/2025/Jun/27/context-engineering/
- Lance Martin, "Context Engineering for Agents" (June 23, 2025): https://rlancemartin.github.io/2025/06/23/context_engineering/
