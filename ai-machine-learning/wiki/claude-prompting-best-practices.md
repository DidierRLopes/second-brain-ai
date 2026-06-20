# Claude Prompting Best Practices

Anthropic's official reference guide for prompt engineering with Claude's latest models (Opus 4.8, Opus 4.6, Sonnet 4.6, Haiku 4.5). This is the single authoritative source for how to steer Claude's behavior across clarity, output formatting, tool use, thinking/reasoning, and agentic systems.

## General Principles

**Be clear and direct.** Claude responds to explicit instructions. Treat Claude as a brilliant new employee who lacks context on your norms — the more precisely you explain what you want, the better the result. The golden rule: show your prompt to a colleague with minimal context; if they'd be confused, Claude will be too.

**Add context and motivation.** Explaining *why* an instruction matters helps Claude generalize it correctly. Don't just say what to do — say why it matters.

**Use examples (few-shot prompting).** Examples are among the most reliable steering tools. Use 3–5 examples, wrap them in `<example>` tags, make them relevant and diverse, and cover edge cases.

**Structure with XML tags.** XML tags help Claude parse complex prompts with mixed instructions, context, and inputs. Use consistent, descriptive tag names (`<instructions>`, `<context>`, `<input>`). Nest tags for hierarchical content.

**Give Claude a role.** A single sentence in the system prompt setting the role (e.g., "You are a helpful coding assistant specializing in Python") noticeably focuses behavior and tone.

**Long context prompting.** Put long documents at the top of the prompt, above the query. Ask Claude to quote relevant passages before answering. Structure multi-document inputs with `<document>` tags.

## Output and Formatting

Claude 4.x models default to a more direct, concise, and conversational style than earlier models. They may skip summaries after tool calls and write shorter answers on simple queries.

Key formatting controls:
- Tell Claude what to do (not what not to do): instead of "don't use markdown," say "write in flowing prose paragraphs."
- Use XML format indicators: "Write the prose sections in `<smoothly_flowing_prose_paragraphs>` tags."
- Match your prompt style to your desired output style — markdown-heavy prompts tend to produce markdown-heavy outputs.

**Prefilled responses removed.** Starting with Claude 4.6 models, prefilled assistant turns are no longer supported and return a 400 error. Migrate to explicit format instructions instead.

## Tool Use

Claude 4.x models are trained for precise instruction following and benefit from explicit guidance on when and how to use tools. If you say "can you suggest some changes," Claude may suggest rather than implement.

For proactive action, add:
```
<default_to_action>
By default, implement changes rather than only suggesting them. If the user's intent is unclear, infer the most useful likely action and proceed.
</default_to_action>
```

**Parallel tool calling.** Claude's latest models excel at parallel tool execution. To boost this to ~100%:
```
<use_parallel_tool_calls>
If you intend to call multiple tools and there are no dependencies between them, make all independent calls in parallel.
</use_parallel_tool_calls>
```

## Thinking and Reasoning

Claude 4.6 models use **adaptive thinking** (`thinking: {type: "adaptive"}`), where Claude dynamically decides when and how much to think. Extended thinking with `budget_tokens` is deprecated — prefer adaptive thinking + the `effort` parameter.

**Effort levels (Opus 4.8):**
- `max` — max effort, potential for overthinking
- `xhigh` — best for coding and agentic use cases
- `high` — minimum for most intelligence-sensitive use cases
- `medium` — cost-sensitive workloads
- `low` — short, scoped, latency-sensitive tasks

Claude Opus 4.8 has thinking *off* by default — you must explicitly set `thinking: {type: "adaptive"}`.

To control overthinking:
```
Thinking adds latency and should only be used when it will meaningfully improve answer quality — typically for problems requiring multi-step reasoning. When in doubt, respond directly.
```

## Agentic Systems

Claude 4.x models have strong long-horizon state tracking. Key patterns:

**Context awareness.** Claude 4.6/4.5 models track their remaining context window. If your harness auto-compacts context, tell Claude explicitly so it doesn't prematurely wrap up work.

**Multi-context window workflows.** Use the first context window to set up tests and scaffolding. Use structured formats (JSON) for state, unstructured text for progress notes. Git is a first-class state-tracking tool for Claude.

**Balancing autonomy and safety.** Claude Opus 4.6 may take hard-to-reverse actions without prompting. Add explicit confirmation requirements for destructive operations (deleting files, force-pushing, external service writes).

**Subagent orchestration.** Claude Opus 4.6 proactively spawns subagents. This can be over-eager — add guidance about when subagents are warranted vs. when a direct tool call suffices.

**Minimizing hallucinations:**
```
<investigate_before_answering>
Never speculate about code you have not opened. Read the file before answering.
</investigate_before_answering>
```

## Claude Opus 4.8-Specific Notes

- **More literal instruction following** — scope instructions explicitly ("Apply this to every section, not just the first one").
- **Fewer subagents by default** — steerable through prompting.
- **Strong design instincts** — defaults to a warm cream/serif aesthetic. Override with concrete specs or ask it to propose options first.
- **Code review:** Opus 4.8 may under-report lower-severity bugs if you've told it to focus on high-severity issues. Prompt for coverage in the finding step, filter in a separate step.

## Related Topics
- [[chain-of-thought-reasoning]] — The reasoning techniques behind thinking modes
- [[llm-agents]] — Applying these principles in agent harnesses
- [[agent-harness-engineering]] — Production engineering context
- [[context-engineering]] — The broader discipline these prompting patterns serve

## Sources
- [Prompting Best Practices — Anthropic Platform Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
