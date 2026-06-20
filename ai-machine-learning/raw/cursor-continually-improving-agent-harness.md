# Continually Improving Our Agent Harness (Cursor)

**Source:** https://cursor.com/blog/continually-improving-agent-harness
**Authors:** Stefan Heule & Jediah Katz
**Published:** April 30, 2026
**Filed under:** Research

## Premise

Cursor builds the agent harness the way they'd build any ambitious software product: vision-driven, with hypotheses tested via offline evals and online A/B experiments. Most progress comes from "obsessively stacking small optimizations" rather than step-change improvements. When new models arrive, weeks are spent customizing the harness to that model's strengths and quirks until the model in the tuned harness is noticeably faster, smarter, and more efficient than the same model in a generic harness.

## Evolving the Context Window

The context window is the central object: system prompt + tool descriptions + conversation state + user request.

- **Late 2024 (first coding agent):** models were much worse at choosing their own context. Cursor invested heavily in **guardrails** — surfacing lint and type errors after every edit, rewriting file reads when the agent requested too few lines, capping the maximum number of tools per turn. They also pre-loaded substantial **static context**: folder layout, semantically matched code snippets, compressed versions of user-attached files.
- **Now:** mostly gone. Static context is reduced to OS, git status, current/recently viewed files. Guardrails are knocked down. Most of the work is on **dynamic context** — letting the agent fetch what it needs while it works (covered in their earlier "dynamic context discovery" post).

## Two Ways of Assessing Harness Changes

### Offline: CursorBench + public benchmarks

Public benchmarks plus their internal eval suite **CursorBench** give a fast, standardized read on quality and let them compare across time. Best benchmarks still only approximate real usage.

### Online: side-by-side A/B harness experiments

Two or more harness variants deployed on real usage. Metrics:

- **Easy/directional:** latency, token efficiency, tool call count, cache hit rate.
- **Keep Rate** of agent-generated code: for a given set of agent-proposed changes, what fraction remain in the user's codebase after fixed time intervals. Manual adjustments or follow-up "fix this" iterations indicate the agent's initial response was lower-quality.
- **LLM-judged user satisfaction:** a model reads the user's response to the agent's initial output. Moving on to the next feature → satisfied. Pasting a stack trace → not satisfied.

Online tests sometimes shelve promising-looking ideas. Example: a more expensive context-summarization model tested negligible quality gain — not worth the cost.

## Tracking and Repairing Degradations

As the harness gets more complex (more models, more capabilities), bug surface area grows.

### Tool call errors are the broadest bug surface

Tool call errors can be extremely harmful to a session. Even when the agent self-corrects, errors stay in context, waste tokens, and cause **context rot** — accumulated mistakes degrade subsequent decisions, sometimes blocking the agent or sending it off the rails.

### Error taxonomy

- **Unknown errors** are always bugs in the harness. Alert whenever the unknown error rate for any tool exceeds a fixed threshold.
- **Expected errors** are classified by cause:
  - `InvalidArguments` — model mistakes
  - `UnexpectedEnvironment` — contradictions in the context window
  - `ProviderError` — vendor outages from tools like `GenerateImage` or `WebSearch`
  - `UserAborted`, `Timeout` — others that round out the taxonomy

### Anomaly detection on expected errors

Whether an expected error spike is a harness bug or just expected behavior is ambiguous (e.g., a grep timeout could be a tool-perf bug or an inefficient model query against a huge codebase). Solution: **per-tool, per-model anomaly baselines.** Alerts fire when the rate significantly exceeds the baseline for that combination.

### Automated triage

A weekly Cloud Agent Automation runs a skill that searches Cursor's logs for new or recently spiked issues, then creates or updates tickets in a backlog with an investigation. Cloud Agents can be triggered directly from Linear to kick off fixes for many issues at once. Cursor calls this an "automated software factory" for the harness. Across one focused sprint they drove unexpected tool call errors down by an order of magnitude.

## Customizing the Harness for Different Models

All harness abstractions are model-agnostic but heavily customized per model.

- **Edit format:** OpenAI models are trained on patch-based edit tools; Anthropic models are trained on string replacement. Either model can use either, but the unfamiliar format costs reasoning tokens and produces more mistakes. Cursor provisions each model with the format it saw in training.
- **Prompt style:** custom prompts per provider and per model version. OpenAI models tend to be literal and precise in instruction-following; Claude is more intuitive and tolerant to imprecise instructions.

### New-model onboarding flow

When Cursor gets early access to a new model:
1. Start from the closest existing model's harness.
2. Run offline evals to find where the model gets confused.
3. Have the team use it and surface problems.
4. Tweak the harness in response.
5. Iterate until model+harness is shippable.

### Quirks the harness mitigates

One model developed what Cursor came to call **"context anxiety"**: as the context window filled up, it would refuse work and hedge that the task seemed too big. Reduced via prompt adjustments.

## Mid-Chat Model Switching

Especially tricky because different models have different behaviors, prompts, and tool shapes.

- **Auto-switch the harness.** When the user switches models, Cursor switches to that model's customized prompts and tools.
- **OOD conversation history.** The new model is now applying its tools to a conversation history produced by a different model — out of distribution from what it was trained on.
- **Mitigation: takeover instructions.** Custom instructions tell the model when it's taking over mid-chat from another model, and steer it away from calling tools that appear in the conversation history but aren't part of its own tool set.
- **Cache penalty.** Caches are provider- and model-specific, so switching means a cache miss → slower, more expensive first turn. Conversation summarization at switch time helps but can drop important detail on deep tasks. Default recommendation: stay with one model unless there's a reason to switch.
- **Sidestep:** use a **subagent**, which starts from a fresh context window. Cursor recently added the ability for users to directly request a subagent run with a particular model.

## Forward View: The Harness Is the Orchestrator

The future of AI-assisted software engineering will be **multi-agent**: not every subtask through a single agent, but specialized delegates — a planner, a fast-edit agent, a debugger — each scoped to what it does best. Making that work is fundamentally a harness problem: the system needs to know which agent to dispatch, how to frame the task for that agent's strengths, and how to stitch results into a coherent workflow.

The orchestration lives in the harness, not in any single agent. Harness engineering has always been important; it's only getting more critical.
