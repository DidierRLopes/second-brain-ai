---
name: handoff
description: Compact the current conversation into a handoff document for another agent to pick up.
argument-hint: "What will the next session be used for?"
---

Write a handoff document summarising the current conversation so a fresh agent can continue the work. Save to the temporary directory of the user's OS - not this repo.

Include a "suggested skills" section in the document, naming which of this repo's skills (`teach`, `interactive-explainer`, or others in `.claude/skills/`) the next session should invoke, and why.

Do not duplicate content already captured in other artifacts specific to this repo: git commits/diffs, `PAPERS_INDEX.md`, wiki pages under `ai-machine-learning/wiki/`, or teaching `learning-records/`. Reference them by path or commit hash instead of re-describing their contents.

Redact any sensitive information, such as API keys, passwords, or personally identifiable information.

If the user passed arguments, treat them as a description of what the next session will focus on and tailor the doc accordingly.
