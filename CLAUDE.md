# llm-knowledge-base

This repository is a personal knowledge base / teaching workspace.

## Use the `teach` skill

A `teach` skill lives in `.claude/skills/teach/`. Use it whenever the user asks
to learn, be taught, be quizzed, or otherwise asks a question that can be
grounded in the material available in this repository (papers, references,
indexes, and notes here).

Concretely, invoke the `teach` skill when the user:

- asks you to teach, explain, walk through, or quiz them on a topic
- asks a question whose answer should be grounded in the files in this repo
  rather than your parametric knowledge

When teaching, follow the skill's philosophy: ground knowledge in trusted
resources (never parametric guesses), tie every lesson to the user's mission,
and produce self-contained lessons. Treat this directory as the teaching
workspace described in `.claude/skills/teach/SKILL.md`.
