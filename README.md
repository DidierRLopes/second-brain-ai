# llm-knowledge-base

A personal AI/ML "second brain" — raw sources get distilled into a cross-linked wiki, which an AI assistant keeps current as new material comes in. Loosely based on the workflow Karpathy described for compounding personal knowledge bases.

## Structure

- **`ai-machine-learning/raw/`** — unprocessed source material (articles, transcripts, notes). Never edited once added.
- **`ai-machine-learning/wiki/`** — the synthesized knowledge base. One `.md` file per topic, cross-linked with `[[wiki-style links]]`, indexed in `wiki/INDEX.md`. This is the part that's actually queried day to day.
- **`ai-machine-learning/interactive/`** — standalone React/JSX visualizations for a few concepts (attention, KV-cache growth, LoRA, quantization).
- **`papers/`** — ~73 arXiv papers organized into thematic folders, indexed in `PAPERS_INDEX.md`. **Not pushed to git** — see below.
- **`PROGRESS.md`** — topic-coverage tracker, including periodic gap-audit notes.
- **`references/starter-prompts.md`** — the prompts used to scrape sources, compile the wiki, query it, and run a monthly health check.
- **`.claude/skills/teach/`** — a Claude Code skill for personalized, mission-tied lessons drawn from this repo.

## Why `papers/` isn't pushed

It's ~210MB of PDFs, which is more than is worth carrying in git history. The folder stays local-only (`.gitignore`'d) — papers are organized on disk for personal reference, and every paper that matters is already cited (with arXiv ID and a link) in `PAPERS_INDEX.md` and the relevant `wiki/` articles. Cloning this repo gets you the full wiki and index, just not the PDF bytes.

## How it grows

1. Drop a source into `raw/`, or ask the assistant to scrape a URL into it.
2. Ask the assistant to compile/update the wiki from `raw/`.
3. Query the wiki directly, or ask for a gap audit (cross-check `raw/`, `papers/`, and recent web sources against `wiki/` coverage) — see `references/starter-prompts.md` for the exact prompts.
