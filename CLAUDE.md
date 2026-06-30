# llm-knowledge-base

This repository is a personal knowledge base / teaching workspace.

## Answering questions grounded in this repo

**Quick questions** ("explain X", "compare X and Y", "what is X") — do NOT
invoke the `teach` skill and do NOT spawn subagents. Instead:

1. `Grep` for the topic across `ai-machine-learning/wiki/` and
   `ai-machine-learning/raw/` (one call, seconds).
2. `Read` the 2-4 most relevant files in parallel.
3. Answer inline in the conversation, citing the source files.

This is fast, cheap, and correct for one-off questions.

**Deep learning requests** ("teach me about X", "quiz me", "walk me through X
over multiple sessions") — invoke the `teach` skill, which produces HTML
lessons, tracks learning records, and manages a mission. Follow the skill's
philosophy: ground knowledge in trusted resources (never parametric guesses),
tie every lesson to the user's mission, and produce self-contained lessons.
Treat this directory as the teaching workspace described in
`.claude/skills/teach/SKILL.md`.

If the `teach` skill fails (e.g. `disable-model-invocation` error), fall back
to the quick-question flow above rather than spawning Explore subagents.

## Auto-ingest papers dropped into `papers/TO-BE-ORGANIZED/`

`papers/TO-BE-ORGANIZED/` is a drop zone for raw, unsorted PDFs. Whenever this
folder is non-empty — whether the user explicitly asks or you simply notice
files sitting there while doing other work in this repo — treat ingesting
them as a standing task. Don't wait to be asked again; run the full workflow
below proactively, the same way you would for any other paper already
correctly filed in `papers/`.

For each PDF in `papers/TO-BE-ORGANIZED/`:

1. **Identify it.** Extract the text (`pdftotext file.pdf -`) and read enough
   of the title page / abstract to determine the real title, authors, and
   arXiv ID. Never guess from parametric memory. If no arXiv ID is
   self-stamped in the PDF text (check the full document, not just page 1 —
   some papers only cite *other* arXiv papers in their bibliography), use
   `WebSearch` with the exact title to find it, and confirm the match by
   comparing the abstract text against the PDF's own abstract before trusting
   the result.
2. **Check for duplicates.** Search the rest of `papers/` (excluding
   `TO-BE-ORGANIZED/`) for the same arXiv ID. If a match exists, verify with
   `pdfinfo` (page count, and title metadata if present) that it's truly the
   same paper, not a coincidental ID collision — then delete the
   `TO-BE-ORGANIZED` copy rather than creating a duplicate entry.
3. **File it.** Rename to the repo convention `<Title> - <arXiv ID>.pdf` and
   move it into the most specific matching category subfolder under
   `papers/` (inspect sibling files in candidate subfolders first to judge
   topical fit, the same way you would when filing any new paper).
4. **Extract insights into the wiki.** Read the paper's full text (not just
   the abstract) and write grounded, specific paragraphs — real numbers,
   named techniques, concrete findings, not generic filler — into the most
   relevant existing page(s) under `ai-machine-learning/wiki/`. Prefer
   extending an existing section or adding a new H2 section over creating a
   new page. Cross-link via `[[wiki-link]]` only to pages that actually exist.
   Update that page's "Related Topics" and "Sources" sections.
5. **Cite correctly, everywhere.** Any mention of the paper — in wiki pages,
   in `PAPERS_INDEX.md`, in conversation — must link to the correct arXiv ID
   using the local PDF now that one exists:
   `[Title (arXiv ID)](../../papers/<category>/<subfolder>/<exact filename>.pdf)`
   in wiki pages, or `[Title](<papers/<category>/<subfolder>/<exact filename>.pdf>) — arXiv:<ID>`
   in `PAPERS_INDEX.md`. Never leave a newly-ingested paper as a bare URL —
   it now has a local PDF, so cite it as one.
6. **Update the indexes.** Add the new entry to `PAPERS_INDEX.md` (the right
   category section, plus the Quick Reference counts), and to
   `ai-machine-learning/wiki/INDEX.md` only if you created a brand-new wiki
   page.
7. **Verify before considering it done.** Confirm `papers/TO-BE-ORGANIZED/`
   is empty, all new `[[wiki-link]]`s resolve, all new local-PDF paths
   resolve on disk, and the arXiv ID in every new citation matches the ID
   embedded in the actual filename.

This workflow fully supersedes any earlier instruction to leave
`papers/TO-BE-ORGANIZED/` untouched — that folder exists specifically so new
papers can be dropped in and then automatically organized.
