---
name: llm-knowledge-base
description: Scaffold a personal LLM knowledge base with folders, schema, and starter prompts. Includes agent-browser for automated web scraping into your knowledge base. Use when someone wants to build a knowledge base, personal wiki, or second brain with AI.
compatibility: Requires npm for agent-browser installation. Works with Claude Code, Cursor, Codex.
---

# LLM Knowledge Base Setup

Build a personal knowledge base that gets smarter every time you use it. Based on the system Karpathy described (4.3M views, 41K bookmarks) and refined with automated web scraping.

## Step 0: Check Dependencies

1. Check if agent-browser is installed:

```bash
which agent-browser
```

2. If the command is not found, install it:

```bash
npm install -g agent-browser && agent-browser install
```

3. Check if the agent-browser skill is installed:

```bash
test -f ~/.claude/skills/agent-browser/SKILL.md && echo "installed" || echo "not installed"
```

4. If not installed:

```bash
npx skills add vercel-labs/agent-browser --skill agent-browser -g -y
```

5. Print: "agent-browser ready. You can scrape any webpage into your knowledge base."

## Step 1: Determine Topic

**If `$ARGUMENTS` is provided:** Use it as the topic name. Slugify for the folder name (lowercase, replace spaces with hyphens, remove special characters). Skip to Step 2.

**If no arguments:** Ask the user these questions one at a time:

1. "What topic do you want to build a knowledge base for?"
2. "List 3-5 specific interests or subtopics within that area."
3. "Do you have any URLs you'd like to scrape into your knowledge base right now? (optional, you can always add more later)"

## Step 2: Scaffold the Knowledge Base

Create the folder structure in the current working directory:

```bash
mkdir -p {topic-slug}/raw {topic-slug}/wiki {topic-slug}/outputs
```

## Step 3: Generate the Schema File

Write `{topic-slug}/CLAUDE.md` with the following content. Replace `{topic}` with the user's topic and `{interests}` with their listed interests (or just "{topic} (general)" if one-shot mode):

```markdown
# Knowledge Base Schema

## What This Is
A personal knowledge base about {topic}.

## How It's Organized
- raw/ contains unprocessed source material. Never modify these files.
- wiki/ contains the organized wiki. AI maintains this entirely.
- outputs/ contains generated reports, answers, and analyses.

## Wiki Rules
- Every topic gets its own .md file in wiki/
- Every wiki file starts with a one-paragraph summary
- Link related topics to each other using [[topic-name]] format
- Maintain an INDEX.md in wiki/ that lists every topic with a one-line description
- When new raw sources are added, update the relevant wiki articles

## My Interests
- {interest 1}
- {interest 2}
- {interest 3}
```

## Step 4: Scrape Initial URLs (If Provided)

If the user provided URLs in the interactive flow:

For each URL:
1. Open the page: `agent-browser open {url}`
2. Wait for content: `agent-browser wait --load networkidle`
3. Extract the main content: `agent-browser get text "article"` (fall back to `agent-browser get text "main"`, then `agent-browser get text "body"`)
4. Get the page title: `agent-browser get title`
5. Save as `{topic-slug}/raw/{slugified-title}.md` with the extracted text
6. Close when done: `agent-browser close`

Print: "Scraped {n} sources into raw/."

## Step 5: Print Starter Prompts

Read `${CLAUDE_SKILL_DIR}/references/starter-prompts.md` and print the full contents so the user can see all four starter prompts.

## Step 6: Print Next Steps

Print:

```
Your knowledge base is ready at: {topic-slug}/

Next steps:
1. Drop your raw sources into raw/ - or scrape them with agent-browser.
   To scrape a webpage: tell me "scrape [URL] into raw/" and I'll handle it.
2. Once raw/ has sources, run the compile prompt to build your wiki.
3. Ask questions against your wiki - answers feed back in and make it smarter.
4. Run a health check once a month to catch errors.

All four prompts are in the starter-prompts reference above. Copy and use them anytime.
```
