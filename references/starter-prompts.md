# Starter Prompts for Your Knowledge Base

## 1. Scrape a Web Source (Use Anytime)

To add a web article or page to your knowledge base:

> Scrape [URL] into raw/. Use agent-browser to open the page, extract the
> main content, and save it as a markdown file in raw/ with a descriptive
> filename.

agent-browser handles JavaScript-heavy sites, pages behind logins, and
content that requires scrolling or clicking "load more." It uses 82% fewer
tokens than alternatives like Playwright.

## 2. Compile Your Wiki (Run This First)

After adding sources to raw/, run this prompt:

> Read everything in raw/. Then compile a wiki in wiki/ following the rules
> in CLAUDE.md. Create an INDEX.md first, then create one .md file per major
> topic. Link related topics. Summarize every source.

## 3. Ask Questions Against Your Wiki

Once wiki/ has 10+ articles, try these:

> Based on everything in wiki/, what are the three biggest gaps in my
> understanding of [topic]?

> Compare what source A says about [concept] vs what source B says.
> Where do they disagree?

> Write me a 500-word briefing on [topic] using only what's in this
> knowledge base.

Save answers back into outputs/ or update wiki articles directly.
This is the compounding loop - every query makes the system smarter.

## 4. Monthly Health Check

Run this once a month to catch errors:

> Review the entire wiki/ directory. Flag any contradictions between
> articles. Find topics mentioned but never explained. List any claims
> that aren't backed by a source in raw/. Suggest 3 new articles that
> would fill gaps.
