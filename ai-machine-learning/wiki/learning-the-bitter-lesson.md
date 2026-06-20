# Learning the Bitter Lesson (AI Engineering)

A post by Lance Martin (LangChain, July 2025) applying Rich Sutton's [Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) to AI engineering — the craft of building applications on top of rapidly improving LLMs.

> The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin.
> — Rich Sutton

## The Bitter Lesson in AI Research

The Bitter Lesson has been learned repeatedly across chess, Go, speech, and vision. Hand-crafted domain knowledge (SIFT features, HOG descriptors, rule-based game heuristics) eventually gets outperformed by general methods + more compute + more data. The "structure" we impose — inductive biases about how we expect models to solve problems — limits their ability to leverage growing computation.

Hyung Won Chung (OpenAI) articulates the engineering implication:
> *Add structures needed for the given level of compute and data available. Remove them later, because these shortcuts will bottleneck further improvement.*

## The Bitter Lesson in AI Engineering

The same dynamic plays out in application development. Martin illustrates this with the history of [open-deep-research](https://github.com/langchain-ai/open_deep_research):

**Phase 1 — Adding Structure (early 2024):** Tool calling was unreliable, so Martin built a workflow: orchestrator LLM decomposes the request into report sections, workers research and write each section in parallel, results combined. The "structure" was: fixed decomposition strategy, no tool calls, parallelized writing. These were reasonable assumptions given the model capabilities of the time.

**Phase 2 — Bottlenecks (late 2024 → early 2025):** Tool calling improved significantly. MCP gained momentum. Agents became well-suited to research tasks. But the imposed structure prevented taking advantage of these gains: no MCP ecosystem access, rigid section decomposition inappropriate for all queries, disjoint parallel-written reports.

**Phase 3 — Removing Structure:** Martin moved to a multi-agent system with tools. But kept one structural artifact: each sub-agent still wrote its own section in parallel. This hit exactly the problem Walden Yan (Cognition) identified — sub-agents without shared context produce inconsistent, disjoint outputs.

**Phase 4 — Removing More Structure:** Writing moved to a final single-shot step after all research was complete. The system could now flexibly plan its research strategy, use multi-agent context gathering, and write the report coherently from the collected context. Result: 43.5% on Deep Research Bench (top 10 open-source), comparable to provider-specific harnesses.

## Key Lessons for AI Engineering

**1. Understand your application structure.** Make the LLM performance assumptions embedded in your design explicit. What were you working around? ("I avoided tool calling because it wasn't reliable in 2023.")

**2. Re-evaluate structure as models improve.** Reassess whether your assumptions still hold. Martin was "a bit slow" to re-evaluate as tool calling improved. Models catch up faster than most developers update their architectures.

**3. Make it easy to remove structure.** Heavy agent abstractions can make it *harder* to remove structure. Martin uses LangGraph but sticks to low-level primitives (nodes and edges) that are easy to reconfigure. Abstract frameworks can entrench the structure you're trying to strip away.

## The Common Failure Mode

We often fail to remove *all* the structure we added. Martin moved from workflow → multi-agent but kept parallel writing. The residual structure was the cause of the residual failure. This is Hyung's point: transitioning between paradigms doesn't automatically clear the old assumptions.

## Connection to Broader Principles

This connects directly to Walden Yan's [[dont-build-multi-agents]] analysis: the parallel-writing failure in open-deep-research is precisely the "conflicting implicit decisions" failure mode Yan describes. Martin cites Yan explicitly. The fix — sequential writing after parallel research — is the same solution: serialize the decision-making step.

It also connects to scaling law intuitions: "build things that don't quite work yet because models will catch up" (Jared Kaplan, Anthropic co-founder). This is the AI engineer's version of Sutton's lesson.

## Related Topics
- [[llm-agents]] — The agent patterns this analysis applies to
- [[dont-build-multi-agents]] — Companion piece on context engineering in multi-agent systems
- [[agent-harness-engineering]] — Production engineering perspective
- [[chain-of-thought-reasoning]] — The reasoning scaffolding that gets stripped away as models improve

## Sources
- [Learning the Bitter Lesson — Lance Martin (July 30, 2025)](https://rlancemartin.github.io/2025/07/30/bitter_lesson/)
- [The Bitter Lesson — Rich Sutton (2019)](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
