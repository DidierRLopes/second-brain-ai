# Agent Harness Engineering

The "harness" is the engineering scaffolding around an LLM that turns it into a working agent: the system prompt, the tool definitions, the context-window strategy, the error handling, the orchestration of multiple agents and roles, and the evaluation infrastructure that measures whether changes help. As of 2026 the dominant view from production agent teams is that, given two competing systems with the same model, the difference between them is the harness — and most of the harness's quality comes from prompts, dynamic context, and error handling rather than novel architectures. Cursor's two posts (Wilson Lin's "Scaling long-running autonomous coding," Jan 2026; Heule & Katz's "Continually improving our agent harness," Apr 2026) are the clearest public account of what production-grade harness engineering looks like.

## Why the Harness Is Now the Product

Cursor's framing: when a new frontier model arrives, weeks of harness customization make the same model "noticeably faster, smarter, and more efficient" inside the tuned harness. The model is the engine; the harness is everything that turns engine output into useful behavior — tool format the model was trained on, prompt style that fits its instruction-following habits, error taxonomies that prevent context rot, orchestration that lets it operate over weeks without drift. The same model in a generic harness vs. a tuned one is effectively a different agent.

This generalizes the [[llm-agents|LLM Agent]] = LLM + Memory + Planning + Tool Use framing: the `+` is the harness, and that's where most of the engineering work lives.

## Context Window Strategy

The historical arc Cursor describes:

- **Late 2024 / early agents:** weak self-context-selection from models. Heavy guardrails (lint/type errors surfaced after every edit, file-read rewriting, hard caps on tools per turn). Pre-loaded static context (folder layouts, semantically matched snippets, compressed user-attached files).
- **2026:** most of that is gone. Static context is reduced to OS, git status, current/recently viewed files. Guardrails knocked down. Most engineering moved to **dynamic context**: letting the model fetch what it needs while it works.

The shape of the trend is "model gets smarter → harness gets thinner on guardrails, thicker on dynamic-fetch primitives."

## Training-Time Harnesses

The same harness idea now shows up inside post-training. [KIMIK2: Open Agentic Intelligence (2507.20534)](../../papers/07-applications/agents-swe/KIMIK2: Open Agentic Intelligence - 2507.20534.pdf) builds agentic capability by constructing the training harness itself: tool specs, simulated users, generated agents, task rubrics, tool simulators, real execution sandboxes, LLM judges, and RL environments with verifiable rewards.

This is not just data generation. It is harness engineering moved into the training loop:

- **Tool repository**: 3,000+ real MCP tools and 20,000+ synthetic tools, so tool-use learning is not overfit to a small API set.
- **Task rubrics**: every generated task includes success criteria and expected tool-use checkpoints, which lets the pipeline filter trajectories rather than blindly imitate them.
- **Hybrid execution**: simulated environments provide breadth; real sandboxes provide ground truth for code and software-engineering tasks.
- **Reward checks**: complex instruction following uses deterministic checks where possible, LLM-as-judge where necessary, and an extra hack-check layer for deceptive claims of compliance.
- **Long-horizon rollout infrastructure**: heavy environments are deployed as scalable services, many concurrent rollouts hide environment latency, and partial rollouts pause/resume long-tail tasks so one slow trajectory does not block the RL iteration.

The practical lesson: training environments are part of the harness surface. A weak verifier, leaky sandbox, or underspecified rubric teaches the model a bad operating policy just as surely as a bad deployment prompt does. This ties directly into [[safety-misalignment]]: reward-hack resistance is a harness requirement, not a cleanup step after training.

## Evaluating Harness Changes

Two layers of measurement:

### Offline

Public benchmarks plus internal eval suites (Cursor's is **CursorBench**). Fast, standardized, comparable across time, but only an approximation of real usage.

### Online (A/B harness experiments)

Deploy harness variants side-by-side on real users. Metrics:
- Easy/directional: latency, token efficiency, tool-call count, cache hit rate.
- **Keep Rate**: fraction of agent-proposed code edits that remain in the user's codebase after fixed intervals. Edits that get manually adjusted or trigger follow-up fix-it iterations show up as low Keep Rate.
- **LLM-judged user satisfaction**: a model reads the user's response to the agent's output. Moving on to the next feature → satisfied; pasting a stack trace → not.

Online testing sometimes overrides offline intuition. Example: Cursor tested a more expensive context-summarization model and found a negligible quality gain not worth the cost — shelved despite seeming promising.

## Single-Shot Build Comparison: GLM-5.2 vs. Claude Opus 4.8 (Southbridge)

Cursor's Keep Rate and LLM-judged satisfaction metrics assume iterative, in-production usage. Southbridge's offmute-v2 case study (May 2026) is the opposite regime: a **single-shot, no-iteration agentic build** — both GLM-5.2 and Claude Opus 4.8 got the same prompt file in Claude Code and built a diarized audio/video transcription tool once, with no back-and-forth. It's a useful complement precisely because it shows what evaluation looks like when you can't lean on online A/B data.

Both models, independently, chose the same core algorithm (Needleman-Wunsch sequence alignment) for aligning diarization segments with transcript text — treated as a signal that some solution patterns are now baked into frontier-model priors for well-known problem classes, not something either model derived from scratch. Where they diverged was failure mode, not raw competence: GLM-5.2 produced the more readable, convention-following codebase but had a **silent caching bug** that quietly returned stale results without erroring; Opus 4.8 had the better self-reported WER and a working ffmpeg.wasm browser-side pipeline but **crashed loudly on audio-only input**. After fixing both models' bugs (GLM also had a WER-inflating double-print bug), the two converged to nearly identical real accuracy (8.3% vs. 7.5% WER) — most of the apparent gap was a measurement artifact, not a capability gap. GLM's build still shipped as the primary release, implicitly weighting "doesn't fail silently" and "maintainable code" above raw self-reported benchmark numbers.

Two takeaways generalize beyond this one comparison:

- **Failure modes matter more than failure counts.** A loud, rare crash (Opus) is operationally safer than a silent, rare wrong-answer bug (GLM), even when a naive bug-count or benchmark-score comparison would treat them as equivalent. This is the single-shot analog of Cursor's `UnexpectedEnvironment`/`InvalidArguments` error taxonomy above — the *kind* of failure is the signal, not just its frequency.
- **"I validated one path and generalized" is a shared root cause.** Both models' bugs trace to testing one code path, confirming it worked, then generalizing that confidence to untested paths (GLM's caching logic, Opus's audio-only branch). Worth watching for in any single-shot agentic build, independent of which model is doing the building — it's a property of how coding agents self-verify, not a model-specific weakness.

See `raw/southbridge-offmute-v2-glm-vs-opus.md` for the full WER and token-cost tables.

## Tool-Call Errors and Context Rot

Tool calls are the broadest bug surface in any agent. Errors stay in context even when the agent self-corrects, wasting tokens and causing **context rot** — accumulated mistakes degrade subsequent decisions, sometimes blocking the agent or pushing it off the rails entirely.

Cursor's error taxonomy:

- **Unknown errors:** always harness bugs. Alert whenever the unknown error rate exceeds a fixed threshold for any tool.
- **Expected errors:** classified by cause.
  - `InvalidArguments` — model mistakes
  - `UnexpectedEnvironment` — contradictions in the context window
  - `ProviderError` — vendor outages from tools like `GenerateImage` or `WebSearch`
  - `UserAborted`, `Timeout` — round out the taxonomy
- **Anomaly detection** on expected errors uses per-tool, per-model baselines. A grep timeout might be a tool perf bug or an inefficient model query against a huge codebase; baselines per (tool, model) make this distinguishable.

Cursor pairs this with weekly Cloud Agent Automations that read logs, surface new/spiking issues, and create tickets — they call it an "automated software factory" for the harness, and credit it with driving unexpected tool-call errors down by an order of magnitude over one focused sprint.

## Per-Model Customization

Cursor's harness abstractions are model-agnostic but heavily customized per model:

- **Edit format.** OpenAI models are trained on patch-based edit tools; Anthropic models on string replacement. Either model *can* use either format, but using the unfamiliar one costs reasoning tokens and produces more mistakes. Provision each model with the format it saw at training time.
- **Prompt style.** OpenAI models are literal and precise in instruction-following; Claude is more intuitive and tolerant to imprecise instructions. Custom prompts per provider and even per model version.
- **Quirk mitigation.** One model developed "context anxiety" — as the context window filled, it began refusing work, hedging that the task seemed too big. Cursor reduced this via prompt adjustments rather than waiting for a model fix.

New-model onboarding flow: start from the closest existing model's harness, run offline evals to find where it gets confused, have the team use it daily and surface problems, tweak the harness in response, iterate.

## Mid-Chat Model Switching

Particularly tricky: different models have different prompts, tool shapes, and behaviors, so the conversation history is out of distribution for whichever model takes over.

- Auto-switch the harness when the user switches models.
- Inject **takeover instructions** telling the new model it's mid-chat from another model, and steering it away from tools that appear in history but aren't in its own toolset.
- Caches are provider- and model-specific: switching = cache miss = slower, more expensive first turn. Conversation summarization at switch time helps but can drop detail on deep tasks. Default guidance: don't switch unless there's a reason.
- **Subagent** as the cleaner sidestep: starts a fresh context window with a chosen model. Cursor exposes this directly to users.

## Multi-Agent Orchestration: Planners + Workers + Judges

Wilson Lin's "Scaling long-running autonomous coding" post is the case study. The team ran agents autonomously for weeks on projects that take human teams months — most prominently building a web browser from scratch (>1M LOC, ~1,000 files, ~1 week, hundreds of concurrent workers).

### Iterations on coordination

1. **Flat self-coordination via shared file + locks.** Failed: agents held locks too long or forgot to release. Twenty agents → effective throughput of two or three. Brittle when agents failed mid-lock.
2. **Optimistic concurrency control.** Simpler and more robust, but with no hierarchy agents became risk-averse — small safe changes only, no agent owning hard end-to-end work, churn without progress.
3. **Planner + Worker + Judge.** What worked.
   - **Planners** explore the codebase continuously and create tasks. Can spawn sub-planners for specific areas → planning is itself parallel and recursive.
   - **Workers** are stateless. Pick up a task, focus only on it, run tests, push, done. Don't worry about the big picture.
   - **Judge agent** at end of each cycle decides whether to continue. Next iteration starts fresh — periodic resets are how the system fights drift and tunnel vision.

### Concrete results

- **Browser from scratch** (`fastrender`): ~1 week, >1M LOC, ~1,000 files, hundreds of concurrent workers pushing the same branch with minimal conflicts.
- **Solid → React migration** of the Cursor codebase: ~3 weeks, +266K / −193K edits, passed CI.
- **Video rendering rewrite**: 25× faster Rust version with smooth zoom/pan and motion blur, merged.
- Other ongoing: Java LSP (550K LoC), Windows 7 emulator (1.2M LoC), Excel (1.6M LoC).

### Key design lessons

- **Removing complexity beat adding it.** An "integrator" role for QC and conflict resolution created bottlenecks rather than solving them; workers were already capable of handling conflicts. Killing that role was a net win.
- **Right-sized structure.** Too little → conflict, duplication, drift. Too much → fragility. The sweet spot is in the middle.
- **Different models for different roles.** GPT-5.2 is a better *planner* than GPT-5.1-Codex, even though Codex is the coding-tuned model. Use the right model per role rather than one universal model.
- **Model fit on long-running work.** GPT-5.2 keeps focus, follows instructions, avoids drift, and implements precisely on extended autonomous tasks. Opus 4.5 stops earlier, takes shortcuts when convenient, yields control quickly.
- **Prompts > harness > models.** A surprising amount of behavior comes down to prompting. Coordination, anti-pathology, focus over long horizons — all needed extensive prompt experimentation.

## Practical Heuristics

Distilled from both posts:

- **Match each model to its training-time tool format.** Don't make Claude eat patches; don't make GPT eat string-replace.
- **Per-(tool, model) error baselines.** Pooled baselines hide model-specific regressions.
- **Treat unknown errors as bugs unconditionally; classify expected errors.** This is the only way to keep an alert system useful as the harness scales.
- **Online A/B with Keep Rate as the load-bearing metric.** Latency and token counts are directional; Keep Rate measures whether work survived contact with the user.
- **Don't switch models mid-conversation if you can avoid it.** Use a fresh-context subagent instead.
- **For long-horizon work, separate planning from execution.** Stateless workers + a periodic judge beat both flat peer coordination and a single super-agent.
- **Remove a role before adding one.** Integrator-style coordinator roles are usually a tax, not a leverage point.
- **Train with the same discipline you deploy with.** Kimi K2's pipeline is a good template: broad tool coverage, explicit rubrics, real execution for code, hack checks around verifiers, and rollout infrastructure that treats environment latency as a first-class systems problem.

## The Harness as Hidden Technical Debt (Lee, 2026)

Han Lee's "Hidden Technical Debt of AI Systems: Agent Harness" (May 2026) frames the harness as the agentic equivalent of Sculley et al.'s 2015 ML systems diagram: the model is the small box, and the harness is the much larger box where technical debt accumulates. The post extends the Sculley analogy and is the third in a series (RL environments → agent runtime → agent harness).

### The Harness Is the OS

A useful analogy: the model is the CPU; the harness is the operating system. The OS provides interrupts and interfaces to the outside world, manages processes and threads, and manages memory so the application sees the illusion of infinite resources. The harness does the same for the agent: it exposes tools, manages context (memory), provides rollout structure, and abstracts the runtime details away from the model.

A harness consists of: system prompt + persona, tool surface + schemas, rollout protocol (single-turn, ReAct, plan-and-execute, deep-research, multi-agent), context manager, memory (short/mid/long-term), sub-agent topology, guardrails and gates, verifiers/judges, and observability.

### First-Party vs. Third-Party Harnesses

When a lab post-trains a model, it does so inside *its* harness — its tool schemas, rollout protocol, system-prompt conventions, context layout, stop conditions. The policy is shaped against that surface. The capability lives in the weights, but how those weights get invoked is part of the training distribution.

**First-party harness advantage:** The same model in a generic third-party harness vs. the first-party harness is effectively a different agent. Empirical benchmark data (posttrain bench): GPT-5.1 Codex 20.2% (first-party) vs 7.7% (third-party); Gemini 3 Pro 18.3% vs 14.9%; Claude Opus 4.5 17.1% vs 17.3%.

**The first-party advantage is not a law.** A third-party harness that invests heavily in a dimension the first-party underweights can win. Letta Code beats Claude Code on Opus 4.5 (59.1% vs 41.6%) on benchmarks that reward durable memory — Claude Code is intentionally thin on memory, Letta is built around a memory substrate. On GPT 5.1 Codex and Gemini 3, Letta lands within a few points. The takeaway: the harness is load-bearing, and a third-party harness with deliberate investment in the right axis can outperform a first-party one that neglects it.

### Training vs. Production Harness Asymmetry

The most under-discussed property of the harness is that the production and training harnesses are not the same artifact, and should not be.

| Dimension | Training / Research Harness | Production Harness |
|-----------|----------------------------|-------------------|
| **Action space** | Maximal — let the model try anything useful | Minimal — explicit allowlist, deny by default |
| **Tools** | Raw, low-level, easy to extend | Wrapped, scoped, versioned, schema-validated |
| **Failures** | Welcome — failure is signal for the optimizer | Suppressed — fail closed, retry, page someone |
| **Network** | Often offline/recorded for determinism | Live, with strict egress policies |
| **Guardrails** | KL caps, reward shaping, curriculum gates | RBAC, JWT scoping, action gates, output filters |
| **State** | Forkable, snapshottable, replayable | Durable, per-user, auditable |
| **What "good" means** | Policy improves on held-out distribution | User's task succeeds without an incident |

**Critical mistake 1 — Over-shackling in training:** Copying the production allowlist into training "for safety." The model never learns to recover from tool errors, choose between tools, or decide when to stop, because the production harness was doing that thinking. The policy never grows those capabilities.

**Critical mistake 2 — Under-fencing in production:** Shipping the open research harness against real systems. The agent is capable and unaligned at the surface. Prompt injection lands. Sensitive operations execute. Post-hoc filters pile up — software engineering substituting for alignment that should have happened in training.

**The bridge** between them is an evaluation harness: mirrors production tightly enough to catch behavioral regressions, run by the same team that owns production prompts and tools. Its job is to ensure research and production don't diverge in behavior between harness releases and model updates.

### Alignment From the Inside Out

Each production guardrail (allowlist, action gate, output filter) is a fence around behavior that hasn't been shaped. A sufficiently capable agent will eventually find a configuration where the fence is wrong. The fence also has a fixed cleverness budget while the model's intelligence grows.

Training does something different: the model explores the action space, observes what behaviors emerge, and those behaviors get shaped with rewards. The fence moves from outside the model to inside. This is the only kind of alignment that scales with capability.

Implication: teams that invest in training harnesses are buying down their production-harness debt, two or three model generations out.

### The Bitter Lesson for Agent Harnesses

Rich Sutton's Bitter Lesson (2019): methods that leverage general computation always eventually beat methods encoding human cleverness. The same is happening to agent harnesses in real time:

- **No-code workflow builders dissolving.** 2024 canvas tools (n8n and peers) delivered repeatability of steps, not quality of output. By 2026, a single long-horizon agent does what dozens of workflow nodes tried to assemble, with the loop running inside the model.
- **Tool wrappers dissolving.** In 2024 teams wrapped raw APIs in cleaner LLM-friendly schemas. By late 2025, the model can read OpenAPI specs — and write the helper it needs when it's missing (Browser Use's ~600-line harness: `helpers.py` the agent can edit, a `daemon.py`, `SKILL.md`, `run.py`).
- **Planner-executor scaffolds dissolving.** In 2024: explicit planner LLM call → executor LLM call → reflection call. By 2026: a single agentic-thinking model interleaves planning, action, and reflection inside its own trace.
- **Memory layers dissolving.** Plain text (`progress.md`, `git log`) beats elaborate vector store stacks because the model already knows how to read, write, and reason about plain text.
- **Multi-agent topologies dissolving.** Elaborate orchestrator-router-judge-critic graphs from 2024 are becoming overhead as context constraints ease (see: Cognition's "Don't build multi-agents").

Hyung Won Chung's rule: *add structure for the level of compute you have, then remove it — the structure becomes the bottleneck for the next level of compute.* This applies to harnesses, not just model architectures.

### Thin Harness, Fat Skills

Three nested optimization surfaces:

| Surface | Cost to change | Iteration cadence | Owned by |
|---------|---------------|-------------------|----------|
| **Skills / prompts** | Cheap — text edits | Hourly to daily | Product builders |
| **Harness** | Medium — ships with the binary | Daily to weekly | Research / Applied AI engineers |
| **Model** | Expensive — post-training compute | Quarterly, lab-side | Research engineers |

The design principle: **push work onto the cheapest optimization surface.** Keep the harness thin, with a small, deliberately under-specified set of primitives mirroring what the lab post-trained against. Put domain expertise into skills, where iteration is fast, the artifact is human-readable, and the cost of being wrong is a text edit.

Auto-harness optimization (Meta-Harness, AutoHarness style outer-loop optimizers) is a legitimate research direction but a production debt risk: the optimized harness is overfit to its training distribution, widens the train/prod gap, and produces structure with no human-readable audit trail. A thin harness with fat skills loses a few benchmark points on niche tasks and wins everywhere the distribution shifts.

**Design test:** For each piece of harness, ask — when this piece becomes obsolete next quarter, how hard is it to remove? If "an hour," you have an option. If "a week," you have debt.

## Related Topics

- [[llm-agents]] — General agent architectures (ReAct, Toolformer, planning + memory + tools)
- [[cloud-agent-infrastructure]] — The runtime layer underneath the harness: VM isolation, snapshotting, sandbox primitives, and Stripe Minions
- [[ai-rd-automation]] — Letting agents run the *research* loop, not just the engineering loop; the same "research intuition" gap shows up in modelcrafting
- [[prompt-optimization]] — Prompt-level improvements as the dominant lever in production agents (matches "prompts > harness > models")
- [[reasoning-models]] — The frontier models (GPT-5.2, Opus 4.5) that production harnesses tune around
- [[swe-agent-benchmarks]] — Executable SWE-agent environments and hybrid verifiers for training/evaluating coding agents
- [[llm-evaluation]] — Closed benchmarks, open-world evals, and trace-analysis norms for agent systems
- [[safety-misalignment]] — Reward hacking, context-dependent misalignment, and agentic safety audits
- [[model-report-case-studies]] — Named-model lessons; the offmute-v2 comparison is a single head-to-head data point in the same spirit

## Sources

- "Scaling long-running autonomous coding" — Wilson Lin, Cursor (Jan 14, 2026): https://cursor.com/blog/scaling-agents
- "Continually improving our agent harness" — Stefan Heule & Jediah Katz, Cursor (Apr 30, 2026): https://cursor.com/blog/continually-improving-agent-harness
- "How we compare model quality in Cursor" — CursorBench background: https://cursor.com/blog/cursorbench
- Companion code: github.com/wilsonzlin/fastrender
- [KIMIK2: Open Agentic Intelligence (2507.20534)](../../papers/07-applications/agents-swe/KIMIK2: Open Agentic Intelligence - 2507.20534.pdf) — agentic data synthesis, verifiable-reward gyms, hack-check verification, and long-horizon rollout infrastructure.
- "Hidden Technical Debt of AI Systems: Agent Harness" — Han Lee (May 8, 2026): https://leehanchung.github.io/blogs/2026/05/08/hidden-technical-debt-agent-harness/
- "Hidden Technical Debt of AI Systems: Agent Runtime" — Han Lee (Apr 24, 2026): https://leehanchung.github.io/blogs/2026/04/24/hidden-technical-debt-agent-runtime/
- "The Bitter Lesson of Agent Harnesses" — Gregor Zunic, Browser Use (2026): https://browser-use.com/posts/bitter-lesson-agent-harnesses
- "Thin Harness, Fat Skills" — gbrain ethos: https://github.com/garrytan/gbrain/blob/master/docs/ethos/THIN_HARNESS_FAT_SKILLS.md
- [offmute v2: GLM-5.2 vs. Claude Opus 4.8 — Southbridge (May 2026)](https://www.southbridge.ai/blog/offmute-v2-glm-vs-opus) — single-shot agentic build comparison; silent-vs-loud failure modes, WER reconciliation, "validated one path and generalized." See `raw/southbridge-offmute-v2-glm-vs-opus.md`.
