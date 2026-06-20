# AI R&D Automation

The use of LLM agents to autonomously perform machine-learning research and engineering work — data curation, training-pipeline construction, reward design, evaluation — that has historically required human researchers. Thoughtful Lab calls the broader vision "modelcrafting": every person and organization shaping their own models end-to-end, with agents doing the hard parts (deciding what to improve, how to measure it, which experiments are worth running). The current evidence as of mid-2026 is that frontier coding agents have the technical capability to write training pipelines but lack the **research intuition** — small habits like sanity-checking outputs, designing curricula, and validating evals against held-out data — that determines whether the loop actually produces a better model.

## Why This Matters

[[alignment-methods|Post-training]] is the expensive, human-intensive last mile of LLM development. Automating it would compound improvements (better agent → better post-training → better agent), reduce the human-researcher bottleneck on shipping new capabilities, and make custom modelcrafting accessible to teams that can't afford a full ML org. The open question is whether agents can run the *research loop*, not just the *training loop* — and recent benchmarks say "not yet."

## FrogsGame-Posttraining (Thoughtful Lab, April 2026)

Mersad Abbasi at Thoughtful Lab built a posttraining task on top of Proximal's FrontierSWE. The premise: give a frontier coding agent the same base model, training API, and time budget a human researcher would have. Can it run modelcrafting end-to-end?

### The Task

Place N frogs on an N×N grid such that no two share a row, column, diagonal, or color region. Trivially solvable with backtracking, easy to grade, within reach of any frontier reasoner. But solving the puzzle and *teaching another model to solve* it are different problems. That gap is what the task measures.

Two variants:
- **Multi-turn:** trained model solves boards via iterative tool calls. Almost every agent failed — failures conflate reasoning quality with format compliance.
- **Single-turn:** model outputs the full placement as a JSON object in one shot. Used in the published results because it isolates reasoning from format.

### Setup

- **Agent:** Claude 4.6 Opus or GPT-5.4
- **Base model:** Qwen3-8B
- **Training API:** Tinker (Thinking Machines Lab) — sandboxed, no local training allowed, no pretrained-weight downloads
- **Time:** 8 or 20 hours
- **Eval:** 500 unseen boards across four difficulty tiers (N=6 through N=13); held-out from the agent
- The agent gets task instructions, a fixed `prepare.py` (game engine + eval harness, hashed to detect tampering), and Tinker API docs. Everything else — data generation, reward design, training loop, evals — goes into `train.py` and is the agent's responsibility.

### Headline Result

Only **4 of 20** single-turn agents reach >25% pass@4. The rest hover near zero. None of the hinted runs improved over the base model in absolute terms; the best gain was +4.8% pass@4.

## Common Failure Modes

These are the recurring patterns across trials, even after the team gave agents a hinted "playbook" addressing the most obvious mistakes.

### 1. No sanity checks on model outputs

In SFT-format-contamination trials, **not a single agent printed one raw decoded sample from its checkpoint.** Most basic check imaginable: `print(repr(tokenizer.decode(...)[:500]))`. Instead they evaluated with regex parsers that matched anywhere in the output, so models emitting narrative with the answer buried somewhere passed. Internal eval looked great; held-out didn't move.

### 2. Evaluating on the training distribution

Every run used the same in-distribution generator for both training and eval. **Trial 13** is the cleanest case: 100% internal eval, 0/500 held-out. Same BFS region-growing algorithm + a fixed seed for the eval boards meant 3 epochs of SFT were enough to memorize the structural patterns. Agent declared victory and sat idle for 10.4 hours. **No agent caught this.**

### 3. No curriculum, no data strategy

Boards span trivial to expert. The right prior is to start small, verify constraint-structure learning, then scale. Most runs either generated tiny datasets (10 boards, 6 boards) or uniform-random ones across all sizes. A few corrected post hoc.

### 4. Misaligned reward functions

The post-playbook failure mode. Agents stop failing in the obvious early-mistake ways but instead optimize against rewards that don't track the actual solve rate. Internal metrics inflate, held-out flatlines.

### 5. No working sense of time

Agents extrapolate training time from initial steps and miss curriculum-induced slowdowns and checkpoint overhead. Opus runs to the clock; Codex stops early; GPT treats the timer as a one-shot orientation step rather than live state. Time mismanagement shapes which technical decisions get made — e.g. trial 11 had 20 hours, sunk 250 minutes into SFT, contaminated the format. The 8-hour version of the same agent skipped SFT entirely and avoided contamination. More time made the agent worse.

### 6. Catastrophic-process inertia

Once committed, agents rarely stop and reflect. One successful 20-hour Opus run spent 61% of its budget evaluating checkpoints and only 3.6% on RL training, because an exceptional step-1 number triggered exhaustive per-checkpoint monitoring instead of more training.

## Where Agents Are Capable

The technical surface is fine. Across trials, agents tried iterative reward sharpening, intermediate representation supervision, iterative LoRA rank scaling, and standard SFT-then-RL recipes. Methods aren't the bottleneck.

The cleanest example of agent capability under pressure: when Tinker's `get_tokenizer()` failed because HuggingFace was blocked in the sandbox, Opus 4.6 runs treated it as a research problem. They built byte-level encoding from scratch using the BPE property that the first 256 token IDs map to raw bytes (in shuffled order), then probed the model's `topk_prompt_logprobs` to discover merged BPE token IDs empirically — and finally noticed that frog coordinates are ASCII digits/commas/brackets, all single-byte tokens, so they could parse outputs straight from the raw token stream and skip BPE decoding entirely. Diagnose → invent two workarounds → spot the structural shortcut.

Agents are clever. They just don't run a research loop.

## Spending Patterns

- GPT-5.4 submits early, barely trains, low spend, low score.
- Opus 4.6 uses the full budget with high variance — same price point lands anywhere from near-zero to top of board.
- **Best 8-hour run ≈ best 20-hour run at ⅓ the cost.** More spend doesn't buy a higher ceiling.

## What's Missing: Research Intuition

The pattern across runs: agents optimize for good-looking metrics rather than systems that actually work. They write evals and trust them blindly. Almost none asks the practitioner questions: *what could make this metric wrong? what should we be measuring at this stage? did I just memorize the eval seed?*

Thoughtful Lab's operationalization of "intuition" as trainable habits: noticing when a result looks off, interrogating a metric before trusting it, running something small before scaling, knowing when to stop a run vs. push further. The thesis is that these habits are themselves a learnable distribution — hundreds of failed experiments are training signal. FrogsGame is a first probe at whether the loop is runnable at all, not whether agents can win this specific puzzle.

## Implications

- The bottleneck for agentic modelcrafting isn't capability or method knowledge — it's research practice.
- Sophisticated methods executed without sanity checks lose to simple methods with them.
- Reward design is where most failures live. Misaligned reward → high internal eval → 0% held-out → "victory" declared.
- Time management is itself a research skill agents currently lack.
- Tinker as an interface for agentic work is good; the deficit is everything around the API call.

## Related Topics

- [[llm-agents]] — General LLM-agent architectures (planning, memory, tools)
- [[agent-harness-engineering]] — The harness side of agentic work; same agents, different problem (engineering loop vs. research loop)
- [[alignment-methods]] — The post-training stage agents are trying to automate
- [[on-policy-distillation]] — A specific post-training method agents could pick
- [[practical-fine-tuning]] — Human-built playbooks the agents are competing against
- [[reasoning-models]] — Most of the agents tested are reasoning-tuned themselves

## Sources

- "What We Learned from Letting AI PostTrain AI" — Mersad Abbasi, Thoughtful Lab (April 2026)
- github.com/Thoughtful-Lab/FrogsGame-Posttraining
- frontierswe.com/frogsgame-rl (Proximal's FrontierSWE)
- Tinker API (Thinking Machines Lab)
