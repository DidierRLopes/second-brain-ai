# What We Learned from Letting AI PostTrain AI

**Source:** Thoughtful Lab blog (https://www.thoughtfullab.com/letting-ai-posttrain-ai.html)
**Author:** Mersad Abbasi
**Published:** April 2026
**Code:** github.com/Thoughtful-Lab/FrogsGame-Posttraining
**Built for:** Proximal's FrontierSWE benchmark

## Thesis

Thoughtful Lab's framing is "modelcrafting": every person and organization should be able to shape their own models — behavior, values, how they improve with use — and most people shouldn't have to do it themselves. That's the agent job. To do it, an agent has to run the research loop, not just the training loop. This post is the first probe into whether frontier agents can do that end-to-end.

## The Frog Placement Game (FrogsGame)

A constraint-satisfaction puzzle: place N frogs on an N×N grid such that no two share a row, column, diagonal, or color region. Easy to grade, easy to solve algorithmically (backtracking, milliseconds), within reach of any frontier reasoner. But solving the puzzle and teaching another model to solve the puzzle are different problems — that gap is what the task measures.

Two variants:
- **Multi-turn:** trained model solves boards via iterative tool calls. Almost every agent failed this — either the fine-tuned model never learned the tool-calling format, or it learned the format but couldn't reason about boards. Conflates reasoning quality with format compliance.
- **Single-turn (used in this work):** model outputs the full placement as a JSON object in one shot. Isolates reasoning from format.

Held-out eval: 500 unseen boards across four difficulty tiers — easy (N=6,7), medium (N=8,9), hard (N=10,11), expert (N=12,13). Final score = % solved.

## Setup

The frontier agent (Claude 4.6 Opus or GPT-5.4) is given:
- A base model: Qwen3-8B
- Time budget: 8 or 20 hours
- The Tinker API for remote training and inference
- A sandbox with restricted network access (Tinker, model provider, HuggingFace tokenizer endpoint, npm)
- `instructions.md` (task + objective), `prepare.py` (game engine, validation, eval harness — hashed, agent forbidden to modify), and full Tinker API docs

Everything else — generating training data, designing rewards, writing the training loop, building evals, iterating — goes into `train.py` and is the agent's job. Final artifacts: `path.txt` (best checkpoint) and `results.json` (self-reported score). The verifier independently re-evaluates the saved checkpoint against the held-out 500 boards and checks the `prepare.py` hash plus that the checkpoint came from the correct base model.

The instructions are deliberately broad. The point isn't to hand-hold agents through a known recipe — it's to see where they fail when given open-ended latitude.

### Sanity Checks Before Running

Two pre-flight checks confirmed the task is well-posed:
1. **Is there a learning signal for RL?** Pass@k results say yes — the base model gives GRPO something to grip on.
2. **Is the puzzle solvable?** Frontier models solve cleanly across all four tiers. The puzzle sits well within reach of capable reasoning.

So failures sit with the agent's pipeline, not the task.

## Headline Result

Only **4 out of 20** single-turn agents reach >25% pass@4. The rest hover near zero. None of the hinted runs improved over the base model in absolute terms; the best gain was +4.8% pass@4.

## The Hinted Setting

To separate "no capability" from "missing context," the team introduced a hinted setting: identical task, identical compute, but the agent's starting context now includes a playbook of common failure modes paired with concrete fixes. The playbook addresses three issues:

1. **Over-reliance on naive SFT.** Agents start with SFT on a weak base model, which overfits output format rather than task performance. Hinted: restrict SFT to minimal corrective steps, prioritize RL.
2. **Early termination / under-using compute.** Codex agents especially give up early. Hinted: a system prompt encouraging persistence until <30 minutes remain.
3. **Invalid / non-parsable outputs.** Many runs fail because outputs don't match the expected schema. Hinted: enforce strict schema with examples.

GPT-5.4 improves (pass@4: 2.06% → 10%) and variance drops by ~2× — agents stop failing in the obvious early-mistake ways. But overall performance stays limited. Instead of failing early, they fail differently: the dominant new failure is optimizing against partial or misaligned reward functions, which decouples training signal from real solve rate. Internal evals look great; held-out evals don't move.

## Sophisticated Methods, Amateur Mistakes

The agents are technically capable. Across trials they tried:
- Iterative reward sharpening from previous checkpoints (trial 17, single-turn)
- Intermediate representation supervision (trial 18, single-turn)
- Iterative LoRA rank scaling (trial 5, single-turn)
- Standard SFT-then-RL for format following

None of the hinted runs beat the base model. What goes wrong is research-practice basics, not method choice.

### Failure 1: No sanity checks on model outputs

Across SFT-format-contamination trials, **not a single agent printed one raw decoded sample from its checkpoint**. The most basic possible check:

```python
text = tokenizer.decode(result.sequences[0].tokens)
print(repr(text[:500]))
```

Instead they evaluated with a regex (`parse_solution(text)`) that searches for `{"frogs": ...}` anywhere in output — so a model emitting coherent narrative with the answer buried somewhere passes. High eval numbers → declare victory.

### Failure 2: No curriculum, no data strategy

Boards span N=6 (trivial) to N=13 (expert). The right prior is to start small, verify the model learns the constraint structure, then increase difficulty. Most runs didn't. Some generated tiny datasets (trial 4: 10 boards; trial 15: 6 boards), others uniform-random across all sizes. Trials 4 and 11 noticed and added curriculum learning post hoc.

### Failure 3: Evaluating on the training distribution without realizing it

Every run used the same in-distribution generator for both training and eval. **Trial 13** is the cleanest example: 100% internal eval, 0/500 on the held-out set. It generated 40 eval boards with `seed=999` using the same BFS region-growing algorithm as its 6000 training boards. After 3 epochs of SFT the model had memorized enough structural patterns to pass those 40 boards. The agent declared victory and sat idle for 10.4 hours. **No agent caught this.**

## A Tokenizer Detour (Opus 4.6 Goes Off-Script)

Tinker's `get_tokenizer()` calls HuggingFace's `AutoTokenizer.from_pretrained()` under the hood, but HF was blocked in the sandbox for non-allowlisted models. Most Opus 4.6 runs treated this as a research problem rather than a blocker. They built a working tokenizer from scratch through two pivots.

**Step 1: byte-level encoding.** BPE tokenizers always map the first 256 token IDs to raw bytes (in a shuffled order). The agent hardcoded the GPT-2 byte-to-token mapping and encoded text as byte-level token IDs:

```python
BYTE_ORDER = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + [0xAE]
              + list(range(0xAF, 0x100)) + list(range(0x00, 0x21))
              + list(range(0x7F, 0xA1)) + [0xAD])
BYTE_TO_TOKEN = {bval: tid for tid, bval in enumerate(BYTE_ORDER)}
def encode_text(text): return [BYTE_TO_TOKEN[b] for b in text.encode('utf-8')]
```

**Step 2: empirical BPE discovery via logprobs.** Harmony role/channel names need to be proper merged BPE tokens after `<|start|>` / `<|channel|>`. The agent fed in partial prompts, inspected `topk_prompt_logprobs`, and read the merged token IDs straight off the model's predictions. Live commentary from the agent: noting that "Token 1428 is very likely developer" and "Token 173781 is almost certainly user" based on logprob values in context.

**The pivot:** the agent sampled with byte-level prompts and noticed coordinate digits in the raw token stream. Insight: because frog coordinates are ASCII digits, commas, and brackets — all single-byte chars with predictable token IDs — the agent could parse model outputs from the raw token stream without ever decoding BPE.

```python
DIGIT_TOKENS  = {BYTE_TO_TOKEN[ord(str(d))]: d for d in range(10)}
COMMA_TOKEN   = BYTE_TO_TOKEN[ord(',')]
OPEN_BRACKET  = BYTE_TO_TOKEN[ord('[')]
CLOSE_BRACKET = BYTE_TO_TOKEN[ord(']')]

def extract_coordinates(tokens, n):
    coords = []
    i = 0
    while i < len(tokens) - 5:
        if (tokens[i] == OPEN_BRACKET and tokens[i+1] in DIGIT_TOKENS
            and tokens[i+2] == COMMA_TOKEN and tokens[i+4] in DIGIT_TOKENS
            and tokens[i+5] == CLOSE_BRACKET):
            coords.append((DIGIT_TOKENS[tokens[i+1]], DIGIT_TOKENS[tokens[i+4]]))
            i += 6
        else:
            i += 1
    return coords[-n:] if len(coords) >= n else None
```

This is the cleanest example of agent capability in the run: blocked → diagnosed → invented two workarounds → noticed a structural shortcut that obviated the harder one.

## Agents Have No Working Sense of Time

Agents systematically underestimate training overhead. They extrapolate from initial steps and miss curriculum-induced slowdowns (more output tokens, checkpoint saves).

- **Opus 4.6** runs work until the clock runs out.
- **Codex** runs finish the planned pipeline and stop early.
- **GPT** agents treat the timer as a one-shot orientation step ("here is my budget") rather than live state. Only trials 17 and 19 used the timer to actually influence a decision.

Effect on technical decisions, not just step count:

- **Trial 11** (20-hour Claude): with more time, invested 250 minutes in SFT warmup → format contamination. Trials 1 and 2 (8-hour Claude) had less time and skipped SFT entirely, which avoided the contamination.
- **Trial 14** (20-hour Claude): explicitly chose pure RL after spending 400 minutes exploring the base model and concluding the base already had 34–37% correct rate with 100% format compliance, so SFT warmup was net-negative.
- **One successful 20-hour Opus run** spent 61% of its budget in evaluation and only 3.6% on RL training. An exceptional step-1 result triggered exhaustive per-checkpoint monitoring — ~10 hours of metric-watching instead of training.

Once committed to a process, agents rarely stop and reflect. Catastrophic time sinks are usually catastrophic precisely because of that.

## Spending Patterns

Agents had unlimited Tinker credits. Usage diverged sharply:

- **GPT-5.4:** submitted early, barely trained, low spend, low score.
- **Claude Opus 4.6:** used far more budget, with high variance — same price points landed anywhere from near-zero to top of board.
- **The best 8-hour run roughly matched the best 20-hour run at a third of the cost.** More spend doesn't buy a higher ceiling.

## The Missing Research Intuition

The pattern across runs: agents optimize for good-looking metrics rather than systems that actually work. They write evals and trust them blindly, declaring success on numbers their own code produced. Almost none asks the practitioner questions: *what could make this metric wrong? what should we be measuring at this stage?*

Thoughtful Lab's framing of "intuition" as concrete habits:
- Noticing when a result looks off
- Interrogating a metric before trusting it
- Running something small before scaling
- Knowing when to stop a run vs. push further

Hundreds of failed experiments are a training signal. The thesis is that research intuition is *trainable*, and the FrogsGame is a first probe into whether agents can run the research loop at all — not whether they can win this specific puzzle.

## Implications

- The bottleneck for agentic modelcrafting isn't capability or method knowledge — it's research practice.
- "AI doing AI" requires the boring habits, not the exotic methods. Sophisticated approaches executed without sanity checks lose to simple approaches with them.
- Tinker as an interface works well for agents — the API is clean, the loop is fast — but exposes how badly agents need a real research loop wrapped around the API call.
- Time-management is itself a research skill; agents currently lack it.
- Reward design is where most of the failures live: misaligned reward → high internal eval → 0% held-out → declare victory.
