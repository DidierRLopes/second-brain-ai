# Prompt Optimization

Prompt optimization is the practice of improving an LLM system's behavior by changing its prompts rather than its weights. For compound AI systems — pipelines of multiple LLM modules glued together by control flow, like DSPy programs — modern prompt optimizers can match or beat reinforcement learning at a fraction of the cost. The core insight: agent rollouts are already rich natural-language artifacts (reasoning steps, tool calls, compiler errors, judge rationales), and a reflection LLM can extract far more signal from one trace than a scalar reward can from thousands.

## GEPA: Reflective Prompt Evolution

GEPA ([GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (2507.19457)](../../papers/05-learning/reasoning/GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning - 2507.19457.pdf)) (UC Berkeley / Stanford / Databricks / MIT; ICLR 2026 oral) is the current default for optimizing compound LLM systems. Where [[alignment-methods|GRPO]] compresses a 5,000-token rollout to a single ±1 reward and back-propagates one bit per trajectory, GEPA hands the entire trace to a *reflection LLM* that diagnoses the failure, localizes it to one module in the pipeline, and rewrites that module's prompt.

The useful shift is that GEPA treats rollouts as *learning material*, not just as reward events. A trace can include the active prompt for each module, model reasoning, retrieval results, tool calls, tool outputs, compiler errors, rubric judgments, and constraint-level pass/fail notes. The reflection model reads that serialized artifact and writes high-level rules back into the prompt, so one failed rollout can teach something general like "retrieve both bridge entities before answering" or "check all privacy constraints before delegating."

**Algorithm:** Pick a candidate prompt set from the population (Pareto sampling). Pick a module to mutate (round-robin). Sample 3 examples, run rollouts, collect traces and natural-language feedback from a feedback function `μ_f`. Feed everything to the reflection LLM, get a new prompt. Re-run on the same 3 examples — if better, keep it; otherwise discard. Repeat until the rollout budget runs out. The search object is the whole prompt vector for a compound system, not a single flat instruction string.

**Pareto candidate selection** is the design choice that separates GEPA from earlier evolutionary prompt methods (EvoPrompt, Promptbreeder). Greedy "always mutate from the best" collapses to local optima; Pareto sampling keeps any candidate that is best on at least one task and weights parent selection by how many tasks each one wins, preserving distinctive strategies for later recombination.

**Feedback function `μ_f`** replaces a scalar metric with a score *plus* a diagnostic natural-language description (which gold docs were retrieved vs. needed, per-constraint pass/fail, actual compiler errors, PII-leakage breakdowns). If the feedback string is just "wrong answer", GEPA degrades to a slower MIPROv2.

**Reported results:** across HotpotQA, IFBench, AIME, LiveBench-Math, PUPA, and HoVer, GEPA beats GRPO by 6% on average and up to 20% while using up to 35× fewer rollouts and no GPU training. It also beats MIPROv2 across the reported benchmarks, including a double-digit gain on AIME-2025. First-class optimizer in DSPy; cookbooks from Hugging Face and OpenAI.

The practical constraint is feedback quality. GEPA is strongest when the task can expose interpretable state — retrieved documents, failing tests, judge rationales, constraint violations, bad tool arguments. It is weaker when the only available signal is a terminal scalar reward or when the base model lacks the underlying skill and no prompt rewrite can elicit it.

## MIPROv2

DSPy's prior flagship optimizer. Bayesian search over instructions plus bootstrapped few-shot demonstrations. Reach for MIPROv2 specifically when you need few-shot exemplars baked into the prompt; reach for GEPA otherwise.

## TextGrad

Propagates "textual gradients" — per-variable critiques — through a computation graph of LLM calls. No population, no Pareto selection. Useful when the graph is deep and you want explicit per-variable feedback rather than population-level evolution.

## Reflexion

Verbal feedback about failed attempts is stored in a per-task memory buffer, letting an agent improve across attempts on the same task. GEPA borrowed Reflexion's reflection idea but moved it from per-task memory to population-level prompt evolution.

## Earlier Generation: APE, OPRO, EvoPrompt, Promptbreeder

APE and OPRO use an LLM as a black-box optimizer over a single prompt with scalar feedback. EvoPrompt and Promptbreeder add evolutionary structure (mutation, crossover, selection) but still optimize a single prompt against a scalar fitness. None reflect on full traces; none target compound systems with multiple modules.

## When to Reach for What

The decision tree for compound-system work in 2026:

- **GEPA** — small training set (often 20–100 examples beats 500; Decagon production ablation, March 2026), expensive rollouts, no access to weights, metric describable in words. Default first try.
- **GRPO** (and other RL) — abundant cheap rollouts, open weights, verifiable terminal reward. Use when the base model genuinely lacks the capability.
- **MIPROv2** — when bootstrapped few-shot exemplars are specifically what you want.
- **TextGrad** — deep computation graph with per-variable critique needs.

The honest framing: RL changes *what the model knows*; prompt optimization changes *how you ask*. Most of what teams currently route to GRPO is the second case, not the first — and reading a rollout costs less than running ten thousand more. The field is increasingly framing this as GEPA *and* RL (hybrid recipes like BetterTogether, mmGRPO) rather than GEPA *vs* RL.

## Speculative: Prompt Optimization for Constructing OPD Teachers

A more recent line of thought (Brown, April 2026) reframes the optimal-teacher question in [[on-policy-distillation|on-policy distillation]] as a Lagrangian — maximize E[Δreward] − β · KL(π_T || π_θ) on the student's rollouts — and proposes using GEPA over that objective on a per-task basis to construct teacher prompts (or hint-rewriters) that move the teacher distribution as little as possible while still improving reward. Speculative, but it's a clean example of GEPA-style search applied somewhere other than end-task prompts.

## Related Topics

- [[alignment-methods]] — GRPO and RL methods that prompt optimization is contrasted with
- [[on-policy-distillation]] — OPD/OPSD, where prompt optimization may help construct better teachers
- [[llm-agents]] — Compound agent pipelines are the primary target for GEPA-style optimization
- [[reasoning-models]] — GRPO's role in eliciting reasoning vs. GEPA's role in shaping how reasoning is invoked
- [[chain-of-thought-reasoning]] — The reasoning patterns that prompt optimizers tune

## Sources

- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (2507.19457)](../../papers/05-learning/reasoning/GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning - 2507.19457.pdf)
- GEPA in DSPy (https://dspy.ai/api/optimizers/GEPA/overview/)
- How to Beat GRPO Without Touching Model Weights — deep-dive article
