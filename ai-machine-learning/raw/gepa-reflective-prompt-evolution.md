# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

**Source:** https://arxiv.org/abs/2507.19457
**Authors:** UC Berkeley (with collaborators)
**Published:** July 2025 (accepted at ICLR 2026)
**Implementation:** https://dspy.ai/api/optimizers/GEPA/overview/

## Key Concepts

GEPA (Genetic-Evolutionary Prompt Adaptation) is a prompt optimizer for compound AI systems (multi-module pipelines like DSPy programs). Instead of compressing each rollout to a scalar reward and back-propagating gradients (as GRPO does), GEPA hands the full natural-language rollout — reasoning steps, tool calls, compiler errors, judge rationales — to a *reflection LLM* that diagnoses the failure, localizes it to a specific module, and rewrites that module's prompt.

The optimizer uses Pareto-based candidate selection (borrowed from quality-diversity optimization): it keeps any candidate that is best on at least one task, then samples parents weighted by how many tasks each one wins. This prevents collapse to local optima that plagues greedy "always mutate from the best" approaches.

## Algorithm in Six Steps

1. Pick a candidate prompt set from the population (Pareto sampling)
2. Pick a module to mutate (round-robin across modules)
3. Sample 3 examples from the training set
4. Run rollouts and collect full traces plus feedback from a feedback function `μ_f`
5. Reflect: feed traces and feedback to a reflection LLM, get a new prompt
6. Accept or reject by re-running on the same 3 examples; keep if better

No gradients, no PPO, no KL penalties — repeat until the rollout budget B is exhausted.

## The Feedback Function `μ_f`

GEPA replaces a scalar metric with a feedback function that returns *both* a score and a natural-language description of what happened:

- HotpotQA: which gold docs you retrieved and which you still need
- Instruction-following: per-constraint pass/fail descriptions
- Code generation: actual compiler errors and profiler traces
- Privacy-preserving rewriting: split scores for quality vs. PII leakage with breakdowns

If feedback is just "wrong answer", GEPA degrades to a slower MIPROv2. Diagnostic, specific feedback is what makes it fly.

## Why It Matters

- Beats GRPO by ~10 points on the same task/base model with 35× fewer rollouts and no GPU training
- Each rollout is roughly a 5,000-token document full of structured signal; GRPO compresses that to ±1, while GEPA reads the trace
- Targets compound AI systems (multi-module pipelines), letting you improve one module without nudging the whole system
- Accepted at ICLR 2026; first-class optimizer in DSPy; cookbooks shipped by Hugging Face and OpenAI; publicly endorsed by Shopify's CEO
- Honest framing: RL changes *what the model knows*; GEPA changes *how you ask*. Use GEPA when the base model can already do the task and the prompt is the bottleneck. Fine-tune when you need new capabilities.

## Worked Example (HotpotQA)

Multi-hop QA agent with separate modules for first-hop query writing, retrieval, summarization, second-hop query writing, and final answering. The seed prompt for the second-hop query writer was the generic DSPy default ("Given the fields question, summary_1, produce the fields query") and scored ~38% on validation.

GEPA observed in traces that the query writer kept paraphrasing the original question and retrieving the same documents already seen. The reflection LLM rewrote the prompt to explicitly target broader/connected entities mentioned in `summary_1` but not in the question (e.g., for "What is the population of the region containing the parish of São Vicente?", search "Madeira archipelago population" rather than "São Vicente population"). New score: 69%.

## Comparison to Related Methods

- **APE, OPRO** — LLM-as-optimizer, single prompt, scalar feedback (older generation)
- **EvoPrompt, Promptbreeder** — evolutionary, single prompt, scalar fitness (no reflection)
- **Reflexion** — verbal feedback but per-task memory, not population evolution
- **TextGrad** — textual gradients propagating through computation graphs (no population)
- **MIPROv2** — Bayesian search over instructions + bootstrapped few-shot demos (DSPy's prior flagship)
- **GRPO** — actual RL with weight updates (different problem class)

GEPA = Reflexion's reflection + EvoPrompt's population + MIPROv2's compound-system focus + novel Pareto selection.
