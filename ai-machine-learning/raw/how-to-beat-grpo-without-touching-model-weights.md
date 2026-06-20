# How to Beat GRPO Without Touching Model Weights

**Source:** Deep-dive article on GEPA (companion to the paper at arxiv:2507.19457)
**Topic:** Reflective prompt evolution as an alternative to RL on compound LLM systems
**Published:** Early 2026

## Key Concepts

A practitioner-oriented walkthrough of GEPA, a prompt optimizer that beats GRPO by ~10 points using ~$1,000 of compute (vs. ~$40,000 and a GPU run for an 8B GRPO fine-tune) by exploiting the natural-language richness of agent rollouts. Where GRPO collapses each 5,000-token rollout to a single ±1 reward and back-propagates one bit per trajectory, GEPA hands the full trace to a reflection LLM that diagnoses the failure, localizes it to one module in the pipeline, and rewrites that module's prompt.

## Why It Matters

- "The signal isn't sparse — RL made it sparse." Single most useful framing in the article.
- GEPA targets compound AI systems (e.g., DSPy programs): pipelines of modules each with their own prompt, glued together by Python control flow. You can target the exact module that's broken instead of nudging the whole system.
- 35× fewer rollouts than GRPO for the same or better benchmark gains, no GPU training needed.
- Since July 2025: paper accepted to ICLR 2026, first-class DSPy optimizer, Hugging Face & OpenAI cookbooks, public endorsement from Shopify's CEO, production ablations from Decagon (Mar 2026), and adoption in agents like Hermes for evolving skills.

## Practical Guidance

**DSPy API** (one-line different from MIPROv2):

```python
optimizer = dspy.GEPA(
    metric=metric_with_feedback,
    auto="medium",
    reflection_minibatch_size=3,
    candidate_selection_strategy="pareto",
    reflection_lm=dspy.LM("gpt-5", temperature=1.0, max_tokens=32000),
    use_merge=True,
    track_stats=True,
)
optimized = optimizer.compile(program, trainset=train, valset=val)
```

The metric function must return a `dspy.Prediction(score=float, feedback=str)`. The feedback string is what the reflection LLM consumes — make it diagnostic ("missed entity X, retrieved doc Y when gold was Z, format violation in step 3"), not just "wrong answer".

**Decagon production finding (March 2026):** more data is *not* always better with GEPA. 20–100 examples often beats 500. The reflector overfits to noise when fed too much. Use small, high-quality training sets.

## Decision Tree

- **Use GEPA when:** small training set, expensive rollouts, no access to weights, metric describable in words. Default first try for compound-system work in 2026.
- **Use GRPO when:** abundant cheap rollouts, open weights, verifiable terminal reward. Fine-tune when the model genuinely lacks the capability.
- **Use MIPROv2 when:** you specifically need bootstrapped few-shot exemplars in your prompts.
- **Use TextGrad when:** computation graph is deep and you want explicit per-variable critique propagation.

## Honest Framing (the line worth keeping)

> "RL changes what the model knows. GEPA changes how you ask."

If the base model genuinely cannot do the task, no amount of prompt evolution will save you — fine-tune. But most of what teams currently route to GRPO is the second case (the model can do it; the prompt is the bottleneck), and reading a rollout costs less than running ten thousand more.

The field has stopped framing this as GEPA *vs* RL and started framing it as GEPA *and* RL — the paper points to hybrid recipes like BetterTogether and mmGRPO as the natural next step.

## References

- Paper: https://arxiv.org/abs/2507.19457
- DSPy GEPA: https://dspy.ai/api/optimizers/GEPA/overview/
