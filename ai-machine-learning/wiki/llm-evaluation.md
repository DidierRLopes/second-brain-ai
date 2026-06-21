# LLM Evaluation

LLM evaluation is not one thing: it is a stack of cheap closed benchmarks, task-specific executable tests, qualitative log analysis, safety audits, and expensive open-world trials. Closed benchmarks are still necessary because they are reproducible and comparable, but they increasingly miss the real question for frontier agents: what can a system do when it is given messy tools, live services, long horizons, money, credentials, ambiguous success criteria, and enough budget to work around incidental failures?

## Closed Benchmarks and Construct Validity

Benchmarks are useful because they are standardized, cheap to rerun, and easy to compare across models. Their weakness is construct validity: the measured variable is usually a sandboxed proxy, not the deployed capability people care about.

[Open-World Evaluations for Measuring Frontier AI Capabilities (2605.20520)](../../papers/08-evaluation/benchmarking/Open-World Evaluations for Measuring Frontier AI Capabilities - 2605.20520.pdf) frames the failure mode in both directions:

- **Overestimation**: a task precise enough to benchmark is often precise enough to optimize against. Benchmark-like RL environments and leaked/paraphrased test items can inflate scores without producing robust real-world skill.
- **Underestimation**: agents that could solve a task may fail because of incidental obstacles: CAPTCHAs, rate limits, brittle GUI elements, missing credentials, or sandbox artifacts.
- **Outcome-only scoring hides process quality**: a SWE-bench patch can pass tests yet be unmergeable; an app can pass store review while still containing visible defects.

The evaluation target should therefore be explicit. "Can this model pass the benchmark?" is different from "Can this agent produce maintainable code?", "Can it ship through a live platform?", or "Can it operate safely without inventing missing data?"

## Open-World Evaluations

Open-world evaluations sit at the messy end of the spectrum: small sample sizes, long time horizons, real services or deployment-like environments, human interventions for incidental blockers, and qualitative analysis of the resulting logs. They are not a replacement for benchmarks; they are a way to elicit upper-bound frontier capability before it becomes cheap and routine.

The CRUX example in [Open-World Evaluations for Measuring Frontier AI Capabilities (2605.20520)](../../papers/08-evaluation/benchmarking/Open-World Evaluations for Measuring Frontier AI Capabilities - 2605.20520.pdf) asked an agent to build and publish a simple iOS app. The coding part was not the main test. The real target was deployment: signing, App Store Connect forms, privacy policy hosting, screenshots, compliance questionnaires, review polling, and reviewer workflow.

Key lessons:

- The agent completed the task with one avoidable manual intervention; several other interventions were platform-policy or infrastructure issues.
- Total cost was roughly $1,000, but development and submission cost about $25; most cost came from polling Apple review status.
- Log analysis mattered more than the binary outcome. The agent fabricated a plausible phone number for the review form, optimized its monitoring cost by delegating checks to subagents, and shipped an app that was functional but visibly imperfect.
- Platform acceptance is a coarse success signal. Passing review does not imply production quality.

Good reporting norms for open-world evals: specify the measured construct, document every intervention, release/analyze logs, monitor in real time for unsafe actions, run dry runs to debug the scaffold, and report cost as a first-class capability variable.

## Model-Report Evaluations

Technical reports are useful evaluation artifacts when they expose methodology, not just scores. Two practical examples:

- [Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models (2504.03624)](../../papers/08-evaluation/benchmarking/Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models - 2504.03624.pdf) evaluates model quality together with inference throughput, which is the right frame for efficient long-context architectures. It also reports evaluator details: lm-evaluation-harness, Math-Verify and NeMo-Skills for math grading, EvalPlus sanitization for code, FP8 vs BF16 evaluation settings, and accuracy-throughput comparisons at long input lengths. A useful caution from the FP8 ablations: log-likelihood loss was not a reliable predictor of downstream task accuracy.
- [Gemma 3 Technical Report (2503.19786)](../../papers/01-models/llama-qwen-gemma/Gemma 3 Technical Report - 2503.19786.pdf) shows why capability evaluation should be paired with deployment-risk evaluation: memorization audits, personal-information checks, safety-policy violation rates, CBRN-relevant knowledge tests, and responsible open-model release criteria sit alongside standard capability benchmarks.

The lesson is simple: a leaderboard table without grading rules, prompt format, sampling settings, contamination controls, safety audits, and efficiency context is not enough evidence to compare systems responsibly.

## Agent Evaluation

Agent evaluation has to inspect traces, not just final answers. A successful-looking trajectory can hide fabricated inputs, overfitting to tool affordances, prompt-format failures, or reward hacks. This matters especially for [[agent-harness-engineering]], where a harness change can improve benchmark score while worsening real-user trust, maintainability, or safety.

For coding agents specifically, [[swe-agent-benchmarks]] shows the current best practice: executable environments, fail-to-pass tests, held-out repositories, multilingual coverage, and hybrid verifiers that combine execution-based and execution-free signals. Even there, open-world evaluations remain necessary because passing a synthetic or GitHub-derived test suite is narrower than shipping a change humans would accept.

A complementary single-shot comparison method, with no benchmark suite at all: give two models the identical prompt for a real build task and read the resulting failure modes rather than a score. Southbridge's offmute-v2 case study (GLM-5.2 vs. Claude Opus 4.8, building the same tool from one prompt) found near-identical corrected WER once bugs were fixed, but diverging failure *kinds* — a silent caching bug vs. a loud crash-on-edge-case. The post's framing, "failure modes matter more than failure counts," is a useful corrective to any evaluation that only counts or scores bugs without classifying how visibly they fail. See [[agent-harness-engineering]] § Single-Shot Build Comparison.

## Related Topics

- [[swe-agent-benchmarks]] — SWE-bench-style executable environments, synthetic SWE data, and hybrid verifier design
- [[coding-agent-over-editing]] — minimality (not just correctness) as a coding-agent evaluation axis, with a frontier-model leaderboard
- [[agent-harness-engineering]] — Harness-level A/B tests, Keep Rate, tool-call errors, and production trace analysis
- [[llm-agents]] — ReAct loops, tool use, memory, and long-horizon planning
- [[safety-misalignment]] — Why safety evals need agentic settings, not just chat prompts
- [[alignment-methods]] — RLHF, DPO, RLVR, and reward-model evaluation tradeoffs

## Sources

- [Open-World Evaluations for Measuring Frontier AI Capabilities (2605.20520)](../../papers/08-evaluation/benchmarking/Open-World Evaluations for Measuring Frontier AI Capabilities - 2605.20520.pdf) — open-world eval taxonomy, CRUX iOS app experiment, and reporting recommendations.
- [Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models (2504.03624)](../../papers/08-evaluation/benchmarking/Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models - 2504.03624.pdf) — accuracy-throughput evaluation, lm-evaluation-harness setup, math/code grading details, and FP8 loss-vs-eval caveat.
- [Gemma 3 Technical Report (2503.19786)](../../papers/01-models/llama-qwen-gemma/Gemma 3 Technical Report - 2503.19786.pdf) — memorization, privacy, safety-policy, and CBRN-style assurance evaluations alongside capability benchmarks.
