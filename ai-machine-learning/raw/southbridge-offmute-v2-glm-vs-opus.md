# offmute v2: GLM-5.2 vs. Claude Opus 4.8 Head-to-Head Agentic Build (Southbridge)

**Source:** https://www.southbridge.ai/blog/offmute-v2-glm-vs-opus
**Filed under:** Agentic coding evaluation, model comparison

## Premise

A practical, single-prompt agentic-coding comparison: both GLM-5.2 and Claude Opus 4.8 were given the **same prompt file** in Claude Code and asked to build **offmute-v2**, a diarized audio/video transcription tool, single-shot with no iterative back-and-forth. The post documents what each model built, where each failed, and what that reveals about evaluating coding agents — not just "which model wins" but "what kind of failure should worry you more."

## Convergent Design Choice

Both models, independently and unprompted to do so, chose the **same core algorithm** for aligning diarization segments with transcript text: **Needleman-Wunsch sequence alignment** (a dynamic-programming algorithm originally from bioinformatics for global sequence alignment). Treated as a notable signal that this is now a "well-known" solution pattern baked into frontier-model priors for this problem class, rather than either model independently deriving it from first principles.

## GLM-5.2's Build

- **Strengths:** most readable and modular codebase of the two; visibly followed and **learned the project's existing conventions** rather than imposing its own style; showed rigorous process discipline (methodical, step-by-step implementation visible in its work).
- **Critical flaw:** a **silent caching bug** — the kind of failure that doesn't crash or error out, just quietly returns wrong/stale results. Classified by the author as the more dangerous failure category precisely because it doesn't announce itself.
- **Secondary bug:** an **overlap-merge double-print bug** that inflated GLM's word-error-rate (WER) numbers by double-printing overlapping speaker segments; fixing this bug brought GLM's real WER down substantially (see table below).

## Claude Opus 4.8's Build

- **Strengths:** best self-reported WER and speaker-attribution accuracy; an elegant **voice-anchored speaker ID** approach; genuine **browser-side ffmpeg.wasm support** that actually worked, letting audio processing run client-side.
- **Critical flaws:** a **crash-on-audio-only-input bug** — loud and immediately visible (the program simply fails) rather than silent; and **model-selection HTTP 400 errors** in certain configurations.
- Framed as the lower-risk failure category specifically because it's loud: a crash is annoying but is caught immediately, unlike GLM's silent caching bug which could ship undetected.

## WER Reconciliation (claimed vs. deduplicated/corrected)

| Model | Claimed WER | Corrected WER (after bug fixes) |
|---|---|---|
| GLM-5.2 | 13.5% | 8.3% |
| Claude Opus 4.8 | 8.1% | 7.5% |

After fixing GLM's overlap-merge double-print bug, the two models' real accuracy converges to **nearly the same level** (8.3% vs. 7.5%) — the large gap in the *claimed* numbers was mostly a measurement artifact of GLM's bug, not a genuine capability gap.

## Cost Comparison

| Model | Total tokens |
|---|---|
| Claude Opus 4.8 | 286.6M |
| GLM-5.2 | 209.0M |

Both totals are dominated by **cache reads** rather than fresh generation, making this primarily a comparison of context-caching efficiency across the two coding sessions rather than raw output volume.

## Decision and Outcome

Despite Opus's higher self-reported accuracy and the appeal of its ffmpeg.wasm browser support, the **GLM-5.2 build became the primary `offmute-v2@latest`** release — implicitly weighting "silent failure risk" and "codebase maintainability/convention-following" above raw benchmark accuracy once the WER gap closed after bug fixes.

## Thesis

Two explicit takeaways structure the whole comparison:

1. **"Failure modes matter more than failure counts."** A model that fails loudly and rarely (Opus's crash) is operationally safer than one that fails silently and rarely (GLM's caching bug), even if a naive bug-count or benchmark-score comparison would treat them as equivalent.
2. **"I validated one path and generalized."** Identified as the shared root cause behind *both* models' first bugs — each model tested its implementation against one code path/input type, confirmed it worked, and then generalized that confidence to other paths it hadn't actually exercised (GLM's caching logic, Opus's audio-only branch). Framed as a generic agentic-coding failure pattern worth watching for in any single-shot build, independent of which model is doing the building.

## Related wiki pages

[[agent-harness-engineering]], [[llm-evaluation]], [[model-release-comparisons]] (candidate new page)
