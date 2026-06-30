# Data Curation and Pre-training Mixtures

Even with the perfect architecture, model performance hinges on training data — no amount of compute or optimization can compensate for training on the wrong content. Frontier-2026 practice is increasingly about **assembling the right multi-stage data mixture**, where you save the highest-quality data for the end (because the data the model sees last most strongly shapes its final behavior), and where domain-specific data (math, code, reasoning) gets allocated in carefully calibrated proportions. **Data quality and mixture dominate architecture tweaks at fixed compute.**

## Multi-stage Training

Evolve the data mixture during training rather than using a static blend. Why this works: a language model's final behavior is heavily dictated by what it sees at the end of training. Saving higher-quality data for late stages amplifies its effect.

This introduces a meta-decision: **when to change mixtures**. A useful heuristic — performance-driven intervention: if a benchmark plateaus, introduce high-quality data for that domain.

### SmolLM3 example (3-stage, 11T tokens)

| Stage | Tokens | Context | Mix |
|---|---|---|---|
| 1 (base) | 8T | 4k | 75% English web / 12% multilingual / 10% code / 3% math |
| 2 (high-quality) | 2T | 4k | + Stack-Edu, FineMath, OpenMathReasoning |
| 3 (reasoning/Q&A) | 1.1T | 4k | Reasoning-heavy + Q&A |

When introducing new stages, they use a **40/60 split between baseline mixture and new dataset** off a mid-training checkpoint (e.g., 7T/11T).

### Qwen3 example (3-stage)

- 30T tokens at 4k context (base)
- 5T tokens reasoning stage (STEM, code) at higher quality
- Long-context stage at 32k context length

## Ablation

Different from architecture ablations:

- **Architecture ablations**: on small models (e.g., 1B for a planned 3B). Cheap, fast.
- **Data mixture ablations**: at scale, because larger models have much larger capacity to understand variety across domains. Small-model ablations on data mixture can mislead.
- **Annealing ablations**: done on checkpoints of the main run (e.g., 7T out of 11T tokens) to determine what datasets to introduce when.

**Validation/holdout loss approaches**: tend to converge to dataset-size distributions and *underperform careful manual ablations*.

## Token Utility

Token efficiency = performance improvement per token consumed during training. Improve via token utility: the effective learning signal per token. This motivates finding the optimal balance of high-quality tokens — leverage them maximally but limit to prevent overfitting and reduced generalization.

### Kimi K2 rephrasing

- **Knowledge domains**: style/perspective-diverse prompting to rephrase texts; chunk-wise autoregressive generation to gradually build rephrased long documents; fidelity verification to ensure semantic alignment. Each corpus rephrased at most twice in the main run.
- **Math**: diversity via "learning-note style" rephrasing + translation into other languages.

## Quality vs Repetition

Naively filtering only for the highest-quality data → too few tokens → repetition → harm. The right mix balances quality and diversity, even at the cost of including some lower-quality data.

## Safety Filtering

For gpt-oss-120b, OpenAI addressed safety at the data-curation stage using **CBRN pre-training filters** (chemical, biological, radiological, nuclear) originally developed for GPT-4o. Filtering at pre-training avoids the harder problem of trying to suppress capabilities through post-training alone.

## Existing Corpora

- **FineWeb / FineWeb2 / FineWeb-Edu**: filtered Common Crawl, with educational/STEM filters.
  - [FineWeb2: One Pipeline to Scale Them All (2506.20920)](../../papers/06-data/curation-filtering/FineWeb2: One Pipeline to Scale Them All - 2506.20920.pdf) extends FineWeb-style Common Crawl processing to multilingual data with language-specific adaptation rather than a single fixed recipe: GlotLID-based language identification, per-language word tokenizers, global MinHash deduplication, heuristic filters calibrated by language statistics, precision filters for low-resource language contamination, and duplication-aware "rehydration" upsampling. The released dataset is 20TB / 5B documents across 1,000+ languages, built from almost 100 Common Crawl snapshots.
- **DCLM**: common-sense reasoning focused subset.
- **The Pile** ([The Pile: An 800GB Dataset of Diverse Text for Language Modeling (2101.00027)](../../papers/06-data/datasets/The Pile: An 800GB Dataset of Diverse Text for Language Modeling - 2101.00027.pdf)): 825 GiB English mixed-domain corpus built from 22 components, including academic/professional sources, code, books, web text, dialogue, legal, patents, and Wikipedia. Its main lesson for mixture design is still useful: broad, high-quality domain diversity can improve cross-domain generalization compared with Common Crawl-only training.
- **The Stack v2 / StarCoder2**: code (16+ languages, PRs, notebooks, issues, StackExchange).
- **FineMath3+ / InfiWebMath3+ / MegaMath**: math.
- **OpenMathInstruct / OpenMathReasoning**: math instruction/reasoning datasets.
- **MEGASCIENCE** ([MEGASCIENCE: Pushing the Frontiers of Large-Scale Data Collection (2507.16812)](../../papers/06-data/curation-filtering/MEGASCIENCE: Pushing the Frontiers of Large-Scale Data Collection - 2507.16812.pdf)): science reasoning post-training mixture, not a generic web pre-training corpus. It combines TextbookReasoning (650k questions extracted from nearly 12k university-level textbooks across 7 disciplines) with selected open scientific reasoning datasets for a 1.25M-instance mixture. The useful curation pattern is dataset-specific selection: response-length, difficulty, and random selection are ablated per source, then LLM-based decontamination is applied against science benchmarks.
- **Pes2o**: scientific articles.

## SmolLM3 mixture details

Stage 1: 75/12/10/3 across English web / multilingual web / code / math.

- **English web**: FineWeb-Edu (educational, STEM benchmarks) + DCLM (common-sense reasoning). Ablations: 60/40 or 50/50 best. Later additions: Pes2o, Wikipedia/Wikibooks, StackExchange.
- **Multilingual**: 5 European languages from FineWeb2-HQ. Smaller portions of Chinese/Arabic to allow downstream continual pretraining. 12% multilingual was best.
- **Code**: Stack v2 + StarCoder2 — 16 languages, GitHub PRs, Jupyter/Kaggle notebooks, issues, StackExchange. Notably, the recommended code mixture *degraded* English benchmarks; Stack-Edu (educationally filtered) was delayed to the late stages.
- **Math**: FineMath3+, InfiWebMath3+, MegaMath, plus OpenMathInstruct and OpenMathReasoning.

## FinePhrase: Synthetic Pretraining via Structured Rephrasing

[How Can We Synthesize High-Quality Pretraining Data? (2604.13977)](https://arxiv.org/abs/2604.13977) is HuggingFace's systematic study of what actually matters when generating synthetic pretraining data. Across 90 experiments generating over 1 trillion tokens (~12.7 GPU years), they varied prompt format, generator model size, model family, and source data quality to isolate each factor's contribution. The result is the **FinePhrase** dataset: 339M documents from FineWeb-Edu rephrased into 1.35 billion samples / 486 billion tokens.

### The Hierarchy of Variables

In order of impact on downstream performance:

1. **Prompt design** — by far the largest lever. Everything else is secondary.
2. **Source data quality** — matters, but a strong prompt can make even low-quality sources work.
3. **Generator model size** — provides zero benefit beyond ~1B parameters. SmolLM2-1.7B-Instruct was optimal and dominated all other model families tested.

### Prompt Formats and Results

Thirteen formats were tested (9 new + prior work from Nemotron, REWIRE, BeyondWeb). Only formats that **restructure** how knowledge is presented beat DCLM — polishing or cleaning the language without changing the structure consistently failed:

| Format | Description | Beats DCLM? |
|---|---|---|
| **Table** | Aggregates scattered info into indexable units | Yes — best overall |
| **FAQ** | Makes implicit questions explicit | Yes |
| **Tutorial** | Externalizes procedural logic | Yes |
| **Math** | Math-problem framing | Yes |
| **Article** | Simple paraphrasing | Yes (marginal) |
| **Commentary** | Review-style summary | Yes (marginal) |
| **Discussion** | Conversational format | Yes (marginal) |
| **Narrative** | Narrative retelling | Yes (marginal) |
| Cleaning/polishing formats | Surface-level edits only | No |

**Best result**: FinePhrase-Table → macro-average **17.18** on 12 lighteval benchmarks, which is +3.41 over DCLM and +3.63 over Nemotron-HQ-Synth.

The final FinePhrase dataset uses the four top-performing structured formats: FAQ, Math, Table, and Tutorial.

### Evaluation Methodology

All experiments run the same eval pipeline:
- **12 English lighteval tasks** evaluated automatically at checkpoints during pretraining.
- Metrics: `prob_norm_token` per task, `agg_score_micro`, `agg_score_macro` (the primary comparison metric), and six category-level aggregates.
- **DCLM** (a curated web baseline strong in commonsense) serves as the reference threshold.
- Contamination auditing uses n-gram overlap detection against benchmark corpora.

### Quality Filters Don't Transfer to Synthetic Data

A critical finding: standard quality proxies that work for web data are nearly useless for evaluating synthetic outputs:
- **FineWeb-Edu-score** correlation with downstream performance: **−0.08** (essentially random)
- **DCLM-score** correlation: only **0.56–0.61**

This means you cannot reuse web-data classifiers to select good synthetic documents. You must eval the actual training signal directly.

### Source Data: Complementary, Not Decisive

DCLM and FineWeb-Edu-HQ have complementary strengths — DCLM skews toward commonsense reasoning, FineWeb-Edu-HQ toward knowledge-heavy tasks. Neither dominates universally. More importantly: **with a strong structured prompt, even low-quality source documents produce competitive synthetic training data**. Source quality is secondary to prompt quality.

### Efficiency

- Generated 486B tokens in ~14,700 GPU hours using SmolLM2-1.7B
- ~30× cheaper than REWIRE, ~13× cheaper than Cosmopedia
- Compared to training on raw DCLM (free to collect), FinePhrase costs compute but gains +3.41 macro-average points

## Hermes 4: synthetic data via DataForge

Nous's pipeline starts with DCLM/FineWeb data:

1. Semantic deduplication using embeddings at cosine similarity 0.7.
2. LLM-as-judge filters incomplete or ill-formatted messages.
3. Pass through **DataForge**, a graph-based synthetic data generator.

DataForge models data generation as a random walk through a directed acyclic graph. Nodes implement struct → struct mappings; an edge from node `A` to node `B` requires that postconditions of `A` satisfy preconditions of `B`. QA pairs are generated with intermediate transformations into other mediums (e.g., a Wikipedia article → a rap song), followed by question generation and LLM-as-judge grading of instruction/response pairs. For data-scarce subdomains, recursive DFS taxonomy generation produces partitions where leaves are prompts.

The DataForge-generated data is used in both pre-training and post-training stages.

## Filtering at Scale: The Bitter Lesson

[A Bitter Lesson for Data Filtering (2605.19407)](../../papers/06-data/curation-filtering/A Bitter Lesson for Data Filtering - 2605.19407.pdf) challenges the assumption that quality filtering always helps. In their high-compute, data-scarce scaling studies, the authors find that sufficiently trained larger models can benefit from the full Common Crawl pool more than from heavily filtered subsets; their scaling-law extrapolation suggests the 240T-token DCLM-Pool Common Crawl pool could become preferable to RefinedWeb around 1e30 FLOPs. They also test "junk" injections such as shuffled-word documents and find that large models can recover signal from noisy additions once compute is high enough. Practical takeaway: deduplication, contamination control, and safety filters remain non-negotiable, but aggressive quality filtering is compute-regime dependent rather than universally good.

## Sequence Composition and Packing Strategies

How documents are packed into training sequences affects performance. [Analysing The Impact of Sequence Composition on Language Model Pre-Training (2402.13991)](../../papers/08-evaluation/analysis/Analysing The Impact of Sequence Composition on Language Model Pre-Training - 2402.13991.pdf) compares packing and masking strategies:

- **MixChunk**: baseline random packing from the whole pre-training corpus, so a single context can mix unrelated sources such as Wikipedia and GitHub.
- **UniChunk**: source-consistent packing, where each chunk samples documents from one corpus/source to reduce distribution jumps.
- **BM25Chunk**: retrieval-based packing where similar documents cluster (implicit topic coherence).
- **Intra-document causal masking**: each token can attend only to earlier tokens from the same document, removing cross-document distraction at the cost of some sparse-mask overhead.

The paper's cleanest result is that standard causal masking lets unrelated previous documents become distracting context. Intra-document masking improves quality but slowed their implementation by about 4%. BM25Chunk keeps the efficient causal-mask path while reducing distraction through related-document packing, improving language modeling (+6.8%), in-context learning (+11.6%), knowledge memorization (+9.8%), and context utilization (+7.2%) in their reported setup. This is directly relevant to long-context pre-training: as context windows grow, document packing stops being a harmless data-loader detail and becomes part of the model's learned attention environment.

## Data Takeaways

- Data quality and mixture often dominate architecture tweaks at fixed compute.
- Multi-stage schedules help: save the best data for late training to shape final behavior.
- Deduplication and contamination checks are non-optional if you care about honest evals.
- Ablate data mixtures at scale; small-model ablations on data mixture can mislead.
- At high compute, diversity may outweigh aggressive quality filtering.

## Related Topics

- [[reasoning-models]] — distilled mid-training (Phi-4-Mini-Reasoning, SmolLM3) and context-length extension (4k → 32k → 64k → 128k via RoPE ABF and YaRN)
- [[reasoning-data-generation]] — teacher traces, synthetic verifier-backed tasks, and cognitive-behavior data
- [[data-quality-vs-diversity]] — stage-dependent tradeoffs between broad coverage and aggressive filtering
- [[long-context-training]] — document packing, long/short mixtures, and document masking
- [[supervised-fine-tuning]] — how mid-training and SFT interact with the data mixture
- [[scaling-laws]] — overtraining vs compute-optimal
- [[frontier-training-playbook]] — where data curation sits in the broader recipe

## Sources

- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- SmolLM3 report. See `raw/smollm3-hugging-face-report.md`.
- Hermes 4 technical report (DataForge).
- Kimi K2 technical report (rephrasing).
- gpt-oss-120b system card (CBRN filtering).
- [A Bitter Lesson for Data Filtering (2605.19407)](../../papers/06-data/curation-filtering/A Bitter Lesson for Data Filtering - 2605.19407.pdf) — Filtering can lose to full-pool Common Crawl in high-compute, data-scarce regimes.
- [FineWeb2: One Pipeline to Scale Them All (2506.20920)](../../papers/06-data/curation-filtering/FineWeb2: One Pipeline to Scale Them All - 2506.20920.pdf) — Adaptive multilingual Common Crawl pipeline; 20TB, 5B documents, 1,000+ languages.
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling (2101.00027)](../../papers/06-data/datasets/The Pile: An 800GB Dataset of Diverse Text for Language Modeling - 2101.00027.pdf) — 825 GiB foundational mixed-domain English corpus with 22 components.
- [MEGASCIENCE: Pushing the Frontiers of Large-Scale Data Collection (2507.16812)](../../papers/06-data/curation-filtering/MEGASCIENCE: Pushing the Frontiers of Large-Scale Data Collection - 2507.16812.pdf) — 1.25M science reasoning post-training mixture with textbook data, selection ablations, and LLM decontamination.
- [Analysing The Impact of Sequence Composition on Language Model Pre-Training (2402.13991)](../../papers/08-evaluation/analysis/Analysing The Impact of Sequence Composition on Language Model Pre-Training - 2402.13991.pdf) — Packing strategies, BM25Chunk, and intra-document causal masking effects.
- [How Can We Synthesize High-Quality Pretraining Data? (2604.13977)](https://arxiv.org/abs/2604.13977) — Prompt format dominates generator size and source quality; structured outputs (Table/FAQ/Tutorial/Math) beat DCLM; standard quality filters break on synthetic data.
