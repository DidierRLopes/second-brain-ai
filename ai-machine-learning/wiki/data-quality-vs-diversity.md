# Data Quality vs Diversity

Data quality versus diversity is the recurring tension behind pretraining, reasoning data, multilingual coverage, small-model training, and SFT. The main lesson across the papers is stage-dependent: broad diversity is often what pretraining needs, high quality is what SFT and late decay phases need, and aggressive filtering can help or hurt depending on scale, contamination risk, and whether the filter removes rare useful signal.

## The Stage-Dependent Rule

The useful heuristic:

| Stage | Prefer | Why |
|---|---|---|
| Early pretraining | Diversity and coverage | The model needs broad world, language, code, and reasoning patterns |
| Late pretraining / decay | Higher-quality targeted data | Late data disproportionately shapes final behavior |
| SFT | Small, high-quality instruction and reasoning traces | Noisy demonstrations are copied directly |
| RLVR | Verifiable tasks and valid rewards | Reward correctness matters more than prose quality |
| Evaluation | Decontaminated held-out data | Leakage invalidates capability estimates |

[Front-Loading Reasoning (2510.03264)](../../papers/05-learning/reasoning/WHY DO REASONING MODELS LOOP - 2510.03264.pdf) gives the cleanest formulation: diversity drives pretraining, quality drives SFT. It also warns that scaling mixed-quality SFT data can erase benefits from earlier reasoning injection.

## Filtering Is Not Always Free

[A Bitter Lesson for Data Filtering (2605.19407)](../../papers/06-data/curation-filtering/A Bitter Lesson for Data Filtering - 2605.19407.pdf) is the cautionary note. Quality filtering can improve small or noisy settings, but at larger scales aggressive filtering can remove diversity that the model would have learned to use or ignore. Filtering for contamination, duplicates, safety, and obvious junk remains important; filtering for generic "quality" should be validated, not assumed.

[The Pile (2101.00027)](../../papers/06-data/datasets/The Pile: An 800GB Dataset of Diverse Text for Language Modeling - 2101.00027.pdf) is the older diversity-first reference: a broad mixed-domain corpus can be more useful than a single clean web source because different components contribute different capabilities.

## Domain Coverage Matters

[FineWeb2 (2506.20920)](../../papers/06-data/curation-filtering/FineWeb2: One Pipeline to Scale Them All - 2506.20920.pdf) makes the multilingual version of the argument. Low-resource language performance needs coverage, language identification, deduplication, and language-specific quality controls. A generic English-centric filter will optimize the visible high-resource slice while damaging the long tail.

[MEGASCIENCE (2507.16812)](../../papers/06-data/curation-filtering/MEGASCIENCE: Pushing the Frontiers of Large-Scale Data Collection - 2507.16812.pdf) makes the domain-specialization version. Scientific reasoning data has to be collected, selected, and decontaminated differently from general web text. "More science tokens" is not enough; task difficulty and solution quality matter.

## Sequence Composition Is Data Quality Too

[Analysing the Impact of Sequence Composition on Language Model Pre-Training (2402.13991)](../../papers/08-evaluation/analysis/Analysing The Impact of Sequence Composition on Language Model Pre-Training - 2402.13991.pdf) shows that data quality is not just document selection. Packing strategy and document boundaries change what the model sees. If unrelated documents share a packed sequence without document masking, the model can learn artificial cross-document dependencies. This connects directly to [[attention-variants]] and [[long-context-training]].

## Small Models Are Less Forgiving

[[small-efficient-models]] makes this tension more concrete. SmolLM2 and MiniCPM both treat small-model capacity as scarce. A 1-3B model cannot absorb unlimited noisy data and recover the useful patterns later, so data curation, stage-specific mixtures, and late high-quality data become first-order design choices.

## Related Topics

- [[data-curation-mixtures]] - concrete dataset mixtures, filtering, and packing details
- [[reasoning-data-generation]] - quality/diversity tradeoffs for reasoning traces
- [[small-efficient-models]] - why small models are especially sensitive to data quality
- [[tokenizers]] - tokenizer choices interact with multilingual and domain coverage
- [[long-context-training]] - document packing and long-data mixture choices
- [[frontier-training-playbook]] - where data choices sit in the full training recipe

## Sources

- [A Bitter Lesson for Data Filtering (2605.19407)](../../papers/06-data/curation-filtering/A Bitter Lesson for Data Filtering - 2605.19407.pdf)
- [FineWeb2: One Pipeline to Scale Them All (2506.20920)](../../papers/06-data/curation-filtering/FineWeb2: One Pipeline to Scale Them All - 2506.20920.pdf)
- [MEGASCIENCE: Pushing the Frontiers of Large-Scale Data Collection (2507.16812)](../../papers/06-data/curation-filtering/MEGASCIENCE: Pushing the Frontiers of Large-Scale Data Collection - 2507.16812.pdf)
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling (2101.00027)](../../papers/06-data/datasets/The Pile: An 800GB Dataset of Diverse Text for Language Modeling - 2101.00027.pdf)
- [Analysing the Impact of Sequence Composition on Language Model Pre-Training (2402.13991)](../../papers/08-evaluation/analysis/Analysing The Impact of Sequence Composition on Language Model Pre-Training - 2402.13991.pdf)
- [Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data (2510.03264)](../../papers/05-learning/reasoning/WHY DO REASONING MODELS LOOP - 2510.03264.pdf)
