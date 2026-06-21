# Reasoning Data Generation

Reasoning data generation is the part of the training stack that turns verifiable tasks, teacher models, synthetic generators, and curated traces into data that actually improves reasoning. It connects [[reasoning-models]], [[supervised-fine-tuning]], [[alignment-methods]], [[knowledge-distillation]], [[data-quality-vs-diversity]], and [[llm-evaluation]]: the key question is not just whether examples are correct, but whether they contain the behaviors and distributional coverage that RL or distillation can amplify.

## The Main Data Sources

Reasoning data now comes from several different pipelines:

| Source | Best use | Main risk |
|---|---|---|
| Teacher-generated long-CoT traces | SFT initialization and distillation | Teacher style and artifacts dominate |
| Verifier-backed synthetic tasks | RLVR and scalable correctness rewards | Narrow task distribution |
| Curated natural math/code/science problems | General reasoning pretraining and SFT | Contamination and uneven difficulty |
| Behavior-enriched web/math text | Installing cognitive behaviors before RL | Weak direct control over final task ability |
| Agent/tool trajectories | Training tool-use and software agents | Harness reward hacks and leakage |

[OpenThoughts: Data Recipes for Reasoning Models (2506.04178)](../../papers/05-learning/reasoning/OpenThoughts: Data Recipes for Reasoning Models - 2506.04178.pdf) is the strongest data-pipeline reference. The OpenThoughts recipe found that sampling 16 answers per question was unusually high leverage, that QwQ-32B was a better teacher than DeepSeek-R1 despite lower benchmark scores, and that answer filtering was much less useful than expected. The pipeline's final shape was 850k math, 250k code, and 100k science examples, with teacher choice and question filtering more important than response filtering.

## Reasoning Behaviors Matter Before RL

[Cognitive Behaviors that Enable Self-Improving Reasoners (2503.01307)](../../papers/05-learning/reasoning/Cognitive Behaviors that Enable Self-Improving Reasoners - 2503.01307.pdf) reframes data quality around behaviors rather than labels. Verification, backtracking, subgoal setting, and backward chaining are the behaviors that let RL improve a model. If the base model almost never emits them, RL has little to amplify.

This is the actionable lesson: before spending RL compute, sample traces from the base model and measure whether the target behaviors appear. If not, enrich pretraining or SFT data with examples that contain those behaviors. Correct answers alone are not enough; the paper found that incorrect traces with the right behaviors could still enable later RL improvement.

## Synthetic Logic And Verifiers

[SynLogic (2505.19641)](../../papers/05-learning/reasoning/SynLogic: A Data Synthesis Framework for Logical Reasoning - 2505.19641.pdf) is the clean synthetic-data case. It generates many logical task families with rule-based verifiers, controls difficulty through task parameters, and trains with binary verifier rewards. The striking result is cross-domain transfer: logic-only training improved math benchmarks, suggesting that some reasoning skills transfer across surface domains.

This makes SynLogic a bridge between [[data-curation-mixtures]] and [[alignment-methods]]. The data is useful not because it resembles every downstream benchmark, but because it supplies verifier-backed search pressure over abstract reasoning operations.

## Where To Put Reasoning Data

[Front-Loading Reasoning (2510.03264)](../../papers/05-learning/reasoning/Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data - 2510.03264.pdf) makes the timing issue explicit. Reasoning data in pretraining creates durable foundations that SFT cannot fully recover later. But pretraining and SFT prefer different data:

- Pretraining benefits from broad, diverse reasoning data.
- SFT benefits from smaller, higher-quality long-CoT data.
- Scaling mixed-quality SFT data can wash out the gains from earlier reasoning injection.

The practical blueprint is: use large diverse reasoning data during pretraining, use small high-quality reasoning traces during SFT, then use [[rl-training-systems|RL with verifiers]] when the model needs to exceed its teacher.

## Small-Model Specialization

[QED-Nano (2604.04898)](../../papers/05-learning/reasoning/QED-Nano: Teaching a Tiny Model to Prove Hard Theorems - 2604.04898.pdf) shows how data generation and distillation can specialize a small model. A 4B model gets SFT initialization from teacher-generated Olympiad proof traces, then RL with rubric rewards, then a reasoning-cache scaffold for test-time adaptation. The important pattern is staged: teacher traces install the proof-writing prior, verifier/rubric RL corrects it, and scaffold training teaches the model to use more inference compute.

## Related Topics

- [[reasoning-models]] - model behavior, failure modes, DeepSeek-R1, GRAM, QED-Nano
- [[supervised-fine-tuning]] - where teacher traces and long-CoT data enter
- [[alignment-methods]] - GRPO, DAPO, RLVR, rubric rewards
- [[knowledge-distillation]] - transferring reasoning traces from larger teachers
- [[data-quality-vs-diversity]] - why pretraining and SFT prefer different data
- [[llm-evaluation]] - contamination, verifier validity, and trace analysis

## Sources

- [OpenThoughts: Data Recipes for Reasoning Models (2506.04178)](../../papers/05-learning/reasoning/OpenThoughts: Data Recipes for Reasoning Models - 2506.04178.pdf)
- [Cognitive Behaviors that Enable Self-Improving Reasoners (2503.01307)](../../papers/05-learning/reasoning/Cognitive Behaviors that Enable Self-Improving Reasoners - 2503.01307.pdf)
- [SynLogic: A Data Synthesis Framework for Logical Reasoning (2505.19641)](../../papers/05-learning/reasoning/SynLogic: A Data Synthesis Framework for Logical Reasoning - 2505.19641.pdf)
- [Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data (2510.03264)](../../papers/05-learning/reasoning/Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data - 2510.03264.pdf)
- [QED-Nano: Teaching a Tiny Model to Prove Hard Theorems (2604.04898)](../../papers/05-learning/reasoning/QED-Nano: Teaching a Tiny Model to Prove Hard Theorems - 2604.04898.pdf)
