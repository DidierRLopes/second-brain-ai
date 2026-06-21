# Knowledge Distillation

Knowledge distillation is the technique of training a smaller "student" model to replicate the behavior of a larger "teacher" model. The student learns from the teacher's soft probability outputs rather than hard labels, capturing richer information about the data structure. This enables deploying large-model capabilities on resource-constrained hardware.

## The Core Idea (Hinton et al., 2015)

Geoffrey Hinton's seminal paper showed that soft targets — the full probability distribution output by the teacher — carry "dark knowledge" that hard labels don't. When a teacher outputs [cat: 0.7, dog: 0.2, car: 0.1], the student learns not just "this is a cat" but also "this looks somewhat like a dog and nothing like a car." This inter-class similarity information transfers structural understanding.

The **temperature** parameter controls softness: higher temperatures spread probability mass more evenly, revealing more relationships between classes. Typical temperatures range from 2-20.

## DistilBERT: Distillation in Practice

DistilBERT (Sanh et al., 2019, Hugging Face) is the canonical success story. It achieves 40% size reduction from BERT while retaining 97% of language understanding and running 60% faster. The approach uses a triple loss: language modeling loss, distillation loss (matching teacher logits), and cosine-distance loss (aligning hidden representations).

This template — combining multiple loss signals — became standard for subsequent distillation work.

## Reasoning Distillation: DeepSeek-R1

The most exciting recent development is distilling **reasoning capabilities**. DeepSeek-R1 first uses RL (GRPO) to develop chain-of-thought reasoning in a large model, then distills the step-by-step reasoning traces into smaller models (1.5B to 70B parameters). The distilled models achieve remarkably strong reasoning — often outperforming the original model on certain benchmarks — because the reasoning traces provide much richer supervision than simple answer labels.

This connects distillation to [[reasoning-models]]: you can train reasoning ability via RL in a large model, then compress it into a deployable-size model via distillation.

[QED-Nano: Teaching a Tiny Model to Prove Hard Theorems (2604.04898)](../../papers/05-learning/reasoning/QED-Nano: Teaching a Tiny Model to Prove Hard Theorems - 2604.04898.pdf) is the small-model theorem-proving version of the same pattern. A 4B Qwen3 model is first initialized with SFT on roughly 7,500 natural-language proofs distilled from DeepSeek-Math-V2, then improved with rubric-based RL and a reasoning-cache scaffold. The important distillation lesson is that teacher-generated proof traces can give a tiny model the proof-writing style and structure needed before RL handles correctness and length control.

## On-Policy Distillation: The Modern Variant

Hinton-style distillation is offline — you generate teacher outputs once, then train the student against them as a fixed dataset. That bakes in an exposure-bias gap: the student trains on the teacher's state distribution but is evaluated on its own. [[on-policy-distillation|On-policy distillation (OPD)]] (Agarwal et al. 2023; Lu et al. 2025; Qwen3 technical report) closes that gap by having the *student* sample its own rollouts, with the teacher providing a per-token reverse-KL signal at each position. Reported gains are roughly 9–30× less compute than RL on AIME-style benchmarks. The catch: OPD essentially requires a same-family teacher (tokenizer- and recipe-matched), since the per-token KL has to be computed over the same vocabulary at the same positions and the signal needs to be informative on capability rather than dominated by stylistic differences.

## Memorization Dynamics in Knowledge Distillation

A natural question about all the distillation recipes above: does compressing a large teacher into a small student also compress *which training examples get memorized verbatim* — and does that change the privacy picture? [Memorization Dynamics in Knowledge Distillation for Language Models (2601.15394)](../../papers/05-learning/fine-tuning/Memorization Dynamics in Knowledge Distillation for Language Models - 2601.15394.pdf) (Borkar, Chadha, Mireshghallah, Zhang et al. — Meta Superintelligence Labs / FAIR / Northeastern / CMU) studies this directly, training teacher/student/baseline triples across three LLM families (Pythia, OLMo-2, Qwen-3) and three datasets (FineWeb, WikiText, Nemotron-CC-v2), using discoverable memorization (does greedy decoding from a 50-token prefix exactly reproduce the 50-token training-data suffix?) as the metric.

**Distillation memorizes substantially less than fine-tuning, without sacrificing quality.** Logit-level distillation (forward-KL between teacher and student softmax distributions, temperature T=2) cuts memorization by more than 50% relative to standard cross-entropy fine-tuning of a same-size baseline — concretely, 2.4× lower on FineWeb and 2.1× lower on WikiText for Pythia, and the gap widens to nearly an order of magnitude on the synthetic Nemotron-CC-v2 dataset (0.0012% vs. 0.0091%). Critically, the student isn't memorizing less by simply learning less: the distilled student achieves *better* validation loss and perplexity than the baseline across Pythia, OLMo-2, and Qwen-3, so distillation is improving generalization and reducing memorization simultaneously rather than trading one for the other. Increasing the distillation temperature further reduces student memorization.

**The student inherits the teacher's competence but rejects its specific memorized examples.** Of 1,955 examples memorized by the teacher but not the same-size baseline, the student inherited only 18 of them — about 0.9%. This directly contradicts the intuitive worry (and prior findings in sequence-distillation-for-MT work by Dankers and Raunak, 2025) that students "inherit" their teacher's memorization; for logit-level LLM distillation the paper finds close to the opposite.

**Memorization is concentrated on a predictable "easy" subset, not spread uniformly.** Across model scales and three independent random seeds, the same examples get memorized again and again — 96% of what a 1B model memorizes persists in a 1.4B model trained the same way, and the 12B teacher recaptures ~80% of the 1.4B baseline's memorized set. These "easy-to-memorize" examples form a tight, separable cluster of low zlib-compressibility and low perplexity (i.e., unusually repetitive/predictable text) versus a random sample of training data. 95.7% of everything the distilled student does memorize falls inside this teacher∩baseline easy-to-memorize set — distillation doesn't introduce new memorization risk so much as it raises the bar for which already-easy examples make it through at all (494 examples the baseline memorizes in all 3 seeds, the student never memorizes in any). Notably, which specific examples count as "easy" is architecture-specific: despite near-perfect cross-model agreement on zlib-entropy/perplexity rankings (Pearson r = 0.95–0.99 between Pythia, OLMo-2, Qwen-3), there is zero overlap in which actual examples each model family memorizes — tokenization and architecture-specific inductive biases determine the selection, not just text-intrinsic difficulty.

**Student memorization is predictable before paying for distillation.** A logistic regression classifier trained on four pre-distillation features — zlib entropy, baseline perplexity, teacher perplexity, and teacher/baseline KL-divergence loss — discriminates memorized vs. non-memorized examples in the student with AUC-ROC of 0.9997 and Recall of 1.0000 (zero missed memorized examples across 100 resampled trials), with zlib entropy by far the dominant feature (coefficient −4.50). This means a practitioner can flag and filter likely-to-be-memorized training examples *before* running the expensive distillation job, rather than auditing after the fact.

**Privacy implication — soft vs. hard distillation aren't equivalent.** Comparing logit-level ("soft," KL-divergence-trained) against sequence-level ("hard," trained on the teacher's generated outputs as if they were labels) distillation: the two have similar *overall* memorization rates, and the soft-distilled student's memorized set captures over 70% of what the hard-distilled student memorizes — but **hard distillation inherits 2.7× more teacher-specific examples than soft distillation**. For privacy-sensitive distillation pipelines (e.g., distilling from a teacher trained on data the student shouldn't be able to leak), this is a concrete, actionable reason to prefer soft logit-matching over hard-label/generated-output distillation, beyond the general utility argument made for soft targets in the Hinton section above.

## Distillation vs Quantization vs Fine-Tuning

These three techniques serve different purposes and are often combined:

**[[quantization-fundamentals|Quantization]]** reduces numerical precision of the same model. No retraining needed, but the architecture stays the same.

**Distillation** trains a genuinely smaller architecture. Requires training compute but produces a faster model with fewer parameters.

**[[parameter-efficient-fine-tuning|Fine-tuning]]** adapts a model to new tasks. Often uses distillation as a data source (training on teacher outputs).

In practice, you might distill a 70B model into a 7B model, then quantize the 7B to 4-bit, then fine-tune with LoRA for your specific use case.

## Related Topics
- [[on-policy-distillation]] — The modern on-policy variant; sits between SFT and RL
- [[reasoning-models]] — Reasoning distillation from DeepSeek-R1
- [[quantization-fundamentals]] — Complementary compression technique
- [[inference-optimization]] — Distillation reduces serving cost
- [[scaling-laws]] — Distillation changes the compute-quality tradeoff
- [[frontier-training-playbook]] — Apple AFM (distill-and-prune 6.4B→3B) and Gemma 2 (27B→9B, 7B→2B) as production pretrain-time distillation case studies
- [[safety-misalignment]] — Memorization and training-data extraction as a privacy/safety concern more broadly

## Sources
- Distilling the Knowledge in a Neural Network — Hinton et al. (arxiv:1503.02531)
- DistilBERT (arxiv:1910.01108)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948)](../../papers/01-models/gpt-deepseek-v2-v3/DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning - 2501.12948.pdf)
- [QED-Nano: Teaching a Tiny Model to Prove Hard Theorems (2604.04898)](../../papers/05-learning/reasoning/QED-Nano: Teaching a Tiny Model to Prove Hard Theorems - 2604.04898.pdf)
- [Memorization Dynamics in Knowledge Distillation for Language Models (2601.15394)](../../papers/05-learning/fine-tuning/Memorization Dynamics in Knowledge Distillation for Language Models - 2601.15394.pdf) — Borkar, Chadha, Mireshghallah, Zhang et al., Meta Superintelligence Labs / FAIR / Northeastern / CMU (2026)
