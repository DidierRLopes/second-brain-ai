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

**Student memorization is predictable before paying for distillation.** A logistic regression classifier trained on four pre-distillation features — zlib entropy, baseline perplexity, teacher perplexity, and teacher/baseline KL-divergence loss — discriminates memorized vs. non-memorized examples in the student with AUC-ROC of 0.9997 and Recall of 1.0000 (zero missed memorized examples across 100 resampled trials), with zlib entropy by far the dominant feature (coefficient −4.50). Acting on this prediction works: removing the pre-identified at-risk examples from the training set before distillation drops the student's total memorized-example count from 1,698 to just 4 — a 99.8% reduction — at the cost of only 4 newly-memorized examples appearing elsewhere, confirming a practitioner can flag and filter likely-to-be-memorized training examples *before* running the expensive distillation job rather than auditing after the fact.

**Why distillation reduces memorization: a confidence/entropy mechanism.** The paper explains the regularizing effect mechanistically by plotting each model's average Shannon entropy (token-level uncertainty) against its sequence log-probability (confidence) on the baseline's hardest-to-fit examples. Three distinct regimes emerge: the 12B teacher is genuinely confident on these examples (high log-probability, low entropy — real competence). The 1.4B baseline, trained with hard cross-entropy targets, is *forced* into high confidence despite high intrinsic uncertainty (high log-probability, high entropy) — cross-entropy's one-hot target compels the model to overfit sequences it doesn't actually have the capacity to model, which the paper calls "forced memorization." The 1.4B student, trained with the same capacity but soft KL targets, stays uncertain (high entropy) *and* stays low-confidence (low log-probability) on those same examples — because KL divergence lets the student match the teacher's own (appropriately uncertain) distribution instead of forcing a hard target, the student is permitted to output a flatter distribution rather than memorizing. The student's actual memorized set clusters tightly with the teacher's, in the low-entropy region, confirming it selectively memorizes only what it's genuinely confident about.

**Hard (sequence-level) distillation carries a distinct, larger inheritance risk than soft (logit-level) distillation.** The paper also trains a student on the teacher's own greedily-decoded output sequences via plain cross-entropy (hard/sequence-level KD, as opposed to matching the teacher's full softmax distribution). Despite the different objective, hard distillation achieves an essentially identical overall memorization rate to soft distillation (0.07% vs. baseline's 0.17%) while still beating the baseline on downstream utility (LAMBADA accuracy 56.65% vs. baseline's 51.85%, perplexity 6.43 vs. 9.41; Winogrande accuracy 57.46% vs. 55.72%). The two distillation modes memorize largely the same examples (~70% overlap), and 90% of what the hard-distilled student memorizes falls in the same teacher∩baseline "easy" set as soft distillation. But on the residual, hard-distillation-only memorized examples, the risk profile diverges sharply: of the 46 examples memorized exclusively by the hard-distilled student, 80% are also memorized by the teacher, and the total count of examples inherited *only* from the teacher (memorized by teacher+student but not baseline) is 50 for hard distillation versus 18 for soft — a **2.7× increase in teacher-specific memorization inheritance**. The mechanism is plausible: training directly on the teacher's generated text (rather than matching its probability distribution) gives the student a more direct path to reproducing the teacher's idiosyncratic memorized continuations.

**Privacy implication — soft vs. hard distillation aren't equivalent.** Comparing logit-level ("soft," KL-divergence-trained) against sequence-level ("hard," trained on the teacher's generated outputs as if they were labels) distillation: the two have similar *overall* memorization rates, and the soft-distilled student's memorized set captures over 70% of what the hard-distilled student memorizes — but **hard distillation inherits 2.7× more teacher-specific examples than soft distillation**. For privacy-sensitive distillation pipelines (e.g., distilling from a teacher trained on data the student shouldn't be able to leak), this is a concrete, actionable reason to prefer soft logit-matching over hard-label/generated-output distillation, beyond the general utility argument made for soft targets in the Hinton section above.

## Proxy-KD for Black-Box Teachers

[Proxy-KD (2401.07013)](../../papers/05-learning/fine-tuning/Knowledge Distillation of Black-Box Large Language Models - 2401.07013.pdf) addresses the missing soft labels of proprietary teachers. A larger white-box proxy is first warmed up on teacher outputs and aligned to the black-box teacher using hard-label NLL plus iterative DPO, treating teacher responses as preferred over the proxy's own samples. The student then combines hard-label NLL on black-box outputs with KL matching to the proxy's token distribution. A weight based on the proxy likelihood of the teacher response downweights examples where proxy-teacher alignment is weak.

Using GPT-4 as teacher, Llama-2-70B as proxy, and Llama-1/2-7B students, Proxy-KD reached average benchmark scores of 52.09 and 56.78, versus 49.11 and 53.66 for vanilla black-box fine-tuning. Alignment was essential: removing it reduced BBH by 10.40 points and GSM8K by 5.53 points for the Llama-2 student. Removing the proxy entirely reduced ARC by 4.24, BBH by 6.72, and GSM8K by 3.56. The method's cost is an additional large-model alignment stage and online proxy sampling; results were limited to Llama-family backbones.

## Compressing Reasoning Traces Before Distillation

[Compress-Distill (2606.05988)](../../papers/05-learning/fine-tuning/Compress-Distill: Reasoning Trace Compression for Efficient Knowledge Distillation - 2606.05988.pdf) tests whether verified teacher reasoning can be rewritten into a shorter target before student fine-tuning. Qwen3.5-397B-A17B and gpt-oss-120B generated roughly 283,000 correct traces each; Llama-3.3-70B and Ministral-3-14B compressed them to 8.6-21.0% of their original character length.

Across 48 main runs and seven length-matched truncation ablations, compressed traces used 12-30% as many training tokens, reduced wall-clock training by 2.0-7.6x, and shortened student reasoning by 3-19x. Students retained up to 96% of raw-trace accuracy and achieved up to 18x higher accuracy per generated token, but **raw traces won on absolute accuracy at every tested scale and under both teachers**. Model-written compression usually beat naive prefix truncation at the same token budget, showing that preservation of answer-bearing reasoning structure matters. Answer-only training was cheapest but performed worst and became unstable under full fine-tuning. The practical choice is a Pareto trade-off: use raw traces for peak quality, compressed traces when training/inference cost or context truncation dominates.

## Hidden Behavioral Transfer

Distillation can transmit more than explicit semantic content. [[subliminal-learning]] documents evidence that same-family students can reconstruct a teacher's activation-space steering direction from semantically unrelated outputs, especially under LoRA with adaptive optimizers. Output filtering is therefore not a complete behavioral-safety check for synthetic distillation data.

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
- [[subliminal-learning]] — hidden trait transfer as steering-vector distillation
- [[reasoning-data-generation]] — generating and filtering teacher reasoning traces

## Sources
- Distilling the Knowledge in a Neural Network — Hinton et al. (arxiv:1503.02531)
- DistilBERT (arxiv:1910.01108)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948)](../../papers/01-models/gpt-deepseek-v2-v3/DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning - 2501.12948.pdf)
- [QED-Nano: Teaching a Tiny Model to Prove Hard Theorems (2604.04898)](../../papers/05-learning/reasoning/QED-Nano: Teaching a Tiny Model to Prove Hard Theorems - 2604.04898.pdf)
- [Memorization Dynamics in Knowledge Distillation for Language Models (2601.15394)](../../papers/05-learning/fine-tuning/Memorization Dynamics in Knowledge Distillation for Language Models - 2601.15394.pdf) — Borkar, Chadha, Mireshghallah, Zhang et al., Meta Superintelligence Labs / FAIR / Northeastern / CMU (2026)
- [Knowledge Distillation of Black-Box Large Language Models / Proxy-KD (2401.07013)](../../papers/05-learning/fine-tuning/Knowledge Distillation of Black-Box Large Language Models - 2401.07013.pdf) — aligned white-box proxy, DPO, sample-weighted KL, and black-box teacher results.
- [Compress-Distill: Reasoning Trace Compression for Efficient Knowledge Distillation (2606.05988)](../../papers/05-learning/fine-tuning/Compress-Distill: Reasoning Trace Compression for Efficient Knowledge Distillation - 2606.05988.pdf) — trace compression ratios, training and inference savings, truncation controls, and the accuracy-efficiency frontier.
- [Subliminal Learning Is Steering Vector Distillation (2606.00995)](../../papers/05-learning/fine-tuning/Subliminal Learning Is Steering Vector Distillation - 2606.00995.pdf) — non-semantic behavioral transfer through activation-space directions.
