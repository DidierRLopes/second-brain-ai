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

## Sources
- Distilling the Knowledge in a Neural Network — Hinton et al. (arxiv:1503.02531)
- DistilBERT (arxiv:1910.01108)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948)](../../papers/01-models/gpt-deepseek-v2-v3/DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning - 2501.12948.pdf)
- [QED-Nano: Teaching a Tiny Model to Prove Hard Theorems (2604.04898)](../../papers/05-learning/reasoning/QED-Nano: Teaching a Tiny Model to Prove Hard Theorems - 2604.04898.pdf)
