# Subliminal Learning and Steering-Vector Distillation

Subliminal learning occurs when a student inherits a teacher's behavioral trait from training outputs that are semantically unrelated to that trait. The phenomenon matters for data provenance and safety because filtering explicit mentions is not sufficient to guarantee that a teacher-generated dataset is behaviorally neutral.

## Steering-Vector Account

[Blank et al. (2606.00995)](../../papers/05-learning/fine-tuning/Subliminal Learning Is Steering Vector Distillation - 2606.00995.pdf) study Qwen2.5-7B-Instruct and Gemma-3-4B-it teachers given trait-inducing system prompts, such as a preference for cats. The teacher generates number sequences or code; explicit trait references are filtered; and a same-family student is fine-tuned on 10,000 examples with rank-8 LoRA.

The proposed mechanism is a residual-stream steering vector. The teacher vector is the mean activation difference between trait-prompted and neutral runs. Fine-tuning installs an aligned student vector:

- adding the teacher vector to the unprompted reference model reproduces the trait;
- ablating it from the prompted teacher suppresses transfer;
- adding the learned student vector to the untrained reference model reproduces the learned preference;
- ablating the student vector from the fine-tuned model reduces the preference.

System prompts that are poorly approximated by a single steering direction are not reliably learned subliminally. Random vectors can also be distilled, but only semantically meaningful directions yield a recognizable behavioral trait. The paper argues that a vector can have both model-independent semantic effects and model-specific non-semantic effects: filtered outputs retain the latter, allowing a same-family student to reconstruct a direction that again has semantic consequences. This also explains weak transfer across unrelated model families.

## Why the Optimizer and Adaptation Method Matter

The effect was strongest under low-rank adaptation with an adaptive optimizer. Plain SGD failed to install the teacher direction in the authors' main settings because a small set of unusually large LoRA-coordinate gradients dominated the update. Adam/RMSProp-style per-parameter scaling attenuated those outliers and preserved the small, consistent component aligned with the steering vector. Full fine-tuning distilled vectors less consistently, and the recognizable subliminal trait appeared only under low-rank training in the main LLM experiments.

The mechanism is not claimed to be exhaustive. On Qwen, vector ablation did not eliminate all trait affinity, Gemma showed weaker effects, the study covered only a small set of prompted teachers, and longer Qwen training made the effect less salient. The result is therefore a causal mechanism for an important component of subliminal transfer, not proof that every hidden training-data influence is one-dimensional.

## Safety Implications

- Semantic filtering of synthetic data does not remove all teacher-specific behavioral signals.
- Same-family distillation and LoRA+Adam are particularly important audit settings.
- Activation-space comparisons between teacher, data-generating policy, and student can complement output-only data inspection.
- Mitigation should test causal ablation and cross-model controls rather than relying only on keyword filters.

## Related Topics

- [[knowledge-distillation]] — the broader teacher-student training process
- [[parameter-efficient-fine-tuning]] — why low-rank adaptation changes the transfer mechanism
- [[safety-misalignment]] — narrow training signals can generalize into broader behavior
- [[alignment-methods]] — behavioral control and unintended transfer during post-training

## Sources

- [Subliminal Learning Is Steering Vector Distillation (2606.00995)](../../papers/05-learning/fine-tuning/Subliminal Learning Is Steering Vector Distillation - 2606.00995.pdf) — causal steering/ablation experiments, model specificity, random-vector distillation, optimizer dependence, and limitations.
