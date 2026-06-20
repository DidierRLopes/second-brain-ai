# New LLM Pre-training and Post-training Paradigms (Sebastian Raschka)

**Source:** https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training
**Author:** Sebastian Raschka, PhD
**Published:** August 17, 2024
**Filed under:** Frontier model training case studies

## Premise

Compares the pre-training and post-training pipelines of four contemporaneous technical reports — Qwen 2, Apple's AFM, Gemma 2, and Llama 3.1 — as a more reliable signal of "what actually works" than the volume of individual papers proposing new techniques.

## Qwen 2 (Alibaba)

- **Specs:** 0.5B/1.5B/7B/72B dense + one 57B MoE (14B active); 151,642-token vocabulary (vs. Llama 2's 32K), strong in 30 languages.
- **Pre-training:** 7T tokens for the 1.5B/7B/72B models (0.5B model trained on 12T tokens, but larger models weren't because no improvement was observed); heavy emphasis on data-filtering-pipeline quality over raw scale. Used previous-generation Qwen models to synthesize additional pretraining data, plus multi-task instruction data woven into pretraining for in-context learning. Two-stage: regular pretraining (4,096 context) → long-context continued pretraining (32,768 context) on high-quality long data.
- **Post-training:** SFT on 500K examples for 2 epochs, then two-stage DPO — offline DPO on an existing preference dataset, then online DPO/rejection sampling where a reward model picks the preferred response from multiple model-generated candidates in real time during training. Used LLM-generated instruction-response pairs for high-quality literary Q&A data specifically.

## Apple Intelligence Foundation Models (AFM)

- **Specs:** 3B on-device model + an unspecified-size server model; respected `robots.txt`; decontaminated against benchmark data; smaller vocab (49K device / 100K server) than Qwen 2.
- **Pre-training (3 stages):**
  1. **Core pretraining** — server model on 6.3T tokens, 4096 batch/sequence length; the on-device 3B model is **distilled and pruned from a 6.4B model**, using a distillation loss that replaces target labels with a convex combination of true labels and the teacher's top-1 prediction (weight 0.9 on the teacher).
  2. **Continued pretraining** — 1T tokens, context to 8,192, math/code up-weighted, web-crawl down-weighted; distillation loss found *not* beneficial at this stage.
  3. **Context lengthening** — 100B tokens, context to 32,768, synthetic long-context Q&A data.
- **Post-training:** SFT + multiple RLHF rounds with two new algorithms: **iTeC** (Rejection Sampling Fine-tuning with Teacher Committee — train independent SFT/DPO/IPO/online-RL models, have a committee generate candidates, human-preference-label them, train a reward model, and use the committee + reward model for rejection sampling) and **RLHF with Mirror Descent Policy Optimization** (chosen over PPO for effectiveness). Data mixture tuned empirically rather than fixed by predetermined ratios.

## Gemma 2 (Google)

- **Specs:** 2B/9B/27B; 256K-token vocabulary (largest of the four); sliding-window attention.
- **Pre-training:** focuses on improvements without scaling data further — the 27B model trained from scratch on 13T tokens; the 9B (8T tokens) and 2B (2T tokens) models trained via **knowledge distillation** from a larger teacher, the same "don't just scale data, distill" theme as AFM.
- **Post-training:** SFT (English-only prompts, human + synthetic, with knowledge-distillation-generated responses during SFT) + RLHF using a reward model **10× larger** than the policy model, then **WARP** (successor to WARM, weight-averaged reward models) to average policy models.

## Llama 3.1 (Meta)

- **Specs:** 8B/70B/405B dense, no MoE, no sliding-window attention — architecturally conservative, effort concentrated on pre/post-training instead. 128K-token vocabulary via OpenAI's tiktoken. License updated to permit synthetic-data generation/distillation from Llama 3 outputs.
- **Pre-training (3 stages):** 15.6T tokens total (vs. Llama 2's 1.8T). Heuristic + model-based filtering (fastText, RoBERTa classifiers) for data-mix categorization. Stage 1: standard pretraining, 8K context, batch size doubled at 252M tokens and again at 2.87T tokens, with a non-static data mix. Stage 2: context lengthening 8K→128K across **six gradual steps** (800B tokens, ~5% of total). Stage 3: annealing on a small high-quality mix (40B tokens assessed for quality, ~40M tokens actually annealed on) — annealing on GSM8K/MATH training sets measurably boosted GSM8K/MATH validation performance.
- **Post-training:** SFT + rejection sampling + DPO, repeated over multiple iterative rounds with both human and synthetic data; explicitly avoided PPO-style RLHF as less stable/harder to scale at this size. Still trains a reward model (from a pretraining checkpoint + human-annotated data) to drive rejection sampling. Applies **model averaging** (weight averaging across recent/previous checkpoints) not just to the reward model but to the SFT and DPO models too.

## Cross-Model Takeaways

- All four use **multi-stage pretraining**: core pretraining → context lengthening → (sometimes) high-quality annealing.
- **Rejection sampling** is now a near-universal post-training staple; no consensus yet on DPO vs. RLHF-with-PPO-style methods.
- **Knowledge distillation** recurs at pretraining time for smaller models in a family (AFM 3B, Gemma 2 9B/2B) as an alternative to scaling raw token count.
- **Model/weight averaging** recurs in post-training (Gemma's WARP/WARM, Llama 3.1's SFT/DPO/RM averaging) as a stabilization technique distinct from RL-time KL regularization.
- No single recipe dominates — the four pipelines diverge significantly in algorithm choice even while converging on the same high-level *stages*.

## Related wiki pages

[[model-report-case-studies]], [[frontier-training-playbook]], [[knowledge-distillation]], [[alignment-methods]], [[scaling-laws]]
