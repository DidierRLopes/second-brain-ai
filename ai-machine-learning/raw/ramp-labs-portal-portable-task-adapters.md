# PorTAL: Portable Task Adapters for LLMs

**Source:** https://labs.ramp.com/research
**Author:** Ben Geist, Ramp Labs
**Published:** June 2026
**Citation:** Geist, B. (2026). *PorTAL: Portable Task Adapters for LLMs*. Ramp Labs.

```bibtex
@techreport{portal2026ramplabs,
  author = {Geist, Ben},
  title = {PorTAL: Portable Task Adapters for LLMs},
  year = {2026},
  month = {June},
  institution = {Ramp Labs},
  url = {https://labs.ramp.com/research}
}
```

## Motivation

A LoRA (or full fine-tune) is locked to the base model it was trained on — when a new, better base model ships, the adaptation must be relearned from scratch. This is a real and growing cost: notable foundation-model releases went from 2/year (2020) to 149/year (2023) [Stanford HAI AI Index 2024], and by 2024-2025 a new SOTA model held the top public leaderboard spot for only ~35 days on average (down from ~1 year for GPT-4) [Chatbot Arena / LMArena leaderboard dataset]. Per-model re-tuning cost therefore scales roughly inversely with time between releases, becoming the dominant ongoing cost of keeping a system specialized while still benefiting from newer, smarter bases.

PorTAL's answer: pay for task adaptation once, in a base-agnostic form, then port it to every future base by refitting only a small per-model component.

## Method

- **Task latent** `z_t`: a base-agnostic vector (dim 256) learned per task, shared across all base models.
- **Decoder** `D_b`: a hypernetwork that generates per-layer LoRA factors (A, B) for a frozen base from `z_t` and a per-layer embedding `e_ℓ`. It has two parts:
  - A **base-agnostic shared core**: a FiLM-conditioned trunk (per-layer embedding as input; task latent scales/shifts hidden features via γ, β) feeding per-module heads that output core-width factors (rank r, fixed width d_c).
  - A **thin per-base converter**: per-module linear "aligner" maps `P_in`, `P_out` that project the core-width factors to the specific base's actual dimensions, plus the per-layer embeddings `e_ℓ` themselves.
- **Training**: `{z_t}` and `D_b` trained jointly, base frozen, minimizing gold-continuation NLL (loss on answer tokens only). Multi-task training uses balanced per-task steps + EMA loss normalization. Multi-base training applies gradient-norm balancing on `z_t` so no single base dominates the shared latent's gradient.
- **Porting to an unseen base `b'`**: freeze the core decoder and `{z_t}`; refit *only* the per-base converter (`e_ℓ`, `P_in`, `P_out`) on a small calibration set via the same NLL objective.
- Initialization: B-heads and FiLM γ, β are zero-initialized so the generated adapter starts as the identity (ΔW = 0).

This is architecturally close to LoRAGen (structural embedding → LoRA hypernetwork) but trained end-to-end on task loss rather than by reconstructing existing LoRAs, and adds the freeze-core/refit-converter porting recipe that LoRAGen doesn't have.

## Experimental Setup

- 14 standard multiple-choice tasks: TruthfulQA, RTE, CB, COPA, WiC, WSC, BoolQ, ARC-Easy, ARC-Challenge, HellaSwag, OpenBookQA, WinoGrande, CommonsenseQA, SciQ.
- Metric: length-normalized log-likelihood (acc_norm), plus held-out log-loss (token-mean NLL of gold continuation). 3-seed means ± std.
- Up to 2,000 examples/task for both source training and converter refit (hard cap). Eval sets: 56 (CB) to 1,000 (BoolQ/WinoGrande/CSQA/SciQ) examples; ~7,200 eval examples total.
- Seen bases: Qwen3-1.7B, Qwen3-4B. Unseen bases: Qwen3-8B (within-family) and Gemma-3-4B (cross-family).
- Per-task LoRA baseline: rank 16 on q/k/v/o + MLP (found to be the strongest per-task config in a sweep). PorTAL / LoRA Hypernet: rank 8 on q/v.
- Hyperparameters: AdamW, LR 1e-3 (decoder) / 2e-3 (latent), 5 epochs, batch size 4, single NVIDIA B200 per run.

## Key Results

**Source base (Qwen3-4B), avg acc_norm over 14 tasks:**
| Method | Avg acc_norm |
|---|---|
| Base | 0.627 |
| Per-task LoRA | 0.765 ± 0.003 |
| LoRA Hypernet (jointly trained z, D on 4B) | 0.757 ± 0.003 (~94% of LoRA's lift; matches/beats LoRA on 6/14 tasks) |

**Within-family portability to unseen Qwen3-8B:**
| Method | Avg acc_norm | Recovered lift |
|---|---|---|
| Base-8B | 0.667 | — |
| Per-task 8B LoRA | 0.795 ± 0.004 | 100% |
| Cross-LoRA transfer (baseline) | 0.685 ± 0.001 | ~14% |
| LoRA Hypernet (jointly trained on 8B) | 0.785 ± 0.002 | ~92% |
| **PorTAL** (frozen z + core from 1.7B+4B, refit converter on 8B) | 0.792 ± 0.004 | **~98%** |

PorTAL's ported result (~98%) is statistically on par with, even slightly above, training the latent/decoder jointly from scratch on the 8B (~92%) — attributed to mild regularization from training across multiple seen bases.

**Cross-family portability to unseen Gemma-3-4B** (core/latent learned only on Qwen3-1.7B + 4B): base 0.595, per-task LoRA 0.778 ± 0.004, PorTAL 0.767 ± 0.004 → **~94% recovered lift**. Cross-family transfer is "nearly lossless."

**Data efficiency (unseen Qwen3-8B, base acc 0.667 / log-loss 3.819):** PorTAL reaches per-task LoRA's peak accuracy with roughly **half the calibration data**, and at every data size shows lower held-out log-loss (better calibration) than a from-scratch LoRA at equal accuracy. Since the frozen base dominates per-step cost, halving the data roughly halves the FLOPs needed to adapt each subsequent base.

## Metric Definition Note

The paper reports *recovered lift* rather than *retention*, arguing retention is non-discriminative in low-headroom settings (where prior cross-model-transfer papers like Cross-LoRA/CAST report only ~1% LoRA gain over base). Definitions, for method m, unadapted base b, from-scratch per-task LoRA L:
- `recovered lift = (acc_m - acc_b) / (acc_L - acc_b)`
- `retention = acc_m / acc_L`

In retention terms, their Cross-LoRA reimplementation scores ~86% (within CAST's reported 85-95% band) while recovering only ~14% of the lift — illustrating why retention looks misleadingly strong when headroom is small. PorTAL: ~99% retention / ~98% recovered lift.

## Comparison to Prior Work (as framed by the paper)

- **Single-base LoRA hypernetworks** (Text-to-LoRA, in-context SHINE, Profile-to-PEFT): amortize across tasks/users but stay locked to one base model.
- **Cross-architecture LoRA generation** (LoRAGen): structural-embedding hypernetwork like PorTAL's decoder, but trained by reconstructing existing LoRAs rather than end-to-end task loss, and without PorTAL's freeze-core/refit-converter porting step.
- **Cross-model LoRA transfer** (Cross-LoRA, LoRA-X, CAST): translate an *already-trained* adapter to a new base via subspace/activation-manifold alignment, with no refitting step. PorTAL's ablation shows this data-free translation approach (Cross-LoRA) recovers only ~14% of LoRA's lift on an unseen 8B model, vs. PorTAL's ~98% — the paper's core empirical argument for why refitting a thin per-base component beats data-free adapter translation.

## Limitations / Future Work Noted by the Authors

- Under best-epoch selection, most tasks reach LoRA's lift, but a few harder tasks underfit: OpenBookQA (~42% of lift), WinoGrande (~57%), HellaSwag (~61%). Hypothesized cause: gradient competition in the shared rank-8 decoder across a 14-task suite (optimization issue, not adapter capacity — neither larger rank nor larger latent helped).
- Proposed extension: replace the free per-task latent with a text-description encoder (`z_t = E(emb(desc_t))`) for zero-shot task adaptation from a description alone, à la Text-to-LoRA.
- Scope acknowledged as limited to multiple-choice tasks on mid-size open models; larger/generative tasks and theory of when a frozen latent suffices are left open.

## Related wiki pages

[[parameter-efficient-fine-tuning]], [[knowledge-distillation]]
