# Parameter-Efficient Fine-Tuning (PEFT)

Parameter-efficient fine-tuning methods adapt large pre-trained models to new tasks by updating only a small fraction of parameters, rather than retraining the entire model. This makes fine-tuning accessible on consumer hardware while achieving performance comparable to full fine-tuning.

## LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2021, Microsoft) is the dominant PEFT method. The core idea: instead of updating the full weight matrix W during fine-tuning, freeze W and learn a low-rank update ΔW = BA, where B is d×r and A is r×d with rank r << d (typically r=8 or 16).

This reduces trainable parameters by **10,000×** while maintaining full fine-tuning quality. Critically, the LoRA matrices can be merged back into the base weights after training, so there's **zero additional inference latency**. LoRA can be applied selectively — typically to attention projection matrices (Q, K, V, O) and sometimes MLP layers.

**Practical tips (via Sebastian Raschka):** Start with r=8, learning rate 1e-4 to 3e-4, and apply to all linear layers. Monitor for overfitting, which is more likely with LoRA than full fine-tuning due to the low-rank constraint.

## QLoRA: Quantized LoRA

QLoRA (Dettmers et al., 2023) combines 4-bit quantization with LoRA, enabling fine-tuning of a **65B model on a single 48GB GPU**. Key innovations:

- **NF4 (4-bit NormalFloat):** Information-theoretically optimal for normally-distributed weights
- **Double quantization:** Quantizes the quantization constants themselves, saving ~0.37 bits per parameter
- **Paged optimizers:** Uses NVIDIA unified memory to handle memory spikes during gradient checkpointing

The base model stays quantized throughout training; only the LoRA adapters train in higher precision. This democratized LLM fine-tuning for researchers without data center hardware.

## Other PEFT Methods

**Prefix tuning:** Learns continuous "soft prompts" prepended to the input, keeping all model weights frozen. Effective but adds latency.

**Adapters:** Small trainable modules inserted between transformer layers. Predates LoRA but adds inference overhead since adapters can't be merged.

**DoRA (Weight-Decomposed LoRA):** Decomposes weights into magnitude and direction, applying LoRA to direction only. Slightly better quality than standard LoRA.

## GaLore: Gradient Low-Rank Projection

GaLore (Zhao et al., ICML 2024) takes a different angle than LoRA: instead of reparameterizing the *weight update* as low-rank, it exploits the fact that the **gradient** of a weight matrix during training is itself slow-changing and low-rank. GaLore periodically computes an SVD of the gradient `G` to find projection matrices `P, Q`, then projects the gradient (and the optimizer's momentum/variance state) into the low-rank subspace `P^T G Q` before the optimizer step, and projects back when applying the update.

This is a meaningful difference from LoRA in practice:
- **Full-parameter training, not reparameterized training.** Every weight in `W` can still update — GaLore compresses the optimizer state, not the model's expressivity. LoRA restricts updates to a low-rank subspace of the weight itself, which is why LoRA sometimes underperforms full fine-tuning on tasks requiring broad weight changes.
- **Optimizer-agnostic.** GaLore is a wrapper around the gradient/optimizer-state pipeline, so it plugs into AdamW or other optimizers with minimal code changes.
- **Memory savings land on optimizer states.** Reported up to **65.5% reduction in optimizer-state memory**, which is what let the original paper pre-train a 7B model on a single 24GB consumer GPU (with activation checkpointing and per-layer weight updates) — without model parallelism or offloading.

Follow-on work: **Q-GaLore** adds INT4 quantization of the projection matrices and layer-adaptive low-rank gradients for further memory reduction; **GaLore 2** (2025) targets large-scale pretraining rather than just fine-tuning, addressing scalability issues (SVD cost, projection update frequency) that limited the original method at larger model/data scale.

GaLore and LoRA are not mutually exclusive — both reduce optimizer/activation memory but via different mechanisms (gradient compression vs weight-update compression), and some recipes combine low-rank gradient projection with low-rank adapters.

## Portable Task Adaptation: PorTAL

A different problem than *how* to fine-tune cheaply is *how often you have to redo it*: a LoRA is locked to the base model it was trained on, so each new model release forces a from-scratch re-tune. This cost doesn't amortize — it's a recurring tax that scales roughly inversely with the time between model releases, and that cadence has been accelerating (notable foundation-model releases: 2/year in 2020 → 149/year in 2023; by 2024-2025 a new SOTA model held the top public leaderboard spot for only ~35 days on average, down from ~1 year for GPT-4).

**PorTAL** (Geist, Ramp Labs, 2026) addresses this by learning a task adaptation once, in a *base-agnostic* form, and porting it to new frozen models by refitting only a small per-model component:

- A **base-agnostic task latent** `z_t` (dim 256) is learned per task, shared across all base models.
- A **hypernetwork decoder** `D_b` generates per-layer LoRA factors (A, B) from `z_t` and a per-layer embedding, split into a **shared core** (a FiLM-conditioned trunk + per-module heads, base-agnostic, producing fixed-width "core" factors) and a **thin per-base converter** (linear projections that map the core-width factors to the target base's actual dimensions, plus the per-layer embeddings themselves).
- To **port to an unseen base**, the task latent and shared core are frozen, and only the thin converter is refit on a small calibration set.

Trained jointly on Qwen3-1.7B and Qwen3-4B, then porting the frozen latent+core to an **unseen Qwen3-8B** by refitting just the converter recovers **~98% of per-task LoRA's accuracy lift** — versus only **~14%** for Cross-LoRA (a data-free adapter-translation baseline that transfers an already-trained adapter without any refitting step). Cross-family transfer (same frozen Qwen latent/core, ported to an unseen **Gemma-3-4B**) recovers **~94%** of the lift, described as "nearly lossless." The converter refit is also more data-efficient than training a LoRA from scratch on the new base: PorTAL reaches from-scratch LoRA's peak accuracy with roughly **half the calibration data**, and is better calibrated (lower held-out log-loss) at every data size along the way — which roughly halves the FLOPs needed to adapt each subsequent base model, since the frozen base dominates per-step cost.

This sits at the intersection of two prior lines of work: single-base LoRA hypernetworks (e.g. Text-to-LoRA), which amortize across tasks but stay locked to one base, and cross-model LoRA transfer methods (Cross-LoRA, LoRA-X, CAST), which translate an existing adapter to a new base without any calibration step. PorTAL's central empirical claim is that a small refit step is what makes cross-model transfer actually work — data-free translation alone leaves most of the accuracy lift on the table.

The paper also reports *recovered lift* — `(acc_method - acc_base) / (acc_LoRA - acc_base)` — rather than *retention* (`acc_method / acc_LoRA`), arguing retention is non-discriminative when headroom over the base model is small (the regime prior cross-model-transfer papers evaluate in).

## Related Topics
- [[quantization-fundamentals]] — QLoRA bridges quantization and fine-tuning
- [[alignment-methods]] — PEFT methods are used in the alignment pipeline (SFT + DPO stages)
- [[practical-fine-tuning]] — Hands-on guidance for applying these methods
- [[optimizers]] — GaLore modifies what the optimizer sees, not which optimizer is used

## Sources
- LoRA: Low-Rank Adaptation (arxiv:2106.09685)
- QLoRA: Efficient Finetuning of Quantized LLMs (arxiv:2305.14314)
- Practical Tips for Finetuning LLMs Using LoRA — Sebastian Raschka
- GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection — Zhao et al., ICML 2024 (arxiv:2403.03507)
- Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients (arxiv:2407.08296)
- GaLore 2: Large-Scale LLM Pre-Training by Gradient Low-Rank Projection (arxiv:2504.20437)
- [PorTAL: Portable Task Adapters for LLMs — Geist, Ramp Labs (2026)](https://labs.ramp.com/research) — no arXiv ID; see local notes at [ramp-labs-portal-portable-task-adapters.md](../raw/ramp-labs-portal-portable-task-adapters.md)
