# Scaling Laws

Scaling laws describe the predictable relationship between model performance and model size, dataset size, compute budget, and increasingly the constraints around them: repeated data, inference cost, MoE sparsity, downstream-task emergence, and test-time search. Understanding these relationships is critical for making efficient training decisions worth millions of dollars, because the practical question is not just "how big can we train?" but "where does the next unit of compute buy the most capability?"

## Chinchilla: Compute-Optimal Training

[Training Compute-Optimal Large Language Models / Chinchilla (2203.15556)](../../papers/03-scaling/scaling-laws/Training Compute-Optimal Large Language Models - 2203.15556.pdf) established the most influential scaling law: for a fixed compute budget, you should scale model size and training data **roughly equally**. The optimal ratio is approximately **20 tokens per parameter**.

This revealed that most existing models were severely undertrained:
- GPT-3 (175B parameters, 300B tokens) was ~4-5× undertrained
- Gopher (280B parameters) used too much compute on a model that was too large relative to its data

Chinchilla (70B parameters, 1.4T tokens) outperformed Gopher despite being 4× smaller, by training on 4× more data. This finding shifted the field from "make models bigger" to "train models longer on more data."

## Impact on Model Design

Chinchilla directly influenced the next generation of efficient models:
- **LLaMA** (Meta): 7B-65B models trained on 1-1.4T tokens — deliberately smaller but better-trained
- **Mistral**: Continued the trend of smaller, well-trained models
- **The open-source ecosystem**: Chinchilla-optimal 7B and 13B models often outperform older 65B+ models

## Data-Constrained Scaling: Repetition Has a Half-Life

The clean Chinchilla recipe assumes enough fresh data. [Scaling Data-Constrained Language Models (2305.16264)](../../papers/03-scaling/compute-optimal/Scaling Data-Constrained Language Models - 2305.16264.pdf) asks what to do when the data budget is fixed and the only way to spend more compute is to repeat data or add parameters. The headline result is pragmatic: **up to about 4 epochs of full-corpus repetition is almost as good as fresh data**, with negligible loss changes and insignificant downstream-performance differences in their experiments. Past that, repeated tokens still help for a while, but the return decays predictably and eventually goes to zero.

Their scaling law replaces raw tokens and parameters with **effective data** and **effective parameters**. Repeated tokens contribute less than fresh tokens according to an exponential decay term, and excess parameters beyond what the unique data can support also have diminishing returns. In their fit, the repeated-data half-life is roughly 15 repetitions, so meaningful gains can continue up to around 16 epochs, but the curve flattens hard after that. This matters because a data-constrained compute-optimal model is not "same model, repeat more"; the efficient frontier shifts toward **smaller models trained for more epochs**, with epochs scaling slightly faster than parameters.

Two operational details matter. First, the paper repeats the **entire available corpus**, shuffled each epoch, not a small upsampled slice. Repeating a tiny subset many times is a different failure mode. Second, code can act like extra data: filling missing natural-language tokens with Python gave roughly a **2x effective-token extension** for natural-language evaluations before degrading. Filtering is more conditional: it helps noisy corpora, but aggressive deduplication can reduce useful data in already-clean settings.

## Beyond Chinchilla

More recent work suggests the Chinchilla ratio isn't universal. **Inference-optimal scaling** considers that smaller models are cheaper to serve, so it may be worth "over-training" a smaller model (e.g., LLaMA 3 trained a 8B model on 15T tokens — far beyond Chinchilla-optimal). If you'll serve the model billions of times, spending extra on training to get a smaller model pays off.

[[reasoning-models]] introduce yet another scaling axis: **test-time compute**. Instead of scaling parameters or data, you scale the amount of thinking at inference time. This is complementary to training-time scaling.

[Scaling Scaling Laws with Board Games (2104.03113)](../../papers/03-scaling/training-optimization/Scaling Scaling Laws with Board Games - 2104.03113.pdf) is a useful non-language example because it scales both the agent and the **problem size**. In AlphaZero-style Hex experiments, compute frontiers fitted on small boards predicted larger boards, and training-time compute could trade against test-time search compute at fixed performance: roughly each extra order of magnitude of train-time compute reduced the needed test-time compute by a similar factor. This should not be imported directly as an LLM law, but it supports the broader lesson behind [[reasoning-models]]: search/inference budget is a real scaling axis, not just an implementation detail.

## Scaling Is Also Systems Engineering

PaLM is a canonical example of scaling laws meeting training infrastructure. [PaLM: Scaling Language Modeling with Pathways (2204.02311)](../../papers/03-scaling/scaling-laws/PaLM: Scaling Language Modeling with Pathways - 2204.02311.pdf) trained a 540B dense decoder-only Transformer on 780B tokens using 6144 TPU v4 chips across two pods. The model was still pre-Chinchilla-undertrained by modern standards, but it proved that a single dense model could be trained efficiently at that scale without pipeline parallelism.

The infrastructure lesson is specific: Pathways used two-way pod-level data parallelism, 12-way model parallelism inside each pod, and fully sharded data parallelism. The hard part was not just FLOPs, but cross-pod gradient transfer: each host pair exchanged about 1.3 GB of gradients per step, creating an aggregate burst around 81 Tbps. By chunking transfers and routing them over multiple data-center-network paths, PaLM reached about **46.2% model FLOPs utilization** and **97% weak scaling** from one pod to two. Scaling laws tell you what run is worth doing; systems work decides whether the run actually fits inside the calendar.

## Frontier-2026 Practice: Overtraining is the Norm

Scaling laws are almost never religiously followed. Recent frontier models *deliberately overtrain* — sometimes by 5× or more relative to compute-optimal — because compute-optimal scaling laws don't account for the inference cost of larger models, and smaller-trained-longer wins on total lifetime cost.

| Model | Active params | Training tokens | Chinchilla ratio |
|---|---|---|---|
| GPT-3 | 175B | 300B | ~1.7 (undertrained) |
| Chinchilla | 70B | 1.4T | 20 (compute-optimal) |
| LLaMA 3 8B | 8B | 15T | ~1875 |
| Qwen 3 | varied | 36T | extreme overtraining |
| SmolLM3 | 3B | 11T | ~3700 |
| Kimi K2 (1T total) | 32B active | 15.5T | overtraining at MoE scale |

## Predicting Downstream Performance

Loss scaling is smoother than benchmark scaling. [Scaling Laws for Predicting Downstream Performance in LLMs (2410.08527)](../../papers/03-scaling/scaling-laws/Towards Greater Leverage: Scaling Laws for Predicting Downstream Performance - 2410.08527.pdf) argues that direct FLOPs-to-performance prediction breaks because downstream tasks have emergence thresholds: below a task-specific compute level, extra FLOPs may not move the benchmark at all. Their FLP method instead predicts **FLOPs → pre-training loss → downstream performance**. The first stage uses fully converged small models to fit a loss law; the second uses intermediate checkpoints that have already crossed the task's randomness threshold to map loss to performance.

The practical result: sampling models up to 3B predicted 7B and 13B downstream performance with error margins around 5% and 10%, outperforming direct FLOPs-to-performance fits. Their FLP-M extension handles data mixtures by predicting **domain-specific validation losses** rather than average loss. For general-text + code mixtures, this better captured capability shifts and predicted most 3B/7B benchmark outcomes within 10% error. This is the scaling-law version of a common frontier-training habit: keep a stable validation suite by domain, not just one global loss number.

## MoE Efficiency Leverage

Dense scaling laws use parameter count as a rough capacity proxy. MoE breaks that proxy because total parameters, active parameters, and FLOPs decouple. [Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models (2507.17702)](../../papers/03-scaling/scaling-laws/Scaling Laws for Predicting Downstream Performance - 2507.17702.pdf) introduces **Efficiency Leverage (EL)**: the dense compute budget divided by the MoE compute budget needed to reach the same loss. In other words, EL = 7 means the MoE gets dense-equivalent loss with about one seventh of the compute.

The Ant Group study trained over 300 models up to 28B parameters and found three useful regularities. First, expert activation ratio is the main driver: lower activation ratio (higher sparsity) predictably improves EL. Second, expert granularity is not monotonic; their best range was about 8-12 under a standard load-balancing loss. Third, EL itself increases with the total compute budget, which means MoE advantages become more important at larger pre-training scales. Their Ling-mini-beta validation model had 17.5B total parameters but only 0.85B active parameters and matched a 6.1B dense model on the same 1T-token dataset while using over 7x less compute.

## Model-Specific Scaling: Kimi K2's Sparsity-Driven Design

Beyond the general scaling laws, Kimi K2's analysis showed model-specific insights that override default heuristics. Their key finding: **increasing sparsity** (total experts / active experts) yields substantial performance improvements for fixed FLOPs. So they:

- Increased MoE experts to 384 (vs 256 in DeepSeek-V3) at sparsity 48.
- Activated only 8 of 384 experts per token.
- *Decreased* attention heads from 128 to 64 to reduce inference compute.

The tradeoff: reducing attention heads cost 0.5–1.2% validation loss but gave a **45% decrease in inference FLOPs** — worth it for a model with billions of inference calls. See [[mixture-of-experts]] for more.

## Does scaling get us to AGI? — The frontier debate

Scaling laws answer "how do losses behave as we scale?" — they do not answer "is scaling sufficient for AGI?" That question is contested. Frontier-lab leaders (Sutskever, Amodei, Hassabis, Schulman) have argued at various times that scaling + RL + "unhobblings" is the path. Skeptics (Sutton, Chollet, Ege/Tamay, Karpathy in late 2025) argue scaling is hitting fundamental limits — data, RL information-efficiency, continual learning — that no amount of compute will paper over.

The most-cited inputs to this debate are Dwarkesh's interviews and essays — they are the primary venue where these positions get articulated and challenged in detail. See [[dwarkesh-podcast]] for the corpus, and [[agi-timelines]] for the synthesis of timeline implications.

Key sources to read in tandem with Chinchilla:
- Dwarkesh, "Will scaling work?" (2023) — the foundational essay laying out cases for and against
- Dwarkesh, "Thoughts on AI progress (Dec 2025)" — late-2025 update on what's been scaling
- Leopold Aschenbrenner, "Situational Awareness" — strong scaling-as-sufficient case
- Ilya Sutskever (2023 vs 2025) — visible shift from "scaling is the answer" to "age of research"
- Ege Erdil & Tamay Besiroglu (Epoch) — quantitative argument that scaling saturates well before AGI
- Richard Sutton — argues the entire scaling paradigm is the wrong architecture

## Related Topics
- [[transformer-architecture]] — The architecture being scaled
- [[mixture-of-experts]] — MoE offers a different scaling trajectory than dense models; Kimi K2 sparsity case
- [[reasoning-models]] — Test-time compute as a third scaling axis
- [[agi-timelines]] — Where the scaling debate translates into timeline forecasts
- [[dwarkesh-podcast]] — Primary source material for the scaling debate
- [[inference-optimization]] — Why inference cost matters for scaling decisions
- [[frontier-training-playbook]] — where scaling-law thinking fits into the broader recipe

## Sources
- [Training Compute-Optimal Large Language Models / Chinchilla (2203.15556)](../../papers/03-scaling/scaling-laws/Training Compute-Optimal Large Language Models - 2203.15556.pdf) — compute-optimal model/data scaling and the 20 tokens-per-parameter rule of thumb.
- [Scaling Data-Constrained Language Models (2305.16264)](../../papers/03-scaling/compute-optimal/Scaling Data-Constrained Language Models - 2305.16264.pdf) — repeated-data limits, data-constrained compute optimality, code as effective extra data.
- [PaLM: Scaling Language Modeling with Pathways (2204.02311)](../../papers/03-scaling/scaling-laws/PaLM: Scaling Language Modeling with Pathways - 2204.02311.pdf) — 540B dense model, Pathways, TPU v4 pod-level scaling, MFU.
- [Scaling Laws for Predicting Downstream Performance in LLMs (2410.08527)](../../papers/03-scaling/scaling-laws/Towards Greater Leverage: Scaling Laws for Predicting Downstream Performance - 2410.08527.pdf) — FLP and FLP-M downstream prediction via validation loss.
- [Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models (2507.17702)](../../papers/03-scaling/scaling-laws/Scaling Laws for Predicting Downstream Performance - 2507.17702.pdf) — MoE Efficiency Leverage, activation ratio, granularity, Ling-mini-beta validation.
- [Scaling Scaling Laws with Board Games (2104.03113)](../../papers/03-scaling/training-optimization/Scaling Scaling Laws with Board Games - 2104.03113.pdf) — problem-size scaling and train-time/test-time compute tradeoffs in AlphaZero Hex.
- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- Kimi K2 technical report (sparsity-driven scaling).
