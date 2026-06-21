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

## Pre-Training Under Infinite Compute: Regularization and Ensembling Beat Naive Scaling

The data-constrained picture above assumes compute is still the scarcer resource. [Pre-training under infinite compute (2509.14786)](../../papers/03-scaling/compute-optimal/Pre-training under infinite compute - 2509.14786.pdf) (Stanford: Kim, Kotha, Liang, Hashimoto) flips the constraint: web text grows ~1.03x/year while pre-training compute grows ~4x/year, so it asks what to do when **data is fixed and compute is effectively unlimited**. This is the regime data-constrained scaling is heading toward, not a separate problem.

Their baseline is the obvious thing to try: fix a 200M-token seed corpus and throw compute at it via more epochs and more parameters (e.g. defaulting to 300M-parameter models on 200M tokens, far past Chinchilla's 20 tokens/parameter). Both knobs **fail by overfitting**: too many epochs eventually increases loss rather than monotonically decreasing it (contradicting the decay-based functional form in the data-constrained scaling-law literature, which excludes overfitting runs when fitting), and scaling parameters alone barely helps — a 1.4B model actually underperforms a 600M model at the same token budget, even after jointly tuning epoch count and learning rate per parameter count.

The fix is regularization tuned for the over-parameterized regime, not copied from standard practice. After jointly tuning learning rate, epoch count, and weight decay via coordinate descent at each parameter count, they find **the optimal weight decay is over 30x larger than the standard default of 0.1** (their tuned values range roughly 0.8 to 3.2 depending on model size, versus the usual 0.1). With this regularization, loss becomes a **clean, monotonically-decreasing power law in parameter count** — `L̂(N) ≈ 1/N^1.02 + 3.43` in their fit — for parameter-to-token ratios up to 140x larger than Chinchilla's regime. Notably the fitted exponent (1.02) is far steeper than Chinchilla's parameter-scaling exponent (0.34), meaning that once data is properly regularized against, bigger models keep paying off much faster than the compute-constrained regime would suggest.

Because loss now decreases monotonically rather than bottoming out at a fixed compute budget, the paper argues you should evaluate a data-constrained recipe by **the asymptote of its power law as N → ∞**, not its loss at some fixed budget — that asymptote is the best a given recipe can ever do with infinite compute and fixed data. For the regularized single-model recipe, that asymptote is a loss of **3.43**.

Ensembling beats the regularized asymptote. Instead of training one ever-larger model, the paper trains `K` independently-seeded models (varying data order and init) of fixed size and averages their logits at inference. Excess loss from ensembling decays roughly as `1/K`, mirroring the `1/N` decay from parameter scaling — but the **ensembling asymptote (N=300M, K→∞) reaches 3.34, lower than the regularized single-model asymptote of 3.43**. Concretely, even a 3-member ensemble already beats the best achievable single large model. The intuition (citing Allen-Zhu & Li's "multi-view" theory) is that a single model is biased toward learning only one of several valid predictive features in the data, while independently-trained ensemble members tend to pick up different features. The hyperparameters that minimize loss for a single model are not the ones that minimize the ensemble asymptote — the best infinite-ensemble members actually want **more epochs and less weight decay per member** than the best standalone model, i.e. slightly more overfit individual members compensated by the ensemble's diversity.

Parameter scaling and ensembling **compose**: taking both `N → ∞` and `K → ∞` jointly pushes the asymptote even lower. The combined recipe (epoching + regularization + parameter scaling + ensemble scaling) reaches the same loss as the 200M-token baseline using an estimated **5.17x less data**, and the paper's data-scaling-law extrapolation predicts this efficiency gain persists at larger token budgets, not just at the 200M-token scale they tested directly. Much of the ensembling benefit is recoverable without paying its inference cost: **distilling an 8-member ensemble into a single 300M student model retains 83% of the ensembling improvement** over the best regularized 300M model, and even a same-size self-distillation step (no ensemble) reduces loss. The validation-loss gains carry over to downstream evaluation too — their best ensemble beats their best unregularized model by 9% on average across PIQA, SciQ, and ARC-Easy, and on continued pre-training on math mid-training data their ensembling recipe needs only 4B tokens to beat default continued pre-training on the full 73B-token budget (a 17.5x data-efficiency improvement).

This connects directly to the [[optimizers]] discussion of weight decay and AdamW defaults: the 30x weight-decay finding is a concrete case where a hyperparameter inherited from compute-constrained pre-training recipes (`λ = 0.1`) is badly miscalibrated once the binding constraint shifts from compute to data.

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

Loss scaling is smoother than benchmark scaling. [Scaling Laws for Predicting Downstream Performance in LLMs (2410.08527)](../../papers/03-scaling/scaling-laws/Scaling Laws for Predicting Downstream Performance in LLMs - 2410.08527.pdf) argues that direct FLOPs-to-performance prediction breaks because downstream tasks have emergence thresholds: below a task-specific compute level, extra FLOPs may not move the benchmark at all. Their FLP method instead predicts **FLOPs → pre-training loss → downstream performance**. The first stage uses fully converged small models to fit a loss law; the second uses intermediate checkpoints that have already crossed the task's randomness threshold to map loss to performance.

The practical result: sampling models up to 3B predicted 7B and 13B downstream performance with error margins around 5% and 10%, outperforming direct FLOPs-to-performance fits. Their FLP-M extension handles data mixtures by predicting **domain-specific validation losses** rather than average loss. For general-text + code mixtures, this better captured capability shifts and predicted most 3B/7B benchmark outcomes within 10% error. This is the scaling-law version of a common frontier-training habit: keep a stable validation suite by domain, not just one global loss number.

## MoE Efficiency Leverage

Dense scaling laws use parameter count as a rough capacity proxy. MoE breaks that proxy because total parameters, active parameters, and FLOPs decouple. [Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models (2507.17702)](../../papers/03-scaling/scaling-laws/Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models - 2507.17702.pdf) introduces **Efficiency Leverage (EL)**: the dense compute budget divided by the MoE compute budget needed to reach the same loss. In other words, EL = 7 means the MoE gets dense-equivalent loss with about one seventh of the compute.

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
- [[optimizers]] — µP/u-µP/CompleteP parametrizations for hyperparameter transfer across width and depth; weight decay tuning for data-constrained regimes

## Sources
- [Training Compute-Optimal Large Language Models / Chinchilla (2203.15556)](../../papers/03-scaling/scaling-laws/Training Compute-Optimal Large Language Models - 2203.15556.pdf) — compute-optimal model/data scaling and the 20 tokens-per-parameter rule of thumb.
- [Scaling Data-Constrained Language Models (2305.16264)](../../papers/03-scaling/compute-optimal/Scaling Data-Constrained Language Models - 2305.16264.pdf) — repeated-data limits, data-constrained compute optimality, code as effective extra data.
- [Pre-training under infinite compute (2509.14786)](../../papers/03-scaling/compute-optimal/Pre-training under infinite compute - 2509.14786.pdf) — overfitting limits of naive epoch/parameter scaling, 30x weight decay finding, power-law asymptote framing, ensembling and distillation results.
- [PaLM: Scaling Language Modeling with Pathways (2204.02311)](../../papers/03-scaling/scaling-laws/PaLM: Scaling Language Modeling with Pathways - 2204.02311.pdf) — 540B dense model, Pathways, TPU v4 pod-level scaling, MFU.
- [Scaling Laws for Predicting Downstream Performance in LLMs (2410.08527)](../../papers/03-scaling/scaling-laws/Scaling Laws for Predicting Downstream Performance in LLMs - 2410.08527.pdf) — FLP and FLP-M downstream prediction via validation loss.
- [Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models (2507.17702)](../../papers/03-scaling/scaling-laws/Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models - 2507.17702.pdf) — MoE Efficiency Leverage, activation ratio, granularity, Ling-mini-beta validation.
- [Scaling Scaling Laws with Board Games (2104.03113)](../../papers/03-scaling/training-optimization/Scaling Scaling Laws with Board Games - 2104.03113.pdf) — problem-size scaling and train-time/test-time compute tradeoffs in AlphaZero Hex.
- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- Kimi K2 technical report (sparsity-driven scaling).
