# Scaling Laws

Scaling laws describe the predictable relationship between model performance and model size, dataset size, compute budget, and increasingly the constraints around them: repeated data, inference cost, MoE sparsity, downstream-task emergence, and test-time search. Understanding these relationships is critical for making efficient training decisions worth millions of dollars, because the practical question is not just "how big can we train?" but "where does the next unit of compute buy the most capability?"

## The Three Scaling Laws: Pretraining, Post-Training, Test-Time

NVIDIA's explainer ["How Scaling Laws Drive Smarter, More Powerful AI"](../../ai-machine-learning/raw/nvidia-ai-scaling-laws.md) (Kari Briski, Feb 2025) gives a clean three-way taxonomy for where compute gets spent, which is a useful frame for the rest of this page:

- **Pretraining scaling** — the original law (below): more data, more parameters, more compute predictably buys more capability, per Chinchilla's compute-optimal ratio.
- **Post-training scaling** — once a foundation model is released, adapting it (fine-tuning, pruning, quantization, distillation, RLHF/RLAIF, best-of-n sampling, search; see [[alignment-methods]] and [[knowledge-distillation]]) for specific domains and use cases is its own compute sink. NVIDIA's estimate: the ecosystem of derivative models built around one foundation model can collectively cost **~30x more compute** than the original pretraining run, since a popular open-weight model spawns hundreds or thousands of fine-tunes.
- **Test-time scaling** ("long thinking") — spending more compute *per query* at inference instead of at training time; see [[reasoning-models]] for the mechanics. NVIDIA's estimate: a hard reasoning query can need **over 100x the compute** of a single traditional inference pass.

The three laws are complementary, not competing — different axes for spending compute, with frontier development blending all three (pretrain a strong base, post-train it into derivatives, then add test-time reasoning on top).

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

Parameter scaling and ensembling **compose**: taking both `N → ∞` and `K → ∞` jointly pushes the asymptote even lower, via a double limit `lim_{N→∞} lim_{K→∞} min_H L(E_A(D,N,K,H))`. Estimating this requires its own two-stage hyperparameter heuristic (2x the epochs and 0.5x the weight decay of the single-model-optimal settings, found to work well across scales, since exhaustively re-running coordinate descent for every (N,K) pair was impractical) — fit a power law in K at each N, take those asymptotes, then fit a second power law in N. The resulting joint-recipe loss asymptote is **3.17**, versus 3.43 for regularized single-model scaling alone and **3.75 for the unregularized standard recipe** — i.e. ensembling and parameter scaling are not redundant, they stack. The combined recipe (epoching + regularization + parameter scaling + ensemble scaling) reaches the same loss as the 200M-token baseline using an estimated **5.17x less data**, and the paper's data-scaling-law extrapolation predicts this efficiency gain persists at larger token budgets, not just at the 200M-token scale they tested directly (their fitted data-scaling exponents cluster between 0.23–0.24 and asymptotes between 1.89–1.96 loss across all three recipes, consistent with all algorithms eventually approaching the entropy of text under infinite data). Without taking the asymptote, the largest concrete ensemble tested (five 1.4B models) was already 3.75x more data-efficient than baseline; the smallest ensemble reaching loss 3.37 needed 1.2B total parameters.

Much of the ensembling benefit is recoverable without paying its inference cost. **Distilling an 8-member ensemble of 300M models (ensemble loss 3.32) into a single 300M student model via sequence-level knowledge distillation (train teacher → sample unconditionally to build a synthetic corpus → train student from scratch on real+synthetic tokens mixed together) reaches student loss 3.36, retaining 83% of the ensembling improvement** over the best regularized 300M model (loss 3.57), despite 8x lower inference compute — the student even matches a 4-ensemble's loss. More surprisingly, **same-size self-distillation (300M teacher distilled into a fresh, identically-sized 300M student) also reduces loss below the teacher's**, despite the data-processing inequality intuition that a student can't exceed its teacher; the authors attribute this to self-distillation implicitly performing ensembling between the teacher and a freshly-initialized student (citing Allen-Zhu & Li), and note it avoids the model-collapse failure mode reported elsewhere because real and synthetic tokens are mixed rather than training purely on self-generated data. The validation-loss gains carry over to downstream evaluation too — their best ensemble beats their best unregularized model by 9% on average across PIQA, SciQ, and ARC-Easy (evaluated only after recipes were locked in by validation loss, to avoid benchmark overfitting), their best distilled 300M model beats the best unregularized 300M model by about 7%, and on continued pre-training on math mid-training data (Llama-3.2-3B + MegaMath-Web-Pro) their ensembling recipe needs only 4B tokens (lower batch size, epoching, 8-way ensembling — weight decay was found *not* to help in the CPT setting) to beat default continued pre-training on the full 73B-token budget, a 17.5x data-efficiency improvement: average accuracy across GSM8K/MATH/MathQA rises from 24.25% (base Llama-3B) to 30.59% (default 4B-token CPT) to 40.58% (their 8-ensemble on 4B tokens), surpassing the 39.23% reached by the reference recipe using the full 73B tokens.

This connects directly to the [[optimizers]] discussion of weight decay and AdamW defaults: the 30x weight-decay finding is a concrete case where a hyperparameter inherited from compute-constrained pre-training recipes (`λ = 0.1`) is badly miscalibrated once the binding constraint shifts from compute to data.

## Beyond Chinchilla

More recent work suggests the Chinchilla ratio isn't universal. **Inference-optimal scaling** considers that smaller models are cheaper to serve, so it may be worth "over-training" a smaller model (e.g., LLaMA 3 trained a 8B model on 15T tokens — far beyond Chinchilla-optimal). If you'll serve the model billions of times, spending extra on training to get a smaller model pays off.

### Vlad Feinberg (Google DeepMind, Princeton talk, April 2025): IsoFlops mechanics and the case against pure Chinchilla-optimal training

Vlad Feinberg, Google DeepMind's Flash Pretraining Lead, gave a talk at Princeton walking through how Gemini Flash-class pretraining decisions actually get made, spanning classical scaling and what he calls inference-aware scaling.

**The FLOPs identity and its derivation.** The talk starts from the standard training-compute approximation **C ≈ 6·N·D** and derives the constant directly: excluding self-attention, an N-parameter decoder-only model costs 6N matmul FLOPs per token seen — 2N for the forward pass, 4N for the backward pass — because every matmul does one multiply and one add per input pair, and backprop requires two matmuls (input-gradient and weight-gradient) for each forward matmul. A more granular slide expands this to `18·B·T·D·F + 24·B·T·D·N·H = 6·B·T·(3DF + 4DNH)`, i.e. the same 6·(tokens)·(params) identity decomposed into FFN-width and attention-head terms.

**The IsoFlops method, spelled out step by step.** Chinchilla's critique of Kaplan et al. (2020) was that Kaplan ran one training run per model size and read off *intermediate* losses at different token counts, which understates achievable loss because proper learning-rate decay to a *final* checkpoint does materially better. Chinchilla's IsoFlops approach instead: (1) fix a target FLOPs budget; (2) train several models at that exact budget, varying model size (so data size moves inversely, since FLOPs = 6ND is fixed); (3) fit a parabola to loss vs. model size and take its minimum as the compute-optimal (N, D) for that budget; (4) repeat across a range of FLOPs budgets; (5) fit a power law N_opt(C); (6) fit a power law D_opt(C). The headline result both power-law exponents land at **~0.5**, i.e. model size and data size should scale at the *same* rate with compute — directly contradicting Kaplan's finding that data should grow only as **D ∼ C^0.27** (with a 10x compute increase implying 5.37x more parameters but only 1.86x more data). Feinberg frames the practical consequence bluntly: Kaplan-style scaling implies models were systematically undertrained, which is "obviously bad" once you account for the fact that bigger models cost more to serve.

**Chinchilla-style scaling ignores inference cost — the talk's central thesis.** Feinberg's explicit claim is that the classical IsoFlops/Chinchilla methodology only optimizes *training* FLOPs and says nothing about the cost of serving the resulting model. He cites **Sardana et al. 2024, "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws" (arXiv:2401.00448)** as the natural next step: globally optimize FLOPs jointly across training and inference rather than training alone. He flags two deeper problems with doing this in practice: (1) compute is non-homogeneous — inference-optimized chips differ from training chips, and organizations don't actually do global cross-org FLOPs optimization; (2) inference-time token demand D_inf is fundamentally non-forecastable, invoking **Jevons's paradox** — making a model cheaper/better to serve tends to increase total demand for it, which can offset or overwhelm the efficiency gains the optimization was chasing.

**Data-constrained framing as a partial answer.** To address what he calls "badness of fit" in naive inference-aware laws, Feinberg points to **Muennighoff et al. 2023, "Scaling Data-Constrained Language Models" (arXiv:2305.16264)** — already covered above in this page's repetition-half-life section — and reframes its **L(N, U, R)** formulation (U = unique tokens, R = repeat factor) as a tool for quantifying training-compute *regret*. His proposed exercise: assume 5 epochs and use the data-constrained law to back out the "ideal shrunk dataset" that would have produced the same loss as 5 epochs over a much larger corpus — i.e., directly estimate how many FLOPs were wasted training on far more unique tokens than actual inference demand (D_inf) would have justified. He contrasts this with Llama 3's stated strategy of treating D_inf as effectively infinite (the Llama 3 paper reports both the 8B and 70B models "continued to improve log-linearly" after 15T tokens), which he frames as a reasonable choice specifically for open-weight releases where serving volume is unknowable in advance, but not a generally optimal one.

**MoE scaling note.** Citing Clark et al. 2022 ("Unified Scaling Laws for Routed Language Models," arXiv:2202.01169), the talk names **alpha** as the parameter-count exponent and **beta** as the data-dependent exponent in MoE scaling laws, and states the empirical result that at matched active-parameter count and a fixed 100B-token training budget, a 64-expert MoE improves on the dense equivalent.

**Distillation scaling laws — a skeptical reading.** Feinberg also critiques **Busbridge et al. 2025, "Distillation Scaling Laws" (arXiv:2502.08606)**, arguing the paper's headline up-trend is only a "very weak effect" outside its narrow tested regime, and that its core equation for predicting student loss from teacher perplexity is sensitive to a knob it doesn't model: raising the teacher's sampling temperature can rescue a "bad" predicted distillation outcome even from an otherwise very strong teacher. His practical fix is to "James-Stein this away with weight tuning" under a supervised objective — i.e., treat the discrepancy as a shrinkage/bias problem rather than trusting the raw equation — and he frames distillation generally as variance reduction (a better teacher mainly reduces bias, not variance).

**Real-time serving constraints push toward smaller models.** As motivation for inference-aware scaling, the talk works a concrete roofline example: a web-interaction agent with 128k-token prefill (8k incremental per step) and 128 decode tokens, under a sub-1-second action-latency budget (250ms assumed consumed by scaffolding/load-balancing/KV-cache retrieval before the model even runs). Assuming the system is compute-bound on prefill and HBM-bound on decode, a Llama3-70B-class model on a single TPU v5e chip would take an estimated **5.7 seconds** to serve this workload — about 11x over budget — meaning a **4×4 topology (16 chips)** is needed just for the prefill station to clear a tightened 0.5-second target. This is the concrete case for why Google's real-time products (cited by name: Astra, Mariner) push toward Flash-class models rather than frontier-scale ones.

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
- [[mixture-of-experts]] — MoE offers a different scaling trajectory than dense models; Kimi K2 sparsity case; Feinberg's alpha/beta MoE-scaling-exponent note
- [[reasoning-models]] — Test-time compute as a third scaling axis
- [[alignment-methods]] — Post-training scaling: RLHF/RLAIF, best-of-n sampling, search
- [[knowledge-distillation]] — Post-training scaling: teacher/student distillation; Feinberg's critique of distillation scaling laws (temperature sensitivity, James-Stein framing)
- [[agi-timelines]] — Where the scaling debate translates into timeline forecasts
- [[dwarkesh-podcast]] — Primary source material for the scaling debate
- [[inference-optimization]] — Why inference cost matters for scaling decisions; Feinberg's roofline-based real-time-latency argument for smaller models
- [[how-to-scale-your-model]] — JAX Scaling Book; Feinberg's talk explicitly credits Jacob Austin's slides and borrows its roofline framing
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
- Kari Briski (NVIDIA), "How Scaling Laws Drive Smarter, More Powerful AI" (Feb 12, 2025) — pretraining/post-training/test-time taxonomy, 30x post-training and 100x test-time compute estimates. See `raw/nvidia-ai-scaling-laws.md`.
- Vlad Feinberg (Google DeepMind), "Gemini Pretraining: Classical and Inference-Optimized Scaling" (Princeton talk, April 2025) — https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf. IsoFlops step-by-step methodology, Chinchilla-ignores-inference-cost thesis, Sardana et al. inference-aware scaling critique, Jevons's-paradox framing of D_inf, data-constrained regret quantification via L(N,U,R), MoE alpha/beta exponents, distillation-scaling-laws critique, Llama3-70B/v5e roofline latency example. See `raw/vlad-feinberg-princeton-talk-2025.md`.
