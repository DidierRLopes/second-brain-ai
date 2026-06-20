# Small Efficient Models

Small language models are not just "scaled-down frontier models"; they are trained under a different economic objective. At 1-3B parameters, the best public recipes deliberately spend more tokens, more data curation, and more scheduler care to buy cheap inference, local deployment, and faster experimentation. The common pattern across SmolLM2 and MiniCPM is: keep the architecture familiar, make the data far cleaner and more targeted than a generic web crawl, and use training schedules that let you delay the final specialization pass until you know what capability gaps remain.

## Design Philosophy

The core tradeoff is inverted from frontier training. A frontier model is expensive to train and expensive to serve, so the lab tries to maximize total capability per training run. A small model can be overtrained because the serving side matters more: a 1-2B model that saw many more tokens than Chinchilla-optimal may be a better product than a larger model that is cheaper to train but more expensive to run.

[SmolLM2: When Smol Goes Big - Data-Centric Training of a Small Language Model (2502.02737)](../../papers/01-models/small-efficient/SmolLM2: When Smol Goes Big - 2502.02737.pdf) is the data-centric version of this philosophy. The 1.7B model follows a Llama-style dense transformer, then pushes capability mostly through 11T tokens, multi-stage mixture changes, and new datasets for math, code, and instruction following. The authors explicitly treat small-model capacity as scarce: noisy or incidental web text is less tolerable because the model cannot memorize everything and still learn robust capabilities.

[MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies (2404.06395)](../../papers/01-models/small-efficient/MiniCPM: Unveiling the Potential of Small Language Models - 2404.06395.pdf) is the scaling-methodology version. Its 1.2B and 2.4B non-embedding-parameter models are used as "model wind tunnel" experiments: tune hyperparameters, batch-size scaling, learning-rate stability, and data/model scaling in the small regime, then reuse the lessons for larger runs. The target is not only a good small model, but a cheaper experimental platform for studying training dynamics.

## Data Quality

SmolLM2's strongest lesson is that small models need data mixtures that are actively managed, not merely sampled. The project starts by comparing FineWeb-Edu and DCLM: FineWeb-Edu wins on educational benchmarks such as MMLU, ARC, and OpenBookQA, while DCLM is stronger on HellaSwag and CommonsenseQA. The final web mix uses those complementary signals rather than picking a single "best" crawl.

The paper then builds missing domain datasets where public data was too small or low quality. FineMath filters Common Crawl for step-by-step mathematical reasoning and produces variants such as FineMath4+ and FineMath3+. Stack-Edu filters StarCoder2Data toward educational and well-documented code, ending at roughly 125B tokens across 15 languages. SmolTalk fills the post-training gap with synthetic and curated instruction data for constraints, rewriting, summarization, math, code, system prompts, function calling, and long-context behavior [SmolLM2: When Smol Goes Big - Data-Centric Training of a Small Language Model (2502.02737)](../../papers/01-models/small-efficient/SmolLM2: When Smol Goes Big - 2502.02737.pdf).

The training mix is also staged around observed weaknesses:

| Stage | Tokens | Main choice | Practical reason |
|---|---:|---|---|
| Stage 1 | 0-6T | Mostly English web, plus 10% code | Build broad language and knowledge before spending small specialized datasets. |
| Stage 2 | 6-8T | Add 5% OpenWebMath and raise code to 20% | Address early code/math gaps without exhausting small math corpora. |
| Stage 3 | 8-10T | Add InfiMM-WebMath, switch to Stack-Edu, add textbooks | Make the stable phase more capability-targeted once general behavior is established. |
| Stage 4 | 10-11T | Decay phase with FineMath, Infi-WebMath3+, Stack-Edu, textbooks | Spend the highest-quality data while the learning rate is decaying, where it shapes final behavior most. |

MiniCPM reaches a similar conclusion from a different path. Its stable stage uses broad, mostly open pretraining data, while the decay stage mixes high-quality knowledge and ability-oriented SFT data into the pretraining stream. The ablations show that introducing high-quality data during decay beats saving it only for the later SFT phase, so specialization should start before instruction tuning [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies (2404.06395)](../../papers/01-models/small-efficient/MiniCPM: Unveiling the Potential of Small Language Models - 2404.06395.pdf).

## Scaling and Optimization Choices

WSD is the shared schedule idea. Warmup-Stable-Decay keeps a high learning rate during a reusable stable phase, then performs a comparatively short decay phase when the run is ready to specialize or finish. SmolLM2 uses AdamW with a 2,000-step warmup, peak learning rate `5.0e-4`, and a final decay over 10% of training. MiniCPM develops the schedule more explicitly: warmup, stable high-LR training, then exponential decay. Its small-scale experiments find that a 10% decay is enough for convergence, while very short decay underperforms [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies (2404.06395)](../../papers/01-models/small-efficient/MiniCPM: Unveiling the Potential of Small Language Models - 2404.06395.pdf).

MiniCPM also supports the muP-style lesson, but the paper's term is Tensor Program scaling. It uses width scaling and depth scaling to make hyperparameters transfer across model sizes, while explicitly not applying the attention softmax scaling technique. In their experiments, the optimal base learning rate remains around `0.01` from 0.04B to 0.5B models, with an additional 2.1B validation run. The practical point is narrower than "all hyperparameters transfer": the recipe is designed to stabilize learning-rate choice across nearby small-model scales.

The most provocative MiniCPM result is the data/model ratio. By using WSD checkpoints to measure scaling laws without retraining every model-token pair from scratch, the authors estimate a much higher compute-optimal data-to-model ratio than Chinchilla: about 192 tokens per parameter on average in their setup, not 20. They also show a 0.036B model can match a 0.17B model with roughly 4x more training compute while saving about 5x inference compute per call. For small models, that is the relevant axis: spend training tokens once to reduce every future inference bill.

Architecturally, both papers stay conservative. SmolLM2-1.7B uses 24 layers, model dimension 2048, FFN dimension 8192, 32 attention heads, RoPE, SwiGLU, tied embeddings, and 2k context before extension to 8k. MiniCPM uses tied input-output embeddings to reduce parameter cost, a deep-and-thin layout, and Grouped Query Attention only for MiniCPM-1.2B; MiniCPM-2.4B keeps full attention. See [[attention-variants]] and [[transformer-architecture]] for the underlying pieces.

## Deployment and On-Device Tradeoffs

The deployment story is why these models exist. MiniCPM frames small models as deployable on personal computers and smartphones, sometimes without a GPU. SmolLM2 makes the same economic argument through overtraining: a 1.7B model trained on 11T tokens costs more upfront than a compute-optimal run, but it is much cheaper to serve than a larger model with similar everyday utility.

Small models also shift where optimization work pays off:

- **Vocabulary and embeddings matter more** because embeddings are a larger share of total parameters. MiniCPM uses tied embeddings and a smaller vocabulary for the 1.2B model to favor efficiency.
- **KV cache choices matter on device**, but simplicity still wins. GQA is useful when it reduces memory without making kernels fragile.
- **Context extension should be staged**, not assumed. SmolLM2 extends from 2k to 8k near the end using a mix with 40% long-context documents; MiniCPM-128K extends the 2.4B model to 128k as a separate family member.
- **Post-training data must fit capacity**. SmolLM2 filters hard or complex instruction examples for the 135M and 360M variants rather than using the exact 1.7B post-training mix.

## Comparison Table

| Dimension | SmolLM2 | MiniCPM |
|---|---|---|
| Main lesson | Data curation and staged mixture changes dominate small-model quality. | Scheduler and hyperparameter-transfer experiments make small models useful scaling probes. |
| Main models | 1.7B, plus 360M and 135M variants. | 1.2B and 2.4B non-embedding-parameter base models. |
| Training tokens | 11T for 1.7B; 4T for 360M; 2T for 135M. | 1.1T for both 1.2B and 2.4B base models. |
| Architecture posture | Llama-style dense transformer; tied embeddings; RoPE; SwiGLU; 2k to 8k context extension. | Deep-and-thin transformer; tied embeddings; GQA for 1.2B; full attention for 2.4B. |
| Data posture | Build FineMath, Stack-Edu, and SmolTalk; rebalance web/math/code by stage. | Broad stable-stage data; high-quality SFT/ability data introduced during decay. |
| Optimization posture | WSD with 10% final decay; online mixture changes based on eval gaps. | WSD as a primary research contribution; Tensor Program scaling for LR transfer. |
| Deployment angle | Strong 1-2B open model with cheap inference and released datasets. | End-device and long-context/MoE/DPO family variants from a small base. |

## Related Topics

- [[frontier-training-playbook]] — Frontier-scale defaults are useful, but small models shift the objective toward inference economics, overtraining, and data quality.
- [[data-curation-mixtures]] — Dataset filtering, mixture staging, and late high-quality data.
- [[scaling-laws]] — Why compute-optimal and inference-optimal choices diverge.
- [[optimizers]] — WSD, cosine schedules, batch size, and learning-rate transfer.
- [[knowledge-distillation]] — Especially relevant when a larger teacher is available for small model training or post-training.
- [[inference-optimization]] — The serving-side reason to overtrain a smaller model.

## Sources

- [SmolLM2: When Smol Goes Big - Data-Centric Training of a Small Language Model (2502.02737)](../../papers/01-models/small-efficient/SmolLM2: When Smol Goes Big - 2502.02737.pdf) — 11T-token overtraining, FineMath, Stack-Edu, SmolTalk, staged mixture design, 2k to 8k context extension.
- [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies (2404.06395)](../../papers/01-models/small-efficient/MiniCPM: Unveiling the Potential of Small Language Models - 2404.06395.pdf) — WSD scheduler, Tensor Program scaling, model wind tunnel experiments, high-quality decay-stage data, MiniCPM-DPO/128K/MoE family.
