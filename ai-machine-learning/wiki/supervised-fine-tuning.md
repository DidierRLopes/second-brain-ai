# Supervised Fine-Tuning (SFT)

SFT is the stable baseline of post-training: cheap relative to RL, insensitive to reward design and hyperparameters, and gives a strong starting point from any base model. Frontier-2026 practice is to **SFT first, then layer preference optimization, RL, distillation, or model-surgery continued training on top** — see [[alignment-methods]] and [[on-policy-distillation]] for what comes after. The interesting choices in SFT today are around chat templates (especially for hybrid reasoning), sequence packing (efficiency vs gradient-update count), masking (loss on assistant tokens only), reasoning-budget signals like Hermes 4's fixed-position `</think>`, and whether to retrofit architectures for more test-time compute.

## Why SFT First

- **Cheap** compared to RL.
- **Stable** — insensitive to reward design and hyperparameters.
- **Gives a strong baseline** off the base model.

Strong models may skip SFT because there are no stronger models to distill from — DeepSeek-R1-Zero is the canonical example. Most teams shouldn't.

SFT typically comes in the form of **distillation from stronger models**, which is why teacher choice often matters more than algorithmic tuning. Base models are usually too unrefined to benefit from more advanced post-training methods without SFT first.

## Dataset Curation

Datasets can seem great on paper but cause models to **overindex on certain domains** (e.g., science). Curation matters as much as algorithm choice.

**SmolLM3 example**: ~100k examples, 76.1M tokens, mostly instruction following + reasoning + steerability for both think and non-think modes. **Pair data across modes** — without this pairing, the model has no indication of when to give a concise answer vs use extended reasoning.

## Chat Templates

Considerations for designing or picking a chat template:
- System role customizability
- Tool calling
- Reasoning (hybrid or always-on)
- Compatibility with inference engines (vLLM, SGLang)

**Qwen3 and GPT-OSS** satisfy all criteria; Qwen3 is designed for hybrid reasoning.

**SmolLM3**: hybrid reasoning, but discards reasoning content for all but the final turn at inference to avoid context blow-up. For training, retain reasoning tokens to condition the model properly. Initial bug: custom instructions not passed into the template; patched fast.

**Intellect-3**: always reasons (not hybrid), trained dominantly on reasoning-only SFT traces. Uses `qwen3_coder` tool call parser and `deepseek_r1` reasoning parser for consistent reasoning chain representation.

**gpt-oss-120b**: uses the **harmony** chat template, which introduces "channels" determining message visibility — `final` for answers shown to the user, `commentary` for tool calling, `analysis` for CoT tokens. This allows the model to interleave tool calls with CoT.

**Hermes 4**: adapts Llama 3's template by changing the assistant token to a first-person identifier. Results in markedly different behaviors (lower refusals, role-play interpretation of fictional prompts, anti-sycophancy effects on CoT). See [[alignment-methods]] for the behavior implications.

**DeepSeek-R1-Zero**: similar to others but adds `\boxed{}` tags for the final answer.

## Sequence Packing

Pack sequences within batches to avoid padding waste, with a constraint of minimizing truncation of documents across batch boundaries.

- **Best-Fit Decreasing** (implemented in TRL): place each sequence in the batch that minimizes remaining space after insertion.
- **First-Fit Decreasing** (Hermes 4's choice): place in the first batch with enough remaining space. ~93% batch efficiency.

Packing can yield up to **33× tokens per batch per optimization step**, but at fixed token budget, more data per batch means fewer gradient updates. This **hurts small datasets** where each sample matters more.

In SmolLM3 ablations: effective batch size 128 hurt IFEval by up to 10%; for effective batch sizes > 32, there was an average performance drop. For large datasets, packing is almost always beneficial.

## Loss Masking

**Mask user turns** so loss is computed only on assistant tokens. Without masking, the model trades off high-quality assistant responses for predicting user queries — which you don't actually want. In practice, masking yields a few-point improvement; not huge but real.

## Hermes 4: Two-Stage SFT for Reasoning Budgets

Hermes 4 does two SFT stages, both on reasoning. They observed: despite training on sequences ≤ 16k tokens, reasoning lengths frequently exceed 41k tokens at inference.

Stage 2 teaches the model to insert `</think>` at 30k tokens — their reasoning budget. **Fixed-token-count insertion** teaches a counting behavior ("when you reach N tokens, stop") while keeping the model's own distribution intact. This avoids the **model-collapse** problem of recursive training on full self-generated outputs, which leads to distribution narrowing and quality degradation.

## Intellect-3: Two-Stage SFT for Agentic Behavior

- **Stage 1 (general reasoning)**: 9.9B tokens from Nemotron post-training (math, code, science, tooling, chat, instruction) + AM-DeepSeek-R1-0528-Distilled.
- **Stage 2 (agentic)**: open-source agentic datasets (SWE-Swiss) + synthetic data from the Environments Hub using DeepSeek-R1. Side effect: pushed context length from 65K → 98K via context parallelism.

## Retrofitted Recurrence for Deeper Thinking

[Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence (2511.07384)](../../papers/05-learning/fine-tuning/TEACHING PRETRAINED LANGUAGE MODELS TO - 2511.07384.pdf) treats recurrence as a post-training/model-surgery option: take a pretrained fixed-depth transformer, keep early layers as a **prelude**, reuse later layers as a **recurrent block** plus **coda**, remove some middle layers, and train the model to loop the recurrent block. This spends more compute at inference by recurring layers rather than by emitting longer chain-of-thought tokens, so it does not increase context length or KV memory in the same way verbal reasoning does.

The recipe is conservative: initialize from pretrained weights, add a small adapter for the recurrent state, and use a recurrence curriculum that gradually increases the average recurrent depth during training. The authors report that this is much more training-efficient than random initialization and that converted TinyLlama, OLMo, and Llama-3.2-1B models improve on GSM8K/MATH at a fixed training-compute budget compared with continued training of the non-recurrent parent. They also found a short **healing** period on broad language data helps recover general LM performance before switching to math-heavy data.

The practical implication for SFT is that "teach the model to think longer" does not always mean "train on longer reasoning traces." Retrofitted recurrence is closer to continued pretraining plus architecture surgery than ordinary instruction SFT, but it belongs in the same post-training decision tree: use it when latent test-time compute is desirable and you can afford careful optimizer/curriculum work. The paper's optimizer result also matches the broader [[optimizers]] page: Muon was more stable than AdamW for recurrent models and avoided loss spikes in their runs.

## Multi-turn Reasoning (SmolLM3 IFThink)

Hugging Face found issues generalising single-turn reasoning data to multi-turn — particularly difficulty differentiating `/think` and `/no_think` tags between turns. They constructed **IFThink**, augmenting single-turn instructions into multi-turn exchanges with verifiable instructions and reasoning traces, generated by Qwen3-32B. Dramatically improved multi-turn reasoning.

## Generalized Advantage Estimation (GAE) and Actor-Critic RL Integration

When post-training shifts toward RL, advantage estimation becomes critical. **GAE** ([High-Dimensional Continuous Control Using Generalized Advantage Estimation (1506.02438)](../../papers/05-learning/reinforcement-learning/High-Dimensional Continuous Control Using Generalized Advantage Estimation - 1506.02438.pdf)) provides a variance-reduction scheme parameterized by γ ∈ [0,1] (discount factor) and λ ∈ [0,1] (trace decay):

Â^GAE(γ,λ)_t = Σ(γλ)^l δ^V_{t+l}, where δ^V_t = r_t + γV(s_{t+1}) - V(s_t)

This is an exponentially-weighted average of k-step advantage estimators. The two extremes: GAE(γ,0) is the 1-step TD residual (low variance, high bias), and GAE(γ,1) is the full return minus baseline (high variance, unbiased). In practice, λ=0.96–0.99 and γ=0.99 balance variance and bias for long-horizon tasks. GAE is now standard in both policy-gradient methods (PPO, TRPO) and [[on-policy-distillation|on-policy distillation]], reducing the number of rollouts needed by substantial margins while maintaining stability.

## Hyperparameters

- **Learning rate**: ~10× smaller than pre-training. Aggressive updates → catastrophic forgetting (the model already has rich representations). SmolLM3 best results at 3e-6 or 1e-5. With packing enabled, **further decrease LR** due to the larger effective batch size and fewer updates for the same token budget.
- **Epochs**: training more than one epoch (vs the single-epoch default in ablations) gives a few-percentage-point lift. On LiveCodeBench v4, performance nearly **doubled from epoch 2 to 3**.
- **Optimizer**: AdamW remains the default for both pre and post-training. For recurrent and depth-stacked models, **Muon outperforms AdamW**, achieving lower loss and eliminating instability spikes typical of second-moment-based optimizers on deep networks.

## Other Engineering Choices

- **Full FT vs LoRA/QLoRA**: see [[parameter-efficient-fine-tuning]].
- **FlashAttention**: reduces memory by recomputing attention on the fly during backward — trades compute for memory.
- **SonicMoE** and similar: more efficient compute for MoE layers.
- **Cute Cross-Entropy (CCE) kernel**: memory-efficient CUDA kernel for cross-entropy loss. Instead of materializing the full logit matrix in global memory, computes only the logit for the correct token and evaluates log-sum-exp over vocab on-the-fly using faster memory tiers. Leverages softmax sparsity by skipping gradient computation for negligible elements. Particularly valuable for large-vocabulary models.
- **Parallelism**: tensor, pipeline, data, expert. Match to your model size and cluster.
- **Sequence length tuning**: match to data distribution to speed up training, especially for larger datasets.

## Related Topics

- [[alignment-methods]] — DPO, KTO, ORPO, APO, RLHF, GRPO; what comes after SFT
- [[on-policy-distillation]] — the dense-supervision alternative or complement to RL
- [[parameter-efficient-fine-tuning]] — LoRA, QLoRA when full FT is too expensive
- [[reasoning-models]] — distilled mid-training, reasoning data preparation
- [[practical-fine-tuning]] — hands-on guidance for smaller fine-tuning projects
- [[optimizers]] — Muon vs AdamW for post-training
- [[frontier-training-playbook]] — where SFT sits in the broader recipe

## Sources

- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- SmolLM3 report. See `raw/smollm3-hugging-face-report.md`.
- Hermes 4 technical report (two-stage SFT, fixed-position `</think>`).
- Intellect-3 (Prime Intellect, two-stage SFT with agentic behavior).
- gpt-oss-120b system card (harmony chat template).
- [Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence (2511.07384)](../../papers/05-learning/fine-tuning/TEACHING PRETRAINED LANGUAGE MODELS TO - 2511.07384.pdf)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (1506.02438)](../../papers/05-learning/reinforcement-learning/High-Dimensional Continuous Control Using Generalized Advantage Estimation - 1506.02438.pdf)
