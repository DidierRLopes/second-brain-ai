# Training Stability

Stable training is mostly about sane defaults, not exotic tricks. As models scale, small numerical issues in attention logits, embeddings, or initialization compound into loss spikes that derail multi-week runs. The frontier reports converge on a handful of stabilization techniques — logit softcapping (Gemma 2/3), z-loss, weight decay handling on embeddings, QK-norm, RMSNorm variants — but the bigger lesson is that **most "mystery failures" are configuration or data issues**, not subtle math problems. Stability work pays off twice: once during the main run, and again when training ops needs to debug a regression.

## Logit Softcapping (preferred)

The Gemma 2/3 approach, and now the modern default for attention/LM-head logit stabilization. Map logits into a bounded range with a smooth, differentiable transform:

```
softcap(z) = cap * tanh(z / cap)
```

Unlike hard clipping (zero gradient at boundaries, can destabilize training), softcap compresses smoothly. The `cap` hyperparameter controls the output range.

Gemma 2 applies softcapping to both **attention logits (pre-softmax)** and the **final LM head**:
- `cap = 50` for attention layers
- `cap = 30` for the final layer

The technique traces back to Bello et al., 2016 (neural machine translation).

**Caveat:** logit softcapping is incompatible with Flash Attention / SDPA during training because those fused kernels assume standard attention. For stable fine-tuning, use `attn_implementation="eager"`. Inference can still use SDPA with minimal quality difference.

## z-loss

A regularization term added to standard cross-entropy that keeps logits from drifting to large magnitudes. The softmax denominator is `Z = sum_v exp(z_v)`; adding `(log Z)^2` penalizes the overall logit scale.

On a 1B Hugging Face model, z-loss didn't impact training loss or downstream performance, so they skipped it to avoid the overhead. Logit softcapping is generally preferred in modern recipes.

## QK-norm

Apply LayerNorm to query and key vectors before computing attention. Prevents attention logits from becoming too large, similar in spirit to z-loss but acting locally on attention.

**Don't assume "always good":** the same paper that proposed RNoPE found QK-norm hurts long-context tasks because normalization de-emphasizes relevant tokens — it strips the query-key dot product of its magnitude.

## RMSNorm and Depth-Scaled Sandwich Norm

**RMSNorm** maintains comparable performance to LayerNorm while being computationally simpler — it avoids the mean-centering step.

**Depth-scaled sandwich norm** (Arcee's Trinity) applies normalization both before and after each attention/MLP block, with the normalization scale adjusted by layer depth. The pattern:

```
h_l = h_{l-1} + γ_l * RMSNorm_post(F(RMSNorm_pre(h_{l-1})))
```

Arcee initializes `γ ~ N(1, 0.02²)` and `γ_depth ~ 1/sqrt(2L)`. Depth-dependent scaling reflects that activations evolve differently across layers. The sandwich pattern (pre-norm + post-norm) is stabilizing in very deep networks where gradient flow becomes challenging. Arcee also applies RMSNorm before the LM head for consistent output activation scales.

## Weight Decay and Embeddings

Removing weight decay from embeddings can improve stability. Weight decay shrinks embedding norm, which causes larger gradients in earlier layers because the LayerNorm Jacobian has a `1/||x||` term — inversely proportional to input norm.

Hugging Face tested baselines with weight decay, without weight decay on embeddings, and a combined recipe; no significant differences in loss or evals, so they removed weight decay from embeddings to be safe.

## MuonClip (Kimi K2)

A purpose-built stabilization technique for the Muon optimizer that prevents exploding attention logits — a common failure mode at large scale. For each head, compute the per-head max logit `S_max^h = max (Q_i K_j / sqrt(d)) / τ` across batch and positions. When `S_max = max_h S_max^h > τ` (hyperparameter threshold), rescale Q and K weight matrices multiplicatively by `(τ/S_max)^α`. Commonly `α = 0.5` for equal Q/K scaling.

Per-head clipping is straightforward for MHA. For MLA, keys are projected from a latent variable rather than materialized directly, so clipping must apply to the latent-to-key projection weights and the latent variable itself, with separate scaling for head-specific Q, K, rotary components, and the shared rotary.

In a 9B-active 53B-total MoE training run, attention logits diverged quickly without MuonClip. With MuonClip and `τ = 100`, max logits decayed to a stable range after ~30% of training steps.

See [[optimizers]] for the rest of MuonClip's integration with Muon.

## Initialization

- **TruncDNormal**: `N(0, σ²)` with clipping at ±3σ. Common `σ = 0.006`. Prevents extreme initial values that destabilize training, particularly for embedding layers where large initial activations propagate through the network.
- **μP (maximal update parametrization)**: dictates how weights and learning rates should scale with width so training dynamics stay comparable across scales.
- **Heuristic**: `σ ~ 1/sqrt(d_model)`, with coefficient that varies in practice.
- **OLMo2 finding**: `σ = 0.02` more stable than the scaled init alternative.

## Embedding Scaling on Forward Pass

Several implementations scale the embedding layer's activations by `sqrt(d_model)` during the forward pass: `embed = E(x) * sqrt(d_model)`. Keeps embedding magnitudes in a stable range relative to the residual stream. Used in Grok-1, Grok-2, Trinity Large, and the first two generations of Gemma.

## Other Design Considerations

- **Activation**: SwiGLU is what most modern LLMs use (including gpt-oss-120b's gated SwiGLU). Exceptions: Gemma 2 uses GeGLU; some NVIDIA models use `ReLU²`.
- **Width vs depth**: deeper outperforms wider at small scales. Larger models trend wider for inference parallelism.
- **Precision**: avoid fp16 (overflow-prone). Use bf16 or mixed-precision recipes.

## Stability Takeaways

- Stabilization is mostly sane defaults, not exotic tricks.
- Logit softcapping (Gemma-style) is the preferred method for attention/LM-head; z-loss and QK-norm are alternatives.
- QK-norm can hurt long-context tasks — don't assume it's "always good".
- Initialization and normalization details matter more as depth grows.
- Track loss spikes early; many "mystery failures" are configuration or data issues.

## Related Topics

- [[optimizers]] — MuonClip's integration with the Muon optimizer
- [[attention-variants]] — gated attention also reduces large activations
- [[training-ops]] — debugging loss spikes; "the usual suspects"
- [[frontier-training-playbook]] — where stability sits in the broader recipe
- [[transformer-architecture|Hyper-Connections / mHC]] — the residual stream's own vanishing-gradient-vs-representation-collapse seesaw, and the manifold constraint DeepSeek added to keep it stable at scale

## Sources

- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- Gemma 2 / Gemma 3 reports (logit softcapping origin in modern LLMs).
- Bello et al., 2016 — original logit softcapping in neural machine translation.
- Kimi K2 technical report (MuonClip).
- OLMo 2 report (`σ = 0.02` init finding).
