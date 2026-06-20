# Quantization Fundamentals

Quantization is the process of reducing the numerical precision of a model's weights (and sometimes activations) — typically from 16-bit floating point to 8-bit or 4-bit integers. This is one of the most effective techniques for making large language models smaller, faster, and runnable on consumer hardware.

## Why Quantization Matters

LLMs are primarily bottlenecked by **memory bandwidth**, not compute. Loading billions of 16-bit parameters from GPU memory is slow. If you can represent each parameter in 4 bits instead of 16, you get a 4× reduction in model size, roughly 2× faster inference, and the ability to run models on hardware that previously couldn't fit them — all while typically losing only 5-10% accuracy.

## How It Works

Instead of storing weights as 32-bit or 16-bit floats, quantization maps them to a smaller set of integer values. During inference, these integers are **dequantized on-the-fly** back to floating point for computation. The key is choosing the mapping (scale and zero-point) to minimize information loss.

**Symmetric quantization** maps values symmetrically around zero. **Asymmetric quantization** allows the zero-point to shift, better handling distributions that aren't centered at zero.

**Per-tensor vs per-channel vs per-group:** Finer granularity (per-channel or per-group) uses more scale factors but preserves more information. Group-wise quantization (e.g., groups of 128 weights) is the standard in modern methods.

## Hardware-Native Microscaling Formats: MXFP4 vs NVFP4

Everything above (symmetric/asymmetric, per-tensor/channel/group) describes *integer* quantization: a scale and zero-point map floats onto an integer grid. The other axis evolving fast since 2025 is **floating-point quantization formats with a shared block exponent** — "microscaling" — where the stored value is itself a tiny float (e.g. 4-bit) and a single scale factor is shared across a small block of elements, giving floats' better dynamic range without integer quantization's single-scale-per-tensor blind spot.

- **MXFP4** (Open Compute Project / OCP standard, multi-vendor): 4-bit float elements (1 sign, 2 exponent, 1 mantissa bit — "E2M1"), grouped in blocks of **32**, sharing one **E8M0** scale (8-bit exponent, no mantissa — a pure power-of-two scale).
- **NVFP4** (NVIDIA, Blackwell-specific — B100/B200/GB200, not adopted by the OCP consortium): same 4-bit element format, but smaller blocks of **16** with a richer **E4M3 FP8** shared scale (plus an optional outer FP32 scale). Smaller blocks track local distribution shifts more tightly, and the FP8 scale (vs MXFP4's power-of-two-only E8M0 scale) gives finer-grained scale resolution.

The tradeoff is storage vs accuracy: NVFP4 costs ~4.5 bits/element effective (vs MXFP4's 4.19) for roughly **0.3-0.5 perplexity improvement** at the same nominal 4-bit width — smaller blocks with finer scales waste less precision on tensors with non-uniform value distributions, at the cost of a bit more scale-metadata overhead. Both formats now have first-class support in Transformer Engine, vLLM, and TensorRT-LLM, and this is the numerical foundation behind native FP4 inference (and increasingly FP4 training) on current-generation hardware — the same drive toward lower-precision-by-default that motivated [[applied-ml-systems|FP8 training]] a generation earlier, pushed one bit further with block-shared scaling doing the work a single per-tensor scale can't.

## Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)

**PTQ** quantizes a pre-trained model without retraining. Faster and simpler. Methods like [[quantization-methods|GPTQ, AWQ, and LLM.int8()]] are PTQ approaches.

**QAT** simulates quantization during training so the model learns to be robust to reduced precision. Higher quality but much more expensive. Less common for LLMs due to training cost.

## The Outlier Problem

Modern transformers develop extreme [[activation-outliers]] — individual activations 100× larger than typical values. These outliers break simple quantization because the range is dominated by a few extreme values, wasting precision on the majority of normal values. Handling outliers is the central challenge in LLM quantization.

## Related Topics
- [[quantization-methods]] — Specific algorithms and their tradeoffs
- [[activation-outliers]] — Why transformers are hard to quantize
- [[practical-quantization]] — Running quantized models in practice
- [[parameter-efficient-fine-tuning]] — QLoRA combines quantization with fine-tuning
- [[applied-ml-systems]] — FP8/BF16/FP16 training-numerics context that MXFP4/NVFP4 extend
- [[kv-cache]] — TurboQuant/PolarQuant apply vector quantization specifically to the KV cache

## Sources
- Quantization from the Ground Up — Sam Rose (ngrok.com/blog/quantization)
- A Minimal Introduction to Quantization — Omar Sanseviero
- OCP Microscaling Formats (MX) Specification — Open Compute Project
- NVFP4 / Blackwell numerics documentation — NVIDIA
