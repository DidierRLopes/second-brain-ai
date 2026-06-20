# Quantization Methods

This article covers the major LLM quantization algorithms, roughly in order of their development and contribution to the field.

## LLM.int8() — Mixed-Precision Decomposition (2022)

Tim Dettmers' LLM.int8() was the first method to successfully quantize very large transformers. It uses a two-part approach: **vector-wise INT8 quantization** for the 99.9% of values that are normal, and **FP16 decomposition** for the rare outlier dimensions. The key insight was that outlier features occur in specific dimensions consistently across tokens, so you can separate them cleanly.

This enabled running models ~2× larger on the same hardware. It also revealed the [[activation-outliers|emergent outlier phenomenon]] — these outliers don't exist in small models and appear suddenly at scale.

## GPTQ — Layer-Wise Optimal Quantization (2022)

GPTQ (Frantar et al.) uses approximate second-order information (the Hessian matrix) to find the optimal way to quantize each weight, minimizing the squared error layer-by-layer. It was the first method to compress LLMs to **3-4 bits** without significant quality loss, processing a 175B model in ~4 hours on a single GPU.

GPTQ became a widely-used format, supported by llama.cpp, vLLM, and many other frameworks. It showed that post-training quantization could be surprisingly effective even at very low bit widths.

## AWQ — Activation-Aware Weight Quantization (2023)

AWQ (Lin et al., MIT) takes a different approach: rather than using weight magnitude to decide importance, it uses **activation statistics**. Weights connected to large activations are more important. AWQ protects just 1% of salient channels via per-channel scaling, achieving 4× compression without backpropagation. Won the **MLSys 2024 Best Paper Award**.

AWQ is more hardware-friendly than mixed-precision approaches like LLM.int8() because it keeps all weights in the same format.

## QuIP / QuIP# — The 2-Bit Frontier (2023-2024)

[QuIP: 2-Bit Quantization of Large Language Models With Guarantees (2307.13304)](../../papers/04-efficiency/quantization/QuIP: 2-Bit Quantization of Large Language Models With Guarantees - 2307.13304.pdf) pushed quantization to the extreme: **2 bits per weight**. The key insight is that quantization error is minimized when weight and Hessian matrices are "incoherent" (no large entries). Random orthogonal transforms achieve this. QuIP consists of two steps: **(1) adaptive rounding** via LDLQ (a theoretically optimal procedure), and **(2) incoherence processing** — pre- and post-multiply weights and Hessian by random orthogonal matrices, ensuring they remain incoherent by a factor of μ = O(1). This spreading of outlier energy across dimensions is the key to 2-bit viability. QuIP# replaced these with faster Hadamard transforms and added lattice codebooks, achieving state-of-the-art at ≤4 bits. Empirically, QuIP achieves the first viable 2-bit compression on OPT-66B and Llama 2 70B without severe accuracy loss, with solid performance at 3-4 bits. This represents the current frontier of how far you can compress without losing too much quality.

## AQLM — Additive Quantization for LLMs (2024)

[Extreme Compression of Large Language Models via Additive Quantization (2401.06118)](../../papers/04-efficiency/quantization/Extreme Compression of Large Language Models via Additive Quantization - 2401.06118.pdf) extends classical **additive quantization** to LLM compression. Instead of representing each weight group with one scalar quantization code, AQLM represents a group as the sum of vectors chosen from **multiple learned codebooks**. This gives a richer approximation at 2-3 bits per weight, where ordinary scalar methods degrade sharply. The algorithm initializes codebooks, uses beam search to select codes, updates codebooks with a least-squares-style objective, and then fine-tunes quantized transformer blocks or the full compressed model against the original model's outputs. Empirically, AQLM is strongest in the extreme-compression regime: it improves over prior 2-bit methods such as QuIP/QuIP# on Llama 2 and Mixtral in the reported setups, and the authors frame its Pareto-optimal point as roughly below 3 bits rather than exactly 2 bits. The tradeoff is calibration cost: AQLM is slower to quantize than direct methods like GPTQ, but it has practical GPU and CPU kernels, including reported ~30% layer-wise GPU speedups in some configurations and up to 4× CPU speedup over FP32.

For most everyday serving, [[practical-quantization]] still favors 4-bit GGUF/GPTQ/AWQ first; QuIP/QuIP# and AQLM are the tools to reach for when memory pressure makes sub-4-bit compression worth the extra calibration and kernel complexity.

## Related Topics
- [[quantization-fundamentals]] — Core concepts behind all these methods
- [[activation-outliers]] — The central challenge these methods address
- [[practical-quantization]] — How to use these methods in practice

## Sources
- LLM.int8() and Emergent Features — Tim Dettmers
- GPTQ (arxiv:2210.17323)
- AWQ (arxiv:2306.00978)
- [QuIP: 2-Bit Quantization of Large Language Models With Guarantees (2307.13304)](../../papers/04-efficiency/quantization/QuIP: 2-Bit Quantization of Large Language Models With Guarantees - 2307.13304.pdf)
- [Extreme Compression of Large Language Models via Additive Quantization (2401.06118)](../../papers/04-efficiency/quantization/Extreme Compression of Large Language Models via Additive Quantization - 2401.06118.pdf)
