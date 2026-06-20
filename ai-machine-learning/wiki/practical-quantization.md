# Practical Quantization

This article covers the practical side of running quantized LLMs: formats, tools, hardware considerations, and deployment guidance.

## Quantization Formats

**GGUF (formerly GGML):** The format used by llama.cpp. CPU-friendly, supports mixed quantization (different bit widths for different layers). Common quantizations: Q4_K_M (4-bit, good balance), Q5_K_M (5-bit, higher quality), Q8_0 (8-bit, near-lossless). Best choice for CPU or CPU+GPU hybrid inference.

**GPTQ:** GPU-optimized format. Quantizes weights to 4-bit using the Hessian-based method. Supported by vLLM, text-generation-inference, AutoGPTQ. Best for pure GPU inference serving.

**AWQ:** Also GPU-optimized but hardware-friendlier than GPTQ. Growing adoption in serving frameworks. Good for production deployment.

## Choosing the Right Quantization

The general rule: use the highest bit width your hardware can fit. For a given model on your GPU, the decision tree is roughly: Can you fit FP16? Use it. Can you fit 8-bit? Use LLM.int8() or Q8. Can you fit 4-bit? Use GPTQ or AWQ or Q4_K_M. Need extreme compression? Try QuIP# at 2-3 bits.

Quality loss is nonlinear — going from 16-bit to 8-bit loses almost nothing, 8 to 4-bit loses a little, and below 4-bit quality drops more steeply.

## Key Tools

- **llama.cpp**: CPU/GPU inference with GGUF. Best for local/laptop use
- **vLLM**: High-throughput GPU serving with GPTQ/AWQ support
- **bitsandbytes**: Hugging Face integration for 4-bit/8-bit loading and QLoRA training
- **AutoGPTQ / AutoAWQ**: Easy quantization of models into GPTQ/AWQ formats
- **ExLlamaV2**: Fast GPU inference with custom GPTQ kernels

## Hardware Considerations

For local use, VRAM is the key constraint. A rough guide for 4-bit quantized models: 7B ≈ 4GB VRAM, 13B ≈ 8GB, 30B ≈ 18GB, 70B ≈ 36GB. CPU inference via llama.cpp is viable but 5-10× slower than GPU.

## Related Topics
- [[quantization-fundamentals]] — The theory behind these practical choices
- [[quantization-methods]] — The algorithms producing these formats
- [[inference-optimization]] — Quantization is one part of the optimization stack
- [[parameter-efficient-fine-tuning]] — QLoRA lets you fine-tune quantized models

## Sources
- The Llama Hitchhiking Guide to Local LLMs — Omar Sanseviero
- A Minimal Introduction to Quantization — Omar Sanseviero
- Quantization from the Ground Up — Sam Rose (ngrok)
