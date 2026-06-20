# QLoRA: Efficient Finetuning of Quantized LLMs

**Source:** https://arxiv.org/abs/2305.14314
**Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
**Published:** 2023

## Key Concepts

Combines 4-bit NormalFloat (NF4) quantization with LoRA for extremely memory-efficient fine-tuning. Enables fine-tuning a 65B parameter model on a single 48GB GPU while preserving full 16-bit fine-tuning performance.

## Technical Innovations

- **NF4 datatype:** Information-theoretically optimal for normally-distributed weights
- **Double quantization:** Quantizes the quantization constants themselves, saving ~0.37 bits per parameter
- **Paged optimizers:** Uses NVIDIA unified memory to handle memory spikes during gradient checkpointing
- Bridges the gap between quantization and fine-tuning — the model stays quantized during training
- Democratized LLM fine-tuning for researchers with limited hardware
