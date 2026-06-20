# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

**Source:** https://arxiv.org/abs/2210.17323
**Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
**Published:** 2022

## Key Concepts

Foundational post-training quantization method enabling compression to 3-4 bits while maintaining accuracy. First method to compress LLMs to the 4-bit range without significant quality loss, allowing inference of 175B parameter models on single GPUs.

## Technical Details

- Based on approximate second-order information (Hessian)
- Quantizes weights one layer at a time
- Uses optimal brain quantization framework adapted for scale
- Processes 175B models in ~4 hours on a single GPU
- Foundational for the GPTQ format widely used in llama.cpp and other frameworks
