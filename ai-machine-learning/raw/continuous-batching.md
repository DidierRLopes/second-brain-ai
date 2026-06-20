# Continuous Batching for LLM Inference

**Source:** https://huggingface.co/blog/continuous_batching
**Authors:** Hugging Face

## Overview

Explains how continuous (or dynamic) batching improves GPU utilization during LLM serving. Unlike static batching (where all requests in a batch must complete before new ones start), continuous batching allows individual requests to finish and be replaced immediately.

## Why It Matters

- Static batching wastes compute because shorter requests pad/wait for the longest request
- Continuous batching can improve throughput by 2-10× depending on request length variance
- Now standard in all modern serving frameworks: vLLM, SGLang, TGI, TensorRT-LLM
- Often combined with PagedAttention for maximum memory efficiency
