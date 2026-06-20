# Ring Attention with Blockwise Transformers for Near-Infinite Context

**Source:** https://arxiv.org/abs/2310.01889
**Authors:** Hao Liu, Matei Zaharia, Pieter Abbeel
**Published:** 2023

## Key Concepts

Ring Attention distributes the KV cache across GPUs in a ring topology with overlapped computation and communication. Each device holds a block of the sequence, computes attention on its local block, and passes KV blocks to the neighbor — communication is hidden behind computation.

Enables training on 100M+ token sequences and inference with 1M+ token contexts by linearly scaling memory with the number of devices. The context length limit becomes the number of available GPUs, not memory per GPU.
