# QuIP: 2-Bit Quantization of Large Language Models With Guarantees

**Source:** https://arxiv.org/abs/2307.13304
**Authors:** Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa (Cornell)
**Published:** 2023

## Key Concepts

First viable 2-bit quantization method using incoherence processing. Includes theoretical analysis showing that quantization error is minimized when weight and Hessian matrices are incoherent. Introduces efficient pre/post-processing via random orthogonal matrices.

## Follow-up: QuIP#

QuIP# (arXiv:2402.04396) by Tseng et al. uses Hadamard transforms for faster incoherence processing, achieving state-of-the-art results in extreme compression (≤4 bits per weight). Represents the frontier of how far quantization can be pushed.
