# Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing

**Source:** https://arxiv.org/abs/2306.12929
**Authors:** Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort

## Abstract

Transformer models have been widely adopted in various domains over the last years, and especially large language models have advanced the field of AI significantly. Due to their size, the capability of these networks has increased tremendously, but this has come at the cost of a significant increase in necessary compute. Quantization is one of the most effective ways to reduce the computational time and memory consumption of neural networks. Many studies have shown, however, that modern transformer models tend to learn strong outliers in their activations, making them difficult to quantize.

## Key Findings

- Strong outliers in transformer activations are related to very specific behavior of attention heads
- These attention heads try to learn a "no-op" or just a partial update of the residual
- The paper proposes methods for removing these outliers by helping attention heads "do nothing" more efficiently
- This makes transformers more amenable to quantization
- Addresses a fundamental challenge in quantizing modern transformer architectures

## Relevance
- Directly addresses why transformers are hard to quantize
- Connects activation outliers to attention head behavior
- Proposes architectural solutions rather than just quantization algorithm improvements
