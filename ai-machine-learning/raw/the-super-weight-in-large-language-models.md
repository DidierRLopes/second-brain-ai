# The Super Weight in Large Language Models

**Source:** https://arxiv.org/abs/2411.07191
**Authors:** Mengxia Yu, De Wang, Qi Shan, Colorado J Reed, Alvin Wan

## Abstract

Recent works have shown a surprising result: a small fraction of Large Language Model (LLM) parameter outliers are disproportionately important to the quality of the model. LLMs contain billions of parameters, so these small fractions, such as 0.01%, translate to hundreds of thousands of parameters. In this work, we present an even more surprising finding: Pruning as few as a single parameter can destroy an LLM's ability to generate text -- increasing perplexity by 3 orders of magnitude and reducing zero-shot accuracy to guessing. We propose a data-free method for identifying such parameters, termed super weights, using a single forward pass through the model.

## Key Findings

- A single parameter ("super weight") can be so important that pruning it destroys an LLM's ability to generate text
- Perplexity increases by 3 orders of magnitude when super weights are removed
- Zero-shot accuracy drops to random guessing level
- Super weights can be identified using a data-free method with a single forward pass
- These super weights induce correspondingly rare and large activation outliers, termed "super activations"
- When preserved with high precision, super activations can improve simple round-to-nearest quantization to become competitive with state-of-the-art methods
- This has major implications for quantization and model compression techniques
