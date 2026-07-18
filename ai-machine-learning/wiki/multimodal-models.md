# Multimodal Foundation Models

Multimodal foundation models process text, images, audio, or other modalities in one system. The central architectural tension is how much computation to share: fully shared parameters encourage cross-modal transfer but force statistically different modalities to compete, while separate towers specialize well but can lose token-level interaction.

## Mixture-of-Transformers

[Mixture-of-Transformers (2411.04996)](../../papers/02-architecture/multimodal/Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models - 2411.04996.pdf) introduces deterministic modality-aware sparsity. Every non-embedding Transformer parameter - Q/K/V/O projections, feed-forward networks, and layer normalization - is separated by modality. Tokens are grouped by modality for those projections, but all queries and keys still participate in **global self-attention**, preserving direct cross-modal interaction.

This differs from ordinary [[mixture-of-experts|MoE]] routing. MoT uses the known modality label rather than a learned router and sparsifies the entire Transformer block rather than only its FFN. The paper's leave-one-modality-out study found that forcing any two of text, image, and speech to share a tower consistently worsened their losses, supporting modality-specific allocation rather than merely adding sparse parameters.

The reported efficiency gains are substantial:

- In a 7B Chameleon text-image setting, MoT matched the dense baseline at 55.8% of its training FLOPs.
- Adding speech, comparable speech performance required 37.2% of the dense baseline's FLOPs.
- In Transfusion, a 7B MoT matched dense image performance with less than one-third of the FLOPs; a 760M MoT outperformed a 1.4B dense model on key image metrics.
- On A100 instances, dense-equivalent image quality took 47.2% of the wall-clock time and text quality took 75.6%.

MoT and MoE were complementary: replacing only the text tower's FFN with a four-expert MoE improved text performance while retaining MoT's image gains. This suggests two distinct sparsity axes - deterministic routing by modality across the block, and learned routing by token within a modality.

## Related Topics

- [[transformer-architecture]] — the shared attention and modality-specific block components
- [[mixture-of-experts]] — learned expert routing and its systems trade-offs
- [[generative-models]] — autoregressive and diffusion objectives used for different modalities
- [[vision-transformers]] — image token processing inside Transformer systems

## Sources

- [Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models (2411.04996)](../../papers/02-architecture/multimodal/Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models - 2411.04996.pdf) — modality-specific parameter decoupling, global cross-modal attention, Chameleon/Transfusion evaluations, and MoT+MoE combination.
