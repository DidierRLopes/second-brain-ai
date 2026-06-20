# DiffusionBlocks: Block-Wise Training via a Diffusion-Model Reframing (Sakana AI)

**Source:** https://pub.sakana.ai/diffusionblocks/
**Filed under:** Memory-efficient training, generative models, ICLR 2026 (arXiv 2506.14202)

## Premise

End-to-end backpropagation through a full deep network requires storing activations for every layer simultaneously, making memory the binding constraint on how large a network can be trained on given hardware. DiffusionBlocks reframes **block-wise training** (training each block of layers somewhat independently, à la greedy/local learning) not as an approximation to end-to-end backprop but as literally the **reverse (denoising) process of a diffusion model** — each block becomes one denoising step, trained with its own independent, local loss, eliminating the need to hold cross-block activations.

## Conceptual Foundation

Builds on the existing observation that **residual connections implement a discretized ODE** — i.e. a residual network's forward pass already looks like the Euler-discretized trajectory of a continuous-time process. DiffusionBlocks pushes this further: if the residual stream already resembles an ODE trajectory, then training it block-by-block can be cast exactly as training a diffusion model's reverse/denoising trajectory, where each block denoises one step further from a "noised" input toward the clean target representation.

## Method: Three-Step Conversion

1. **Partition** the network into contiguous blocks of layers (the unit that will become one denoising step).
2. **Assign a noise range** to each block — i.e. give each block responsibility for denoising a specific portion of the noise schedule, analogous to how diffusion timesteps are partitioned across a U-Net's stages.
3. **Add conditioning**, e.g. **AdaLN** (adaptive layer normalization), so each block knows which noise level / step it's responsible for and can modulate its computation accordingly.

Each block can then be trained with a fully **local, independent loss** — no gradient needs to flow across block boundaries, so the only activations that must be held in memory at any time are those of the single block currently being trained.

## Memory and Validated Architectures

Achieves a **B× memory reduction** relative to standard end-to-end backprop, where B is the number of blocks the network is partitioned into. Validated across five distinct architecture families to demonstrate generality rather than a single-architecture trick:

- **ViT** (Vision Transformer)
- **DiT** (Diffusion Transformer)
- **Masked Diffusion** models
- **AR (autoregressive) Transformer**
- **Recurrent-depth / Looped Transformer**

## Special Case: Recurrent-Depth Models

For recurrent-depth (looped/weight-shared) transformers, standard training requires **backpropagation through time (BPTT)** across all K iterations of the loop, which is both memory- and compute-heavy. Because DiffusionBlocks already treats each iteration as a separate local-loss denoising step, it lets recurrent-depth models be trained with a **single forward pass per step** instead of full K-step BPTT — a direct practical win for this architecture family specifically (recurrent-depth models being an active area of efficient-test-time-compute research).

## Related Prior Work

**NoProp** is cited as related prior work also recasting layer-wise/block-wise training through a diffusion-like local-loss lens, but its validation was limited to image classification; DiffusionBlocks' contribution is generalizing the idea across five architecture families including generative and autoregressive models, not just classifiers.

## Future Work

The authors flag two open directions: (1) a **theoretical analysis** of why memory reduction and performance co-improve rather than trade off — floated as possibly an **implicit curriculum** effect, where training each block on its assigned noise range acts like an easy-to-hard curriculum across the network's depth; and (2) extending the method to **fine-tune existing large pretrained models** rather than only training from scratch, which would be the more immediately impactful direction for practitioners who don't control pretraining.

## Related wiki pages

[[generative-models]] (diffusion models background), [[deep-learning-fundamentals]] (residual connections / ODE view), [[memory-efficient-training]] (candidate new page), [[transformer-architecture]]
