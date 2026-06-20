# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)

**Source:** https://arxiv.org/abs/2010.11929
**Authors:** Alexey Dosovitskiy et al.
**Published:** ICLR 2021

## Key Concepts

Demonstrates that a pure transformer applied directly to sequences of image patches achieves excellent results on image classification. Images are split into fixed-size 16x16 patches treated as tokens. Shows superior performance compared to convolutional networks with substantially fewer computational resources for training when pre-trained on large datasets.

## Significance

- Proved transformers are not limited to NLP — they work for vision too
- Spawned an entire family of vision transformers (DeiT, Swin, BEiT, etc.)
- Led to unified architectures across modalities
