# Distilling the Knowledge in a Neural Network

**Source:** https://arxiv.org/abs/1503.02531
**Authors:** Geoffrey Hinton, Oriol Vinyals, Jeff Dean (Google)
**Published:** 2015

## Key Concepts

The seminal knowledge distillation paper. A smaller "student" model is trained to match the soft probability outputs (logits) of a larger "teacher" model, rather than just the hard labels. Soft targets carry richer information — they encode which classes the teacher considers similar, transferring "dark knowledge" about the data structure.

The temperature parameter controls how soft the distributions are: higher temperature spreads probability mass more evenly, revealing more inter-class relationships. This simple technique enables significant model compression with minimal quality loss.

## Impact

- Foundation for all subsequent distillation work
- Enabled deployment of large model capabilities in resource-constrained settings
- Applied to BERT (→ DistilBERT), reasoning models (DeepSeek-R1 distillation), and more
