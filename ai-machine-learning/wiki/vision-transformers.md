# Vision Transformers

Vision Transformers (ViT) demonstrated that the transformer architecture is not limited to language — it works remarkably well for computer vision when applied to image patches treated as tokens.

## ViT: An Image is Worth 16x16 Words

Dosovitskiy et al. (2020) showed that a pure transformer applied directly to sequences of 16×16 image patches achieves excellent image classification results, challenging the dominance of convolutional neural networks (CNNs). Images are split into fixed-size patches, linearly embedded, and processed exactly like word tokens in an NLP transformer.

Key finding: ViT outperforms CNNs when pre-trained on sufficient data (e.g., JFT-300M), but underperforms with smaller datasets due to fewer inductive biases than convolutions. It uses substantially fewer computational resources for training at scale.

## DeiT: Data-Efficient Image Transformers

DeiT (Touvron et al., 2021) solved ViT's data hunger by introducing knowledge distillation specific to transformers. It achieved 84.2% ImageNet top-1 accuracy training only on ImageNet (no external data) in just 3 days on 8 GPUs. This made vision transformers practical without Google-scale datasets.

## Impact on the Field

ViT spawned an entire family of models and led to the idea that transformers are a **universal architecture** across modalities. Swin Transformer added hierarchical features and shifted windows. BEiT applied BERT-style masked pre-training to vision. Eventually, this convergence enabled multimodal models (GPT-4V, Gemini) that process text and images with a single transformer backbone.

## Related Topics
- [[transformer-architecture]] — The core architecture adapted for vision
- [[positional-encodings]] — 2D positional encodings for image patches
- [[model-report-case-studies]] — Qwen-VLA extends this cross-modal unification idea from perception into robot action generation, attaching a DiT-based action decoder to a vision-language backbone so one model handles manipulation, navigation, and human egocentric demonstrations

## Sources
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (arxiv:2010.11929)
