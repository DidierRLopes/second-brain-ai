# RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)

**Source:** https://arxiv.org/abs/2104.09864
**Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
**Published:** 2021

## Key Concepts

Rotary Position Embeddings (RoPE) encode absolute positions with rotation matrices while incorporating explicit relative position dependency. This offers sequence length flexibility and naturally decaying inter-token dependency with distance.

## Why It Matters

- Widely adopted in LLaMA, PaLM, GPT-NeoX, and most modern LLMs
- More elegant than learned positional embeddings or sinusoidal encodings
- Enables better length extrapolation than absolute positional embeddings
- Key enabler for extending context lengths via techniques like NTK-aware scaling and YaRN
