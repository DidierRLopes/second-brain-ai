# LoRA: Low-Rank Adaptation of Large Language Models

**Source:** https://arxiv.org/abs/2106.09685
**Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)
**Published:** 2021

## Key Concepts

Foundational parameter-efficient fine-tuning (PEFT) method that reduces trainable parameters by 10,000x while maintaining full fine-tuning performance. Injects trainable low-rank decomposition matrices (A and B) into each transformer layer alongside frozen pre-trained weights.

## Key Properties

- No additional inference latency (weights can be merged)
- Works by decomposing weight updates: ΔW = BA where B is d×r and A is r×d (r << d)
- Typical rank r = 8 or 16 is sufficient for most tasks
- Can be applied selectively to attention weights (Q, K, V, O) and/or MLP layers
- Spawned many variants: QLoRA, DoRA, AdaLoRA, LoRA+
- The de facto standard for efficient LLM fine-tuning
