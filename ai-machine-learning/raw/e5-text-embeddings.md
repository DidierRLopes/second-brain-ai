# E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training

**Source:** https://arxiv.org/abs/2212.03533
**Authors:** Liang Wang, Nan Yang, et al. (Microsoft, 2022)

## Key Concepts

E5 uses contrastive pre-training on 1 billion multilingual text pairs with specialized prefixes ("query:" and "passage:"). Achieves best MTEB results when fine-tuned, beating larger models. Available in small/base/large variants. Multilingual E5 (2024) extended this to cross-lingual settings.

## Significance

- Demonstrated that large-scale weak supervision produces excellent embeddings
- The prefix-based approach (query: vs passage:) became standard practice
- Strong zero-shot transfer across tasks
