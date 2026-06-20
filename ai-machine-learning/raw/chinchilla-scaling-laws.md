# Training Compute-Optimal Large Language Models (Chinchilla)

**Source:** https://arxiv.org/abs/2203.15556
**Authors:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, et al. (DeepMind)
**Published:** 2022

## Key Concepts

Foundational scaling laws paper establishing that compute-optimal training requires roughly equal scaling of model size and training data (approximately 20 tokens per parameter). Showed that many existing models were severely undertrained.

## Key Findings

- Chinchilla (70B parameters, 1.4T tokens) outperforms Gopher (280B) with same compute budget
- Established the "Chinchilla optimal" training ratio: ~20 tokens per parameter
- Proved GPT-3 (175B, 300B tokens) was 4-5x undertrained
- Shifted the field from "bigger model" to "more data" for same compute
- Directly influenced LLaMA, Mistral, and other efficient model designs
