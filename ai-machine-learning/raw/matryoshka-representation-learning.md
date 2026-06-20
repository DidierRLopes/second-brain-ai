# Matryoshka Representation Learning

**Source:** https://arxiv.org/abs/2205.13147
**Authors:** Aditya Kusupati, Gantavya Bhatt, et al. (NeurIPS 2022)

## Key Concepts

MRL enables flexible-dimension embeddings by training models to encode the most critical information in the first dimensions of the embedding vector. Like Russian nesting dolls, you can truncate the embedding to any prefix length and still get useful representations.

Achieves up to 14× smaller embeddings with comparable accuracy, and 14× speed-ups for retrieval. Powers OpenAI's text-embedding-3 models which allow users to choose embedding dimensions at query time.

## Why It Matters

- Practical: choose embedding size at inference time based on latency/cost requirements
- No need to retrain for different dimension targets
- Adopted by OpenAI, Cohere, and many open-source models
