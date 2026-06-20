# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**Source:** https://arxiv.org/abs/2305.13245
**Authors:** Joshua Ainslie, James Lee-Thorp, Michal de Jong, Yinfei Yang, Siddhartha Reddy Jonnalagadda, Santiago Ontañón (Google Research)
**Published:** 2023

## Key Concepts

Grouped Query Attention (GQA) partitions query heads into groups that share key and value heads, striking a balance between the quality of multi-head attention and the efficiency of multi-query attention. This significantly reduces KV-cache memory during inference.

## Adoption

- Adopted by Llama 2 and Llama 3
- Standard in most modern open-source LLMs
- Significantly improves inference throughput and reduces memory for long sequences
