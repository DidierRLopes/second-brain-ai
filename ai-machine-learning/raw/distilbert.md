# DistilBERT: A Distilled Version of BERT

**Source:** https://arxiv.org/abs/1910.01108
**Authors:** Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf (Hugging Face)
**Published:** 2019

## Key Concepts

DistilBERT achieves 40% size reduction from BERT while retaining 97% of language understanding capabilities and running 60% faster. Uses a triple loss combining language modeling, knowledge distillation (matching teacher logits), and cosine-distance loss (aligning hidden representations).

## Why It Matters

- One of the most successful practical distillation applications
- Proved that significant compression is possible with minimal quality loss
- Still widely used in production where BERT-level quality is needed with lower latency
- The triple loss approach became a template for later distillation work
