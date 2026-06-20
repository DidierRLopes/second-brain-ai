# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**Source:** https://aclanthology.org/N19-1423.pdf
**Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google)
**Published:** 2018

## Key Concepts

BERT is an encoder-only transformer that pre-trains bidirectional representations using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). Unlike unidirectional models like GPT, BERT considers both left and right context at all levels, enabling superior performance on natural language understanding tasks.

## Key Contributions

- Introduced Masked Language Modeling (MLM) pre-training objective
- Bidirectional attention over full input sequence
- Fine-tuning paradigm: pre-train once, fine-tune for many downstream tasks
- Set new SOTA on 11 NLP benchmarks at time of release
- Spawned numerous variants (RoBERTa, ALBERT, DistilBERT, etc.)
