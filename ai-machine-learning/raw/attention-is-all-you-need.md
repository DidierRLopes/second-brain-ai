# Attention Is All You Need

**Source:** https://arxiv.org/abs/1706.03762
**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Łukasz Kaiser, Illia Polosukhin
**Published:** 2017
**Citations:** 173,000+

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

## Key Contributions

- Introduced the Transformer architecture based solely on attention mechanisms, eliminating recurrence and convolutions
- Introduced scaled dot-product attention and multi-head attention
- Achieved state-of-the-art results on machine translation: 28.4 BLEU on WMT 2014 English-to-German
- The foundational paper for all modern LLMs, BERT, GPT, T5, and beyond
- Introduced positional encodings to inject sequence order information
- Demonstrated that attention alone is sufficient for sequence-to-sequence tasks
