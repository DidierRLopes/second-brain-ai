# Toolformer: Language Models Can Teach Themselves to Use Tools

**Source:** https://arxiv.org/abs/2302.04761
**Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (Meta AI)
**Published:** 2023

## Key Concepts

Demonstrates a self-supervised approach where LLMs learn to call external APIs without human labeling. The model decides when and how to use tools (calculators, search engines, translation systems, Q&A systems) by learning from its own generated examples.

## Key Properties

- No human annotation required for tool use training
- Model learns when NOT to use tools (only uses them when beneficial)
- Covers: calculator, Q&A, search, translation, calendar APIs
- Zero-shot performance competitive with much larger models
- Foundational for function calling in modern LLMs (GPT-4, Claude, etc.)
