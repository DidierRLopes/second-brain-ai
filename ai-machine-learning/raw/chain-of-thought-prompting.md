# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**Source:** https://arxiv.org/abs/2201.11903
**Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (Google Research)
**Published:** 2022

## Key Concepts

Foundational paper showing that prompting models to generate intermediate reasoning steps ("chain of thought") dramatically improves accuracy on multi-step reasoning problems. Simply adding "Let's think step by step" or providing few-shot examples with reasoning traces unlocks latent reasoning ability.

## Key Findings

- Reasoning ability is emergent — appears at ~100B parameter scale
- Works across math, commonsense, and symbolic reasoning tasks
- Zero-shot CoT ("Let's think step by step") surprisingly effective
- Few-shot CoT with exemplars even more powerful
- Spawned entire field: Tree of Thoughts, Self-Consistency, Least-to-Most, etc.
