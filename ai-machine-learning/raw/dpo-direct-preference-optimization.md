# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**Source:** https://arxiv.org/abs/2305.18290
**Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (Stanford)
**Published:** 2023 (NeurIPS 2023 Outstanding Paper Award)

## Key Concepts

Simplifies RLHF by showing that the optimal policy can be derived in closed form from the reward function, eliminating the need for a separate reward model. Uses a direct classification loss on preference pairs to align language models.

## Why It Matters

- Dramatically simpler than PPO-based RLHF (no reward model, no RL training loop)
- More stable training — avoids reward hacking and mode collapse
- Equivalent to RLHF under certain assumptions but much easier to implement
- NeurIPS 2023 Outstanding Paper Award
- Became the dominant alignment method, with variants like IPO, KTO, ORPO
