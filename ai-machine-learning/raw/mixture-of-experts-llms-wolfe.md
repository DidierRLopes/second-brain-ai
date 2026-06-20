# Mixture-of-Experts (MoE) LLMs — Deep Dive

**Source:** https://cameronrwolfe.substack.com/p/moe-llms
**Author:** Cameron R. Wolfe, Ph.D.

## Key Concepts

Deep dive into MoE architectures as a transformer modification, explaining sparse routing mechanisms, token-to-expert assignment, and how MoE achieves better quality-to-inference-efficiency trade-offs compared to dense models.

## Topics Covered

- Switch Transformers: simplified routing with one expert per token
- Mixtral 8x7B: practical open-source MoE deployment
- Load balancing and expert utilization challenges
- Sparse vs dense scaling tradeoffs
- How MoE allows massive parameter counts with manageable compute
