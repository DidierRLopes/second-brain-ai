# Chain-of-Thought Reasoning

Chain-of-thought (CoT) prompting is the technique of getting language models to generate intermediate reasoning steps before arriving at an answer. Discovered by Wei et al. (2022, Google), it was one of the most impactful findings in LLM capabilities — simply asking a model to "think step by step" dramatically improves accuracy on complex tasks.

## The Core Discovery

When you ask an LLM a math problem directly, it often fails. But if you prompt it to show its work — or include examples with reasoning traces — accuracy jumps dramatically. This works for math, commonsense reasoning, symbolic reasoning, and multi-step logic.

Crucially, CoT is an **emergent ability**: it only works in models above roughly 100B parameters. Smaller models produce incoherent reasoning traces that don't improve answers. This suggests that step-by-step reasoning is a capability that develops at scale.

## Key Variants

### Zero-Shot CoT
Simply append "Let's think step by step" to your prompt. Surprisingly effective and requires no examples. Discovered by Kojima et al. (2022).

### Few-Shot CoT
Provide 3-8 examples with detailed reasoning traces. More powerful than zero-shot but requires careful example selection.

### Self-Consistency
Wang et al. (2022) showed that sampling multiple reasoning paths and taking the **majority vote** on the final answer significantly improves reliability. On GSM8K (math), self-consistency improved CoT by +17.9 percentage points. The intuition: correct reasoning paths tend to converge on the same answer, while errors are more random.

### Tree of Thoughts
Yao et al. (2023, Princeton) generalized CoT from a linear chain to a **tree** of reasoning paths. The model generates multiple candidate next steps, evaluates them, and uses search (BFS/DFS) to explore the most promising branches with backtracking. Achieved 74% on Game of 24 vs 4% for standard CoT.

## From Prompting to Learning

The progression from CoT prompting to [[reasoning-models]] like DeepSeek-R1 and o1 represents a paradigm shift: instead of prompting the model to reason, the model is **trained** to reason through RL. The reasoning is no longer a prompting trick but a learned behavior that emerges from reinforcement learning on outcome rewards.

## Related Topics
- [[llm-agents]] — Agents use CoT-style reasoning for planning and decomposition
- [[reasoning-models]] — Models that internalize reasoning via training
- [[scaling-laws]] — CoT is an emergent ability tied to model scale

## Sources
- Chain-of-Thought Prompting Elicits Reasoning (arxiv:2201.11903)
- Tree of Thoughts (arxiv:2305.10601)
- Self-Consistency (arxiv:2203.11171)
