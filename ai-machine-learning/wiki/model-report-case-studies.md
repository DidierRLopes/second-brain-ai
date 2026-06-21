# Model Report Case Studies

Model report case studies are the named-model anchors for the wiki: GPT-4, DeepSeek, Llama, Qwen, Gemma, Nemotron-H, SmolLM2, MiniCPM, and Kimi K2 each expose a different slice of the training stack. This note is intentionally a map rather than a full rewrite of [[frontier-training-playbook]]: use it to jump from a model name to the design choices it teaches.

## Case Study Map

| Report | Main lesson | Primary wiki nodes |
|---|---|---|
| GPT-4 | Predictable scaling and capability forecasting can be locked before the main run | [[scaling-laws]], [[frontier-training-playbook]], [[llm-evaluation]] |
| DeepSeek-V2 | MLA plus MoE can cut KV cache and training cost while preserving quality | [[attention-variants]], [[kv-cache]], [[mixture-of-experts]] |
| DeepSeek-V3 | Auxiliary-loss-free MoE, MTP, FP8, and low-cost frontier-scale training | [[mixture-of-experts]], [[frontier-training-playbook]], [[training-ops]] |
| DeepSeek-R1 | Reasoning behaviors can emerge from verifier-based RL | [[reasoning-models]], [[alignment-methods]], [[rl-training-systems]] |
| Llama 3 | Dense simplicity, disciplined data filtering, and scaling-law prediction remain strong | [[frontier-training-playbook]], [[data-quality-vs-diversity]], [[scaling-laws]] |
| Qwen3 | Extreme token training, thinking/non-thinking modes, and strong-to-weak distillation | [[frontier-training-playbook]], [[reasoning-models]], [[knowledge-distillation]] |
| Gemma 2 / 3 | Local/global attention, logit soft-capping, distillation, and deployment-minded evaluation | [[attention-variants]], [[training-stability]], [[llm-evaluation]] |
| Nemotron-H | Hybrid Mamba-Transformer models should be judged by accuracy-throughput, not accuracy alone | [[hybrid-architectures]], [[inference-optimization]], [[llm-evaluation]] |
| SmolLM2 / MiniCPM | Small models are a separate optimization regime: overtraining, clean data, WSD, and deployment | [[small-efficient-models]], [[data-quality-vs-diversity]], [[optimizers]] |
| Kimi K2 | Agentic training, high-sparsity MoE, MLA, rubric rewards, and harness design co-evolve | [[llm-agents]], [[agent-harness-engineering]], [[mixture-of-experts]] |
| Apple AFM | Distill-and-prune for on-device models, committee-based rejection sampling (iTeC), Mirror Descent over PPO | [[knowledge-distillation]], [[alignment-methods]], [[frontier-training-playbook]] |

## Patterns Across Reports

Three patterns recur:

- **The model architecture is rarely the whole story.** The reports that matter pair architecture with data, optimizer, systems, eval, and post-training choices.
- **Named models become useful when they teach a transferable design rule.** DeepSeek-V2 teaches MLA/KV cache economics; Llama 3 teaches managing complexity; Qwen3 teaches staged data and distillation; Gemma teaches attention/locality and safety evaluation.
- **Evaluation details are part of the technical contribution.** GPT-4's prediction methodology, Llama's downstream scaling predictions, Nemotron-H's accuracy-throughput framing, and Gemma 3's safety/privacy audits are all model-report lessons, not appendix trivia.

## How To Use This In Obsidian

Use this page as a graph hub for named models. When a model comes up in a paper or podcast note, link to this note first if the question is "what does this model teach?" Link to the deeper technical note if the question is about a mechanism:

- use [[mixture-of-experts]] for routing and load balancing;
- use [[attention-variants]] for GQA, MLA, SWA, and RNoPE;
- use [[reasoning-models]] for DeepSeek-R1/Qwen-style reasoning;
- use [[small-efficient-models]] for 1-3B deployment models;
- use [[llm-evaluation]] for model-report benchmark and safety methodology.

## Related Topics

- [[frontier-training-playbook]] - the main synthesized training recipe
- [[small-efficient-models]] - small-model case studies
- [[llm-evaluation]] - model-report evaluation methodology
- [[long-context-training]] - context-extension case studies
- [[rl-training-systems]] - reasoning and agent RL systems
- [[data-quality-vs-diversity]] - data lessons across reports

## Sources

- [GPT-4 Technical Report (2303.08774)](../../papers/01-models/gpt-deepseek-v2-v3/GPT-4 Technical Report - 2303.08774.pdf)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (2405.04434)](../../papers/01-models/gpt-deepseek-v2-v3/DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model - 2405.04434.pdf)
- [DeepSeek-V3 Technical Report (2412.19437)](https://arxiv.org/abs/2412.19437) — not currently in the repo as a PDF; arXiv link only
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2501.12948)](../../papers/01-models/gpt-deepseek-v2-v3/DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning - 2501.12948.pdf)
- [The Llama 3 Herd of Models (2407.21783)](../../papers/01-models/llama-qwen-gemma/The Llama 3 Herd of Models - 2407.21783.pdf)
- [Qwen3 Technical Report (2505.09388)](../../papers/01-models/llama-qwen-gemma/Qwen3 Technical Report - 2505.09388.pdf)
- [Gemma 2: Improving Open Language Models at a Practical Size (2408.00118)](../../papers/01-models/llama-qwen-gemma/Gemma 2: Improving Open Language Models at a Practical Size - 2408.00118.pdf)
- [Gemma 3 Technical Report (2503.19786)](../../papers/01-models/llama-qwen-gemma/Gemma 3 Technical Report - 2503.19786.pdf)
- [OLMo 2 Furious (2501.00656)](../../papers/01-models/llama-qwen-gemma/OLMo 2 Furious - 2501.00656.pdf)
- [Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models (2504.03624)](../../papers/08-evaluation/benchmarking/Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models - 2504.03624.pdf)
- [SmolLM2: When Smol Goes Big (2502.02737)](../../papers/01-models/small-efficient/SmolLM2: When Smol Goes Big - 2502.02737.pdf)
- [MiniCPM: Unveiling the Potential of Small Language Models (2404.06395)](../../papers/01-models/small-efficient/MiniCPM: Unveiling the Potential of Small Language Models - 2404.06395.pdf)
- [KIMIK2: Open Agentic Intelligence (2507.20534)](../../papers/07-applications/agents-swe/KIMIK2: Open Agentic Intelligence - 2507.20534.pdf)
