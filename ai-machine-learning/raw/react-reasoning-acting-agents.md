# ReAct: Synergizing Reasoning and Acting in Language Models

**Source:** https://arxiv.org/abs/2210.03629
**Authors:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
**Published:** 2022

## Key Concepts

Core paper introducing the ReAct framework that interleaves reasoning traces and action generation in LLMs. Shows how agents can dynamically reason while acting on external environments (like Wikipedia APIs), overcoming hallucination issues in pure chain-of-thought reasoning through grounding in external information.

## Why It Matters

- Foundation for modern LLM agent architectures
- Showed that reasoning + acting together outperforms either alone
- Reasoning traces provide interpretability and error diagnosis
- Actions ground the model in real information, reducing hallucination
- Directly influenced LangChain, AutoGPT, and other agent frameworks
