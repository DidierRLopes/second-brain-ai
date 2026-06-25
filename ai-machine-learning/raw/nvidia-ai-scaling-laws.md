# How Scaling Laws Drive Smarter, More Powerful AI (NVIDIA)

**Source:** https://blogs.nvidia.com/blog/ai-scaling-laws/
**Authors:** Kari Briski
**Published:** February 12, 2025
**Filed under:** Scaling laws / AI fundamentals

## Premise

An NVIDIA explainer arguing that AI development now runs on **three distinct, complementary scaling laws** rather than one: pretraining scaling, post-training scaling, and test-time scaling (also called "long thinking"). Each describes a different place to spend compute for predictable capability gains, and the rise of AI reasoning models is framed as the moment test-time scaling became a mainstream third axis alongside the other two.

## Pretraining Scaling

The original scaling law: increasing training-dataset size, model parameter count, and compute together produces predictable improvements in model intelligence and accuracy. The post cites [Training Compute-Optimal Large Language Models / Chinchilla (2203.15556)](../../papers/03-scaling/scaling-laws/Training Compute-Optimal Large Language Models - 2203.15556.pdf) as the underlying result — see `raw/chinchilla-scaling-laws.md` and [[scaling-laws]] for the deep dive on compute-optimal ratios. NVIDIA credits pretraining scaling with driving the architectural innovations of the last few years: billion- and trillion-parameter transformer models, mixture-of-experts architectures, and new distributed training techniques — all consequences of organizations chasing the next unit of pretraining compute.

## Post-Training Scaling

Once a foundation model is pretrained and released, adapting it for specific applications is its own scaling axis. NVIDIA's estimate for the size of this axis: **developing the ecosystem of derivative models for a variety of use cases can take around 30x more compute than pretraining the original foundation model**, because a popular open-source release can spawn hundreds or thousands of downstream derivatives across the developer community.

The named post-training technique menu: fine-tuning (additional training data, either internal datasets or input/output pairs, to specialize a model for a domain), pruning, quantization, distillation, reinforcement learning, and synthetic data augmentation. Two techniques get a more detailed treatment:

- **Distillation** uses a paired large "teacher" model and lightweight "student" model. In the most common variant — **offline distillation** — the student learns to mimic the pretrained teacher's outputs.
- **Reinforcement learning** trains an agent (the model) to make decisions that maximize cumulative reward as it interacts with an environment. **RLHF** (reinforcement learning from human feedback) is the standard form — e.g. a chatbot LLM reinforced by user "thumbs up" reactions. **RLAIF** (reinforcement learning from AI feedback) is a newer variant that substitutes AI-model feedback for human feedback to streamline post-training.
- **Best-of-n sampling** generates multiple outputs from a model and selects the one with the highest reward-model score — improving outputs without touching model parameters, as an alternative to RL fine-tuning.
- **Search methods** explore a range of potential decision paths before selecting a final output, iteratively improving the model's responses.

Synthetic data is called out as a complement to all of the above: AI-generated data augmenting a real-world fine-tuning dataset helps models handle edge cases that are underrepresented or missing in the original data.

## Test-Time Scaling ("Long Thinking")

Test-time scaling happens during inference, not training: instead of generating a one-shot answer, a model allocates extra computational effort to reason through multiple potential responses before settling on the best one. NVIDIA's analogy: answering "what's two plus two" needs no reasoning, but drafting a business plan to grow a company's profit by 10% does — most humans reason through several options first. The post's headline compute estimate: this reasoning process **can take multiple minutes or even hours and can require over 100x the compute of a single traditional inference pass** for a hard query — a query a standard one-shot LLM would be unlikely to answer correctly on the first try regardless.

Named techniques: **chain-of-thought prompting** (decomposing a complex problem into a series of simpler steps), **sampling with majority voting** (generating multiple responses to the same prompt and returning the most frequently recurring answer), and **search** (exploring and evaluating multiple paths through a tree-like structure of possible responses). Post-training's best-of-n sampling is noted as reusable at test time too, for the same "improve without retraining" reason.

The post frames the practical upside across domains: healthcare (predicting disease progression or treatment complications, reasoning through clinical-trial matches), retail/supply-chain (multi-scenario demand forecasting and routing), and general enterprise use (business plans, debugging complex code, logistics optimization for delivery trucks, warehouses, and robotaxis).

As of the post's writing (Feb 2025), this framing had just gone mainstream: OpenAI's o1-mini and o3-mini, DeepSeek-R1, and Google DeepMind's Gemini 2.0 Flash Thinking had all launched within the preceding weeks — all classed as reasoning models that trade extra inference-time compute for accuracy on complex, multistep queries.

## Related wiki pages

[[scaling-laws]], [[reasoning-models]], [[alignment-methods]], [[knowledge-distillation]]
