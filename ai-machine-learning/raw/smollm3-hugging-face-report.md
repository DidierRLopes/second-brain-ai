# SmolLM3: smol, multilingual, long-context reasoner

**Source URL:** https://huggingface.co/blog/smollm3
**Fetch date:** 2026-05-25
**Published:** July 8, 2025
**Authors (HF team):** Elie Bakouch, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Lewis Tunstall, Carlos Miguel Patino, Edward Beeching, Aymeric Roucher, Aksel Joonas Reedi, Quentin Gallouedec, Kashif Rasul, Nathan Habib, Clementine Fourrier, Hynek Kydlicek, Guilherme Penedo, Hugo Larcher, Mathieu Morlon, Vaibhav Srivastav, Joshua Lochner, Xuan-Son Nguyen, Colin Raffel, Leandro von Werra, Thomas Wolf

---

## Headline Summary

- **3B model trained on 11T tokens** (11.2T exactly), SoTA at the 3B scale and competitive with 4B models (Qwen3-4B, Gemma3-4B).
- **Instruct model with dual-mode reasoning,** supporting `/think` / `/no_think` modes.
- **Multilingual support for 6 languages:** English, French, Spanish, German, Italian, Portuguese.
- **Long context up to 128k** using NoPE and YaRN extrapolation.
- Outperforms Llama-3.2-3B and Qwen2.5-3B, competitive with 4B alternatives.

Model artifacts:
- Base model: https://hf.co/HuggingFaceTB/SmolLM3-3B-Base
- Instruct + reasoning: https://hf.co/HuggingFaceTB/SmolLM3-3B

---

## Pretraining

### Architecture and training details

SmolLM3 follows a **transformer decoder architecture with tied embedding** similar to SmolLM2, building on the Llama architecture with modifications optimized for efficiency and long context.

**Grouped Query Attention (GQA):**
- Replaced multi-head attention with grouped-query attention using **4 groups**.
- Ablations on a 3B model trained with 100B tokens from FineWeb-Edu showed GQA matches multi-head attention performance while significantly reducing KV cache size at inference.

**NoPE (No Positional Encoding, hybrid):**
- Implementation of "RoPE to NoRoPE and Back Again: A New Hybrid Attention Strategy" (Yang et al., 2025, paper 2501.18795).
- **Selectively removes rotary position embeddings from every 4th layer.**
- Improves long context performance without affecting short context capabilities (confirmed via ablations).

**Intra-Document Masking:**
- Following "Analysing The Impact of Sequence Composition on Language Model Pre-Training" (paper 2402.13991).
- During training, attention masking ensures tokens from different documents in the same training sequence don't attend to each other.
- Similar to Llama 3 - helps with faster and more stable long context training while maintaining short context performance.

**Training Stability:**
- Following OLMo 2, **weight decay is removed from embedding layers** to improve training stability.
- Contributed to more stable training dynamics; embedding norms naturally stabilized at healthier values during training without impacting overall performance.

All changes were validated through ablations using the same 3B architecture trained on 100B tokens from FineWeb-Edu.

### Training Configuration

- **Global batch size:** 2.36M tokens
- **Sequence length:** 4096
- **Learning rate:** 2e-4
- **Optimizer:** AdamW (beta1: 0.9, beta2: 0.95)
- **Weight decay:** 0.1
- **Gradient clipping:** 1
- **Scheduler:** WSD (Warmup-Stable-Decay) with 2000 warmup steps, linear decay to 0 in the final 10% of training steps.

**Tooling:**
- `nanotron` framework for training
- `datatrove` for data processing
- `lighteval` for evaluation

**Compute:** 384 H100 GPUs for 24 days.

**Distributed setup (from comments / author confirmation):**
- Tensor Parallelism (TP) of 2, Data Parallelism (DP) of 4 on each 8-GPU node.
- Model does not fit per GPU for training without ZeRO/activation recomputation, and TP=2 did not impact throughput much.

### Data Mixture and Training Stages

Total: **11.2T tokens** in a three-stage training strategy mixing web, math, and code with evolving proportions. Mixture and ratios determined via extensive ablations on 3B models trained on 50B-100B tokens.

**Stage 1: Stable phase (0T -> 8T tokens)** - foundation stage:
- **Web: 85%** (12% multilingual) - FineWeb-Edu, DCLM, FineWeb2, FineWeb2-HQ
- **Code: 12%** - The Stack v2 (16 programming languages), StarCoder2 pull requests, Jupyter and Kaggle notebooks, GitHub issues, StackExchange
- **Math: 3%** - FineMath3+ and InfiWebMath3+

**Stage 2: Stable phase (8T -> 10T tokens)** - introduces higher-quality math/code:
- **Web: 75%** (12% Multilingual)
- **Code: 15%** - adding Stack-Edu
- **Math: 10%** - introducing FineMath4+, InfiWebMath4+, MegaMath (Qwen Q&A, Pro synthetic rewrites, text-code interleaved blocks)

**Stage 3: Decay Phase (10T -> 11.1T tokens)** - further upsamples math and code:
- **Web: 63%** (12% Multilingual)
- **Code: 24%** - upsampling of high-quality code data
- **Math: 13%** - upsampling math data and introducing instruction/reasoning datasets such as OpenMathReasoning

Nanotron training configs with exact data weights: https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs (training logs and intermediate checkpoints to be shared).

---

## Mid-Training

Mid-training = long-context adaptation + reasoning adaptation. Shorter than main pretraining but still general, not domain-specific.

### Long Context Extension

- After main pretraining, trained on an **additional 100B tokens** to extend context length.
- **Sequential two-stage extension, 50B tokens each:**
  - Stage A: 4k -> 32k context, **RoPE theta increased to 1.5M**
  - Stage B: 32k -> 64k context, **RoPE theta increased to 5M**
- Both stages upsampled math, code, and reasoning data.
- **Ablation finding:** Upsampling specific long-context data such as code repositories, books, and long web pages (beyond the naturally long samples) did NOT further boost performance on RULER and HELMET benchmarks.
- Using NoPE + training on the decay mixture with longer sequences and increased RoPE theta was sufficient for competitive long context performance up to 64k.
- **Following Qwen2.5, YaRN is used to extrapolate beyond the training context length.** Model handles up to 128k at inference (2x extension beyond 64k training length).

### Reasoning Mid-training

- Mid-training stage targets general reasoning capability, not a particular domain (not math-only, not code-only).
- **Dataset: 35B tokens** sourced from:
  - Open Thought's OpenThoughts3-1.2M
  - A subset from NVIDIA's Llama-Nemotron-Post-Training-Dataset-v1.1 (reasoning traces from R1)
- Used **ChatML chat template** with **wrapped packing** to avoid providing too much structure.
- Trained for **4 epochs (~140B tokens)**; this checkpoint feeds subsequent SFT stages.

---

## Post-training

Goal: build a dual-mode (reasoning + non-reasoning) instruction model with fully open recipe, contrasting with proprietary RL pipelines used by DeepSeek R1 / Qwen3-style releases.

Pipeline: mid-training for general reasoning -> SFT with synthetic data generation -> alignment with Anchored Preference Optimization (APO, a DPO variant).

### Building the Chat Template

- Users activate reasoning/non-reasoning via `/think` and `/no_think` flags in the **system prompt**.
- In non-reasoning mode, the model's response is pre-filled with **empty think blocks** (similar to Qwen3) to ensure direct answers without explicit reasoning.
- Supports tool calling with two distinct sections: **XML Tools** and **Python Tools**. Separation found beneficial for accurate interpretation of each tool format.
- Default system message includes metadata: date, knowledge cut-off date, current reasoning mode.
- Users can override system message via `system` role. Metadata section can be excluded via `/system_override` flag.

### Supervised Finetuning

- After reasoning mid-training (140B tokens), SFT incorporates capabilities across both modes for: math, code, general reasoning, instruction following, multilinguality, tool calling.
- Tracked SFT performance across: math, code, general reasoning, instruction following, multilinguality.
- **Challenge:** Scarcity of reasoning-trace datasets in some domains.
- **Solution: Synthetic data generation** - prompted Qwen3-32B in reasoning mode with prompts from existing non-reasoning datasets to fill gaps (multi-turn conversations, multilinguality, everyday conversations).

**Final SFT mixture:**
- **1.8B tokens total**
  - 1B in non-reasoning mode
  - 0.8B in reasoning mode
- **12 non-reasoning datasets + 10 datasets with reasoning traces**
- Trained for **4 epochs (~8B tokens)**
- Used **BFD (best-fit decreasing) packing** with loss masked on user turns and tool-call results.

Datasets released in SmolTalk2 (https://huggingface.co/datasets/HuggingFaceTB/smoltalk2); dataset card flags which were Qwen3-32B-generated.

### Off-policy Alignment with Anchored Preference Optimization (APO)

After SFT:
- **Non-reasoning preferences:** Tulu3 preference dataset (allenai/llama-3.1-tulu-3-8b-preference-mixture)
- **Reasoning preferences:** synthetic preference pairs generated from Qwen3-32B (chosen) and Qwen3-0.6B (rejected)
- Generated complementing thinking-mode preference pairs to ensure full domain coverage.

**APO (paper 2408.06266) vs DPO (paper 2305.18290):**
- DPO reward `r_theta(x, y)` is the log-ratio of sequence probability during training vs the reference model.
- `beta` controls how much the model being optimized can change relative to reference model.
- DPO loss optimizes triplets (prompt x, chosen y_w, rejected y_l).
- **APO objective shown to be more stable;** authors observed higher downstream performance vs DPO in internal ablations.

**Observed issue:** APO improvements across math/science/IF/coding/chat/multilingual came with **degradation on long context benchmarks (RULER).** Traced back to reasoning mid-training stage. Additionally, **APO training data was limited to 24k tokens** since the vast majority of reasoning data fell below that length.

### Model Merging (to recover long context performance)

- Used **MergeKit** library (linear and non-linear methods supported).
- **Two-step recipe:**
  1. Take each APO checkpoint and create a model "soup".
  2. **Linear merge: APO soup (weight 0.9) + mid-training checkpoint with strong long-context performance (weight 0.1).**
- Result: **recovered base model's RULER score on contexts up to 128k tokens.**
- This is the released checkpoint.

---

## Evaluation

### Base Model

- Win-rate plot across **12 benchmarks**: HellaSwag, ARC, Winogrande, CommonsenseQA, MMLU-CF, MMLU Pro CF, PIQA, OpenBookQA, GSM8K, MATH, HumanEval+, MBPP+.
- Consistently outperforms other 3B models; competitive with 4B models including Qwen3-4B and Gemma3-4B.
- First or second place on knowledge/reasoning (HellaSwag, ARC, BoolQ).
- Math/code competitive within the 3B class.
- Long-context: strong on Ruler 64k.
- Multilingual: strong performance across five major European languages (Global MMLU, MLMM HellaSwag, Flores-200, Belebele - tests knowledge, commonsense reasoning, text understanding, translation).

### Dual Instruct / Reasoning Model

**No extended thinking (non-reasoning mode):**
- Outperforms other 3B non-reasoning models (Llama3.2 3B Instruct, Qwen2.5 3B Instruct).
- Significantly outperforms Qwen3 1.7B; approaches 4B model performance at lower cost.

**Extended thinking (reasoning mode):**
- Substantial gains vs non-reasoning counterpart:
  - **AIME 2025: 36.7% vs 9.3%**
  - **LiveCodeBench: 30.0% vs 15.2%**
  - **GPQA Diamond: 41.7% vs 35.7%**
- Qwen3 4B still tops most leaderboards in both modes, but SmolLM3 is competitive within the 3B class, particularly in mathematical reasoning and complex problem-solving.

---

## How to Run Locally

Requires **transformers v4.53.0** or newer; also loadable with latest `vllm` (transformers backend).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "HuggingFaceTB/SmolLM3-3B"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
messages_think = [{"role": "user", "content": "Give me a brief explanation of gravity in simple terms."}]
text = tokenizer.apply_chat_template(messages_think, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

Recommended sampling: `temperature=0.6`, `top_p=0.95`.

**Toggling extended thinking:** include `/think` or `/no_think` in the system prompt. Extended thinking is enabled by default.

**Tool calling:** pass tools via `xml_tools` (standard tool-calling) or `python_tools` (Python-function calls inside `<code>` snippets).

---

## Notes from Author Q&A in Comments

- **Tied embeddings confirmed** - explains why HF model page double-counts parameters for some related Qwen3 1.7B figure.
- **GQA setup is classical GQA** (query heads > KV heads); a figure in the post showing the opposite was a typo per author.
- **Released datasets:** all training data published as **SmolTalk2** (https://huggingface.co/datasets/HuggingFaceTB/smoltalk2). Dataset card identifies Qwen3-32B-generated subsets.

---

## Resources

- Models collection (incl. quantized): https://huggingface.co/collections/HuggingFaceTB/smollm3-686d33c1fdffe8e635317e23
- GitHub repo (pretraining configs + eval code): https://github.com/huggingface/smollm
- HF org: https://huggingface.co/HuggingFaceTB
- Nanotron configs / exact data weights: https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs
- Distributed pre-training reference: https://huggingface.co/spaces/nanotron/ultrascale-playbook

## Papers Referenced

- "RoPE to NoRoPE and Back Again: A New Hybrid Attention Strategy" (Yang et al., 2025) - paper 2501.18795
- "Analysing The Impact of Sequence Composition on Language Model Pre-Training" - paper 2402.13991
- DPO - paper 2305.18290
- APO (Anchored Preference Optimization) - paper 2408.06266
- DeepSeek R1 - arxiv 2501.12948
- Qwen3 - arxiv 2505.09388

## Citation

```bibtex
@misc{bakouch2025smollm3,
  title={{SmolLM3: smol, multilingual, long-context reasoner}},
  author={Bakouch, Elie and Ben Allal, Loubna and Lozhkov, Anton and Tazi, Nouamane and Tunstall, Lewis and Patino, Carlos Miguel and Beeching, Edward and Roucher, Aymeric and Reedi, Aksel Joonas and Gallouedec, Quentin and Rasul, Kashif and Habib, Nathan and Fourrier, Clementine and Kydlicek, Hynek and Penedo, Guilherme and Larcher, Hugo and Morlon, Mathieu and Srivastav, Vaibhav and Lochner, Joshua and Nguyen, Xuan-Son and Raffel, Colin and von Werra, Leandro and Wolf, Thomas},
  year={2025},
  howpublished={\url{https://huggingface.co/blog/smollm3}}
}
```

---

## Note on Source Coverage

Note: several items mentioned in the original task brief (vanishing throughput infrastructure issue, dataloader bug, tensor parallelism seed bug, "RNoPE" name, "IFThink" dataset, SFT mix of 100k examples / 76.1M tokens, DPO learning rate of 1e-6, 60/40 FineWeb-Edu/DCLM split) do not appear in the HuggingFace blog post at https://huggingface.co/blog/smollm3 as fetched on 2026-05-25. Those details are likely from a different/companion source (e.g., a separate engineering writeup or talk). What is captured above represents the full technical content of the public blog post; SFT figures here are the blog's stated 1.8B tokens / 22 datasets / 4 epochs, not 100k examples / 76.1M tokens.
