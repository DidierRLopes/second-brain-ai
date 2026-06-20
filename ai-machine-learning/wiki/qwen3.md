# Qwen3: Think Deeper, Act Faster

Released April 29, 2025 by the Qwen Team. Qwen3 is the latest generation of Alibaba's Qwen model family, introducing hybrid thinking modes (thinking vs non-thinking per turn), a massive pre-training expansion to ~36 trillion tokens across 119 languages, and a four-stage post-training pipeline that merges long-CoT reasoning with fast-response capabilities.

Source: [Qwen3 Blog — qwen.ai](https://qwenlm.github.io/blog/qwen3/) | [Qwen.ai redirect](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd)

---

## Models Released (all Apache 2.0)

### Dense Models

| Model | Layers | Q/KV Heads | Tie Embedding | Context |
|---|---|---|---|---|
| Qwen3-0.6B | 28 | 16/8 | Yes | 32K |
| Qwen3-1.7B | 28 | 16/8 | Yes | 32K |
| Qwen3-4B | 36 | 32/8 | Yes | 32K |
| Qwen3-8B | 36 | 32/8 | No | 128K |
| Qwen3-14B | 40 | 40/8 | No | 128K |
| Qwen3-32B | 64 | 64/8 | No | 128K |

### MoE Models

| Model | Layers | Q/KV Heads | Experts (Total/Active) | Context |
|---|---|---|---|---|
| Qwen3-30B-A3B | 48 | 32/4 | 128/8 | 128K |
| Qwen3-235B-A22B | 94 | 64/4 | 128/8 | 128K |

**Flagship:** Qwen3-235B-A22B achieves competitive results against DeepSeek-R1, o1, o3-mini, Grok-3, and Gemini-2.5-Pro on coding, math, and general capabilities benchmarks.

**Efficiency:** Qwen3-30B-A3B outperforms QwQ-32B with only 10× fewer activated parameters. Qwen3-4B rivals Qwen2.5-72B-Instruct.

**Distribution:** Hugging Face ([collection](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)), ModelScope ([collection](https://modelscope.cn/collections/Qwen3-9743180bdc6b48)), Kaggle ([models](https://www.kaggle.com/models/qwen-lm/qwen-3))

**Demo:** [Hugging Face Space](https://huggingface.co/spaces/Qwen/Qwen3-Demo) | **Chat:** [chat.qwen.ai](https://chat.qwen.ai) | **GitHub:** [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3)

---

## Key Features

### 1. Hybrid Thinking Modes

The defining feature of Qwen3: every model supports both a **Thinking Mode** (extended step-by-step reasoning before answering) and a **Non-Thinking Mode** (fast, near-instant responses). The choice is made per-request, not per-model.

- **Thinking Mode:** ideal for complex problems requiring deeper reasoning; produces visible `<think>...</think>` chain-of-thought blocks before the final answer
- **Non-Thinking Mode:** for simpler questions where speed dominates

The integration of both modes enables **scalable thinking budget control**: Qwen3 shows smooth, monotonically improving performance as reasoning budget increases. Users can configure task-specific budgets to balance cost and quality.

The `</think>` token has ID **151668**.

### 2. Multilingual Support: 119 Languages and Dialects

Full breakdown by family:

| Family | Languages |
|---|---|
| Indo-European | English, French, Portuguese, German, Romanian, Swedish, Danish, Bulgarian, Russian, Czech, Greek, Ukrainian, Spanish, Dutch, Slovak, Croatian, Polish, Lithuanian, Norwegian (Bokmål/Nynorsk), Persian, Slovenian, Gujarati, Latvian, Italian, Occitan, Nepali, Marathi, Belarusian, Serbian, Luxembourgish, Venetian, Assamese, Welsh, Silesian, Asturian, Chhattisgarhi, Awadhi, Maithili, Bhojpuri, Sindhi, Irish, Faroese, Hindi, Punjabi, Bengali, Oriya, Tajik, Eastern Yiddish, Lombard, Ligurian, Sicilian, Friulian, Sardinian, Galician, Catalan, Icelandic, Tosk Albanian, Limburgish, Dari, Afrikaans, Macedonian, Sinhala, Urdu, Magahi, Bosnian, Armenian |
| Sino-Tibetan | Chinese (Simplified, Traditional, Cantonese), Burmese |
| Afro-Asiatic | Arabic (Standard, Najdi, Levantine, Egyptian, Moroccan, Mesopotamian, Ta'izzi-Adeni, Tunisian), Hebrew, Maltese |
| Austronesian | Indonesian, Malay, Tagalog, Cebuano, Javanese, Sundanese, Minangkabau, Balinese, Banjar, Pangasinan, Iloko, Waray |
| Dravidian | Tamil, Telugu, Kannada, Malayalam |
| Turkic | Turkish, North Azerbaijani, Northern Uzbek, Kazakh, Bashkir, Tatar |
| Tai-Kadai | Thai, Lao |
| Uralic | Finnish, Estonian, Hungarian |
| Austroasiatic | Vietnamese, Khmer |
| Other | Japanese, Korean, Georgian, Basque, Haitian, Papiamento, Kabuverdianu, Tok Pisin, Swahili |

### 3. Agentic Capabilities + MCP Support

Qwen3 is optimized for coding and agentic tasks, with strengthened support for MCP (Model Context Protocol). The recommended agentic framework is [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent), which encapsulates tool-calling templates and parsers to reduce coding complexity.

---

## Pre-training

Pre-training data expanded from **18 trillion tokens** (Qwen2.5) to approximately **36 trillion tokens**, covering 119 languages. Data sources include web, PDF-like documents (text extracted with Qwen2.5-VL and quality-improved with Qwen2.5), and synthetic data (Qwen2.5-Math and Qwen2.5-Coder generated textbooks, QA pairs, code snippets).

### Three-Stage Pre-Training

**Stage 1:** >30 trillion tokens at 4K context length. Establishes basic language skills and general world knowledge.

**Stage 2:** Knowledge-intensive data mix augmented (STEM, coding, reasoning tasks). Additional ~5 trillion tokens trained.

**Stage 3:** High-quality long-context data extends context to 32K tokens.

### Base Model Performance

- Dense base models match Qwen2.5 base models one size class larger: Qwen3-1.7B ≈ Qwen2.5-3B, Qwen3-4B ≈ Qwen2.5-7B, Qwen3-8B ≈ Qwen2.5-14B, Qwen3-14B ≈ Qwen2.5-32B, Qwen3-32B ≈ Qwen2.5-72B
- In STEM, coding, and reasoning specifically, Qwen3 dense base models exceed larger Qwen2.5 models
- Qwen3 MoE base models match Qwen2.5 dense base models while using only **10% of active parameters** — significant savings in training and inference cost

---

## Post-Training: Four-Stage Pipeline

Qwen3 post-training produces a hybrid model capable of both step-by-step reasoning and rapid responses.

**Stage 1 — Long CoT Cold Start:**
Fine-tune on diverse long chain-of-thought data covering mathematics, coding, logical reasoning, and STEM. Establishes fundamental reasoning abilities.

**Stage 2 — Reasoning-Based RL:**
Scale up compute for RL training using rule-based rewards. Enhances exploration and exploitation capabilities.

**Stage 3 — Thinking Mode Fusion:**
Integrate non-thinking capabilities into the thinking model by fine-tuning on a mix of long CoT data and standard instruction-tuning data. The instruction data is generated by the Stage 2 enhanced thinking model, ensuring a seamless blend.

**Stage 4 — General RL:**
Apply RL across **more than 20 general-domain tasks** to strengthen general capabilities and correct undesired behaviors. Tasks include instruction following, format following, and agent capabilities.

---

## Deployment and Usage

### Recommended Frameworks

- **Serving at scale:** [SGLang](https://github.com/sgl-project/sglang) (`>=0.4.6.post1`), [vLLM](https://github.com/vllm-project/vllm) (`>=0.8.4`)
- **Local:** [Ollama](https://ollama.ai), [LMStudio](https://lmstudio.ai), [MLX](https://github.com/ml-explore/mlx), [llama.cpp](https://github.com/ggerganov/llama.cpp), [KTransformers](https://github.com/kvcache-ai/ktransformers)

### Hugging Face Transformers (Qwen3-30B-A3B example)

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen3-30B-A3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Your prompt here"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # True = thinking mode, False = non-thinking mode
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# Split thinking vs response content
try:
    index = len(output_ids) - output_ids[::-1].index(151668)  # 151668 = </think> token
except ValueError:
    index = 0
thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
```

### SGLang and vLLM Server

```shell
# SGLang
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --reasoning-parser qwen3

# vLLM
vllm serve Qwen/Qwen3-30B-A3B --enable-reasoning --reasoning-parser deepseek_r1

# Ollama
ollama run qwen3:30b-a3b
```

### Soft Switch Mechanism (Per-Turn Thinking Control)

When `enable_thinking=True`, users can dynamically control thinking mode per turn using `/think` and `/no_think` tokens in prompts or system messages. The model follows the most recent instruction in multi-turn conversations.

### Agentic Usage with Qwen-Agent

```python
from qwen_agent.agents import Assistant
bot = Assistant(
    llm={"model": "qwen3-30b-a3b"},
    function_list=[
        {"mcpServers": {"time": {"command": "uvx", "args": ["mcp-server-time", "--local-timezone=Asia/Shanghai"]}}},
        "code_interpreter"
    ]
)
for response in bot.run(messages=[{"role": "user", "content": "What time is it?"}]):
    ...
```

---

## Future Direction

> "We believe we are transitioning from an era focused on training models to one centered on training agents."

Stated goals for next iterations: scaling data, increasing model size, extending context length, broadening modalities, and **advancing RL with environmental feedback for long-horizon reasoning**.

---

## Related Topics
- [[reasoning-models]] — The long-CoT RL training paradigm Qwen3 extends; DeepSeek-R1 comparison point
- [[mixture-of-experts]] — Qwen3's MoE architecture (128 experts, 8 activated)
- [[alignment-methods]] — The four-stage post-training pipeline; rule-based RL rewards, general RL
- [[frontier-training-playbook]] — Where Qwen3 fits in the broader frontier training recipe landscape
- [[frontier-async-rl]] — Qwen 3.5 is one of the labs listed as using async RL for post-training
- [[inference-optimization]] — SGLang, vLLM, Ollama deployment; KV cache; quantized rollouts
- [[llm-agents]] — Qwen3's agentic improvements and MCP support
- [[claude-prompting-best-practices]] — Hybrid thinking mode parallels Claude's adaptive thinking; compare approaches

## Sources
- [Qwen3: Think Deeper, Act Faster — Qwen Team (April 29, 2025)](https://qwenlm.github.io/blog/qwen3/)
- [GitHub: QwenLM/Qwen3](https://github.com/QwenLM/Qwen3)
- [Hugging Face: Qwen3 collection](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)
- [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
- [Demo Space](https://huggingface.co/spaces/Qwen/Qwen3-Demo)
- [Discord](https://discord.gg/yPEP2vHTu4)
