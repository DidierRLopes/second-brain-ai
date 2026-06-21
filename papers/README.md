# papers/

This folder holds 88 arXiv papers organized into thematic subfolders. The full per-paper listing lives in `../PAPERS_INDEX.md`; this file documents the categorization scheme itself, so a paper can be filed correctly without re-deriving the taxonomy from scratch.

The PDF files themselves are gitignored and not pushed — see the root [README.md](../README.md#why-papers-isnt-pushed) for why. If you've cloned this repo, the folder structure and index are here, but you'll need to fetch the actual PDFs from arXiv yourself (the index links straight to each one).

## Categorization

Eight numbered top-level categories, each with topic-specific subfolders. Numbering is fixed — don't renumber existing categories when adding a new one (append `09-...` instead).

```
papers/
├── 01-models/                Named model releases & full technical reports
│   ├── gpt-deepseek-v2-v3/     GPT and DeepSeek-V2/V3/R1
│   ├── llama-qwen-gemma/       Llama, Qwen (incl. Qwen-VLA), Gemma, OLMo
│   └── small-efficient/        Sub-3B "small/efficient" releases (MiniCPM, SmolLM2)
├── 02-architecture/          Architecture-level techniques, not full model releases
│   ├── transformers/           Foundational attention-head variants (MQA, GQA)
│   ├── attention-variants/     Positional encoding & attention mechanisms (RoPE, FlashAttention)
│   └── alternatives/           Non-transformer / hybrid sequence architectures (SSMs)
├── 03-scaling/                Scaling laws & compute-optimal training
│   ├── scaling-laws/           Parameter/data/compute scaling laws (Chinchilla, PaLM)
│   ├── compute-optimal/        Compute-optimal recipes under data or compute constraints
│   ├── sparse-moe/             MoE scaling (Switch, GShard, routed scaling laws)
│   └── training-optimization/  Optimizers, parametrization, pipeline/systems-level training
├── 04-efficiency/             Quantization, long-context, inference speed
│   ├── quantization/           Post-training quantization methods (QuIP, additive quantization)
│   ├── context-extension/      Long-context training/extension (YARN, RULER, Cartridges)
│   └── inference-kernels/      Inference-time kernel/caching speedups (IndexCache)
├── 05-learning/               Preference optimization, RL, reasoning, fine-tuning
│   ├── alignment-preferences/  Preference-optimization objectives (DPO, KTO, ORPO)
│   ├── reinforcement-learning/ Core RL algorithms & training frameworks (GAE, DAPO)
│   ├── reasoning/              Reasoning-specific training, data synthesis, behavior analysis
│   └── fine-tuning/            SFT, distillation, imitation-learning methods
├── 06-data/                   Data curation, datasets, tokenization
│   ├── curation-filtering/     Data filtering/curation pipelines (FineWeb2)
│   ├── datasets/               Dataset release papers (The Pile)
│   └── tokenization/           Tokenizer design & cross-lingual transfer
├── 07-applications/           Applied/agentic systems
│   └── agents-swe/             Coding agents, SWE benchmarks, agent harnesses
├── 08-evaluation/             Benchmarking, training-dynamics analysis, safety
│   ├── benchmarking/            Capability / post-training benchmarks
│   ├── analysis/                Training-dynamics analysis
│   └── safety/                  Safety / misalignment
└── TO-BE-ORGANIZED/           Drop zone for new, unsorted PDFs (never a permanent home —
                                see CLAUDE.md's auto-ingest workflow, which files these
                                automatically into the structure above)
```

When filing a paper (new or from `TO-BE-ORGANIZED/`): pick the most specific subfolder by topical fit, checking sibling files in candidate subfolders first rather than guessing from the category name alone — several boundaries are easy to blur (e.g. a paper introducing both a new optimizer *and* a scaling law belongs with whichever is its main contribution). Rename to `<Title> - <arXiv ID>.pdf` and add the entry to `../PAPERS_INDEX.md` (the category section plus the Quick Reference count).
