# Tokenizers

A tokenizer is the layer between raw text and the model's vocabulary embeddings; choices here propagate through compute, memory, KV cache, and downstream accuracy on numeric and multilingual tasks. Frontier-2026 practice: **use an existing tokenizer (GPT-4's `o200k`, Gemma 3's, Llama 3) unless you're training for low-resource languages or have an unusual data mixture**. Vocab size trades embedding-matrix cost against compression efficiency; large models benefit more from large vocabs because the per-token forward-pass savings (project to QKV, attention, MLP) outweigh the extra embedding parameters, and a smaller token count also means a smaller KV cache.

## Three Considerations

- **Domains.** In math and code, digits and special characters need careful treatment. Most modern tokenizers do **single-digit splitting**, which helps arithmetic patterns more effectively and prevents memorization of numbers. Llama 3 goes further: it encodes integers 1–999 as unique tokens.
- **Supported languages.** An English-trained tokenizer is extremely inefficient on Mandarin or Farsi — far more tokens per word, blowing up both context and KV cache.
- **Target data mixture.** When training from scratch, train on a sample that mirrors the final training mix. Otherwise the tokenizer's frequency statistics will be wrong.

## Vocabulary Size

Larger vocabularies compress text more efficiently, but the embedding matrix grows as `2 × vocab × d_model` (input + output, or `vocab × d_model` if tied). For small LMs, the embedding can be up to 20% of parameters; for large models, it's a small fraction. Compression gains from larger vocabs decrease exponentially.

Rules of thumb:
- **English-only**: ~50K often enough.
- **Multilingual**: >100K (some go past 150K).
- **Large models benefit from large vocabs** because the per-token compression saves more on the forward pass than the extra embedding parameters cost during softmax.
- For memory, larger vocab → fewer tokens → smaller KV cache.

## BPE (Byte-Pair Encoding)

BPE remains the de facto standard. Start with tiny units (characters or bytes), repeatedly merge the most common adjacent pair into a new token. Variants like SentencePiece BPE handle whitespace and Unicode normalization.

[Decoupling the Benefits of Subword Tokenization for Language Model Training via Byte-level Simulation (2604.27263)](../../papers/06-data/tokenization/Decoupling the Benefits of Subword Tokenization for Cross-Lingual Transfer - 2604.27263.pdf) asks why subword tokenizers beat raw byte models by simulating individual subword benefits inside a byte-level pre-training setup. The strongest effects are **sample throughput** (subword compression lets the model process more raw text per FLOP) and **subword boundary information**. End-of-subword boundaries are especially helpful because they leak a limited non-causal prior about upcoming bytes; start boundaries also help, suggesting a structural inductive bias rather than pure compression alone. The paper is a useful caution for tokenizer-free designs: matching the architecture is not enough if the training stream loses both compression and boundary priors.

## Evaluation Metrics

- **Fertility**: average number of tokens needed to encode a word. Lower = more efficient.
- **Proportion of continued words**: percentage of words that get split into multiple pieces. Lower = more efficient.
- **Normalized Sequence Length (NSL)**: average tokens needed to represent a normalized document. Accounts for preprocessing (punctuation, whitespace) unlike raw fertility.

Both metrics avoid the limitations of characters-to-tokens or bytes-to-tokens ratios (word length variability, byte representations).

## Tokenizer Optimization and Trade-offs

[Getting the most out of your tokenizer for pre-training and domain adaptation (2402.01035)](../../papers/06-data/tokenization/Getting the most out of your tokenizer for pre-training and domain adaptation - 2402.01035.pdf) studies compression-performance trade-offs across vocabulary size, pre-tokenization regexes, tokenizer training data, and domain adaptation, with code-generation experiments as the main testbed. Key findings:

- Larger vocabularies compress text more aggressively (fewer tokens per document) but incur embedding matrix cost and slower convergence early in training.
- Larger models can afford larger vocabularies more easily because embedding parameters are a smaller fraction of total parameters, but the optimal vocabulary depends on whether you are optimizing inference time or memory.
- **Domain-specific tokenizer switching** can work during continued pre-training / fine-tuning if the run is long enough. The paper's recommendation is conservative: changing the tokenizer of a pre-trained model becomes viable when fine-tuning on more than roughly 50B tokens, and they validate 500B-token code fine-tuning runs at 1.5B and 7B scale.
- The pre-tokenization regex matters. A GPT-4-style regex gives strong code compression without the performance cost seen when skipping pre-tokenization entirely.

The paper uses **bytes-per-token** and **NSL** (Normalized Sequence Length) to account for preprocessing overhead, revealing that raw compression metrics can be misleading — a tokenizer that handles punctuation and spacing efficiently may outperform one with nominally lower fertility.

## When to Use an Existing Tokenizer

For most projects, an existing tokenizer is enough. Modern reference tokenizers:
- `o200k_harmony` (GPT-4 family, used by gpt-oss-120b)
- Gemma 3 tokenizer
- Llama 3 tokenizer (used by SmolLM3)
- `cl_100k` (OLMo 3)
- `tokenization_kimi` (Kimi-K2)

Train your own only if:
- You target a low-resource language poorly served by existing tokenizers.
- Your data mixture is unusual enough that fertility on your corpus is much worse than alternatives.

## Freeze Vocab Early

A practical reminder from the frontier-training playbook: **choose a tokenizer matched to your target languages and domains, then freeze vocab and special tokens early**. Changing tokenizer mid-project means re-tokenizing the entire dataset and invalidating any ablations done against the previous version.

## Related Topics

- [[embeddings]] — input/output embedding matrices and tied embedding tradeoffs
- [[kv-cache]] — token count directly affects KV cache memory
- [[frontier-training-playbook]] — where tokenizer choice sits in the recipe

## Sources

- Alex Wa, "Frontier model training methodologies" (Jan 31, 2026). See `raw/alex-wa-frontier-model-training-methodologies.md`.
- SmolLM3, gpt-oss-120b, Kimi K2, OLMo 3, Trinity Large reports for specific tokenizer choices.
- [Decoupling the Benefits of Subword Tokenization for Language Model Training via Byte-level Simulation (2604.27263)](../../papers/06-data/tokenization/Decoupling the Benefits of Subword Tokenization for Cross-Lingual Transfer - 2604.27263.pdf) — Byte-level simulations isolating sample throughput and subword-boundary priors.
- [Getting the most out of your tokenizer for pre-training and domain adaptation (2402.01035)](../../papers/06-data/tokenization/Getting the most out of your tokenizer for pre-training and domain adaptation - 2402.01035.pdf) — Compression trade-offs, NSL metric, vocabulary size optimization, pre-tokenization regexes, and long-run domain tokenizer switching.
