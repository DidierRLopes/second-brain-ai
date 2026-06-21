# Long-Context Training

Long-context training is the cross-cutting recipe for making a model actually use more tokens, not merely advertise a larger context window. It sits between [[positional-encodings]], [[attention-variants]], [[kv-cache]], [[data-curation-mixtures]], [[training-ops]], and [[llm-evaluation]]: extend RoPE or switch to hybrid position schemes, train on the right long/short mixture, prevent packed-document leakage, keep KV cache affordable, and verify effective context length on real tasks.

## The Practical Recipe

The most reliable long-context path is staged continuation from a shorter model:

1. Start from a strong short-context checkpoint.
2. Extend positional behavior with RoPE base scaling, YaRN, or RNoPE.
3. Continue training at the target or above-target context length.
4. Mix long documents with high-quality short data rather than training only on long data.
5. Use document masking when packing unrelated documents.
6. Evaluate effective context length, not claimed context length.

[Effective Long-Context Scaling of Foundation Models / Llama Long (2309.16039)](../../papers/04-efficiency/context-extension/Effective Long-Context Scaling of Foundation Models - 2309.16039.pdf) is the clean case for continuation: bumping Llama 2's RoPE base from 10,000 to 500,000 and continuing at 32k matched from-scratch long-context training while saving about 40% FLOPs. Its most useful negative result is that long-text data was not the core ingredient. Quality mattered more than length distribution.

[How to Train Long-Context Language Models (Effectively) / ProLong (2410.02660)](<../../papers/04-efficiency/context-extension/How to Train Long-Context Language Models (Effectively) - 2410.02660.pdf>) gives the more detailed recipe. Code repositories plus books were the best long-data sources, a 60/40 long/short mix beat long-only training, and training longer than the evaluation length improved 64k results. It also found that short-context instruction data was enough for SFT, while synthetic long-context SFT did not reliably help.

## Position Encoding Is Necessary But Not Sufficient

RoPE extension methods solve the out-of-distribution angle problem, but they do not guarantee retrieval or reasoning over long spans. [YaRN (2309.00071)](../../papers/04-efficiency/context-extension/YARN: Efficient Context Window Extension of Large Language Models - 2309.00071.pdf) is the production-grade RoPE extension: NTK-by-parts interpolation preserves high-frequency relative-position dimensions while stretching dimensions that encode absolute position, and attention-temperature scaling stabilizes softmax behavior at longer contexts.

RNoPE changes the architecture-level division of labor. [Rope to Nope and Back Again (2501.18795)](../../papers/02-architecture/attention-variants/Rope to Nope and Back Again: A New Hybrid Position Encoding for Efficient Context Scaling - 2501.18795.pdf) showed that NoPE layers specialize in long-range retrieval while RoPE layers retain local aggregation. The practical pattern is RNoPE plus sliding-window attention on RoPE layers, not simply "increase the RoPE base."

## Attention Pattern And KV Cache Constraints

Long context is expensive because attention and KV cache scale with sequence length. [[attention-variants]] covers the architectural levers: document masking, sliding-window attention, chunked attention, DCA, and interleaved local/global attention. [[kv-cache]] covers the serving-side constraint: a model can train at 128k and still become impractical if per-request KV memory dominates.

The useful mental model:

| Constraint | Typical tool | Failure if ignored |
|---|---|---|
| Positional extrapolation | RoPE base scaling, YaRN, RNoPE | Good short-context model fails beyond training length |
| Quadratic attention cost | SWA, chunking, DCA, local/global interleaving | Training and inference cost explode |
| KV cache memory | GQA, MQA, MLA, PagedAttention, KV compression | Claimed context is too expensive to serve |
| Packed-document leakage | Document masking | Model learns cross-document artifacts |
| Real retrieval ability | RULER, HELMET, task-specific evals | Long window exists but is not usable |

## Effective Context Must Be Measured

[RULER (2404.06654)](../../papers/04-efficiency/context-extension/RULER: What's the Real Context Size of Your Long-Context Language Models - 2404.06654.pdf) is the warning label for this whole topic: many models claiming 32k+ windows failed well before their advertised length on retrieval, tracing, aggregation, and QA tasks. Effective context is task- and model-dependent.

The operational lesson for [[llm-evaluation]] is to report effective length separately from maximum supported length. The operational lesson for [[frontier-training-playbook]] is to treat context extension as its own training stage, not as a config flag.

## Related Topics

- [[positional-encodings]] - RoPE, YaRN, NoPE, RNoPE, and extrapolation behavior
- [[attention-variants]] - document masking and long-context attention patterns
- [[kv-cache]] - serving memory, GQA/MQA/MLA, PagedAttention, compression
- [[data-curation-mixtures]] - long/short mixture design and document packing
- [[training-ops]] - throughput and storage issues at long sequence length
- [[llm-evaluation]] - effective-context evaluation and open-world trace analysis

## Sources

- [Effective Long-Context Scaling of Foundation Models / Llama Long (2309.16039)](../../papers/04-efficiency/context-extension/Effective Long-Context Scaling of Foundation Models - 2309.16039.pdf)
- [How to Train Long-Context Language Models (Effectively) / ProLong (2410.02660)](<../../papers/04-efficiency/context-extension/How to Train Long-Context Language Models (Effectively) - 2410.02660.pdf>)
- [RULER: What's the Real Context Size of Your Long-Context Language Models (2404.06654)](../../papers/04-efficiency/context-extension/RULER: What's the Real Context Size of Your Long-Context Language Models - 2404.06654.pdf)
- [YaRN: Efficient Context Window Extension of Large Language Models (2309.00071)](../../papers/04-efficiency/context-extension/YARN: Efficient Context Window Extension of Large Language Models - 2309.00071.pdf)
- [Rope to Nope and Back Again (2501.18795)](../../papers/02-architecture/attention-variants/Rope to Nope and Back Again: A New Hybrid Position Encoding for Efficient Context Scaling - 2501.18795.pdf)
- [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism (2401.02954)](../../papers/04-efficiency/context-extension/DeepSeek LLM: Scaling Open-Source Language Models with Longtermism - 2401.02954.pdf)
