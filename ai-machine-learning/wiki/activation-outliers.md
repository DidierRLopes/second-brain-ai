# Activation Outliers

Activation outliers are extreme values — often 100× larger than typical activations — that emerge in transformer models at scale. They are the central reason why quantizing LLMs is hard, and understanding them connects quantization to fundamental questions about how transformers work.

## The Phenomenon

Tim Dettmers discovered that large transformers (roughly 6B+ parameters) develop **emergent outlier features**: specific hidden dimensions that produce extremely large activation values across all tokens. These outliers don't exist in smaller models — they appear suddenly as models scale, making them an emergent property of scale.

These outlier dimensions are consistent: the same dimensions produce outliers regardless of the input. This makes them predictable but also means they carry critical information — removing or damaging them destroys model quality.

## Super Weights

Yu et al. (2024) discovered an even more extreme version: **super weights** — individual parameters so critical that pruning a single one can increase perplexity by 3 orders of magnitude and reduce accuracy to random guessing. These super weights induce "super activations" — the most extreme outlier activations. They can be identified with a data-free single forward pass.

The positive implication: preserving super activations with high precision during quantization can improve even simple round-to-nearest quantization to competitive state-of-the-art quality.

## Why Do Outliers Exist?

Bondarenko et al. (2023) showed that strong outliers are connected to **attention heads trying to learn a "no-op"** — when an attention head has nothing useful to contribute to a token's representation, it tries to pass the residual through unchanged. The mechanism it uses to do this creates extreme activation values. This is an architectural limitation, not a quantization problem.

## Implications for Quantization

Every successful quantization method must handle outliers:

- **LLM.int8()**: Separates outlier dimensions into FP16, keeps 99.9% in INT8
- **AWQ**: Protects salient channels (those producing outliers) via per-channel scaling
- **GPTQ**: The Hessian naturally assigns more precision to important weights
- **QuIP**: Incoherence processing spreads outlier energy across all dimensions

The general strategy is either (a) separate outliers and handle them at higher precision, or (b) transform the weight matrix so outliers are spread out before quantization.

## Related Topics
- [[quantization-fundamentals]] — The core concepts outliers complicate
- [[quantization-methods]] — How each method handles outliers differently
- [[transformer-architecture]] — Outliers emerge from attention head behavior

## Sources
- The Super Weight in Large Language Models (arxiv:2411.07191)
- LLM.int8() and Emergent Features — Tim Dettmers
- Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing (arxiv:2306.12929)
