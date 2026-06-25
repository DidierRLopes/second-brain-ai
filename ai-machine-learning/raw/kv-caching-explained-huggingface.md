# KV Caching Explained: Optimizing Transformer Inference Efficiency

**Source:** https://huggingface.co/blog/not-lain/kv-caching
**Author:** Not Lain (Hugging Face Community Article), with support from Aritra Roy Gosthipaty
**Published:** January 30, 2025

## Overview

A Hugging Face community blog post explaining KV caching from first principles. During autoregressive generation, a model normally repeats the same key/value computations for every previously-seen token at every new decoding step. KV caching remembers those intermediate key (K) and value (V) projections from previous steps instead of recomputing them, so each new step only computes K/V for the newest token. The post frames the core motivation simply: "Instead of recomputing everything from scratch, the model reuses what it has already calculated, making text generation much faster and more efficient."

## Prerequisites Assumed

The post assumes familiarity with the Transformer attention mechanism, autoregressive modeling (how GPT-style models generate sequences), and basic linear algebra (matrix multiplication/transposition). It points to the author's companion post "[Mastering Tensor Dimensions in Transformers](mastering-tensor-dimensions-transformers.md)" for background, and highlights three essentials from it: the attention-weight tensor has shape `[batch, h, Seq_len, Seq_len]`; masked multi-head attention lets each token be represented by itself and all preceding tokens; and generating a new token requires the model to look at all previous tokens and their representations conditioned on their own preceding tokens.

## How Does KV Caching Work? Step-by-Step Process

The post lays out KV caching as an explicit four-step loop:

1. **First Generation**: when the model sees the first input, it calculates and stores its keys and values in the cache.
2. **Next Words**: for each new word, the model retrieves the stored keys and values and adds the new ones instead of starting over.
3. **Efficient Attention Computation**: calculate attention using the cached K and V along with the new Q (query) to compute the output.
4. **Update Input**: add the newly generated token to the input and go back to step 2, repeating until generation finishes.

The post visualizes the cache's growth directly:

```
Token 1: [K1, V1] ➔ Cache: [K1, V1]
Token 2: [K2, V2] ➔ Cache: [K1, K2], [V1, V2]
...
Token n: [Kn, Vn] ➔ Cache: [K1, K2, ..., Kn], [V1, V2, ..., Vn]
```

(The post's illustrative figures use a small head dimension of `d_k = 5` purely for visual clarity, noting the real number is typically much larger.)

## Comparison: KV Caching vs. Standard Inference

The post's comparison table frames the tradeoff along five axes:

| Feature | Standard Inference | KV Caching |
| --- | --- | --- |
| Computation per Word | Repeats the same calculations for every word | Reuses past calculations for faster results |
| Memory Usage | Uses less memory at each step, but memory grows with longer texts | Uses extra memory to store past information, but keeps things efficient |
| Speed | Gets slower as text gets longer because it repeats work | Stays fast even with longer texts by avoiding repeated work |
| Efficiency | High computational cost and slower response times | Faster and more efficient since the model remembers past work |
| Handling Long Texts | Struggles with long texts due to repeated calculations | Perfect for long texts as it remembers past steps |

## Practical Implementation

The post gives a simplified PyTorch pseudocode class showing the cache's update/retrieve contract:

```python
# Pseudocode for KV Caching in PyTorch
class KVCache:
    def __init__(self):
        self.cache = {"key": None, "value": None}

    def update(self, key, value):
        if self.cache["key"] is None:
            self.cache["key"] = key
            self.cache["value"] = value
        else:
            self.cache["key"] = torch.cat([self.cache["key"], key], dim=1)
            self.cache["value"] = torch.cat([self.cache["value"], value], dim=1)

    def get_cache(self):
        return self.cache
```

It then notes that in the `transformers` library this behavior is enabled by default via the `use_cache` parameter, and that multiple caching strategies are selectable via the [`cache_implementation`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.cache_implementation) `GenerationConfig` parameter. The minimal usage example given:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B')
model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-1.7B').cuda()

tokens = tokenizer.encode("The red cat was", return_tensors="pt").cuda()
output = model.generate(
    tokens, max_new_tokens=300, use_cache = True # by default is set to True
)
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
```

## Benchmark: With vs. Without KV Caching

The author benchmarked this exact code (300 new tokens, `HuggingFaceTB/SmolLM2-1.7B`) with and without KV caching on a single **T4 GPU**:

| With KV Caching | Standard Inference | Speedup |
| --- | --- | --- |
| 11.7 s | 1 min 1 s (61 s) | ~5.21× faster |

This single data point is the post's concrete empirical anchor for the more general "stays fast even with longer texts" claim in the comparison table above.

## Conclusion and Framing

The post closes by reiterating that KV caching is "simple but powerful" — it trades extra memory for large generation-time savings, which is "especially useful for long conversations." It explicitly flags this as a memory-for-speed tradeoff rather than a free lunch: caching requires extra memory that grows with sequence length, even though it eliminates redundant compute.

## References Cited by the Post

The post's own further-reading list: "Transformers KV Caching Explained" (Medium, João Lages), "Transformers Key-Value Caching Explained" (neptune.ai), "Mastering LLM Techniques: Inference Optimization" (NVIDIA Developer Blog), and the Hugging Face Transformers documentation page on KV caching in generation strategies.

## Related wiki pages

[[kv-cache]], [[transformer-architecture]], [[inference-optimization]]
