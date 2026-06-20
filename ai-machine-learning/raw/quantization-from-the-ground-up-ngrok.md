# Quantization from the Ground Up

**Source:** https://ngrok.com/blog/quantization
**Author:** Sam Rose (ngrok)

## Overview

A comprehensive guide to what quantization is, how it works, and how it's used to compress large language models. Quantization can make LLMs 4x smaller and 2x faster — enough to run very capable models on a laptop — while losing only 5-10% accuracy.

## How Quantization Works

Parameters (also called "weights") are the majority of what an LLM is when it's in memory or on disk. Instead of floats, small integers are what get stored and loaded into memory, and when the time comes to use the quantized values, they are dequantized on the fly.

## Key Concepts

### Handling Outliers
Real-world quantization schemes sometimes do extra work to preserve outlier values by not quantizing them at all, or by saving their location and value into a separate table. During dequantization, this table is consulted and the outliers are restored.

### Quality Measurement
The article covers a number of ways quality loss in LLMs can be measured, with each measure having pros and cons. If evaluating quantized models for a critical use-case, nothing beats creating your own benchmark for the specific task.

### Key Takeaways
- Quantization reduces model size by using lower-precision number formats
- Most common approaches: INT8, INT4, and mixed-precision methods
- Trade-off between model size reduction and quality preservation
- Outlier handling is critical for maintaining model quality
- Custom benchmarks recommended for evaluating quantized models in production
