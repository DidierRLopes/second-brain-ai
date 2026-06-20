# LLM.int8() and Emergent Features

**Source:** https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/
**Author:** Tim Dettmers
**Published:** August 17, 2022

## Overview

This blog post introduces LLM.int8() and provides an introduction into the emergent features discovered in language models at scale. It is described as a more speculative companion to the paper, exploring curious details about the fascinating properties surrounding the emergent outlier features found in transformers.

## Key Technical Details

### The LLM.int8() Method
- A two-part quantization procedure for transformer models
- Uses vector-wise quantization for most features
- Employs a mixed-precision decomposition scheme that isolates outlier feature dimensions into 16-bit multiplication
- Keeps more than 99.9% of values in 8-bit precision
- Enables inference of ~2x larger transformers with fixed memory budget

### Emergent Features
- Transformer models at scale develop strong outlier activations in specific feature dimensions
- These outliers are related to very specific behavior of attention heads
- The outliers dominate attention and performance
- They emerge at a certain model scale and are not present in smaller models

### Implications
- Standard INT8 quantization breaks at scale because of these outlier features
- Mixed-precision decomposition (separating outliers) solves this problem
- Opens interesting questions about emergent properties in transformers
- Endorsed by Andrej Karpathy as "beautiful work"

## Related Paper
- arXiv: 2208.07339 - "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"
