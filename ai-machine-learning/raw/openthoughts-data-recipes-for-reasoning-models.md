# OpenThoughts: Data Recipes for Reasoning Models

**Source:** https://arxiv.org/abs/2506.04178
**Authors:** OpenThoughts Team

## Abstract

Reasoning models have made rapid progress on many benchmarks involving math, code, and science. Yet, there are still many open questions about the best training recipes for reasoning since state-of-the-art models often rely on proprietary datasets with little to no public information available. To address this, the goal of the OpenThoughts project is to create open-source datasets for training reasoning models.

## Key Findings

- After initial explorations, the OpenThoughts2-1M dataset led to OpenThinker2-32B, the first model trained on public reasoning data to match DeepSeek-R1-Distill-32B on standard reasoning benchmarks such as AIME and LiveCodeBench
- The dataset was further improved by systematically investigating each step of the data generation pipeline with 1,000+ controlled experiments, which led to OpenThoughts3
- Scaling the pipeline to 1.2M examples and using QwQ-32B as teacher yields the OpenThoughts3-7B model
- OpenThoughts3-7B achieves state-of-the-art results: 53% on AIME 2025, 51% on LiveCodeBench

## Key Contributions
- Open-source datasets for training reasoning models
- Systematic investigation of data generation pipeline with 1,000+ controlled experiments
- Demonstrated that public reasoning data can match proprietary distilled models
- Provides reproducible "data recipes" for the community
- Scaling analysis of training data quality vs quantity
