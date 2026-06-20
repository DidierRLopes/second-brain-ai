# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Source:** https://arxiv.org/abs/2205.14135
**Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
**Published:** 2022

## Key Concepts

FlashAttention introduces an IO-aware tiling algorithm that reduces memory reads/writes between GPU HBM and SRAM. Rather than materializing the full attention matrix, it computes attention in tiles, dramatically reducing memory usage from O(N²) to O(N).

## Key Results

- 15% end-to-end speedup on BERT-large training
- 3× speedup on GPT-2 (sequence length 1K)
- 2.4× speedup on long-range tasks
- Enables much longer context lengths without running out of memory
- FlashAttention-2 and FlashAttention-3 offer further improvements
- Now standard in virtually all modern LLM training and inference frameworks

## Why It Matters

FlashAttention solved one of the key bottlenecks in transformer scaling — the quadratic memory cost of attention. By making attention IO-aware rather than compute-aware, it enabled the long-context revolution (100K+ token models).
