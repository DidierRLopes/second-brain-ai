# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

**Source:** https://arxiv.org/abs/2306.00978
**Authors:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (MIT)
**Published:** 2023 (MLSys 2024 Best Paper Award)

## Key Concepts

Hardware-efficient quantization approach that protects salient weights based on activation statistics rather than weight magnitude. Achieves 4x memory reduction without requiring backpropagation or reconstruction.

## Key Insights

- Not all weights are equally important — importance correlates with activation magnitude
- Protecting just 1% of salient channels preserves model quality
- Uses per-channel scaling to reduce quantization error of important weights
- More hardware-friendly than mixed-precision approaches
- MLSys 2024 Best Paper Award winner
