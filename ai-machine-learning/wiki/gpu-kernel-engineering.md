# GPU Kernel Engineering

GPU kernel engineering is the layer where model architecture meets the hardware memory hierarchy. It connects [[attention-variants]], [[training-ops]], [[inference-optimization]], [[hybrid-architectures]], and [[long-context-training]]: the algorithm may be mathematically simple, but performance depends on HBM traffic, SRAM reuse, tensor-core scheduling, dispatch overhead, dynamic shapes, and whether the operation maps cleanly to hardware.

## TPU and GPU Architecture (Hardware Primer)

From "How to Scale Your Model" (Austin et al., 2025). Source: https://jax-ml.github.io/scaling-book/training

### TPU Architecture

A single TPU chip has three main components:

| Component | Description | Notes |
|---|---|---|
| **MXU** (Matrix Multiply Unit) | Systolic array; bulk of compute | Executes `bf16[128,128] × bf16[128,128]` tile operations |
| **VPU** (Vector Processing Unit) | Elementwise + reduction ops | Layernorm, softmax, activations run here |
| **VMEM** (Vector Memory) | Fast on-chip scratchpad SRAM | ~32MB; pipelining buffer between HBM and MXU |
| **HBM** | High-bandwidth off-chip memory | 96 GB on v5p, 2 TB/s bandwidth |

**Key insight:** The MXU feeds on data from VMEM, not HBM. Loading from HBM is the bottleneck for memory-bound ops. High arithmetic intensity ops (large matmuls) keep the MXU busy while HBM streams in the next tile.

### TPU Specs

| Chip | FLOPs/s (bf16) | HBM | ICI Bandwidth |
|---|---|---|---|
| TPUv5p | 4.59×10¹⁴ | 96 GB | 90 GB/s bidirectional |
| TPUv5e | 1.97×10¹⁴ | 16 GB | lower |
| TPUv6e | 9.20×10¹⁴ | 32 GB | higher |

**ICI (Inter-Chip Interconnect):** Dense, low-latency links between chips within a pod. Arithmetic intensity = FLOPs/ICI_bandwidth = 4.59e14 / 1.8e11 = **2550** (TPUv5p). Any operation with arithmetic intensity > 2550 can overlap communication with compute.

**DCN (Data Center Network):** Cross-pod interconnect. Only ~6.25 GB/s per TPU — 14× lower than ICI. This is the bottleneck for scaling beyond a single pod.

### GPU Architecture (vs TPU)

NVIDIA GPUs use a different hierarchy:

| Component | GPU equivalent | Key difference |
|---|---|---|
| MXU | Tensor Cores | Multiple SM warp-level execution |
| VPU | CUDA cores / SFUs | More flexible but less efficient |
| VMEM | Shared memory / L1 cache | Smaller per-SM; managed by programmer |
| HBM | HBM | Similar capacity; H100 = 80GB/3.35 TB/s |

**GPU interconnect:**
- NVLink (intra-node): 900 GB/s bidirectional on H100
- NVSwitch (intra-rack): all-to-all within a node
- InfiniBand (inter-node): 400 Gb/s — much lower than ICI cross-pod

**Key GPU differences:**
- Higher raw FLOPs (H100: 9.89e14 bf16 FLOPs/s) but NVLink bandwidth limits inter-GPU parallelism more than ICI limits TPUs
- More flexible memory model (software-managed L1 vs TPU's VMEM) → more programmer responsibility
- Pipeline parallelism more often necessary on GPUs because ICI equivalent is NVLink (intra-node) + InfiniBand (inter-node)

### Roofline on Real Hardware

For a matmul `[B,D]×[D,F]`:
- Bytes read: `2 × (BD + DF)` ≈ `2DF` when D,F >> B
- FLOPs: `2BDF`
- Arithmetic intensity: `BDF / DF = B` (arithmetic intensity ≈ batch size)

**TPUv5p critical batch size:** ~2550 (need batch > 2550 to be compute-bound)
**H100 critical batch size:** ~9.89e14 / (2 × 3.35e12) ≈ **~150** (H100 has higher BW per FLOP)

This is why large matmuls on TPUs need much larger batches to saturate compute than on H100s.

## The Memory Hierarchy Is The Model Constraint

FlashAttention made the core lesson visible: attention is often memory-bandwidth bound, not pure FLOP bound. The winning move is to avoid materializing the full attention matrix in HBM and instead tile the computation through fast on-chip memory.

[FlashAttention-4 (2603.05451)](../../papers/02-architecture/attention-variants/FlashAttention-4: Algorithm and Kernel Pipelining Co-Design - 2603.05451.pdf) shows how hardware-specific this has become. On Blackwell, tensor-core throughput improved faster than shared-memory bandwidth and exponential units. The bottleneck shifted toward shared-memory traffic and softmax exponentials, so the kernel had to co-design around asynchronous MMA, polynomial exponential emulation, conditional softmax rescaling, and tensor memory.

The practical point: "use FlashAttention" is no longer a single technique. It is a moving family of hardware-specific kernels.

## Kernel Abstractions

[ThunderKittens (2410.20399)](../../papers/03-scaling/training-optimization/ThunderKittens: Simple, Fast, and Adorable AI Kernels - 2410.20399.pdf) argues that the right abstractions can make fast kernels maintainable. Its abstractions map to GPU levels:

- warp-level 16x16 matrix tiles and PyTorch-like operations;
- thread-block templates for overlapping asynchronous work;
- grid-level scheduling to hide launch and memory costs.

The result matters because production training stacks cannot rely only on heroic one-off kernels. When architectures change quickly, kernel DSLs and reusable templates become part of research velocity.

## Long Context And MoE Stress Kernels Differently

Long context stresses attention and KV memory. MoE stresses dispatch, all-to-all communication, and small GEMMs. [Scalable Training of MoE Models with Megatron Core (2603.07685)](../../papers/03-scaling/training-optimization/Scalable Training of Mixture-of-Experts Models with Efficient Sparsity - 2603.07685.pdf) is the systems-side complement: grouped GEMM, fused router/permutation kernels, optimized dispatchers, communication overlap, CUDA Graphs, FP8/NVFP4 support, and parallel folding all exist because sparse activation creates work that does not look like one big dense matmul.

This creates a useful architecture heuristic:

| Architecture choice | Kernel/system pressure |
|---|---|
| Full attention at long context | Attention tiling, KV memory, softmax bottlenecks |
| Sliding-window or chunked attention | Mask/layout complexity, less generic fused-kernel support |
| MQA/GQA/MLA | KV cache savings, projection/layout complexity |
| MoE | All-to-all dispatch, grouped GEMM, expert imbalance |
| State-space or hybrid layers | Scan/recurrence kernels, linear attention kernels |

## Inference Kernels And Model Decomposition

[[inference-optimization]] is not only about running one LLM faster. [Interfaze (2602.04101)](../../papers/04-efficiency/inference-kernels/Interfaze: The Future of AI is Built on Task-Specific Small Models - 2602.04101.pdf) argues for decomposing systems into task-specific smaller models plus a larger language model. That changes the kernel problem: OCR, detection, retrieval, ranking, ASR, and LLM decoding have different performance profiles. A fast system may come from routing around the LLM, not just optimizing the LLM kernel.

## Related Topics

- [[attention-variants]] - FlashAttention, GQA, MQA, MLA, local/global attention
- [[training-ops]] - throughput, dataloaders, MoE systems, pipeline RL
- [[inference-optimization]] - serving, batching, speculative decoding, task-specific models
- [[hybrid-architectures]] - Mamba/state-space kernels and transformer hybrids
- [[long-context-training]] - long-window attention and KV constraints
- [[mixture-of-experts]] - sparse dispatch, grouped GEMM, expert parallelism

## Sources

- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design (2603.05451)](../../papers/02-architecture/attention-variants/FlashAttention-4: Algorithm and Kernel Pipelining Co-Design - 2603.05451.pdf)
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels (2410.20399)](../../papers/03-scaling/training-optimization/ThunderKittens: Simple, Fast, and Adorable AI Kernels - 2410.20399.pdf)
- [Scalable Training of Mixture-of-Experts Models with Megatron Core (2603.07685)](../../papers/03-scaling/training-optimization/Scalable Training of Mixture-of-Experts Models with Efficient Sparsity - 2603.07685.pdf)
- [Interfaze: The Future of AI is Built on Task-Specific Small Models (2602.04101)](../../papers/04-efficiency/inference-kernels/Interfaze: The Future of AI is Built on Task-Specific Small Models - 2602.04101.pdf)
