# Applied ML Systems

The engineering foundations for training large neural networks efficiently: distributed training, numerical precision, memory management, and profiling.

---

## Floating Point Representation

Modern ML uses several floating point formats, each with different precision/range tradeoffs.

### IEEE 754 Basics

Floating point numbers: `(-1)^s × m × 2^e`

- **Sign** (s): 1 bit
- **Exponent** (e): determines range
- **Mantissa/Significand** (m): determines precision

**FP32 (32-bit float)**: 1 sign, 8 exponent, 23 mantissa bits. ~7 decimal digits of precision. Dynamic range: ~1.2e-38 to 3.4e38. The default in PyTorch.

**FP16 (16-bit float)**: 1 sign, 5 exponent, 10 mantissa bits. Range: ~6e-5 to 65504. Limited range causes overflow/underflow during training. Used in mixed-precision.

**BF16 (Brain Float 16)**: 1 sign, 8 exponent (same as FP32!), 7 mantissa bits. Same range as FP32, less precision. Native on TPUs and A100/H100. Generally preferred over FP16 for training — same range avoids overflow.

**FP8 (E4M3 / E5M2)**: 8-bit float with two variants. Used for inference and increasingly training (H100 support). Requires careful scaling.

**INT8**: 8-bit integer. Used for inference quantisation. Values in [-128, 127] (signed) or [0, 255] (unsigned). Fast compute but requires calibration to determine scale factors.

### Numerical Issues

**Overflow**: value exceeds maximum representable number (→ inf). Common with FP16 and large logits.

**Underflow**: value too small to represent (→ 0). Gradients becoming 0 from underflow in FP16.

**Catastrophic cancellation**: subtracting two nearly-equal large numbers loses significant digits. `(1e8 + 1) - 1e8 = 0.0` in FP32.

**NaN propagation**: NaN (Not a Number) from 0/0 or inf-inf spreads through all downstream computations, causing silent training failures. Always check for NaN in loss.

---

## Mixed Precision Training

Train with FP16/BF16 (faster arithmetic, less memory) while maintaining numerical stability with FP32 master weights.

### AMP (Automatic Mixed Precision)

**Architecture:**
- **FP32 master weights**: stored for accumulation of small gradient updates
- **FP16/BF16 forward/backward passes**: faster compute, less memory for activations
- **FP32 optimizer step**: prevents small gradient updates from underflowing

**Loss scaling** (for FP16): scale loss up by a large constant (e.g., 2^16) before backward pass. This scales gradients up by the same factor, preventing underflow. Unscale before optimizer step. BF16 doesn't need loss scaling due to wider exponent range.

**Dynamic loss scaling (DLS)**: adaptively adjust the loss scale. Start high, halve when overflow (inf/NaN detected), multiply by 2 after N clean steps. This is what `torch.cuda.amp.GradScaler` does automatically.

```python
scaler = GradScaler()
with autocast():
    loss = model(x)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Memory savings**: FP16/BF16 weights are 2× smaller than FP32. Activations (stored for backprop) are also 2×. Total memory reduction: ~2× for model + activations, less for optimizer states (still FP32).

---

## Gradient Checkpointing

**The memory problem**: backprop requires storing all intermediate activations from the forward pass. For a transformer with L layers, sequence length T, batch B, hidden dim d: activation memory = O(L × T × B × d). For LLaMA-70B training: can be tens of GB just for activations.

**Solution**: recompute activations during backward pass instead of storing them.

```
Forward:  Only store checkpoints at K intervals (not all activations)
Backward: When an activation is needed, recompute it from the nearest checkpoint
```

**Memory-compute tradeoff**:
- **No checkpointing**: memory = O(L), compute = O(L)
- **Full checkpointing**: memory = O(1), compute = O(L²/K) [must recompute K layers for each of L backward steps]
- **Selective checkpointing**: checkpoint every √L layers → memory = O(√L), compute = O(L × √L)

**Practical impact**: gradient checkpointing typically increases compute by ~30-40% but reduces activation memory by ~10-20×. Almost universally used in large model training.

**What to checkpoint**: in transformers, typically checkpoint at transformer block boundaries (attention + FFN layer). Some frameworks checkpoint selectively (only FFN, not attention, because attention activations are smaller).

**FlashAttention** (see [[gpu-kernel-engineering]]): recomputes attention activations during backward pass via kernel fusion — achieves the memory savings of checkpointing without extra compute overhead for attention.

---

## Gradient Accumulation

Train with larger effective batch sizes without requiring more GPU memory.

**Method**: run K forward/backward passes without an optimizer step, accumulate gradients, then step.

```python
optimizer.zero_grad()
for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps
    loss.backward()           # accumulate gradients
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()       # update every K steps
        optimizer.zero_grad()
```

Effective batch size = (minibatch size) × (accumulation steps).

**Why needed**: batch size is limited by GPU memory. A100 80GB can fit batch=4 for a large model. To get the stability and convergence benefits of batch=64, accumulate 16 steps.

**Caveats:**
- BatchNorm stats become incorrect (computed per-step, not per-effective-batch). Use SyncBatchNorm or LayerNorm instead.
- Gradient accumulation ≠ large batch exactly (different random seeds per step, not one large batch). Usually equivalent for most purposes.
- Reduces training speed to 1/K of theoretically possible throughput — use FSDP or model parallelism instead when possible.

---

## Gradient Clipping

Prevent **exploding gradients** (see below) by capping the gradient norm.

**Global gradient norm clipping**:

```
if ||g||₂ > max_norm:
    g ← g × (max_norm / ||g||₂)
```

This scales the entire gradient vector down if its L2 norm exceeds the threshold.

**PyTorch**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

**Typical values**: max_norm=1.0 for most transformer training. Some aggressive setups use max_norm=0.3 or lower.

**When it's critical**: large models with MoE architectures, early training before the model stabilises, RL training (gradients can be highly variable). Gradient clipping is one of the most important stability tricks — forgetting it is a common cause of training divergence.

**Connection to MuonClip**: see [[optimizers]] — MuonClip for Kimi K2 extends this to per-attention-head logit clipping, targeting the specific instability mode of attention logit explosion.

---

## Exploding and Vanishing Gradients

**Vanishing gradients**: as gradients are backpropagated through many layers, the product of Jacobians can shrink exponentially. ∂L/∂θᴸ = (∂L/∂θᴺ) × Π_{l=L}^N (∂hˡ/∂hˡ⁻¹). If each factor has spectral radius < 1, the product → 0.

Symptoms: loss stops decreasing, layers close to the input don't learn, gradients near zero for early layers.

Solutions: residual connections (add identity path for gradient highway), LSTM gating, careful weight initialisation, shorter gradient paths.

**Exploding gradients**: the product of Jacobians grows exponentially. Loss becomes NaN.

Symptoms: loss suddenly becomes inf/NaN, parameters jump to large values, model breaks.

Solutions: gradient clipping, careful initialisation (spectral norm < 1 initially), weight decay, gradient norm monitoring, smaller learning rate.

**Residual connections**: the primary solution to both. The skip connection y = x + F(x) ensures the gradient ∂y/∂x = I + ∂F/∂x — the identity term provides a "gradient highway" that doesn't vanish, regardless of how small ∂F/∂x is.

**Gradient norm monitoring**: log `||∇L||` at each step. Sudden spikes indicate exploding gradients; persistent near-zero values indicate vanishing.

---

## DDP (Distributed Data Parallel)

**Goal**: train one model on N GPUs in parallel, each processing a different batch.

**Architecture:**
1. Each GPU gets the same model replica (identical weights)
2. Each GPU processes a different mini-batch
3. After backward pass, **all-reduce** gradients across all GPUs (sum/average)
4. Each GPU applies the same gradient update → all replicas stay in sync

**All-Reduce**: communication primitive that computes the sum (or average) of tensors across all GPUs and distributes the result to all. Ring all-reduce is most efficient: GPUs form a ring, each sends and receives in overlapping rounds. Time: O(d/N) per GPU (d = gradient size).

**When DDP is enough**: the model fits on a single GPU, you just want to process more data in parallel. Scale by number of examples per step, not by model size.

**Gradient compression**: reduce all-reduce communication by compressing gradients (e.g., 1-bit SGD, PowerSGD). Trades accuracy for communication efficiency.

---

## Communication Primitives

The building blocks of distributed training communication.

**All-Reduce**: every device contributes a tensor; all devices receive the sum/average. Used in DDP for gradient synchronisation.

**All-Gather**: every device has a shard; all devices collect all shards. Used in FSDP to reconstruct full model parameters before forward/backward.

**Reduce-Scatter**: every device contributes a tensor; each device receives one shard of the sum. Used in FSDP to shard the gradient results.

**Broadcast**: one device sends to all others. Used for initial model weight distribution.

**All-to-All**: each device sends different data to each other device. Used in Mixture of Experts routing — tokens need to be redistributed to their assigned expert GPUs.

**Topology matters**: the cost of communication depends on whether devices are:
- On the same GPU (NVLink / NVSwitch): ~600 GB/s bidirectional
- On the same node, different GPUs: NVLink or PCIe
- Across nodes: InfiniBand (~200 Gb/s per link) or Ethernet (slower)

FSDP and tensor parallelism work best within a node (NVLink bandwidth). Pipeline parallelism works across nodes (less communication). See [[how-to-scale-your-model]] for the full parallelism breakdown.

---

## Mixed Precision and Numerical Precision Tricks

Beyond AMP, several numerical tricks prevent training failures.

**Log-sum-exp trick**: compute log(Σᵢ exp(xᵢ)) stably:

```
log(Σᵢ exp(xᵢ)) = m + log(Σᵢ exp(xᵢ - m)), where m = max(xᵢ)
```

Subtracting max prevents overflow. This is exactly what numerically stable softmax does.

**Gradient norm logging**: track training health. Plot gradient norm per layer. Instability shows as sudden spikes.

**Z-loss / logit softcapping**: see [[training-stability]] — prevents logit explosion in MoE architectures and attention.

**FP32 upcast for critical ops**: even in BF16 training, some operations are computed in FP32: loss computation, softmax in attention, LayerNorm reduction, optimizer states. Most frameworks handle this automatically.

**Stochastic rounding**: instead of always rounding to nearest (deterministic), round up or down with probability proportional to the fractional part. Reduces systematic bias from repeated rounding.

**Symlog (DreamerV3 trick)**: for regression with wildly varying target scales, apply symlog(x) = sign(x) log(|x|+1) to targets. Makes the model robust to targets spanning many orders of magnitude without manual normalisation.

---

## Profiling

Identifying bottlenecks in model training.

**CUDA profiling tools:**
- `torch.profiler`: records CPU and GPU events, generates flame graphs
- `nvprof / Nsight Systems`: NVIDIA's GPU profiler
- `cudnn.benchmark = True`: automatically finds fastest algorithms for fixed-size inputs

**Key metrics to profile:**
- **GPU utilisation**: should be ~90%+. Low utilisation = bottleneck elsewhere (data loading, CPU, communication).
- **Memory bandwidth utilisation**: is the GPU memory bandwidth saturated?
- **Compute vs memory bound**: is the bottleneck FLOP throughput or memory bandwidth?
- **Communication overhead**: what % of time is spent on all-reduce / all-gather?

**Roofline model**: given the ops/byte ratio of a kernel, the arithmetic intensity, and the hardware's peak FLOP/s and bandwidth, predict the achievable throughput. Kernels below the roofline are memory-bound; kernels at the compute roof are compute-bound.

**Common bottlenecks:**
- **Data loading**: always prefetch data (DataLoader num_workers, pin_memory=True)
- **Python overhead**: eliminate Python loops in the critical path; use compiled kernels
- **Memory fragmentation**: use memory pooling, avoid frequent allocate/free
- **Communication/compute overlap**: use async collectives (`async_op=True`) and overlap with computation

---

## JIT Compilation

**PyTorch `torch.compile`** (PyTorch 2.0): compiles a PyTorch model using TorchDynamo (captures computation graph) + TorchInductor (generates optimised kernels, often using Triton).

```python
model = torch.compile(model)  # wraps the model
```

**What it does:**
- Fuses adjacent operations (e.g., relu + add → single fused kernel)
- Eliminates Python overhead by tracing and compiling to C++/CUDA
- Generates optimised loops via Triton (NVIDIA GPU kernel language)
- Typically 10-50% speedup on standard training loops, sometimes 2×+

**`torch.jit.script` (TorchScript)**: older approach. Converts Python code to a serialisable intermediate representation. More limited but more portable. Used for production deployment.

**JAX's JIT**: `@jax.jit` decorator traces and XLA-compiles Python functions. XLA (Accelerated Linear Algebra) compiles to optimal HLO (High-Level Operations) for TPUs and GPUs. Extremely fast for fixed-shape computations.

**Key JAX features:**
- **Functional**: pure functions, no side effects → composable transformations
- **`jax.grad`**: automatic differentiation, composable with vmap and jit
- **`jax.vmap`**: vectorise a function over a batch dimension (maps single-example code to batch)
- **`jax.pmap`**: parallelize across devices

---

## JAX vs PyTorch vs TensorFlow

| Feature | PyTorch | JAX | TensorFlow |
|---------|---------|-----|------------|
| Default mode | Eager | Eager (with JIT) | Graph (eager since TF2) |
| Primary abstraction | Tensor + autograd | Functional transformations | Keras / tf.function |
| JIT | torch.compile | @jax.jit (XLA) | tf.function (XLA) |
| Distribution | DDP, FSDP, deepspeed | pmap, jit-pjit | tf.distribute |
| Ecosystem | Dominant in research | Dominant at Google | Production, mobile (TFLite) |
| Debug | Easy (eager) | Harder (jit breaks pdb) | Hard (graph mode) |
| TPU support | Limited | Native | Native |
| Flexibility | High | Very high (composable) | Lower |

**PyTorch**: dominates academic research and most LLM training (DeepSeek, LLaMA, Mistral all use PyTorch). `torch.compile` has largely closed the performance gap with JAX.

**JAX**: dominant at Google DeepMind. Used for Gemini/Gemma training (on TPUs). Best for research requiring custom autodiff or complex functional transformations (e.g., meta-learning, physics simulations).

**TensorFlow**: declining in research, still used for mobile deployment (TFLite) and Google production systems. Keras high-level API sits atop both TF and JAX.

**Flax/Equinox/Haiku**: neural network libraries built on JAX. Equinox (from Patrick Kidger) is the cleanest: fully PyTree-based, arbitrary Python classes as models.
