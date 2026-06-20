# Deep Learning Fundamentals

The core building blocks of modern neural networks: computation graphs, gradient flow, architectural patterns, and normalisation.

---

## Backpropagation

Backprop is the algorithm for computing gradients of the loss with respect to all parameters in a neural network. It applies the chain rule of calculus through the computation graph.

### The Chain Rule

For a composed function L = f(g(h(x))):

```
dL/dx = dL/df · df/dg · dg/dh · dh/dx
```

In a neural network, each layer is a function, and backprop computes these products from the output (loss) backwards to the inputs (parameters).

### Forward Pass

Compute and store the activations at each layer:

```
a₀ = x                      (input)
z¹ = W¹a⁰ + b¹             (pre-activation)
a¹ = σ(z¹)                 (activation)
...
L = loss(aᴸ, y)            (loss)
```

### Backward Pass

Starting from ∂L/∂aᴸ = 1, propagate backwards:

```
∂L/∂zᴸ = ∂L/∂aᴸ · σ'(zᴸ)     (element-wise for activation)
∂L/∂Wᴸ = ∂L/∂zᴸ · (aᴸ⁻¹)ᵀ    (matrix multiply)
∂L/∂bᴸ = ∂L/∂zᴸ              (sum over batch)
∂L/∂aᴸ⁻¹ = (Wᴸ)ᵀ · ∂L/∂zᴸ   (backprop to previous layer)
```

**Key insight**: the backward pass reuses the stored forward activations. Memory cost is O(depth × batch × width) — this motivates gradient checkpointing (see [[applied-ml-systems]]).

**Computational graphs**: modern frameworks (PyTorch, JAX) build a dynamic computation graph during the forward pass and automatically compute gradients via autograd. You just define the forward computation; backprop is automatic.

---

## Activation Functions

Nonlinearities that allow neural networks to learn arbitrary functions (without them, stacking linear layers is still linear).

### ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
```

- Dead neurons: if x < 0, gradient is exactly 0 — neuron never activates again
- No saturation for positive values (unlike sigmoid/tanh)
- Default choice for hidden layers in deep networks

### Variants

**Leaky ReLU**: `max(αx, x)` with small α (0.01). Prevents dead neurons, small negative gradient.

**ELU** (Exponential Linear Unit): `x if x>0 else α(eˣ-1)`. Smooth, negative saturation, outputs can have negative mean (unlike ReLU).

**GELU** (Gaussian Error Linear Unit): `x · Φ(x)` where Φ is the Gaussian CDF. Approximated as `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`. The default in transformers (GPT, BERT). Smoother than ReLU, probabilistic interpretation.

**SiLU / Swish**: `x · σ(x)`. Used in LLaMA's feedforward (via SwiGLU).

**SwiGLU** (Shazeer 2020): `SiLU(xW₁) ⊗ (xW₂)`. Gated variant — multiply two linear transformations, one passed through SiLU. Used in LLaMA 2/3, PaLM, Gemma. Better than standard FFN, requires 2/3 scaling of hidden dim to maintain parameter count.

**Sigmoid**: `σ(x) = 1/(1+e^{-x})`. Range (0,1). Saturates at extremes → vanishing gradients. Used for output probabilities in binary classification, gate values in LSTMs.

**Tanh**: `(e^x - e^{-x})/(e^x + e^{-x})`. Range (-1,1). Zero-centred (unlike sigmoid), but also saturates. Used in LSTMs and old-style RNNs.

**Softmax**: `softmax(x)_i = e^{x_i} / Σ_j e^{x_j}`. Converts logits to probability distribution. Output layer for multi-class classification and next-token prediction in LLMs. Numerically stable via `x - max(x)`.

---

## Loss Functions

### Cross-Entropy Loss

For classification with true distribution p and predicted distribution q:

```
H(p, q) = -Σ_i p_i log q_i
```

For one-hot labels: CE = -log q_y (where y is the true class). This penalises low probability assigned to the correct class.

**Binary cross-entropy**: CE = -y log ŷ - (1-y) log(1-ŷ). For binary classification or multi-label.

**Label smoothing**: replace hard one-hot targets with (1-ε)·one_hot + ε/K. Prevents overconfidence, improves calibration. Used in most modern LLM training.

### Mean Squared Error (MSE)

```
L = (1/n) Σᵢ (ŷᵢ - yᵢ)²
```

Regression. Penalises outliers heavily (squared). Combined with L2 loss on regression outputs, equivalent to assuming Gaussian noise.

**MAE (L1 loss)**: Σ|ŷᵢ - yᵢ|. More robust to outliers. Not differentiable at 0 (use Huber loss: L1 below threshold, L2 above).

### KL Divergence Loss

KL(p||q) = Σ pᵢ log(pᵢ/qᵢ). Used in VAEs, knowledge distillation (student learns from teacher's soft logits), RLHF KL penalties.

### Contrastive Losses

**InfoNCE / NT-Xent (SimCLR)**: pull together augmented views of same image, push apart views of different images. Foundation of self-supervised representation learning.

**Triplet loss**: anchor should be closer to positive than negative by a margin.

---

## Weight Initialisation

Poor initialisation → vanishing or exploding gradients before training even starts.

### Xavier / Glorot Initialisation

Designed for linear and tanh/sigmoid activations. Keeps variance consistent across layers:

```
W ~ Uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
or
W ~ N(0, 2/(fan_in + fan_out))
```

Ensures Var(output) ≈ Var(input) for each layer, preventing exponential growth/shrinkage of activations.

### He Initialisation (Kaiming)

For ReLU activations (which kill half the variance). Doubles the variance:

```
W ~ N(0, 2/fan_in)
```

"fan_in" is the number of input connections. This is the standard for networks with ReLU/GELU.

### Why Initialisation Matters

Consider a deep network with L layers. If each layer multiplies variance by c:
- c > 1: variance explodes as cᴸ → gradient explodes in backprop
- c < 1: variance shrinks as cᴸ → gradient vanishes in backprop
- c = 1: variance is preserved (the goal of Xavier/He)

Modern initialisation in transformers: typically He/Glorot for MLP layers, small std (0.02) for embedding matrices, scaled by 1/√(2L) for residual projections to account for residual stream accumulation.

---

## BatchNorm, LayerNorm, RMSNorm

Normalisation layers stabilise training by controlling the distribution of activations.

### Batch Normalisation (BatchNorm)

For a mini-batch of size B, normalise across the batch dimension:

```
μ_B = (1/B) Σᵢ xᵢ                    # batch mean
σ²_B = (1/B) Σᵢ (xᵢ - μ_B)²          # batch variance
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)       # normalise
yᵢ = γ x̂ᵢ + β                        # scale and shift
```

γ and β are learnable parameters. At inference, use running statistics (EMA of batch stats during training).

**Benefits**: reduces sensitivity to weight initialisation, allows higher learning rates, mild regularisation effect.

**Problems**: doesn't work for small batches, sequence models (variable length), or inference-only settings (the running statistics can differ from training statistics). Bad for transformers where batch sizes vary.

### Layer Normalisation (LayerNorm)

Normalise across the feature dimension (per sample):

```
μ = mean(x)  over features
σ² = var(x)  over features
x̂ = (x - μ) / √(σ² + ε)
y = γ x̂ + β
```

No batch dimension involved — works for any batch size, including 1. Standard in transformers (BERT, GPT, and nearly all LLMs).

**Pre-norm vs Post-norm:**
- **Post-norm** (original Transformer): LN applied after residual: y = LN(x + sublayer(x)). Harder to train deep models.
- **Pre-norm** (modern default): LN applied before sublayer: y = x + sublayer(LN(x)). More stable training, default in GPT-2+.

### RMSNorm

Simplified LayerNorm — only normalise by RMS (root mean square), skip mean subtraction:

```
x̂ᵢ = xᵢ / RMS(x) = xᵢ / √(mean(x²) + ε)
y = γ x̂
```

No bias parameter β. Slightly faster (no mean computation). Used in LLaMA, Gemma, Qwen. RMSNorm works as well as LayerNorm in practice — mean subtraction doesn't matter much.

---

## CNNs (Convolutional Neural Networks)

CNNs exploit the spatial structure of images (and audio, text) through:
- **Local connectivity**: each neuron connects to a small receptive field (kernel)
- **Weight sharing**: same kernel applied across all spatial positions
- **Translation equivariance**: shift input → shift output (feature map)

### Convolution Operation

```
(f * g)[i,j] = Σ_{m,n} f[i+m, j+n] · g[m,n]
```

A kernel g (e.g., 3×3) slides across the input, computing dot products. Learnable kernels learn to detect edges, textures, shapes at different scales.

**Key dimensions:**
- Kernel size: 3×3, 5×5 (larger = larger receptive field)
- Stride: step size of sliding window (stride 2 = halve spatial dimensions)
- Padding: zero-padding input to control output size
- Channels: number of parallel feature maps

### CNN Architectures

**LeNet (1998)**: conv → pool → conv → pool → FC. Digit recognition.

**AlexNet (2012)**: deep CNN that won ImageNet, triggered the deep learning revolution. 60M params, ReLU, dropout, data augmentation.

**VGG (2014)**: very deep (16-19 layers) using 3×3 convolutions only. Shows depth matters.

**ResNet (2015)**: introduced **skip connections** (residual connections) enabling training of 100+ layer networks. The residual block: y = F(x) + x. Skip connections ensure gradient flow even when deep layers are uninitialised. ResNet-50/101/152 are still widely used backbones.

**EfficientNet**: compound scaling of depth, width, and resolution. Achieves SOTA at given parameter budget.

Modern vision is dominated by Vision Transformers (ViT) for large-scale tasks, but CNNs remain preferred for edge deployment and tasks with limited data (inductive bias toward local patterns helps).

---

## RNNs and LSTMs

Recurrent Neural Networks process sequences by maintaining a hidden state:

```
h_t = tanh(W_h h_{t-1} + W_x x_t + b)
y_t = W_y h_t + b_y
```

The hidden state h_t is a compressed representation of the sequence so far.

**Problems with vanilla RNNs:**
- **Vanishing gradients**: gradient of loss w.r.t. h_0 involves product of T Jacobians ∂h_t/∂h_{t-1}. If eigenvalues < 1, gradient vanishes; if > 1, it explodes.
- **Long-term dependencies**: practically can't remember information from >10-20 steps ago

### LSTM (Long Short-Term Memory)

LSTMs (Hochreiter & Schmidhuber, 1997) add a **cell state** C_t and gating mechanisms:

```
f_t = σ(W_f [h_{t-1}, x_t] + b_f)    # forget gate
i_t = σ(W_i [h_{t-1}, x_t] + b_i)    # input gate
g_t = tanh(W_g [h_{t-1}, x_t] + b_g) # candidate values
o_t = σ(W_o [h_{t-1}, x_t] + b_o)    # output gate

C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t    # cell state update
h_t = o_t ⊙ tanh(C_t)               # hidden state
```

The cell state C_t is the "memory highway" — information flows through largely unchanged (only forget gate acts on it), enabling gradients to flow back through time without vanishing.

**GRU (Gated Recurrent Unit)**: simplified LSTM with 2 gates instead of 3. Slightly fewer parameters, often comparable performance.

**Bidirectional RNNs**: process sequence forward and backward, concatenate hidden states. Useful when you have the full sequence (not causal). Standard in BERT-era encoder models.

RNNs were dominant for sequence modelling pre-2018, then displaced by transformers. They're still used in edge/streaming scenarios requiring very low latency and memory.

---

## S4 (Structured State Space Sequence Model)

S4 (Gu et al., 2021) is a state space model designed to efficiently handle very long sequences.

### Continuous State Space Model

The underlying continuous system:
```
x'(t) = Ax(t) + Bu(t)
y(t) = Cx(t) + Du(t)
```

where u is input, x is latent state, y is output. A, B, C, D are learned matrices.

### The Problem and Solution

Naively, discretising this system and running as an RNN is O(n) per step, O(n²) for a sequence of length n in parallel. Naively computing as a convolution is O(n log n) with FFT but the kernel needs to be computed.

S4's key insight: choose A as the **HiPPO matrix** (High-Order Polynomial Projection), which has a special structure allowing the kernel to be computed efficiently. With the HiPPO-LegS matrix, the model can remember all history optimally under L² projection.

**S4 in practice:**
- Can be computed as a **convolution** during training (O(n log n))
- Can be run as a **recurrence** during inference (O(1) per step, O(n) total)
- Handles sequences 10× longer than transformers at equivalent compute
- State space allows exact computation (no approximation)

**Mamba (SSM + selective mechanism)**: adds input-dependent (selective) transitions — the matrices A, B, C depend on the input. This breaks the fixed-kernel property but enables the model to focus on relevant inputs. Mamba achieves transformer-competitive performance on language modelling.

See [[hybrid-architectures]] for Mamba-Transformer hybrids used in modern LLMs.

---

## Autoencoders

Autoencoders learn to compress data into a bottleneck representation and reconstruct it.

```
Encoder: x → z = e_φ(x)     (compressed representation)
Decoder: z → x̂ = d_θ(z)    (reconstruction)
Loss: L = ||x - x̂||²        (MSE) or cross-entropy
```

The bottleneck forces the network to learn the most informative features. Uses:
- Dimensionality reduction (PCA without the linear constraint)
- Denoising: train on corrupted input, predict clean output → denoising autoencoder
- Anomaly detection: high reconstruction error = anomalous

**Undercomplete autoencoders**: dim(z) < dim(x). Forced compression.

**Overcomplete autoencoders**: dim(z) > dim(x). Must regularise to prevent learning identity:
- Sparse autoencoder: L1 penalty on z (encourages sparse activations)
- Contractive autoencoder: penalise ||∂z/∂x||_F
- Denoising autoencoder: add noise to input during training

**Variational Autoencoders (VAEs)**: see [[generative-models]] — VAEs add a probabilistic bottleneck with KL regularisation.

**Sparse autoencoders for mechanistic interpretability**: train overparameterised sparse autoencoders on LLM activations to find monosemantic features (features that activate for single interpretable concepts). Key tool in Anthropic's interpretability work — the LLM's internal representation is superposed (polysemantic), and sparse autoencoders decompose it.

---

## Gumbel-Softmax

The Gumbel-Softmax trick allows backpropagation through **discrete sampling operations** — a fundamental challenge since sampling from a categorical distribution is not differentiable.

### Gumbel Max Trick

The argmax of (log π_k + Gumbel_k) has the same distribution as sampling from a categorical distribution with probabilities π = softmax(logits).

```
z = argmax_k [log π_k + g_k], where g_k ~ Gumbel(0,1) = -log(-log Uniform(0,1))
```

This converts the sampling problem into a deterministic operation (argmax) plus independent noise.

### The Softmax Relaxation

Replace argmax with softmax (differentiable):

```
y_k = softmax((log π_k + g_k) / τ)_k
```

As temperature τ → 0: approaches a one-hot vector (like argmax)
As τ → ∞: approaches uniform distribution

**Straight-Through Estimator**: in the forward pass, use the hard argmax (one-hot); in the backward pass, backprop through the soft Gumbel-Softmax. This gives gradients while maintaining the discrete forward behaviour.

### Applications

- **Discrete VAEs**: discrete latent space (DALL-E uses a discrete codebook; Gumbel-Softmax allows training)
- **Differentiable text generation**: generate from discrete token distributions while backpropagating loss
- **Neural architecture search**: discrete architectural choices become differentiable
- **VQ-VAE training**: straight-through estimator for the vector quantisation step

The core insight: many discrete operations have useful continuous relaxations, and Gumbel noise is the natural relaxation for categorical sampling. Temperature annealing (start high, reduce to 0) during training gives the network time to learn which discrete choices to make before committing.
