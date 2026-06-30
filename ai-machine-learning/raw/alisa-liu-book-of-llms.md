# Alisa's Book of LLMs

**Source:** https://alisawuffles.notion.site/alisa-s-book-of-llms
**Author:** Alisa Liu
**Published (per page metadata):** Nov 20, 2025

> **Capture note:** This is a very long, dense Notion reference page, fetched via a Jina Reader proxy (`r.jina.ai`) because the live page is JS-rendered. The fetch tool truncates output at a fixed length; three independent fetch attempts (including two with different URL fragments, to test whether scrolling/virtualization would surface different content) all returned byte-identical text truncated at the exact same point, confirming this is a hard limit of the fetch tool rather than something retriable. The capture below is complete and faithful through "Accounting → FLOPs in forward pass" (cut off mid-formula on the FFN down-projection FLOPs term). Per this repo's grounding rule, the remaining TOC sections are **not fabricated** and are listed at the end of this file as not-yet-retrieved, for a future pass (e.g. once a JS-rendering browser tool is available):
> FLOPs in backward pass; Inference memory use; Train memory use; the dedicated **Attention**, **RMSNorm**, **SwiGLU FFN**, **RoPE** sub-pages; all of **Inference** (Batching & packing, Speculative decoding, KV cache, Reducing KV cache size, Sampling strategies, Flash Attention); **Scaling laws**; **GPUs**; **Other architectures** (RNNs, vanilla RNN, LSTM, vs. transformers, State space models); **Post-training** (policy gradients, PPO, RLHF, GRPO, DPO); **Precision**; **Parallelism** (core collective operations, data/pipeline/tensor parallelism); **Multimodality**.

## Full Table of Contents (for reference — sections marked ✅ are captured below, others are not yet retrieved)

- ✅ Neural net basics: Multi-layer perceptrons, Activation functions, Gradients, Backpropagation, Optimizers, Learning rate
- ✅ Mathy things: Information theory, Numerical stability and other tricks, Basic statistics, Gradient flow through sampling
- ✅ Theoretical CS
- 🔶 The modern transformer LM: ✅ Architecture, ✅ Implementation notes, 🔶 Accounting (✅ Model parameters, ✅ Model activations, 🔶 FLOPs in forward pass [cut off mid-formula], ❌ FLOPs in backward pass, ❌ Inference memory use, ❌ Train memory use), ❌ Attention, ❌ RMSNorm, ❌ SwiGLU FFN, ❌ RoPE
- ❌ Inference: Batching & packing, Speculative decoding, KV cache, Reducing KV cache size, Sampling strategies, Flash Attention
- ❌ Scaling laws
- ❌ GPUs
- ❌ Other architectures: RNNs (Vanilla RNN, LSTM), vs. transformers, State space models
- ❌ Post-training: policy gradients, PPO, RLHF, GRPO, DPO
- ❌ Precision
- ❌ Parallelism: Background (core collective operations), Data parallelism, Pipeline parallelism, Tensor parallelism
- ❌ Multimodality

---

## Neural Net Basics

### Multi-layer Perceptrons

A multi-layer perceptron (MLP) is a fully-connected network with an input layer, at least one hidden layer, and an output layer. Often used synonymously with "feed-forward network," even though FFN is technically a broader category where information flows in one direction.

A single neuron computes a weighted sum of its inputs, adds a bias, and passes the result through an activation function. For input vector $x\in\mathbb{R}^n$, weight vector $w\in\mathbb{R}^n$, bias $b\in\mathbb{R}$, and activation $f$:
$$y = f\left(\sum_{i=1}^n w_ix_i + b\right) = f(w^\top x + b)$$

A layer with $n_{in}$ inputs and $n_{out}$ neurons is computed via matrix multiplication. Stack weight vectors into $W\in\mathbb{R}^{n_{out}\times n_{in}}$ (each row = weights into one neuron) and biases into $b\in\mathbb{R}^{n_{out}}$: $h = f(Wx+b)$.

In practice we process a batch of $m$ inputs at once: arrange inputs as rows of $X\in\mathbb{R}^{m\times n_{in}}$, and conventionally change $W$ to shape $\mathbb{R}^{n_{in}\times n_{out}}$ (each column = weights into one neuron): $H = f(XW+b)$, with $b$ broadcast to shape $m\times n_{out}$.

**PyTorch convention note**: PyTorch actually stores the weight matrix as $n_{out}\times n_{in}$. The forward pass transposes $W$, computing `X @ W.T` — the transpose is free (only changes the stride). This is so the gradient for $W$ naturally comes out as $n_{out}\times n_{in}$, matching $W$'s own shape.

**Backprop for $Z=XW+b$** (math convention, $W\in\mathbb{R}^{n_{in}\times n_{out}}$):
$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z}W^\top \quad (m,n_{out})\times(n_{out},n_{in})=(m,n_{in})$$
$$\frac{\partial L}{\partial W} = X^\top\frac{\partial L}{\partial Z} \quad (n_{in},m)\times(m,n_{out})=(n_{in},n_{out})$$

The bias $b\in\mathbb{R}^{n_{out}}$ is added to every sample, and each sample produces its own gradient contribution for $b$, so these **accumulate** (sum over the batch):
$$\frac{\partial L}{\partial b_j} = \sum_{i=1}^m \frac{\partial L}{\partial z_{ij}}\cdot\frac{\partial z_{ij}}{\partial b_j} = \sum_{i=1}^m \frac{\partial L}{\partial z_{ij}}$$

Intuition for the batched gradient shapes: derive the Jacobian for a single example first (clean, 2-D), then generalize. If a tensor is **shared** across the batch (like $W$), the batch dimension gets summed out (contract — matmul with the batch dim as the inner dimension). If a tensor is **not shared** (like activations $X$), the batch dimension is preserved (stack — matmul with the batch dim on the outside).

PyTorch's actual convention ($Z=XW^\top$, $W\in\mathbb{R}^{n_{out}\times n_{in}}$):
$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z}W \quad (m,n_{out})\times(n_{out},n_{in})=(m,n_{in})$$
$$\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Z}\right)^\top X \quad (n_{out},m)\times(m,n_{in})=(n_{out},n_{in})$$

### Activation Functions

**Sigmoid** $\sigma(x)\in(0,1)$: $\sigma(x) = \frac{1}{1+e^{-x}}$. Good for interpreting outputs as probabilities; not used for hidden layers in neural nets. Suffers vanishing gradients since its derivative $\sigma(x)(1-\sigma(x))\le 0.25$. Not zero-centered, so downstream gradients for a single node are either all-positive or all-negative (depending on the upstream gradient).

**Tanh** $\in(-1,1)$: $\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}} = 2\sigma(2x)-1$. Derivative peaks at 1.0 (at $x=0$) but can still vanish: $\tanh'(x) = 1-\tanh^2(x)\in(0,1]$ (factors only ever shrink).

**Softmax** → probability distribution: $\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$. With temperature: $\text{softmax}(x/T)_i = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}}$.

**ReLU** $\in[0,\infty)$: $\text{ReLU}(x) = \max(x,0)$. Derivative is 1 for $x>0$, 0 for $x<0$. **Dying ReLUs**: if a pre-activation becomes permanently negative (negative for every input), it receives zero gradient forever — a fraction of the network can go dead during training.

**Leaky ReLU** $\in(-\infty,\infty)$: $x$ if $x>0$, else $\alpha x$. Fixes the dying-ReLU problem.

**Swish** (smooth, non-monotonic): $\text{Swish}(x) = x\cdot\sigma(x)$.

**GLU**: uses one linear projection for "content" and another for the "gate": $\text{GLU}(x) = xW_1 \odot \sigma(xW_2)$.

**SwiGLU**: plugs Swish in as the GLU's activation: $\text{SwiGLU}(x) = (xW_1)\odot\text{Swish}(xW_2)$.

Without non-linearities, neural nets can't do anything more than a linear transform — extra layers compile down to a single linear transform $W_1W_2x = Wx$. With non-linearities, stacking layers lets the network approximate any complex function.

### Gradients

The derivative of a variable tells you the sensitivity of the whole expression to its value: if $\partial f/\partial x=3$, changing $x$ by small $h$ changes $f(x)$ by $\sim 3h$. $\frac{df(x)}{dx} = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$.

The **gradient** $\nabla f$ is the vector of partial derivatives. For a function with $m$ outputs and $n$ inputs, the **Jacobian** is an $m\times n$ matrix of partial derivatives $\left(\frac{\partial f}{\partial x}\right)_{ij} = \frac{\partial f_i}{\partial x_j}$. For a function with $n$ inputs and scalar output, the **Hessian** is the $n\times n$ matrix of second partials $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ — it describes the curvature of the loss landscape.

**Chain rule**: for one-variable compositions, multiply derivatives ($x=3y, y=x^2 \Rightarrow \frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}=6x$); for multivariable functions, multiply Jacobians ($h=f(z), z=Wx+b \Rightarrow \frac{\partial h}{\partial x} = \frac{\partial h}{\partial z}\frac{\partial z}{\partial x}$).

**Worked neural-net-setup example**: $x\in\mathbb{R}^d$, $h=f(Wx+b)\in\mathbb{R}^k$ ($W\in\mathbb{R}^{k\times d}, b\in\mathbb{R}^k$), $s=u^\top h\in\mathbb{R}$ ($u\in\mathbb{R}^k$).

For an element-wise activation $h=f(z)$, $h,z\in\mathbb{R}^n$: $\left(\frac{\partial h}{\partial z}\right)_{ij} = f'(z_i)$ if $i=j$, else 0 — i.e. the Jacobian is **diagonal**: $\frac{\partial h}{\partial z} = \text{diag}(f'(z))$.

Useful Jacobians: $\frac{\partial}{\partial x}(Wx+b) = W$; $\frac{\partial}{\partial b}(Wx+b)=I$; $\frac{\partial}{\partial z}f(z) = \text{diag}(f'(z))$; $\frac{\partial}{\partial u}(u^\top h) = h^\top$.

Other helpful derivatives: $\frac{d}{dx}\frac1x = -\frac{1}{x^2}$; $\frac{d}{dx}e^x=e^x$; $\frac{d}{dx}\sigma(x) = (1-\sigma(x))\sigma(x)$; $\frac{d}{dx}\log x = \frac1x$; $\frac{d}{dx}\tanh(x) = 1-\tanh^2(x)$.

**Derivative of sigmoid, derived**: $\frac{d}{dx}\sigma(x) = \frac{d}{dx}(1+e^{-x})^{-1} = (1+e^{-x})^{-2}e^{-x} = \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}} = \sigma(x)(1-\sigma(x))$.

**Derivative of Swish, derived** (product rule + sigmoid derivative): $\frac{\partial}{\partial x}\text{Swish}(x) = \sigma(x) + x\sigma'(x) = \sigma(x)+x\sigma(x)(1-\sigma(x)) = \sigma(x)+\text{Swish}(x)(1-\sigma(x))$.

**Gradient of softmax + cross-entropy loss, full derivation**: let $z\in\mathbb{R}^V$ be logits, $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$, $L=-\log p_t$ ($t$ = correct class).

$\frac{\partial L}{\partial p_i} = -\frac{1}{p_t}$ if $i=t$, else 0. Using the chain rule, most terms vanish since $\partial L/\partial p_j$ is only non-zero at $j=t$: $\frac{\partial L}{\partial z_i} = \sum_j \frac{\partial L}{\partial p_j}\frac{\partial p_j}{\partial z_i} = -\frac{1}{p_t}\frac{\partial p_t}{\partial z_i}$.

Softmax's own Jacobian: $\frac{\partial p_j}{\partial z_i} = p_j(1-p_j)$ if $i=j$, else $-p_jp_i$.

Putting it together: for the true token ($i=t$): $\frac{\partial L}{\partial z_t} = -\frac1{p_t}\cdot p_t(1-p_t) = p_t - 1$. For all other tokens: $\frac{\partial L}{\partial z_i} = -\frac{1}{p_t}\cdot(-p_tp_i) = p_i$.

**Very clean overall result**: $\frac{\partial L}{\partial z} = p - \text{one\_hot}(t)$.

### Backpropagation

Neural-net equations are represented as a **computation graph**; which sub-expressions are treated as "gates" is just a matter of convenience (typically: parts with easy local gradients). Backprop can be thought of as gates communicating to each other (via the gradient signal) whether they want their outputs to increase or decrease — and how strongly — to decrease the loss. This is achieved via repeated chain-rule application, decomposing each gradient into the **upstream gradient** (already computed) times the **local gradient**: *downstream gradient = upstream gradient × local gradient*.

**Gradients sum at outward branches**: if $y$ feeds into both $a$ and $b$: $\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial y}$.

**Node intuitions**: `+` distributes the upstream gradient to each summand; `max` "routes" the upstream gradient to exactly one of the input arguments; `×` switches the forward coefficients in the downstream gradient.

**Backprop algorithm**: initialize the output gradient as 1; visit nodes in reverse topological order, computing the gradient w.r.t. each node from the gradients w.r.t. its successors. Done correctly, the big-O complexity of backprop matches forward prop.

**Automatic differentiation**: the gradient computation can be automatically inferred from the symbolic forward-pass expression — each node type needs to know how to compute its output and how to compute gradients w.r.t. its inputs given gradients w.r.t. its outputs (the local gradient is written by the programmer).

**Manual gradient checking**: for every parameter $x$, recompute $f$ at $x-h$ and $x+h$ and check $f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}$.

**Activation/gradient checkpointing**: in the backward pass, intermediate activations are needed, so NNs typically store all of them during the forward pass. Checkpointing trades compute for memory by storing only a subset of activations ("checkpoints") — activations that weren't saved are recomputed on the fly via a partial forward pass from the nearest checkpoint. For a model with $N$ layers checkpointed into $K$ segments: memory goes from $O(N)$ to $O(K+N/K)$; backward compute goes from $O(N)$ to $O(N + N(K-1)/K)$. **Optimal choice** is $K=\sqrt N$: $O(\sqrt N)$ memory, $\sim O(2N)$ backward compute.

**Worked "do for practice" example**: $f(x,y) = \frac{x+\sigma(y)}{\sigma(x)+(x+y)^2}$ — computing $\partial f/\partial x$ explicitly would be extremely complex, but unnecessary: construct intermediate variables in the forward pass, each a simple expression with a known local gradient.

Backprop must start from a **scalar**, because it computes $\partial L/\partial \theta$ — a single number per parameter $\theta$ — which only makes sense when $L$ is scalar. Calling `.backward()` on a scalar implicitly seeds the backward pass with $\partial L/\partial L = 1$.

For per-token losses $\ell_1,\dots,\ell_n$ with mean reduction $L=\frac1N\sum_i \ell_i$, by linearity of derivatives: $\frac{\partial L}{\partial\theta} = \frac1N\sum_{i=1}^N \frac{\partial \ell_i}{\partial\theta}$ — the gradient from the mean loss is exactly the mean of the per-example gradients.

Upstream gradients are always w.r.t. **activations**; gradients w.r.t. **parameters** are used for the update and "end there," since parameters are leaf nodes in the computation graph:
```
loss
  │  dL/dy2 (activation grad)
  ▼
Layer 2 ──→ dL/dW2 (param grad, stored)
  │  dL/dy1 (activation grad)
  ▼
Layer 1 ──→ dL/dW1 (param grad, stored)
  │  dL/dx  (activation grad — usually discarded)
  ▼
input
```

### Optimizers

Vanilla SGD update: $\theta \leftarrow \theta - \eta g_t$. An optimizer determines the direction and magnitude of parameter updates.

**Adam** keeps, for every parameter tensor: the parameter $\theta$, the gradient $g$, a first moment (momentum) $m$, and a second moment (variance) $v$.

Update: $\theta \leftarrow \theta - \eta \frac{\hat m}{\sqrt{\hat v}+\epsilon}$.

First moment (running mean of gradients): $m \leftarrow \beta_1 m + (1-\beta_1)g$ — if gradients have consistently pointed one direction, the optimizer moves more confidently that way; accumulating velocity helps barrel through noisy gradients.

Second moment (running mean of squared gradients): $v \leftarrow \beta_2 v + (1-\beta_2)g^2$ — different weights effectively get different learning rates (consistently large gradients → smaller steps); effectively normalizes gradients onto the same scale.

**Bias correction** (corrects for initialization bias early in training, since $m,v$ both init to 0), for time step $t$ starting at 1: $\hat m_t = \frac{m_t}{1-\beta_1^t}$, $\hat v_t = \frac{v_t}{1-\beta_2^t}$. So memory per parameter ≈ **4× parameter size** (param + grad + m + v).

**AdamW** modifies Adam by adding weight decay toward 0: $\theta \leftarrow \theta - \eta\frac{\hat m}{\sqrt{\hat v}+\epsilon} - \eta\lambda\theta$.

The optimizer is initialized with the model's parameters (so it knows what to optimize) and the `lr` hyperparameter (size of the update). Rule of thumb for "LR schedule vs. optimizer": if a quantity depends on the time step $t$ alone, it's probably an LR schedule; if it requires per-parameter history, it's an optimizer.

`params` creates parameter groups, each with its own hyperparameters (e.g. different LRs per layer); `torch.optim.AdamW(model.parameters())` creates a single parameter group. In practice we usually don't want weight decay on biases and LayerNorm params:
```python
torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0},
])
```
The `defaults` dict provides fallback values for any hyperparameter not explicitly specified in a parameter group. In practice it's better to apply weight decay *before* the Adam update, because weight decay depends on the parameter itself.

Full from-scratch AdamW implementation given in the notes:
```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 < betas[0] < 1 or not 0 < betas[1] < 1:
            raise ValueError(f"Invalid beta values: {betas}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:  # for every group of parameters
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:  # for every parameter in the group
                if p.grad is None:
                    continue
                state = self.state[p]

                # state initialization with 0s
                t = state.get("t", 0)
                m, v = state.get("m", torch.zeros_like(p.data)), state.get("v", torch.zeros_like(p.data))

                # weight decay
                p.data -= lr * weight_decay * p.data

                # Adam update
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1 ** (t + 1))
                v_hat = v / (1 - beta2 ** (t + 1))
                p.data -= lr * m_hat / (v_hat.sqrt() + eps)

                # update optimizer state
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
```

**Gradient clipping** constrains the size of the grad norm: compute the global norm of all gradients; if it exceeds a max, scale all parameters down by the same value to stay under the max. Prevents any individual step from being catastrophically large.

### Learning Rate

Warmup reduces the primacy effect of early training examples.

## Mathy Things

### Information Theory

**Cross-entropy**: $\text{CE}(p,q) = -E_p[\log q] = -\sum_{x\in X}p(x)\log q(x)$.

**KL divergence**: $\text{KL}(p\|q) = \sum_{x\in X}p(x)(\log p(x)-\log q(x))$.

**Entropy**: $H(p) = -\sum_{x\in X}p(x)\log p(x)$. Another common form, given logits $x_i$: $H(p) = \log\sum_i e^{x_i} - \frac{\sum_i e^{x_i}x_i}{\sum_i e^{x_i}}$ (the second term is $E[x]$ under the softmax distribution).

**Identity**: cross-entropy between $p$ and $q$ is KL between $p$ and $q$ plus the irreducible entropy of $p$: $\text{CE}(p,q) = \text{KL}(p\|q) + H(p)$.

Proof: $\text{KL}(p\|q) = \sum_x p(x)(\log p(x)-\log q(x)) = \sum_x p(x)\log p(x) - \sum_x p(x)\log q(x) = -H(p) + \text{CE}(p,q)$.

**Cross-entropy loss**: when the target distribution is one-hot, cross-entropy loss is the negative log-likelihood of the next token: $L(x) = -\sum_{t=1}^T \log p(x_t\mid x_{<t})$ — also equivalent to a KL divergence.

Implementing the loss — if using `F.cross_entropy()`, logits and labels are shifted internally:
```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=pad_idx)
```
Manual version (e.g. for masked / per-token-weighted loss):
```python
shift_logits = logits[:, :-1, :]
shift_labels = input_ids[:, 1:]
logprobs = F.log_softmax(shift_logits, dim=-1)
token_logprobs = logprobs.gather(index=shift_labels.unsqueeze(-1), dim=-1).squeeze(-1)
# build loss mask
masked_logprobs = -token_logprobs * mask.float()
return masked_logprobs.sum() / mask.sum()
```

### Numerical Stability and Other Tricks

General things to watch for: `exp(x)` for large $x$ → overflow to $\infty$; `log(x)` near 0 → underflow to $-\infty$ ($\log(0)=-\infty$); `log(x)` near 1 → precision issues ($\log(1)=0$).

**Computing softmax(x)** — naively unstable because for large $x_i$, $\exp(x_i)$ overflows. Softmax is invariant to subtracting a constant $c$ from all inputs:
$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i-c}\cdot e^c}{e^c\cdot\sum_j e^{x_j-c}} = \frac{e^{x_i-c}}{\sum_j e^{x_j-c}} = \text{softmax}(x-c)_i$$
Subtracting $x_{\max}$ from all $x_i$ ensures no $x_i$ is large (largest exponent is $\exp(0)=1$), giving the numerically-stable form: $\text{softmax}(x)_i = \frac{e^{x_i-x_{\max}}}{\sum_j e^{x_j-x_{\max}}}$.

**Computing log(softmax(x))** — doing log-softmax directly is bad because log of near-zero inputs (low-probability classes) is unstable. Use $x-\text{logsumexp}(x)$ instead, avoiding materializing the tiny probability:
$$\log(\text{softmax}(x))_i = \log\frac{e^{x_i}}{\sum_j e^{x_j}} = x_i - \log\sum_j e^{x_j}$$

**Computing logsumexp(x)** — why naive computation is unstable: any large $x_i$ overflows $\exp(x_i)$ (e.g. in float32, $x_i\approx 83$ overflows); if all $x_i$ are very negative, $\log(0)$ underflows to $-\infty$; and if the sum is close to 1, $\log(1)$ has precision issues (the author notes: "this is the problem we got in our distillation project"). Intuition: subtract a constant to make the $x_i$ small, then add it back at the end:
$$\log\sum_i e^{x_i} = \log\sum_i\left(e^{x_i-x_{\max}}\cdot e^{x_{\max}}\right) = x_{\max} + \log\sum_i e^{x_i-x_{\max}}$$
The largest term is now $e^0=1$ (no overflow), and the sum is at least 1 (never $\log(0)$).

Naively, computing softmax this way requires **two passes**: one to compute $x_{\max}$, one for the denominator $\sum_j e^{x_j-x_{\max}}$.

**Online softmax trick**: fuse the computation of $x_{\max}$ and the denominator into a *single pass*. Idea: compute the denominator using the running max-so-far, continuously rescaling it as the max-so-far updates. Maintain a running max $m_k = \max(x_1,\dots,x_k)$ and running shifted denominator $d_k = \sum_j e^{x_j-m_k}$. Update rule on encountering $x_{k+1}$:
$$m_{k+1} \leftarrow \max(m_k, x_{k+1}) \qquad d_{k+1} \leftarrow d_k\cdot e^{m_k-m_{k+1}} + e^{x_{k+1}-m_{k+1}}$$

Derivation of the $d_{k+1}$ update (split off the last term, rewrite the exponent as $x_j-m_k+m_k-m_{k+1}$, pull out the constant factor $e^{m_k-m_{k+1}}$, and recognize the remaining sum as $d_k$):
$$d_{k+1} = e^{x_{k+1}-m_{k+1}} + \sum_{j=1}^k e^{x_j-m_{k+1}} = e^{x_{k+1}-m_{k+1}} + e^{m_k-m_{k+1}}\underbrace{\sum_{j=1}^k e^{x_j-m_k}}_{d_k} = \underbrace{d_k\cdot e^{m_k-m_{k+1}}}_{\text{rescaling of prev terms}} + \underbrace{e^{x_{k+1}-m_{k+1}}}_{\text{new term}}$$
When the max doesn't change, $m_k=m_{k+1}$ and the rescaling factor $e^{m_k-m_{k+1}}=1$.

To actually return the softmax, one more pass over the logits computes $e^{x_i-m_S}/d_S$.

**Connection to FlashAttention** — the same trick computes a weighted sum given a *stream* of logits $x_i$ and values $v_i$: $o = \sum_i p_i v_i = \frac{\sum_i e^{x_i}v_i}{\sum_i e^{x_i}}$. For FlashAttention, $o$ is the attention output for query $q$, where $x_i = q\cdot k_i$ and $v_i$ are value vectors. Numerically-stable version multiplies num/denom by $e^{-x_{\max}}$: $o = \frac{\sum_i e^{x_i-x_{\max}}v_i}{\sum_i e^{x_i-x_{\max}}}$.

In addition to $m_k,d_k$, FlashAttention maintains a running numerator $o_k\in\mathbb{R}^H$ ($H$ = head dimension): $o_k = \sum_{i=1}^k e^{x_i-m_k}v_i$, updated the same way as $d_k$:
$$o_{k+1} = o_k\cdot e^{m_k-m_{k+1}} + \underbrace{e^{x_{k+1}-m_{k+1}}v_{k+1}}_{\text{new term}}$$
After all $N$ logits, we have $(m_N,d_N,o_N)$ and the true attention output is $o_N/d_N$.

General-purpose stability tip: when an expression involving $e^x$ needs to be made numerically stable, multiply by $e^{-m}$ (commonly $m=x_{\max}$) and see what survives.

### Basic Statistics

**$p$-value**: probability of seeing data (at least this extreme) given the null hypothesis. Crucially: it's the probability of the *data given the null*, not the probability of the *null given the data*.

**"Are these groups different?"**
- **Kolmogorov-Smirnov test**: given two groups of observations of a *continuous* variable, were they drawn from the same distribution? Measures the maximum vertical distance between the two samples' CDFs.
- **Chi-squared test**: same question for a *categorical* variable.
- **T-test**: are the means of two groups of continuous observations different? Assumes normally-distributed data. One-sample version (group mean equals some value); two-sample version (two groups have the same mean); paired version (whether two measurements on the *same* items are systematically different — this is a one-sample t-test in disguise: given per-example differences, test whether the differences are significantly different from 0).
- **ANOVA (F-test)**: generalization of the t-test to more than two groups — do any of $k$ groups have different means?
- **McNemar's test**: compares two classifiers on the same dataset. Called out as "probably the best test for a standard setup with two models being evaluated on the same test set, where each example can be answered correctly or incorrectly."

**"Are these variables related?"**
- **Pearson correlation test**: tests for a *linear* relationship between two continuous variables; returns a correlation coefficient $r$ and a p-value for whether $r$ differs significantly from 0. Misses non-linear relationships entirely (a perfect parabolic relationship → $r\approx 0$). Visualize as a line fit on a scatterplot.
- **Spearman correlation test**: same idea but for *monotonic* associations — converts both variables to ranks, then computes Pearson correlation on the ranks. Only order matters, not specific values.
- **Pearson vs. Spearman**: Pearson is more sensitive to outliers (Spearman is robust to them); Pearson underestimates non-linear monotonic relationships; Pearson has a cleaner interpretation.
- **Mutual information**: captures *any* dependency between two variables, including non-linear ones — not really a statistical test per se.

### Gradient Flow Through Sampling

**Gumbel-Max trick**: given logits $z_1,\dots,z_k$, sample from the corresponding categorical distribution by drawing independent noise $g_1,\dots,g_k\sim\text{Gumbel}(0,1)$ and taking $\arg\max(z_1+g_1,\dots,z_k+g_k)$.

**Gumbel-Softmax** replaces the argmax with a softmax: $\text{softmax}((z_1+g_1,\dots,z_k+g_k)/\tau)$. High temperature → smooth gradients; low temperature → near-discrete samples. The point isn't to make something differentiable (plain softmax already is) — it's to make sampling **stochastic** while remaining differentiable.

Three-way comparison given in the notes:
- Plain softmax $y=\text{softmax}(\alpha)$: deterministic (always the same soft mixture).
- True categorical sampling $y=\text{one\_hot}(\text{sample}(\text{softmax}(\alpha)))$: stochastic but **not** differentiable.
- Gumbel-softmax $y=\text{softmax}((\alpha+G)/\tau)$: stochastic (Gumbel noise), approximately discrete (low temperature), and differentiable — "you get exploration AND gradients!"

**Straight-through estimator (STE)**: pretend a non-differentiable function $f$ was the identity on the backward pass. Forward: $y=f(x)$ (apply the actual non-differentiable function). Backward: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}$ — pass the upstream gradient straight down. This would be *wrong* if $f(x)$ were decreasing (gradient direction would be backwards), but STE is usually applied to monotonically-increasing operations (rounding, quantization, step functions), where the direction is right even if the magnitude is approximate.

## Theoretical CS

A **regular language** is one recognizable by a finite-state machine (finite automaton) — e.g. a deterministic finite automaton (DFA). A **context-free language** is one recognizable by a pushdown automaton (a finite-state machine plus a stack); the stack gives unbounded memory but only top-of-stack access. Programming-language syntax (matched parens, nested function calls, balanced HTML tags) is built from context-free languages. Any finite language is trivially regular (enumerate all valid strings with enough states).

**Any DFA can be encoded by a ReLU RNN.** Given a DFA with states $Q=\{q_1,\dots,q_k\}$, alphabet $\Sigma=\{\sigma_1,\dots,\sigma_m\}$, transition function $\delta: Q\times\Sigma\to Q$, start state $q_1$, accept states $F\subseteq Q$: build an RNN with hidden dimension $k$ where the hidden state is a one-hot encoding of the current DFA state. Construct $W_h\in\mathbb{R}^{k\times k}$, $W_x\in\mathbb{R}^{k\times m}$, $b\in\mathbb{R}^k$ such that for each transition $\delta(q_i,\sigma_k)=q_j$: set $(W_h)_{ji}=1$, $(W_x)_{jk}=1$, $b_j=-1$.

## The Modern Transformer LM

### Architecture

Notation used throughout:

| Symbol | Dimension |
|---|---|
| B | number of sequences in the batch |
| L | number of layers |
| T | sequence length (number of tokens to generate) |
| S | sequence length (provided context) |
| V | vocab size |
| D | hidden dimension |
| H | head dimension |
| F | MLP hidden dimension, generally F = 4D |
| N | number of query heads, N·H = D |
| K | number of key/value heads, K < N in GQA |
| G | group size in GQA = N // K |

**Token embedding**: embedding matrix $W_e\in\mathbb{R}^{V\times D}$, initial hidden states $X^{(0)}\in\mathbb{R}^{B\times S\times D} = W_e[\text{tokens}]$.

**Layer loop**, for $\ell\in[0,\dots,L-1]$:

*RMSNorm* divides every element of $X^{(\ell)}$ by the RMS of $X^{(\ell)}$ (so the hidden state has unit RMS) then multiplies by a learned rescaling $\gamma$:
$$\bar X^{(\ell)} = \frac{X^{(\ell)}}{\text{RMS}(X^{(\ell)})+\epsilon}\odot\gamma^{(\ell)}_{\text{attn}} \qquad \text{RMS}(X) = \sqrt{\frac1D\sum_{i=1}^D x_i^2}$$

*Attention projections*: each head projects $X^{(\ell)}$ via $W_Q^{(\ell)}\in\mathbb{R}^{D\times D}$, $W_K\in\mathbb{R}^{D\times KH}$, $W_V^{(\ell)}\in\mathbb{R}^{D\times KH}$ ($H=D/N$) into the head's lower-dimensional subspace:
$$Q=\bar XW_Q\in\mathbb{R}^{B\times T\times D},\quad K=\bar XW_K\in\mathbb{R}^{B\times S\times D},\quad V=\bar XW_V\in\mathbb{R}^{B\times S\times D}$$

*[optional] QK norm*: RMSNorm applied to query/key vectors to control the magnitude of vectors going into the dot product.

Reshape to expose the head dimension ($D\to N\times H$, and $K\cdot H \to K\times H$), then transpose the sequence-length and head-count dims:
$$Q\in\mathbb{R}^{B\times T\times D}\to\mathbb{R}^{B\times N\times T\times H} \qquad K,V\in\mathbb{R}^{B\times S\times(K\cdot H)}\to\mathbb{R}^{B\times K\times S\times H}$$
Expand $K,V$ for GQA: $\mathbb{R}^{B\times K\times S\times H}\to\mathbb{R}^{B\times N\times S\times H}$.

*RoPE*: at each position $m$, rotate a query vector $q_m\in\mathbb{R}^H$ (or key vector $k_m$) by $R_m$. For dimension pair $i$ (indices $(2i,2i+1)$ of $q_m$), rotate by angle $m\theta_i$ where $\theta_i = \Theta^{-2i/H}$ ($\Theta$ controls the base rotation frequency, $H$ is head dim):
$$R_m = \begin{bmatrix}\ddots & & \\ & R_m^{(i)} & \\ & & \ddots\end{bmatrix}\in\mathbb{R}^{H\times H} \quad\text{where}\quad R_m^{(i)} = \begin{bmatrix}\cos(m\theta_i) & -\sin(m\theta_i)\\ \sin(m\theta_i) & \cos(m\theta_i)\end{bmatrix}$$
$$q_m \leftarrow R_mq_m, \qquad k_m \leftarrow R_mk_m$$

*Attention scores*: divide by $\sqrt H$ (otherwise dot products scale with $\sqrt H$; large inputs to softmax → peakier distributions → resistant to gradient updates):
$$A = \frac{QK^\top}{\sqrt H}\in\mathbb{R}^{B\times N\times T\times S}$$
Apply the causal mask ($A_{ij}\leftarrow A_{ij}$ if $j\le i$, else $-\infty$), apply softmax:
$$A = \text{softmax}(A) = \frac{\exp(A_{ij})}{\sum_{k=1}^S \exp(A_{ik})}$$
Get the attention output as a weighted sum of values, $O=AV\in\mathbb{R}^{B\times N\times T\times H}$, reshape to $\mathbb{R}^{B\times T\times D}$, apply the output projection $W_O^{(\ell)}\in\mathbb{R}^{D\times D}$ to mix information across heads: $O_{\text{proj}} = OW_O$.

*Residual connection*: $X^{(\ell)} \leftarrow X^{(\ell)} + O_{\text{proj}}$.

*Feed-forward network*: RMSNorm $\bar X^{(\ell)} = \frac{X^{(\ell)}}{\text{RMS}(X^{(\ell)})+\epsilon}\odot\gamma^{(\ell)}_{\text{ffn}}$. Gate and up projections (expansion) using $W_{\text{up}}^{(\ell)}, W_{\text{gate}}^{(\ell)}\in\mathbb{R}^{D\times F}$: $U=\bar XW_{\text{up}}$, $G=\bar XW_{\text{gate}}$. SwiGLU activation: $\text{Swish}(G) = G\odot\sigma(G) = G\odot\frac{1}{1+e^{-G}}$, then $H_{\text{ffn}} = \text{Swish}(G)\odot U \in\mathbb{R}^{B\times T\times F}$. Down projection using $W_{\text{down}}^{(\ell)}\in\mathbb{R}^{F\times D}$: $F_{\text{out}} = H_{\text{ffn}}W_{\text{down}}$. Residual: $X^{(\ell+1)} = X^{(\ell)}+F_{\text{out}}$.

**Final layer norm**: $X_{\text{final}} = \frac{X^{(L)}}{\text{RMS}(X^{(L)})+\epsilon}\odot\gamma_{\text{final}}$.

**Unembedding**: project onto vocab dimension using $W_u\in\mathbb{R}^{D\times V}$: $Z = X_{\text{final}}W_u \in\mathbb{R}^{B\times T\times V}$.

### Implementation Notes

`scores.masked_fill(~mask, -torch.inf)` for making pre-softmax attention scores, assuming the convention that `mask` is True for positions that *can* be attended to (`tensor.masked_fill(mask, value)` fills `tensor` with `value` wherever `mask` is True).

**RoPE implementation**: cache $\cos(m\theta_i)$ and $\sin(m\theta_i)$ for every (position, index) pair $(m,i)$ upon initialization:
```python
positions = torch.arange(max_seq_len, device=device)  # shape (max_seq_len)
thetas = self.theta ** (-torch.arange(0, d_k, 2, device=device) / d_k)  # shape (d_k // 2)
angles = positions.unsqueeze(-1) * thetas.unsqueeze(0)
```
In practice, instead of doing many 2×2 matmuls, express the rotation via dot products. Extract even/odd indices of $Q,K$ by reshaping the head dimension $H$ into $(H/2,2)$:
```python
x_pairs = x.reshape(*x.shape[:-1], -1, 2)
x_even = x_pairs[..., 0]
x_odd = x_pairs[..., 1]
```
Compute rotated even/odd positions:
```python
x_out_even = x_even * cos - x_odd * sin
x_out_odd = x_even * sin + x_odd * cos
```
Then interleave by stacking and flattening (`torch.stack()` adds a new dimension):
```python
torch.stack([x_out_even, x_out_odd], dim=-1).flatten(start_dim=-2)
```

**Full attention block implementation**: needs `.reshape()` to expand $D$ into `num_heads × head_dim`; `qkv.unbind()` to split Q/K/V (optional, implementation-dependent); `.transpose()` to swap `num_heads` and `seq_len` dims for the attention computation; then `.transpose()`/`.reshape()` again afterward to recover the original shape:
```python
batch_size, seq_len, _ = x.shape

x_norm = self.norm(x)

qkv = self.qkv_proj(x_norm)  # (batch, seq_len, 3 * d_model)

qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
q, k, v = qkv.unbind(dim=2)  # (batch, seq_len, num_heads, head_dim)
q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
k = k.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
v = v.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
output = scaled_dot_product_attention(q, k, v, mask)  # (batch, num_heads, seq_len, head_dim)

output = output.transpose(1, 2)  # (batch, seq_len, num_heads, head_dim)
output = output.reshape(batch, seq_len, d_model)  # (batch, seq_len, d_model)
output = self.out_proj(output)  # (batch, seq_len, d_model)

return x + output
```
And the core attention function:
```python
def scaled_dot_product_attention(q, k, v, mask):
    """
    k, q: (batch_size, ..., seq_len, d_k)
    v: (batch_size, ..., seq_len, d_v)
    returns o (batch_size, ..., seq_len, d_v)
    """
    d_k = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(~mask, -torch.inf)
    return softmax(scores, dim=-1) @ v
```

### Accounting

#### Model Parameters

- Embedding: $(V,D)$
- Attention: $2D^2 + 2DKH \approx 4D^2$ (for $N=K$, i.e. standard multi-head attention) — $Q$ is $(D,D)$, $K$ is $(D,KH)$, $V$ is $(D,KH)$, $O$ is $(D,D)$
- FFN: $3DF$ — up projection $(D,F)$, gate projection $(D,F)$, down projection $(F,D)$
- Layer norm: $2D$ at each layer (plus the final norm) — pre-attention and pre-FFN layernorm each have $D$ parameters ($\gamma$ for each of the $D$ dimensions)
- Unembedding: $(V,D)$

**Total**: $2VD + L(4D^2+2D+3DF) \approx 2VD + 12LD^2$ (for $F=8D/3$). So **total model parameters $\approx 2VD + 12LD^2$**.

#### Model Activations

- Attention activations: $6BSD + BNS^2$ — layernorm input $(B,S,D)$, layernorm output $(B,S,D)$, Q/K/V outputs $(B,S,D),(B,S,KH),(B,S,KH)$, attention scores $(B,N,S,S)$, attention output $(B,S,D)$
- FFN activations: $2BSD + 2BSF \approx 8BSD$ (for $F=8D/3$) — layernorm input $(B,S,D)$, gate/up projection outputs $(B,S,F)$ each, down-projection output $(B,S,D)$
- **Per-layer activations**: $14BSD + BNS^2$

#### FLOPs in Forward Pass

(Assumes the prefill stage, so $S=T$.)

**Attention**: $8BSD^2 + 4BS^2D$ per layer:
- $Q$ projection: $(B,S,D)\times(D,D) \to 2BSD^2$ FLOPs
- $K$ projection: $(B,S,D)\times(D,KH) \to 2BSDKH \approx 2BSD^2$ (for $K=N$)
- $V$ projection: $(B,S,D)\times(D,KH) \to 2BSDKH \approx 2BSD^2$ (for $K=N$)
- $QK^\top$: $(B,N,S,H)\times(B,N,H,S) \to 2BNS^2H = 2BS^2D$ (since $D=NH$)
- $AV$: $(B,N,S,S)\times(B,N,S,H) \to 2BS^2D$
- $O$ projection: $(B,S,D)\times(D,D) \to 2BSD^2$ FLOPs

**FFN**: $6BSDF \approx 16BSD^2$ (for $F=8D/3$) per layer:
- Up projection: $(B,S,D)\times(D,F) \to 2BSDF$
- Gate projection: $(B,S,D)\times(D,F) \to 2BSDF$
- Down projection: $(B,S,F)\times(F,D) \to 2BSD\dots$

*(Capture ends here — the down-projection FLOPs term and everything after it in the source, listed at the top of this file, were not retrieved.)*
