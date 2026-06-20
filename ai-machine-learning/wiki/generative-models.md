# Generative Models

Generative models learn the distribution of training data p_data(x) and can sample new examples from it. The major families: GANs (adversarial training), VAEs (variational inference), Diffusion models (denoising), and Flow Matching (ODE-based transport).

---

## GANs (Generative Adversarial Networks)

GANs (Goodfellow et al., 2014) pit two networks against each other in a minimax game:
- **Generator G(z; θ_G)**: maps noise z ~ p(z) to fake samples
- **Discriminator D(x; θ_D)**: distinguishes real from fake

### Training Objective

```
min_G max_D  E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
```

At Nash equilibrium: G generates samples indistinguishable from real data, and D outputs 1/2 everywhere (can't distinguish). The optimal generator distribution matches the true data distribution.

### Training Procedure

Alternate between:
1. Update D: maximise log D(x) + log(1-D(G(z))) — make it better at distinguishing
2. Update G: minimise log(1-D(G(z))) — or equivalently maximise log D(G(z)) (non-saturating)

The non-saturating loss for G avoids gradient saturation early in training when D easily rejects all fake samples.

### GAN Problems

**Mode collapse**: G generates only a few modes of the distribution, ignoring the rest. The generator finds a single "cheat" that fools D.

**Training instability**: the minimax game is notoriously hard to balance. G and D need to improve at the same rate.

**Vanishing gradients**: when D is too powerful, log(1-D(G(z))) ≈ 0 and G gets no gradient signal.

### Improvements

**DCGAN** (2015): convolutional architecture with batch norm, stable training on images.

**Wasserstein GAN (WGAN)** (2017): replace JS divergence with Wasserstein-1 distance (Earth Mover's Distance). The critic (renamed from discriminator) is 1-Lipschitz constrained (gradient clipping or spectral normalisation). This removes mode collapse and provides meaningful loss throughout training.

**WGAN-GP**: gradient penalty term instead of weight clipping: `λ E[(||∇_x̂ D(x̂)||₂ - 1)²]`.

**StyleGAN / StyleGAN2/3** (NVIDIA): state-of-the-art image synthesis. Progressively growing, style injection at each resolution, mapping network from latent to "style" space. StyleGAN2 removes progressive growing in favour of skip connections and residuals.

**BigGAN**: class-conditional, large-scale, spectral normalisation.

GANs have largely been supplanted by diffusion models for high-quality image synthesis, but remain relevant in video generation and settings requiring fast sampling.

---

## VAEs and the ELBO

VAEs (Kingma & Welling, 2013) are directed latent variable models with learned approximate inference.

### Setup

Assume data x is generated from a latent variable z:
- **Prior**: p(z) = N(0, I)
- **Likelihood**: p_θ(x|z) (decoder network)
- **True posterior**: p_θ(z|x) = p_θ(x|z)p(z) / p_θ(x) — intractable

We introduce an **approximate posterior** (encoder): q_φ(z|x) = N(μ_φ(x), σ²_φ(x))

### The ELBO

We want to maximise log p_θ(x). Using Jensen's inequality:

```
log p_θ(x) = log E_{z~q_φ(z|x)} [p_θ(x,z) / q_φ(z|x)]
           ≥ E_{z~q_φ(z|x)} [log p_θ(x,z) / q_φ(z|x)]
           = E_{z~q_φ(z|x)} [log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

This lower bound is the **Evidence Lower BOund (ELBO)**:

```
ELBO = E_{z~q_φ}[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
       = Reconstruction term - KL regularisation
```

Maximising the ELBO:
- Forces the decoder to reconstruct x well (reconstruction term)
- Forces the posterior q_φ(z|x) to stay close to the prior p(z) (KL term)
- The KL has a closed form for Gaussians: KL(N(μ,σ²) || N(0,I)) = ½ Σ(μ² + σ² - log σ² - 1)

### Reparameterisation Trick

The ELBO requires E_{z~q_φ(z|x)}[log p_θ(x|z)]. We need to backprop through the sampling operation z ~ N(μ_φ(x), σ²_φ(x)).

Solution: reparameterise z = μ_φ(x) + σ_φ(x) ⊙ ε, where ε ~ N(0, I). Now z is a deterministic function of φ and a fixed noise sample — gradients flow through μ and σ.

### Decoder and Generation

- Gaussian decoder: p_θ(x|z) = N(μ_θ(z), I) → reconstruction loss is MSE
- Bernoulli decoder: binary cross-entropy
- Categorical (discrete VAE): used in DALL-E, VQ-VAE

**VQ-VAE** (Vector Quantised VAE): discrete latent space. Encoder output is mapped to nearest codebook vector. Used to compress images into discrete tokens for transformer-based generation (e.g., DALL-E, image tokens in LLMs).

### VAE vs GAN

| | VAE | GAN |
|--|-----|-----|
| Training | Stable | Unstable |
| Sample quality | Blurry (Gaussian decoder) | Sharp |
| Latent space | Smooth, interpolatable | Less structured |
| Density estimation | Yes (via ELBO) | No |
| Mode coverage | Good | Mode collapse |

---

## Score Function and Score Matching

The **score function** of a distribution p(x) is:

```
s(x) = ∇_x log p(x)
```

It points in the direction of increasing probability density — toward the high-density regions of the distribution.

**Score matching** (Hyvärinen, 2005): instead of learning the density directly, learn the score function. This avoids the normalisation constant (which is typically intractable).

The score matching objective:

```
L = E_x [ ||s_θ(x) - ∇_x log p_data(x)||² ]
```

Can be rewritten (via integration by parts) to avoid needing the true score:

```
L = E_x [ tr(∇_x s_θ(x)) + ½ ||s_θ(x)||² ]
```

**Denoising Score Matching** (Vincent, 2011): add noise to data, train a denoising network to predict the noise. This is equivalent to score matching with a Parzen window estimator.

**Connection to diffusion models**: diffusion models are score-based models. The neural network at each noise level learns ∇_x log p(x_t | x_0) — the score of the noisy distribution. Langevin dynamics then uses the score to generate samples:

```
x_{t+1} = x_t + (δ/2) ∇_x log p(x_t) + √δ · ε
```

---

## Diffusion Models

Diffusion models (Ho et al., 2020; Song et al., 2020) are currently the dominant approach for image/video/audio generation.

### Forward Process (Noising)

Add Gaussian noise incrementally over T steps, gradually turning data into pure noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

where β_t is a noise schedule (linear, cosine, etc.). After T steps, x_T ≈ N(0, I).

**Key closed form**: can jump directly from x_0 to x_t without iterating:

```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)
```

where ᾱ_t = Π_{s=1}^t (1 - β_s). This allows efficient training without running the full T-step forward process.

Equivalently: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε, where ε ~ N(0,I).

### Reverse Process (Denoising / DDPM)

The reverse process p_θ(x_{t-1} | x_t) removes noise step by step. We parameterise it as:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**DDPM** (Denoising Diffusion Probabilistic Models, Ho et al., 2020) simplifies: instead of predicting μ directly, predict the noise ε_θ(x_t, t), then compute μ from it:

```
μ_θ(x_t, t) = (1/√α_t) (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
```

Training loss simplifies to noise prediction:

```
L_simple = E_{x_0, ε, t} [ ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||² ]
```

Just train a U-Net to predict the noise added at step t. This is equivalent to score matching.

**DDPM generation**: start from x_T ~ N(0,I), run the reverse process T=1000 steps. Slow (1000 NFEs per sample).

### DDIM (Denoising Diffusion Implicit Models)

DDIM (Song et al., 2020) reinterprets the diffusion process as a **non-Markovian** process with the same marginals but faster sampling. The reverse step:

```
x_{t-1} = √ᾱ_{t-1} · (x_t - √(1-ᾱ_t) ε_θ) / √ᾱ_t
         + √(1-ᾱ_{t-1} - σ_t²) · ε_θ
         + σ_t · ε
```

When σ_t = 0: fully deterministic. This allows:
- **Skipping steps**: only compute at a subset of timesteps (e.g., 50 instead of 1000)
- **Deterministic sampling**: same x_T always gives same x_0 (useful for editing/interpolation)
- **Inversion**: encode x_0 → x_T exactly (for image editing)

DDIM reduces sampling from 1000 NFEs to 50 with comparable quality.

### Diffusion as SDEs

Song et al. (2021) unified diffusion models as **Stochastic Differential Equations**:

**Forward SDE** (adding noise):
```
dx = f(x, t) dt + g(t) dW
```
- For DDPM: f(x,t) = -½ β(t) x, g(t) = √β(t)
- dW is Brownian motion (Wiener process)

**Reverse SDE** (denoising):
```
dx = [f(x,t) - g(t)² ∇_x log p_t(x)] dt + g(t) dW̄
```

The term ∇_x log p_t(x) is the score function — the same quantity learned by score matching.

**Probability Flow ODE** (deterministic reverse):
```
dx = [f(x,t) - ½ g(t)² ∇_x log p_t(x)] dt
```

This ODE has the same marginals as the reverse SDE but is deterministic. Solving it with numerical ODE solvers gives faster sampling and exact likelihood computation.

---

## Flow Matching

Flow Matching (Lipman et al., 2022; Liu et al., 2022 "Rectified Flow") is an alternative to diffusion that directly learns an ODE that transports noise to data.

### Continuous Normalising Flows (CNF)

Learn a time-dependent vector field v_θ(x, t) such that integrating the ODE:

```
dx/dt = v_θ(x, t), with x(0) ~ p_0 (noise), x(1) ~ p_1 (data)
```

transforms noise to data. The change in log-likelihood:

```
d log p(x(t))/dt = -tr(∂v_θ/∂x)
```

**Training CNFs** requires computing tr(∂v_θ/∂x) (expensive). Flow matching avoids this.

### Flow Matching Objective

**Conditional flow matching**: define a per-sample flow from x_0 ~ p_0 to x_1 = x_data:

```
x_t = (1-t) x_0 + t x_1   (straight-line interpolation)
```

The conditional vector field: u_t(x | x_1) = x_1 - x_0 (constant velocity field — straight line).

Train a network to match this field:

```
L_FM = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - u_t(x_t | x_1)||² ]
```

**Advantages over diffusion:**
- **Straight paths**: optimal transport interpolation → fewer NFEs at inference
- **Simpler training**: no noise schedule tuning, no variance-exploding/preserving choices
- **Exact target**: the conditional target is deterministic given (x_0, x_1), no need to estimate scores
- **Faster generation**: can solve the ODE in 1-10 steps

**Rectified Flow** (Liu et al.) emphasises making the paths as straight as possible via iterative "reflow": generate (x_0, x_1) pairs from the trained flow, retrain on straight paths between them. Converges to 1-step generation.

**Stable Diffusion 3, FLUX, Sora** are all flow matching models. Flow matching has largely replaced diffusion for new high-quality image/video generation systems.

---

## Classifier-Free Guidance (CFG)

CFG (Ho & Salimans, 2021) is the key technique for conditioning generation on text or class labels, without training a separate classifier.

### The Problem

To condition a diffusion/flow model on prompt c (e.g., "a photo of a cat"), we want samples from p(x | c) rather than p(x). Bayes: ∇_x log p(x|c) = ∇_x log p(x) + ∇_x log p(c|x). The second term (score of a classifier) can be large and informative.

**Classifier guidance** trains a separate classifier p(c|x_t) on noisy images and uses its gradient. Effective but cumbersome (classifier must handle all noise levels).

### Classifier-Free Guidance

Train a **single network** that can operate either conditionally or unconditionally, by randomly dropping the conditioning signal during training (replace c with null token ∅ with probability p_uncond, typically 10-20%).

At inference, compute:

```
ε_guided = ε_θ(x_t, ∅) + w · (ε_θ(x_t, c) - ε_θ(x_t, ∅))
```

Or equivalently, the guided score:

```
ŝ(x_t, c) = s(x_t, ∅) + w · (s(x_t, c) - s(x_t, ∅))
```

where w > 1 is the **guidance scale** (CFG weight). This extrapolates the conditional score away from the unconditional score.

**Effect of guidance scale w:**
- w = 0: pure unconditional generation
- w = 1: standard conditional generation
- w > 1: high prompt adherence, but reduced diversity and potential artefacts
- w = 7-9: typical for high-quality image generation

**Why it works**: extrapolating the conditional score is equivalent to sampling from a sharpened distribution p(x|c)^(1/T) with temperature 1/w — high guidance sharpens the distribution, increasing mode probability at the cost of diversity.

**Negative prompts**: some implementations use a negative prompt c_neg instead of unconditional: ε_guided = ε_θ(x_t, c_neg) + w · (ε_θ(x_t, c) - ε_θ(x_t, c_neg)).

CFG is universal across diffusion models (DALLE-2, Stable Diffusion, Imagen) and flow matching models (Stable Diffusion 3, FLUX). It's essentially the standard inference-time conditioning technique.
