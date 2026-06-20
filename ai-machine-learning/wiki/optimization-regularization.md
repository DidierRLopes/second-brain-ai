# Optimisation and Regularisation

How to train models efficiently and prevent them from overfitting. Core techniques: gradient-based optimisation, regularisation, and the strategies used when models fail to generalise.

---

## Gradient Descent and SGD

**Gradient Descent**: iteratively update parameters in the direction of the negative gradient of the loss:

```
θ ← θ - η ∇_θ L(θ)
```

where η is the learning rate.

**Full Batch GD**: compute gradient on the entire dataset each step. Exact gradient, but expensive for large datasets. Not used in practice.

**Stochastic Gradient Descent (SGD)**: compute gradient on a **single** randomly chosen example each step:

```
θ ← θ - η ∇_θ L(θ; xᵢ, yᵢ)
```

Noisy but fast. The noise has a regularising effect (implicit regularisation through noise). Provably converges for convex objectives.

**Mini-batch SGD**: gradient on a mini-batch of B examples (typically B=32–2048). The standard in deep learning. Balance between:
- Parallelism (larger batch = more GPU utilisation)
- Gradient noise (smaller batch = more exploration)
- Update frequency (larger batch = fewer updates per epoch)

**SGD with Momentum:**

```
v ← β v - η ∇L
θ ← θ + v
```

Momentum (typically β=0.9) accumulates gradient history, accelerating in consistent directions and damping oscillations. Essential for deep network training.

**Nesterov Momentum**: compute gradient at the "lookahead" position θ + β v rather than at θ. Slightly better convergence in theory and often in practice.

See [[optimizers]] for AdamW, Muon, and learning rate schedules.

---

## Newton's Method and Second-Order Methods

**Newton's Method** uses second-order information (the Hessian of the loss):

```
θ ← θ - H⁻¹ ∇L
```

where H = ∇²L is the Hessian matrix (matrix of second derivatives). This takes into account the curvature of the loss landscape, scaling the gradient update by the local curvature.

**Advantages:**
- Quadratic convergence near the optimum (vs linear for gradient descent)
- Automatically adapts step size to curvature — no learning rate needed
- Exact optimum in one step for quadratic loss

**Disadvantages:**
- H is d×d (d = number of parameters). For a 70B parameter model: completely infeasible.
- Computing and inverting H is O(d³) — prohibitive even for small models
- Hessian can be indefinite (not PSD) far from the optimum

**Quasi-Newton methods** (L-BFGS): approximate H⁻¹ using gradient history, without ever forming H. Compact representation using the last m gradient updates. O(md) memory instead of O(d²). Standard in scientific computing, sometimes used for small ML problems or second-phase fine-tuning.

**K-FAC (Kronecker-Factored Approximate Curvature)**: approximates the Fisher information matrix (which equals the Hessian at the MLE) as a Kronecker product of smaller matrices. Each layer's curvature is approximated by the Kronecker product of input covariance and gradient covariance. Practical second-order method for neural networks.

**Natural Gradient**: updates in the direction of the steepest ascent in distribution space (Fisher-Rao metric), rather than parameter space. Invariant to parameterisation. Foundation of TRPO/PPO in RL and connection to natural policy gradients.

---

## Overfitting and Underfitting

**Underfitting**: model is too simple to capture the patterns in the data.
- Train error: high
- Test error: high (close to train error)
- Cure: more model capacity, better features, more training, reduce regularisation

**Overfitting**: model memorises training data but doesn't generalise to new examples.
- Train error: low
- Test error: much higher than train error (large generalisation gap)
- Cure: regularisation, more data, simpler model, early stopping, dropout

**The validation set**: never use the test set for hyperparameter selection. Use a held-out **validation set** for model selection and hyperparameter tuning. The test set is used once for final evaluation.

**Double descent**: in overparameterised models (parameters >> data points), the test error initially decreases, then increases as model size approaches n (interpolation threshold), then decreases again into the "overparameterised regime." Large modern neural networks operate in this regime — they interpolate training data but still generalise. This apparently contradicts the classical bias-variance picture and is an active research area.

---

## Regularisation Methods

Regularisation adds constraints or penalties to prevent overfitting.

### L2 Regularisation (Weight Decay / Ridge)

Add λ||θ||² to the loss:

```
L_reg = L + λ Σ θᵢ²
```

Effect: parameters are pulled toward zero. Gradient: ∇L_reg = ∇L + 2λθ → update: θ ← θ(1-2λη) - η∇L. The (1-2λη) term is weight decay. Prevents any single weight from becoming too large.

**In transformers**: weight decay is essential. Typically λ = 0.1. Applied to weight matrices but not biases or LayerNorm parameters.

### L1 Regularisation (Lasso)

Add λ||θ||₁ = λ Σ |θᵢ| to the loss. Gradient has sign discontinuity at 0, which creates exact sparsity — some weights become exactly zero. Good for feature selection.

### Dropout

During training, randomly set each neuron's output to 0 with probability p (dropout rate). At test time, scale outputs by (1-p) to maintain expected activation magnitude.

**Intuition**: forces the network to learn redundant representations, prevents co-adaptation of neurons. Each forward pass trains a different sub-network. Equivalent to training an ensemble of exponentially many sub-networks.

Typical rates: p=0.1–0.5 for hidden layers, p=0.1 for attention in transformers. Not commonly used in modern LLMs (replaced by weight decay), but still standard in image models.

### Data Augmentation

Artificially expand training data with label-preserving transformations:
- Images: random crop, flip, rotation, colour jitter, cutout, mixup, cutmix
- Text: synonym replacement, back-translation, paraphrasing
- Audio: time stretching, pitch shifting, noise addition

Data augmentation is the most effective regularisation for computer vision. MixUp creates convex combinations of pairs of examples: x̃ = λxᵢ + (1-λ)xⱼ, ỹ = λyᵢ + (1-λ)yⱼ.

---

## Cross Validation

**The problem**: a single train/test split has high variance — the test set might be unusually easy or hard.

**K-fold cross-validation:**
1. Split data into K folds of equal size
2. For each fold k: train on the other K-1 folds, evaluate on fold k
3. Report mean ± std of performance across K folds

Typical K=5 or K=10. K-fold gives K different train/test splits, averaging out the variance. Computationally K× more expensive.

**Leave-One-Out (LOO)**: K=n (one example per fold). Unbiased but expensive. Rarely used for large datasets.

**Stratified K-fold**: for classification, maintain the same class distribution in each fold. Important for imbalanced datasets.

**Nested cross-validation**: outer loop evaluates models, inner loop tunes hyperparameters. Prevents evaluation contamination from hyperparameter selection.

---

## Early Stopping

Stop training when validation performance stops improving.

**Algorithm**: track validation loss at each epoch. If validation loss hasn't improved in `patience` epochs, stop and restore the weights from the best epoch.

**Why it works**: as training progresses, the model increasingly memorises training data. Validation loss reaches a minimum then starts rising. Early stopping is equivalent to a form of regularisation (limits the effective capacity of the optimised model).

**Practical implementation:**
- Monitor validation metric every N steps (not just epoch end)
- Keep a "best checkpoint" and restore it at the end
- Patience: how many steps/epochs to wait before stopping (typically 3-10 epochs)

**Caveat**: for LLM training, early stopping is uncommon because validation loss generally decreases monotonically with scale. Early stopping is more relevant for fine-tuning (which can overfit in a few epochs).

---

## Gradient Descent Variants

See [[optimizers]] for AdamW, Muon, and learning rate schedules. Brief comparison:

| Method | Memory | Per-step cost | Curvature | Notes |
|--------|--------|--------------|-----------|-------|
| SGD | O(d) | O(d) | No | Simple, often surprisingly good |
| SGD + Momentum | O(2d) | O(d) | No | Standard for CNNs |
| Adam/AdamW | O(3d) | O(d) | Diagonal | Default for transformers |
| L-BFGS | O(md) | O(md) | Approximated | Small models, scientific ML |
| Newton | O(d²) | O(d³) | Full | Infeasible for neural networks |

---

## Transfer Learning

Use a model pre-trained on one task/domain as a starting point for another.

**Why it works**: lower layers learn general features (edges, textures, syntax, semantics) that transfer across tasks. Upper layers learn task-specific patterns. Fine-tuning updates all or some layers to the new task.

**Strategies:**
1. **Feature extraction**: freeze all pre-trained layers, only train a new head. Fast, works when source and target domains are similar.
2. **Fine-tuning**: train all layers, initialised from pre-trained weights. Higher performance, more compute, risk of catastrophic forgetting.
3. **Gradual unfreezing**: start by training only the head, then unfreeze layers from top to bottom. Reduces forgetting.

**Pre-trained foundations of modern ML:**
- CV: ImageNet pre-trained ResNet/EfficientNet → fine-tune on target
- NLP: BERT/GPT pre-trained on web text → fine-tune on downstream NLP
- LLMs: pre-trained on next-token prediction → RLHF/SFT alignment → deployment

The scaling hypothesis argues: enough pre-training creates general capabilities that transfer to almost any task via prompting or lightweight fine-tuning. This is why ChatGPT-like systems work — pre-training provides the knowledge, alignment provides the interface.

---

## Domain Adaptation

Adapting a model trained on a source domain to perform well on a target domain, when there's a **distribution shift** between source and target.

**Types of shift:**
- **Covariate shift**: P(x) changes, P(y|x) unchanged. Importance weighting: weight source examples by P_target(x)/P_source(x).
- **Label shift**: P(y) changes, P(x|y) unchanged. Common in real-world deployment.
- **Concept drift**: P(y|x) itself changes over time.

**Methods:**
- **Domain-adversarial training (DANN)**: train a feature extractor that is useful for the main task but confuses a domain classifier. Forces domain-invariant representations.
- **Fine-tuning on target data**: even a small amount of labelled target data dramatically helps.
- **Self-training**: generate pseudo-labels for unlabelled target data, retrain on them.
- **Prompt tuning**: for LLMs, adapt to new domain with in-context examples without weight updates.

**In LLM training context**: domain adaptation = continued pre-training on domain-specific text (medical records, code, legal documents), followed by SFT and alignment on domain-specific instructions.

---

## Few-Shot and Zero-Shot Learning

**Zero-shot**: perform a task without any task-specific training examples. Requires the model to have relevant knowledge and understand task descriptions from language alone.

**Few-shot**: use a small number of labelled examples (1-100) — not enough for standard fine-tuning but used for rapid adaptation.

**In-context learning (ICL)**: GPT-3 demonstrated that large language models can perform few-shot learning just from examples in the prompt, without any weight updates. The model "conditions" on the demonstrations at inference time.

**Prompt engineering as zero-shot**: instruction following models can generalise to new tasks with appropriate natural language descriptions.

**Meta-learning**: "learning to learn" — train explicitly to generalise to new tasks from few examples. MAML (Model-Agnostic Meta-Learning) finds initialisation θ such that one gradient step on K examples from a new task achieves good performance.

**Prototypical networks**: embed examples into a metric space, classify by nearest prototype (mean embedding of each class). Clean, effective for few-shot image classification.

---

## Dimensionality Reduction

Project high-dimensional data to lower dimensions while preserving structure.

**PCA (Principal Component Analysis)**: find the orthogonal directions of maximum variance. Project onto the top-k eigenvectors of the covariance matrix.

```
Σ = (1/n) XᵀX   (centered data)
Σ = U S Uᵀ      (eigendecomposition)
Z = X U_k        (project onto top-k eigenvectors)
```

Linear, fast (SVD), exact. Optimal for Gaussian data. Whitening: divide by singular values so variance is 1 in each dimension.

**t-SNE**: nonlinear. Models pairwise similarities as probabilities in high and low dimensions, minimises KL divergence between them. Excellent for visualisation (2D/3D), but not good for general dimensionality reduction (not interpretable embedding, non-parametric so can't embed new points).

**UMAP**: similar to t-SNE but faster, more scalable, and preserves global structure better. Default for visualising embeddings.

**Autoencoders**: nonlinear dimensionality reduction. The bottleneck z learns a low-dimensional representation. More expressive than PCA.

**Applications**: visualise embeddings, speed up downstream algorithms, remove noise (since top PCs capture signal, bottom PCs capture noise), feature engineering.

---

## Data Whitening

Whitening (sphering) transforms data to have zero mean and identity covariance:

```
z = Σ^{-1/2} (x - μ)
```

After whitening: E[z] = 0, Cov(z) = I.

**Why**: removes correlations between features and normalises variance. PCA whitening first decorrelates (PCA), then scales by eigenvalues. ZCA whitening additionally rotates back to the original space (minimises change from original data).

**Use in neural networks**: whitening input features accelerates training (loss landscape better conditioned). BatchNorm can be seen as approximate whitening of hidden layer activations during training.

**PCA whitening for images**: used in early deep learning (AlexNet era) as preprocessing. Largely replaced by data augmentation + BatchNorm, but still theoretically motivated.
