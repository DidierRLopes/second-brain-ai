# Classical Machine Learning

The pre-deep-learning toolkit. Still broadly applicable and often underrated — SVMs work extremely well on structured/tabular data, boosting dominates Kaggle competitions, and the bias-variance intuitions from these methods underpin all of deep learning theory.

---

## Supervised vs Unsupervised Learning

**Supervised learning**: learn a mapping f: X → Y from labelled examples {(x_i, y_i)}. The goal is to generalise — predict y for unseen x.

- Regression: Y is continuous (predict house price)
- Classification: Y is discrete (predict spam/not-spam)
- Key challenge: overfitting to training labels

**Unsupervised learning**: find structure in unlabelled data X without knowing the "right answer".

- Clustering: group similar examples
- Density estimation: learn p(x)
- Dimensionality reduction: find compact representations
- Key challenge: the objective is unclear — what counts as "good" structure?

**Semi-supervised learning**: combine small amounts of labelled data with large amounts of unlabelled data. Unlabelled data helps regularise and provides structure, labelled data provides the target signal.

**Self-supervised learning** (the foundation of modern DL): design labels automatically from the data itself — next-token prediction, masked image modelling, contrastive objectives. The model must understand the data to perform the self-supervised task.

---

## Linear Regression

The simplest model: predict y as a linear function of features x.

```
ŷ = wᵀx + b = w₁x₁ + w₂x₂ + ... + wₐxₐ + b
```

**Ordinary Least Squares (OLS)**: minimise mean squared error:

```
L(w) = (1/n) ||Xw - y||² = (1/n) Σᵢ (ŷᵢ - yᵢ)²
```

**Closed-form solution**: set ∂L/∂w = 0:

```
w* = (XᵀX)⁻¹ Xᵀy
```

This is the normal equations. Requires XᵀX to be invertible (full column rank), which fails when features are collinear or n < d.

**Assumptions**: linearity, homoscedasticity (constant variance), independence, normally distributed residuals (for inference). Violations don't necessarily break prediction but do break statistical tests.

**Ridge Regression** (L2 regularisation): adds λ||w||² to prevent overfitting and handle ill-conditioned XᵀX:

```
w* = (XᵀX + λI)⁻¹ Xᵀy
```

**Lasso** (L1): adds λ||w||₁, which induces sparsity — some weights exactly zero. Useful for feature selection.

**Elastic Net**: combines L1 + L2.

Linear regression is the starting point for understanding gradient descent, regularisation, and the bias-variance tradeoff. The analytical solution also demonstrates why model complexity (too few/many features) matters.

---

## K-Nearest Neighbours (KNN)

The simplest non-parametric classifier: to classify a new point, find the k nearest training examples and take a majority vote (classification) or average (regression).

```
ŷ(x) = (1/k) Σ_{i ∈ kNN(x)} yᵢ
```

Distance metric choices: Euclidean, Manhattan, Cosine, Mahalanobis.

**Properties:**
- **Non-parametric**: no training, just memorise data
- **Lazy learning**: all computation at inference time
- **Decision boundary**: can approximate any decision boundary as k→1 (but overfits)
- **k=1**: Bayes error rate approaches 2× optimal (asymptotically) — surprisingly powerful
- **Large k**: smooth, lower variance but higher bias

**Problems:**
- **Curse of dimensionality**: in high dimensions, all points are roughly equidistant from each other. Nearest neighbours become meaningless.
- **Computational cost**: O(nd) per query for brute force. Approximate NN (FAISS, HNSW) is essential at scale.
- **Storage**: must keep all training data

KNN is still used in recommendation systems, anomaly detection, and as a baseline. The curse of dimensionality is why dimensionality reduction (PCA, autoencoders) is often applied before KNN.

---

## Support Vector Machines (SVMs)

SVMs find the **maximum-margin hyperplane** separating two classes.

### Hard-Margin SVM (Linearly Separable)

Find hyperplane wᵀx + b = 0 such that:
- yᵢ(wᵀxᵢ + b) ≥ 1 for all training points
- Maximise margin = 2/||w||

Equivalent to: minimise ½||w||² subject to yᵢ(wᵀxᵢ + b) ≥ 1.

Support vectors are the training points that lie exactly on the margin boundary. The decision boundary depends **only** on support vectors — all other points can be removed without changing the classifier.

### Soft-Margin SVM (Non-Separable)

Allow some misclassification with slack variables ξᵢ:

```
minimise ½||w||² + C Σᵢ ξᵢ
subject to yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

C controls the tradeoff: large C → small margin but fewer violations; small C → large margin but more violations.

Hinge loss: L = max(0, 1 - yᵢ(wᵀxᵢ + b)). The SVM primal is equivalent to regularised hinge loss minimisation.

### The Kernel Trick

SVMs become powerful in the dual formulation, where the objective depends on data only through inner products xᵢᵀxⱼ. Replace with a kernel function K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ) — implicitly mapping to a higher-dimensional feature space without computing φ explicitly.

Common kernels:
- **Linear**: K(x,x') = xᵀx'
- **Polynomial**: K(x,x') = (xᵀx' + c)^d
- **RBF / Gaussian**: K(x,x') = exp(-γ||x-x'||²) → infinite-dimensional feature space
- **Sigmoid**: K(x,x') = tanh(αxᵀx' + c)

The RBF kernel SVM can learn any smooth boundary and is still competitive with neural networks on structured data when samples are few (< 10K).

**SVMs in 2025**: largely displaced by gradient boosting and neural networks for tabular data, but still taught as the canonical example of maximum-margin classifiers and the kernel trick.

---

## Decision Trees

Decision trees partition the feature space with axis-aligned splits, recursively:
1. Choose the feature and threshold that best splits the data
2. Recurse on each side until a stopping criterion is met

**Split criteria:**
- **Gini impurity**: G = Σ_k p_k(1-p_k) = 1 - Σ_k p_k² (used in CART)
- **Information gain (entropy)**: H = -Σ_k p_k log p_k (used in ID3, C4.5)
- **Variance reduction** (regression): MSE of subsets

At each node, choose the split that maximally reduces impurity.

**Advantages**: interpretable, handles mixed feature types, no scaling needed, automatic feature selection.

**Disadvantages**: prone to overfitting (a deep tree memorises training data), sensitive to small changes in data ("unstable"), axis-aligned splits miss diagonal patterns.

**Regularisation**: max depth, min samples per leaf, min impurity decrease, cost complexity pruning.

---

## Ensembles, Bagging, and Boosting

Single models have high variance or high bias. Ensembles combine multiple models to reduce error.

### Bagging (Bootstrap Aggregating)

Train multiple independent models on different bootstrap samples (random sample with replacement) of the training data. Aggregate by voting (classification) or averaging (regression).

- Reduces variance without significantly increasing bias
- Models are trained in parallel
- Each model sees ~63.2% of training data (probability of being included = 1-(1-1/n)^n → 1-1/e)

**Random Forest** = bagging + random feature subsets at each split. At each node, consider only √p (classification) or p/3 (regression) randomly chosen features. This decorrelates the trees, reducing variance further.

Random Forests are extremely robust, handle missing data, give feature importance, and are competitive with deep learning on tabular data. They're the "ensemble duct tape" — throw them at any problem as a strong baseline.

### Boosting

Train models **sequentially**, each focusing on examples the previous models got wrong.

**AdaBoost** (Freund & Schapire, 1995):
1. Initialise weights wᵢ = 1/n
2. Train weak learner h_t on weighted data
3. Compute error ε_t = Σᵢ wᵢ 𝟙[h_t(xᵢ) ≠ yᵢ]
4. Compute learner weight α_t = ½ log((1-ε_t)/ε_t)
5. Update: wᵢ ← wᵢ exp(-α_t yᵢ h_t(xᵢ)), renormalise
6. Final: H(x) = sign(Σ_t α_t h_t(x))

AdaBoost minimises exponential loss. The "boosting" idea: even weak learners (only slightly better than random) can be combined into a strong learner.

**Gradient Boosting** (Friedman, 2001): fit each new learner to the **residuals** (negative gradient of the loss) of the current ensemble:

```
F_m(x) = F_{m-1}(x) + ν h_m(x)
```

where h_m fits the negative gradients -[∂L/∂F(xᵢ)] at xᵢ.

**XGBoost / LightGBM / CatBoost**: highly optimised implementations of gradient boosting with regularisation, efficient sparse feature handling, and fast training. These dominate structured/tabular ML competitions. Key features:
- Second-order Taylor expansion for loss approximation
- L1/L2 regularisation on tree weights
- Column and row subsampling
- Efficient histogram-based split finding (LightGBM)
- Native categorical handling (CatBoost)

---

## Clustering

### K-Means

Partition n points into k clusters to minimise within-cluster variance (inertia):

```
min_{C,μ} Σ_{j=1}^k Σ_{x∈C_j} ||x - μ_j||²
```

**Lloyd's algorithm**:
1. Initialise k centroids (random or K-means++)
2. Assign each point to its nearest centroid
3. Recompute centroids as mean of assigned points
4. Repeat until convergence

**K-means++** initialisation: choose centroids proportional to distance squared from existing centroids. Avoids poor random initialisation, gives O(log k) approximation guarantee.

**Choosing k**: elbow method (plot inertia vs k), silhouette score, gap statistic. No universally right answer.

**Limitations**: assumes spherical clusters, sensitive to scale (normalise features), requires specifying k, can get stuck in local minima (run multiple times).

**Other clustering algorithms:**
- **DBSCAN**: density-based, finds arbitrary shapes, identifies outliers, doesn't require k
- **Hierarchical clustering**: builds a dendrogram, no k required, but O(n²) or O(n² log n)
- **GMM (Gaussian Mixture Models)**: soft clustering with probabilistic assignments, EM algorithm
- **Spectral clustering**: uses graph Laplacian eigenvectors, handles non-convex clusters

---

## Precision, Recall, F1, AUC-ROC

For binary classification (and extended to multi-class):

```
Precision = TP / (TP + FP)   — of all predicted positive, how many actually positive
Recall    = TP / (TP + FN)   — of all actual positive, how many did we catch (sensitivity)
F1        = 2 * Precision * Recall / (Precision + Recall)   — harmonic mean
```

**Precision vs Recall tradeoff**: changing the classification threshold moves along a precision-recall curve. High threshold → high precision, low recall; low threshold → low precision, high recall. F1 summarises both.

**ROC curve**: plots True Positive Rate (Recall) vs False Positive Rate at all thresholds.

```
TPR = TP / (TP + FN) = Recall
FPR = FP / (FP + TN)
```

**AUC-ROC** (Area Under the Curve): probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example. AUC = 1.0: perfect classifier. AUC = 0.5: random.

**When each metric matters:**
- Precision matters when false positives are costly (spam detection: don't put good email in spam)
- Recall matters when false negatives are costly (cancer screening: don't miss real cases)
- F1 balances both
- AUC is useful for ranking/sorting and when classes are imbalanced

**For imbalanced datasets**: accuracy is misleading (99% accuracy by always predicting majority class). Use F1, precision-recall AUC, or balanced accuracy instead.
