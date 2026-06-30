# Optimization: Backpropagation

**Source:** https://cs231n.github.io/optimization-2/
**Authors:** Stanford CS231n staff (Andrej Karpathy et al.), Stanford University
**Course:** CS231n: Deep Learning for Computer Vision

## Introduction / Problem Statement

The core problem: given a function \(f(x)\) where \(x\) is a vector of inputs, compute the gradient \(\nabla f(x)\). In neural networks, \(f\) corresponds to the loss function \(L\), and the inputs consist of training data \((x_i, y_i)\) and the weights/biases \(W, b\). Training data is treated as fixed; in practice we usually only compute gradients with respect to the parameters (\(W, b\)) for use in parameter updates, though the gradient on \(x_i\) can be useful for visualization/interpretation.

Backpropagation is presented as a way of computing gradients of expressions through recursive application of the **chain rule**, framed as backward flow in real-valued circuits.

## Simple Expressions and Interpreting the Gradient

For \(f(x,y) = xy\): \(\frac{\partial f}{\partial x} = y\), \(\frac{\partial f}{\partial y} = x\).

The derivative is defined via the limit \(\frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}\). Key interpretive point: "the derivative on each variable tells you the sensitivity of the whole expression on its value."

Worked numeric example: if \(x = 4, y = -3\), then \(f(x,y) = -12\) and \(\frac{\partial f}{\partial x} = -3\). Increasing \(x\) by a tiny amount \(h\) decreases the output by \(3h\) (negative sign). Since \(\frac{\partial f}{\partial y} = 4\), increasing \(y\) by \(h\) increases output by \(4h\). This follows from rearranging \(f(x+h) = f(x) + h\frac{df(x)}{dx}\).

The gradient \(\nabla f\) is the vector of partial derivatives: \(\nabla f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}] = [y, x]\).

Addition: \(f(x,y) = x+y \rightarrow \frac{\partial f}{\partial x} = 1, \frac{\partial f}{\partial y} = 1\) — constant regardless of input values, unlike multiplication.

Max: \(f(x,y) = \max(x,y) \rightarrow \frac{\partial f}{\partial x} = \mathbb{1}(x \geq y)\), \(\frac{\partial f}{\partial y} = \mathbb{1}(y \geq x)\). The (sub)gradient is 1 on the larger input and 0 on the other. Example: if \(x=4, y=2\), the max is 4 and the function is insensitive to small changes in \(y\) — gradient on \(y\) is zero. Derivatives only describe infinitesimally small changes, not large ones.

## Compound Expressions, Chain Rule, Backpropagation

Worked example: \(f(x,y,z) = (x+y)z\), decomposed into \(q = x+y\) and \(f = qz\). Known local derivatives: \(\frac{\partial f}{\partial q} = z\), \(\frac{\partial f}{\partial z} = q\), \(\frac{\partial q}{\partial x} = 1\), \(\frac{\partial q}{\partial y} = 1\). Chain rule combines them via multiplication: \(\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q}\frac{\partial q}{\partial x}\).

Concrete numeric walkthrough (code from the notes):

```python
# set some inputs
x = -2; y = 5; z = -4

# forward pass
q = x + y   # q becomes 3
f = q * z   # f becomes -12

# backward pass (in reverse order)
dfdz = q          # df/dz = q, gradient on z becomes 3
dfdq = z          # df/dq = z, gradient on q becomes -4
dqdx = 1.0
dqdy = 1.0
dfdx = dfdq * dqdx  # chain rule
dfdy = dfdq * dqdy
```

This yields gradients `[dfdx, dfdy, dfdz]`. The notes adopt a shorthand going forward: write `dq` instead of `dfdq`, always assuming the gradient is with respect to the final output.

This is visualized as a circuit diagram: inputs x=-2, y=5 feed a `+` gate producing q=3, which together with z=-4 feeds a `*` gate producing f=-12. The forward pass (green) computes left-to-right; the backward pass (red) propagates gradients right-to-left: gradient on f is 1, on q is -4, on z is 3, on x is -4, on y is -4.

## Intuitive Understanding of Backpropagation

Backpropagation is described as "a beautifully local process." Every gate in a circuit diagram, given its inputs, can immediately compute (1) its output value and (2) the *local* gradient of its output with respect to its inputs — independently of the rest of the circuit. Only later, during the backward pass, does the gate learn the gradient of its output with respect to the final circuit output, and the chain rule says to multiply that incoming gradient into each locally-computed gradient.

> "This extra multiplication (for each input) due to the chain rule can turn a single and relatively useless gate into a cog in a complex circuit such as an entire neural network."

Worked intuition using the running example: the add gate received inputs [-2, 5] and computed output 3; its local gradient for both inputs is +1 (it's an add gate). During the backward pass, the add gate learns the gradient on its output is -4 (from the multiply gate above it). Anthropomorphizing: the circuit "wants" the add gate's output to be lower, with "force" 4. The add gate multiplies that -4 into both of its local gradients (1 each), giving gradient -4 on both x and y. The intended effect: if x, y decrease (following their negative gradient), the add gate's output decreases, which in turn makes the multiply gate's output increase.

> "Backpropagation can thus be thought of as gates communicating to each other (through the gradient signal) whether they want their outputs to increase or decrease (and how strongly), so as to make the final output value higher."

## Modularity: Sigmoid Example

Worked example of a 2D neuron with sigmoid activation:

\[f(w,x) = \frac{1}{1+e^{-(w_0x_0 + w_1x_1 + w_2)}}\]

Four additional unary gate derivatives introduced (beyond add/mul/max):
- \(f(x) = 1/x \rightarrow df/dx = -1/x^2\)
- \(f_c(x) = c + x \rightarrow df/dx = 1\) (translate by constant c)
- \(f(x) = e^x \rightarrow df/dx = e^x\)
- \(f_a(x) = ax \rightarrow df/dx = a\) (scale by constant a)

Circuit walkthrough with concrete numbers: weights \(w = [2, -3, -3]\), inputs \(x = [-1, -2]\). Forward pass through the circuit shows intermediate values: w0=2.00, x0=-1.00 multiply to -2.00; w1=-3.00, x1=-2.00 multiply to 6.00; these combine with w2=-3.00 via two add gates to reach 1.00, then negated to -1.00, exponentiated (`exp`) to 0.37, plus 1 to 1.37, then `1/x` to give final output **0.73**.

Sigmoid function \(\sigma(x) = \frac{1}{1+e^{-x}}\) has a derivative that simplifies remarkably:

\[\frac{d\sigma(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = \left(\frac{1+e^{-x}-1}{1+e^{-x}}\right)\left(\frac{1}{1+e^{-x}}\right) = (1-\sigma(x))\,\sigma(x)\]

Numeric check against the circuit: the sigmoid expression receives input 1.0 and outputs 0.73 on the forward pass; the local gradient is then \((1-0.73) \times 0.73 \approx 0.2\), matching the value computed step-by-step through the four individual gates in the circuit diagram — but obtainable via one simple, efficient, numerically-better-behaved expression. This motivates grouping multiple gates into a single composite gate when convenient.

Code for backprop through this neuron:

```python
w = [2,-3,-3]  # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0]*x[0] + w[1]*x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot))  # sigmoid function

# backward pass through the neuron
ddot = (1 - f) * f                          # gradient on dot, via sigmoid derivative
dx = [w[0] * ddot, w[1] * ddot]              # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot]  # backprop into w
```

**Implementation protip — staged backpropagation**: break the forward pass into stages with easily-derived local gradients (e.g., the intermediate `dot` variable above), then successively compute `ddot`, `dw`, `dx` in reverse order.

## Backprop in Practice: Staged Computation

Worked "useless but illustrative" example:

\[f(x,y) = \frac{x+\sigma(y)}{\sigma(x)+(x+y)^2}\]

With \(x=3, y=-4\), the forward pass is staged explicitly:

```python
x = 3; y = -4

sigy = 1.0 / (1 + math.exp(-y))   # (1) sigmoid in numerator
num = x + sigy                     # (2) numerator
sigx = 1.0 / (1 + math.exp(-x))   # (3) sigmoid in denominator
xpy = x + y                        # (4)
xpysqr = xpy**2                    # (5)
den = sigx + xpysqr                # (6) denominator
invden = 1.0 / den                 # (7)
f = num * invden                   # (8) done!
```

Backward pass, staged in exact reverse, each line a local-gradient times incoming gradient:

```python
# backprop f = num * invden                                        (8)
dnum = invden
dinvden = num
# backprop invden = 1.0 / den                                      (7)
dden = (-1.0 / (den**2)) * dinvden
# backprop den = sigx + xpysqr                                     (6)
dsigx = (1) * dden
dxpysqr = (1) * dden
# backprop xpysqr = xpy**2                                         (5)
dxpy = (2 * xpy) * dxpysqr
# backprop xpy = x + y                                             (4)
dx = (1) * dxpy
dy = (1) * dxpy
# backprop sigx = 1.0 / (1 + math.exp(-x))                         (3)
dx += ((1 - sigx) * sigx) * dsigx   # note += !
# backprop num = x + sigy                                          (2)
dx += (1) * dnum
dsigy = (1) * dnum
# backprop sigy = 1.0 / (1 + math.exp(-y))                        (1)
dy += ((1 - sigy) * sigy) * dsigy
```

Two key practical notes called out explicitly:

1. **Cache forward pass variables** — variables computed during the forward pass (e.g., `sigx`, `sigy`, `xpy`, `den`) are needed during the backward pass; structure code to cache them, or recompute them wastefully if necessary.
2. **Gradients add up at forks** — because `x` and `y` are each used in multiple places in the forward expression, the backward pass must use `+=` rather than `=` to accumulate their gradients (otherwise earlier contributions get overwritten). This follows from the multivariable chain rule: when a variable branches/forks into different parts of the circuit, the gradients flowing back to it **add**.

## Patterns in Backward Flow

The three most common gates — **add, mul, max** — have simple, intuitive backward-pass interpretations, illustrated with an example circuit (inputs x=3, y=-4, z=2, w=-1; a `max` gate combines two upstream products; final output via `*` and `+` gates reaches -20.00 with an incoming gradient of 2.00 from above).

- **Add gate distributes gradient.** It takes the gradient on its output and distributes it *equally and unchanged* to all of its inputs, regardless of their forward-pass values — because its local gradient is exactly +1.0 for every input. In the example circuit, the `+` gate routes a gradient of 2.00 to both of its inputs unchanged.

- **Max gate routes gradient.** Unlike add, max routes the gradient *unchanged to exactly one* of its inputs — the one that had the highest value during the forward pass — since its local gradient is 1.0 for the max input and 0.0 for the rest. In the example, the max operation routed gradient 2.00 to variable **z** (which was larger than **w** on the forward pass); the gradient on **w** remains zero.

- **Multiply gate switches (and scales) gradient.** Its local gradients equal the *other* input's value (the inputs "switched"), multiplied by the gradient on the output via chain rule. In the example, the gradient on **x** is -8.00, computed as -4.00 × 2.00 (the other input's value times the incoming gradient).

**Unintuitive consequence of the multiply gate**: if one input is very small and the other very large, the multiply gate assigns a relatively huge gradient to the small input and a tiny gradient to the large input. Consequence for linear classifiers computing \(w^Tx_i\): the scale of the input data directly affects the magnitude of the gradient on the weights. Concrete example given: if all input examples \(x_i\) are multiplied by 1000 during preprocessing, the gradient on the weights becomes 1000× larger, requiring a correspondingly lower learning rate to compensate. This is offered as a reason data preprocessing matters, sometimes in subtle ways.

## Gradients for Vectorized Operations

All the same concepts extend to matrices/vectors, but require care with dimensions and transposes.

**Matrix-Matrix multiply gradient** (the "most tricky" vectorized operation):

```python
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# gradient on D from upstream
dD = np.random.randn(*D.shape)   # same shape as D
dW = dD.dot(X.T)
dX = W.T.dot(dD)
```

**Tip: use dimension analysis.** You don't need to memorize the `dW`/`dX` formulas — they can be re-derived from shape constraints. Concretely: `X` is [10×3], `dD` is [5×3]; since `dW` must match `W`'s shape [5×10], the only way to combine `dD` and `X` via matrix multiplication to get a [5×10] result is `dD.dot(X.T)`. Similarly for `dX = W.T.dot(dD)`.

**Tip: work with small, explicit examples.** Recommended workflow for unfamiliar vectorized gradients: write out a minimal explicit example, derive the gradient by hand, then generalize the pattern to the efficient vectorized form.

(References Erik Learned-Miller's notes on matrix/vector derivatives, linked at http://cs231n.stanford.edu/vecDerivs.pdf.)

## Summary (as stated in the notes)

- Developed intuition for what gradients mean, how they flow backward through a circuit, and how they communicate which parts of the circuit should increase/decrease (and how strongly) to raise the final output.
- Emphasized **staged computation**: break a function into modules with easily-derived local gradients, then chain them via the chain rule. You almost never need to symbolically differentiate the full expression on paper — decompose into stages (matrix-vector multiplies, max operations, sum operations, etc.) and backprop through variables one step at a time.

## Reference cited in the notes

- "Automatic differentiation in machine learning: a survey" — http://arxiv.org/abs/1502.05767
