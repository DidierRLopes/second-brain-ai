# Neural Geometry

Neural geometry is the study of the *shape* that concepts take in a neural network's activation space. The empirical claim, made forcefully by the Goodfire research group in their May 2026 series ("The World Inside Neural Networks"), is that concepts inside trained networks live on *curved manifolds* — loops for cyclic concepts (months, days, hours), smooth 1D curves for ordinal concepts (years, position on a number line), more complex surfaces for color, and graph-like structures for things like the tree of life. The corollary for interpretability and control: linear methods (steering vectors, [[knowledge-distillation|distillation]] of single directions, single-feature SAE readouts) are useful but lossy approximations. Methods that *respect* the underlying geometry — manifold steering, unsupervised pipelines that find manifolds rather than directions — promise both deeper understanding and more reliable, fine-grained control.

## The Core Empirical Claim

Across modalities and domains, geometric structure inside trained networks recapitulates structure in the world:

- **Language models.** Months of the year and days of the week form circular loops in activation space; numbers form smooth ordered curves; years in history sit on smooth curves; character positions in a line of text trace a 1D structure.
- **Image models.** Spatial arrangement of objects is recapitulated in activation space; colors lie on smooth surfaces organized by hue/saturation/lightness.
- **Genomic / epigenomic models.** The tree of life sits on a complex manifold; a previously-unknown class of Alzheimer's biomarkers was found as a clean curve in an epigenomic model.

The claim is "abundant and undeniable" per the Goodfire team — the question now is how to *use* this structure systematically.

## Three Levels of Analysis: Representation, Computation, Behavior

The argument mirrors Marr's levels:

- **Representation.** The data structures (geometries) the model uses to encode information.
- **Computation.** The operators (attention heads, MLP blocks, residual stream interactions) that consume representations and produce new ones.
- **Behavior.** What the model outputs.

Computation cannot be fully understood without understanding representation — "imagine trying to figure out how a laptop sorts a list of numbers without knowing what bits are." The field has made meaningful progress on computation (circuits, induction heads, [[mixture-of-experts|MoE]] routing analysis, etc.); the geometric structure of representations is the matching half of the picture.

If we understood both, we could: know when models will fail, debug them when they do, evaluate and audit with confidence, reshape their training, and "generally build them with less mystery and more engineering."

## Where Does This Geometry Come From?

Concepts don't appear by magic — they arise from the structure of the *world* reflected in *training data*, mediated by optimization pressure and finite parameters.

Mechanism (months-of-year example): months form a cycle in the world (January ≈ February, December; far from June, July). That cyclic structure leaves statistical traces in text (nearby months co-occur in similar contexts). The model is optimized to capture these regularities efficiently within a finite parameter budget, and so its internal representation *recapitulates* the cyclic geometry.

The lesson generalizes: where the data has structure (continuous, cyclic, hierarchical, manifold-shaped), the model's representation tends to mirror it.

## Manifold Steering vs. Linear Steering

The Goodfire team grounds the abstract claim in a controlled experiment: an image-action world model trained on the classic mountain car RL environment.

**Finding 1.** Embedding vectors of frames at different car positions trace an obvious *string-like* (1D manifold) structure in activation space — the model's representation of position.

**Finding 2 (manifold steering works).** Fit a 1D manifold to the curve; intervene on activations to "steer" along it; the predicted image shows the car moving smoothly up and down the hill. Geometry hypothesis confirmed by *causal* control.

**Finding 3 (linear steering fails).** The standard contrastive technique (compute the activation difference between two states, push along that direction) takes a *linear* shortcut between two points on the curve. Two failure modes:
- The linear path crosses "voids" in activation space where the model's output is garbled.
- The linear path intersects a *different* valid activation, causing the car to teleport to that unrelated location.

The takeaway: when a model has learned a scalar concept (here, position), the concept may live on a curved manifold, not a straight line. Following the geometry gives precise, effective interventions. Ignoring it pushes the model into unnatural, incoherent states — or worse, accidentally steers toward an unrelated concept.

## Why Sparse Autoencoders Aren't Enough

SAEs are the dominant unsupervised feature-extraction tool in 2025–2026 mechanistic interpretability. They are useful — they break down a model into features at scale without supervision. But they have a structural limitation that neural-geometry analysis exposes clearly.

Goodfire's illustration: a *slant-rhyme* manifold, where words rhyming with "-ore" arrange continuously from perfect rhymes (door) to near-rhymes (car) to poor rhymes (wire). When they list the SAE features that reconstruct this manifold and read the auto-interp labels, every feature describes a *string-local* property — "Words beginning with 'Hor'", "Tokens starting with 'Mor'", "Word-initial 'Cor' in names and words", and ~20 more variants. None capture the unifying semantic structure (phonological endings) that is obvious when the manifold is viewed as a whole.

SAEs "shatter" manifolds into many small, apparently-unrelated pieces. They're a coordinate system in the wrong basis. For deeper precision in understanding and control, the field needs supervised *and* unsupervised methods that respect manifolds, not pull them apart.

The Goodfire group has a forthcoming unsupervised pipeline for finding manifolds directly; they also promise a follow-up on how to reconstruct manifolds *from* SAE features (rather than treating SAEs as the endpoint).

## Implications for Interpretability and Control

Linear methods underestimate what's there. Wherever a "linear feature" or "linear steering vector" appears to work, that's often only the tangent of a curved structure. Three concrete implications:

1. **Audit / evaluation.** Probing for a concept along a single direction tests a 1D slice; if the concept lives on a curved manifold, off-manifold paths may produce false negatives (concept apparently absent) or false positives (a different concept fires).
2. **Steering / control.** Manifold steering moves smoothly along the concept; linear steering can teleport, garble, or no-op. For safety-critical interventions, linearity-by-default is brittle.
3. **Knowledge extraction.** Foundation models contain scientifically novel structure (e.g., the Alzheimer's biomarker curve). Curve- and manifold-aware analysis can extract knowledge that survey-style feature lists miss.

## Open Directions (from the Goodfire series)

- **Unsupervised manifold discovery.** A pipeline that finds the manifolds without needing the experimenter to pre-specify the concept. Goodfire previewed this with the slant-rhyme example; forthcoming work details the method.
- **Reconstruction of manifolds from SAE features.** Bridging existing SAE infrastructure to the geometry view rather than treating them as competing approaches.
- **Manifold-respecting steering at production scale.** The mountain-car experiment is small; scaling manifold steering to frontier LLMs is the natural next test.
- **Theory.** General laws connecting world structure → training pressure → internal geometry. The series quotes Darwin — "a machine for grinding general laws out of large collections of facts" — and argues that research agents armed with geometry-respecting tools can now collect the data at scale.

## Related Topics

- [[transformer-architecture]] — The architecture whose activation spaces these manifolds live in.
- [[embeddings]] — Word- and sentence-level vector representations are the entry point to this kind of geometric analysis.
- [[mixture-of-experts]] — Sparse routing creates additional structure in activation space worth analyzing.
- [[activation-outliers]] — Outliers in activation space connect to the structure of representations (super weights, emergent features).

## Sources

- **The World Inside Neural Networks** — Geiger, Lubana, Fel, Merullo, Byun, Lewis, McGrath (Goodfire, May 7, 2026). The framing piece for the Neural Geometry series.
- **Steering Along Manifolds to Control Neural Networks** — Wurgaft et al. (Goodfire, May 7, 2026). The companion paper formalizing manifold steering.
- **A Geometric Calculator Inside a Neural Network** — Feucht et al. (Goodfire, May 14, 2026). A worked example of geometry-as-computation.
- **Interpreting Language Model Parameters** — Bushnaq, Braun, Clive-Griffin, Bussman, Hu, Ivanitskiy, Linsefors, Sharkey (paper summary, May 5, 2026).
