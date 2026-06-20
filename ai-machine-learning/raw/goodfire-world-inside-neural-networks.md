# The World Inside Neural Networks

**Source:** https://www.goodfire.ai/research/the-world-inside-neural-networks
**Authors:** Atticus Geiger, Ekdeep Singh Lubana, Thomas Fel, Jack Merullo, Michael Jae Byun, Owen Lewis, Tom McGrath (Goodfire)
**Published:** May 7, 2026
**Series:** The Neural Geometry Series (Goodfire Research)
**Correspondence:** atticus@goodfire.ai, ekdeep@goodfire.ai

## Key Concepts

How neural geometry will unlock understanding and control of AI. The thesis: the inner world of neural networks is full of *structure* that reflects the structure of the outer world. Concepts inside models do not live on straight linear directions in activation space — they live on curved manifolds (loops for cyclical concepts like months, smooth curves for ordinal concepts like years, complex surfaces for color, etc.). To understand, audit, and control models, we need methods that respect this geometry rather than methods that pretend everything is linear.

## Evidence of Curved Geometric Structure (Across Models, Modalities, Domains)

- **Language models:** numbers, days of the week, months of the year form circular loops; smooth curves represent years in history and character positions in a line of text.
- **Image models:** spatial arrangement of objects is recapitulated in activation space; colors lie on smooth surfaces structured by hue, saturation, and lightness.
- **Genomic model:** the entire tree of life lies on a complex manifold.
- **Epigenomic model:** a previously-unknown class of Alzheimer's biomarkers found as a clean curve.

Conclusion: "the evidence for curved geometric structure inside neural networks is abundant and undeniable."

## The Mountain Car Worked Example

The post grounds the abstract claim in a small, controlled experiment.

**Setup.** Train an image-action model on the classic mountain car RL environment to predict the next frame given current state (position + momentum) and action (accelerate left/right). The model becomes a world model of mountain car physics.

**Finding 1 — geometry of representations.** Plotting the embedding vectors of frames at different car positions reveals an obvious *string-like* (1D manifold) structure in activation space. Nearby positions on the string correspond to images with the car in nearby physical positions. The natural hypothesis: this curve is *how* the model represents position.

**Finding 2 — geometry validates via intervention.** Fit a 1D manifold (smooth curve) to the string. Intervene on hidden activations to "steer" representations *along the manifold*. Result: the car moves smoothly up and down the hill in the predicted image. The geometry hypothesis is confirmed by causal control.

**Finding 3 — naive linear steering fails.** The standard technique (linear steering vectors built from contrastive activation differences) takes *linear* paths between two positions. In the mountain-car model:
- Some linear paths cross through "voids" where the model output is garbled — the model can't handle these regions.
- Some linear paths intersect a different valid activation, causing the car to *teleport* to that other location.

Lesson: when the underlying geometry is curved, linear steering either (a) does nothing, (b) makes outputs incoherent, or (c) accidentally lands on an unrelated concept. Following the manifold is precise and effective; ignoring it pushes the model into unnatural states.

## Representation, Computation, Behavior — Three Levels of Analysis

The argument structurally mirrors Marr's levels:

- **Representation:** the data structures (geometries) the model uses to encode information.
- **Computation:** the operators (push, pop, read, write, attention, MLPs) that consume representations and produce new ones.
- **Behavior:** what the model actually outputs.

You cannot fully understand computation without understanding representation — like trying to figure out how a laptop sorts numbers without knowing what bits are. The field has made progress on computation (circuits, etc.); geometric structures are the *data structures* of neural networks, so they are required to fill out the picture.

If we understood both: we could know when models will fail, debug them when they do, evaluate and audit with confidence, reshape training, and "generally build them with less mystery and more engineering."

## Where Does Neural Geometry Come From?

Concepts don't appear in network internals by magic — they arise from the structure of the world reflected in training data.

Take months of the year: in the world, months form a cycle (January is near February and December, far from June and July). That cyclic structure leaves statistical traces in text (nearby months appear in similar contexts). During training, the model is optimized to capture these regularities efficiently within a finite parameter budget, and so its internal representation recapitulates the cyclic geometry of the underlying concept.

Optimization pressure + finite parameters + structured world → inner geometry that mirrors outer structure.

## What About Sparse Autoencoders?

The Goodfire group uses an unsupervised pipeline (forthcoming work) to find manifolds directly in activation space. As an example, they show a slant-rhyme manifold: words rhyming with "-ore" arranged from perfect rhymes (door) at one end to near-rhymes (car) in the middle to poor rhymes (wire) at the other end — a clean 1D structure organized by phonological ending.

**The SAE failure mode.** When they list the SAE features that reconstruct this manifold and read their auto-interp labels, they get features like:
- "Words beginning with 'Hor'"
- "Words beginning with 'Mor-'"
- "Words beginning with 'Sor-'"
- "Tokens starting with 'Dor'"
- "Word-initial token fragment 'Cor' in names and words"
- ... and roughly 20 more, each describing a string-local property.

The auto-interp labels capture properties *local to a point on the curve* — what the first few letters of a word are — but completely miss the overarching semantic structure (the manifold *is* about phonological endings).

**The general claim.** SAE features "shatter" manifolds into many small, apparently-unrelated pieces, obscuring the unifying structure that becomes clear when you view the manifold as a whole. SAEs are useful tools (especially for unsupervised feature extraction at scale), but for deeper precision, we need methods — supervised and unsupervised — that respect neural geometry. (A future post in the series will discuss how to reconstruct manifolds from SAE features.)

## Why This Matters for Control

A simple example of how neural geometry unlocks control:

> Even when a model has learned a scalar concept like position, that concept may live on a curved manifold rather than along a straight line in activation space. By following the geometry, we can intervene in ways that are precise and effective; by ignoring it, we risk pushing the model into unnatural and incoherent states.

Applications the authors highlight:
- **Extracting scientific knowledge** from foundation models (e.g., the new Alzheimer's biomarker class).
- **Intentional design** of better, safer models — fine-grained, robust control that linear steering vectors can't deliver.

## Conclusion

> "Neural networks have structured inner worlds with geometry that reflects the structure of reality. By developing theories and methods that respect neural geometry, we will unlock deeper interpretability, more reliable control, and safer, better AI."

The closing rhetorical move quotes Darwin — "My mind seems to have become a kind of machine for grinding general laws out of large collections of facts" — and argues that with research agents armed with geometry-respecting tools, we can now collect the empirical data at scale to grind out general laws of (model) thought.

## Related Goodfire Posts in the Series

- **Steering Along Manifolds to Control Neural Networks** (May 7, 2026) — Wurgaft, Rager, Kowal, Shyam, Feucht, Bhalla, Haklay, Bigelow, Sarfati, McGrath, Lewis, Merullo, Goodman, Fel, Geiger, Lubana
- **A Geometric Calculator Inside a Neural Network** (May 14, 2026) — Feucht, Haklay, Bhalla, Wurgaft, Rager, Sarfati, Merullo, McGrath, Lewis, Lubana, Fel, Geiger
- **Paper Summary: Interpreting Language Model Parameters** (May 5, 2026) — Bushnaq, Braun, Clive-Griffin, Bussman, Hu, Ivanitskiy, Linsefors, Sharkey
