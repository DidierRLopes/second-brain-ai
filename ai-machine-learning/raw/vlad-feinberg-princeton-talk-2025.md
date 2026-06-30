# Gemini Pretraining: Classical and Inference-Optimized Scaling

**Source:** https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf
**Author:** Vlad Feinberg (vladf@google.com), Google DeepMind — Flash Pretraining Lead
**Date:** April 2025 (Princeton talk)
**Format note:** This is a slide deck. Text extraction via `pdftotext -layout` (through a
proxy-mediated web fetch, since direct curl access to the domain was blocked) yielded
unusually dense per-slide text — speaker annotations, bullet points, and citations are
mostly intact. However, this is still a *slide deck*: charts, plots (e.g., IsoFlops parabola
fits, the loss-vs-compute curves, the Llama3-70B/v5e roofline diagrams, Fig4/Fig5/Table1
references from cited papers, the MoE alpha/beta exponent figure, the "Quality x Model Size"
plot) are **not** captured — only slide titles, bullets, and occasional speaker-note-style
asides survive. Several slides are clearly progressive reveals of the same diagram (e.g. six
near-duplicate "IsoFlops approach" slides each adding one bullet) — these have been
collapsed below into a single ordered list rather than repeated. Where a slide is just a
figure reference with no surrounding prose (e.g. "Fig4, eq8" for distillation), that
sparseness is flagged inline rather than invented.

## About the Speaker (from slide 2, "About Me")

Vlad Feinberg's career timeline as given on the slide:
- 2017: Princeton COS + SML (undergrad), worked with Kai on 3D CNNs for MRI segmentation
  for connectome reconstruction.
- Dropped out for a startup: Sisu Data, with Peter Bailis — efficient DB cubing with FDR
  (false discovery rate) control via a custom distributed lasso engine.
- ML Systems PhD at Berkeley RISE Lab, with Ion (Stoica), Joey (Gonzalez), Mike (Jordan,
  presumably) — model-based deep RL, MuJoCo work (with Sergey Levine).
- Google Cerebra (Ads): quantizing an Ads DNN for pCTR (predicted click-through rate)
  serving efficiency.
- Google Brain → Google DeepMind (GDM): optimizer work with Elad (Hazan, presumably);
  inference-efficient LLMs; currently "Flash Pretraining Lead" (i.e., leads pretraining for
  Gemini Flash).

## Talk Scope (slides 3-5)

Explicit agenda (5 sections): (1) Introduction, (2) Classical Scaling, (3) Small Model
Customers, (4) Inference-optimized Scaling, (5) Closing Thoughts.

**What the talk covers:**
- Classic scaling: basic methodologies for pretraining LLMs, foundational lessons learned.
- Inference-optimized scaling: how those methodologies interact with practical serving
  needs; "basic roofline methodology, but without sharding."

**What the talk explicitly does NOT cover:** specialized-capability research (image/audio/
video in/out, long context), "thinking"/post-training, evals, language-modeling basics
(decoder-only transformer mechanics), distributed computing.

**Presumed background reading** cited directly on the slide: DeepSeek-V3 Technical Report
(DeepSeek-AI et al., 2024, arXiv:2412.19437) and GSPMD: General and Scalable Parallelization
for ML Computation Graphs (Yuanzhong Xu et al., 2021, arXiv:2105.04663).

Feinberg credits slide material borrowed from colleagues: Jean-Baptiste Alayrac, Sebastian
Borgeaud, and Jacob Austin (the last is a co-author of the JAX Scaling Book, already in this
wiki as [[how-to-scale-your-model]]).

## Section 2: Classical Scaling

### The core question and the FLOPs identity

Framing question: given compute budget C (e.g., 1000 H100s for 30 days), what model size N
and token count D should you train on to get the best model?

States the standard approximation for transformer training FLOPs:
**C ≈ 6 · N · D**

Derivation sketch given on slide ("How many FLOPs in each training step?"): excluding
self-attention, an N-parameter decoder-only model requires 6N matmul FLOPs per token seen —
2N for the forward pass, 4N for the backward pass — because each matmul performs one
multiply and one add per pair of input values, and the backward pass requires two matmuls
for every one in the forward pass.

A more granular FLOPs breakdown is given (presumably decomposing into FFN width F, number of
heads/head-dim terms H, batch B, sequence length T, and depth/layers via D used in two
different senses — the slide text is terse here and the exact variable bindings are not
fully spelled out in the extracted text):

**18·B·T·D·F + 24·B·T·D·N·H = 6·B·T·(3DF + 4DNH) = 6 · (num tokens) · (parameter count)**

(the slide also notes "or 2·... for forward pass" — i.e., the forward-only term is 2 instead
of 6). *Note: the exact meaning of each symbol (F, H) in this expanded formula is not fully
disambiguated in the extracted text — flagged as a sparse/unclear point.* There's also an
audience Q&A slide: "What About MoEs? Why not Attn?" — posed as a question with no recorded
answer text extracted (likely answered verbally).

### Why extrapolation-based scaling matters

Slide ("Why do we ask ourselves this question?") contrasts old vs. new ML workflows:
- **Old:** maybe 2 stages — a toy problem for iteration (e.g. CIFAR-10), then apply learnings
  to ImageNet; LR (learning rate) searches done via multiple "final runs," where "the last
  data point is our test set."
- **Now:** every next run requires *extrapolation* from smaller experiments. This analysis
  is made "in the context of a parameterized LLM training recipe" — you must already have
  architecture scaling and the schedule defined as functions of N and D. The talk stresses:
  "Loss forecast implies model/recipe selection capability."

### Kaplan et al. (2020) scaling laws

Cites Kaplan et al. (2020), "Scaling Laws for Neural Language Models" (arXiv:2001.08361):
showed that for autoregressive transformers, the performance of much larger models is
accurately predicted by extrapolating from a series of smaller models.

Key quoted finding: "Maximally compute-efficient training would therefore be far more
sample efficient than one might expect based on training small models to convergence, with
data requirements growing very slowly as **D ∼ C^0.27** with training compute."

Concretely: with a **10x compute budget**, Kaplan et al. found parameters should increase by
**5.37x** and data by **1.86x**. Slide draws the industry consequence directly: "We should
heavily invest in scaling the model size rather than the data size!" — with two caveats
flagged: these laws are empirical only, and the fit depends heavily on the experimental
setup and implicit assumptions.

### Chinchilla (Hoffmann et al., GDM, March 2022) and the IsoFlops method

Chinchilla (cited as "[b]," GDM, March 2022 — i.e. Hoffmann et al.) challenges a Kaplan-paper
assumption: Kaplan et al. ran a *single* training run per model size and used *intermediate*
losses to estimate the loss at different token horizons. Chinchilla's critique: this is a
poor approximation because you can get materially better losses through proper learning-rate
decay — only the *final* loss value (after full decay) is the optimal one for that model/data
size.

**The IsoFlops approach** (one of three approaches in the Chinchilla paper), reconstructed
from the talk's step-by-step slide sequence:
1. Fix a target FLOPs budget.
2. Train several models at that fixed budget, varying model size (and therefore data size,
   since FLOPs = 6ND is fixed).
3. Fit a parabola to the loss-vs-model-size points and find its minimum — this gives the
   compute-optimal (N, D) for that FLOPs budget.
4. Repeat steps 1-3 across a range of different FLOPs budgets.
5. Fit a power law relating FLOPs budget to optimal model size N_opt(C).
6. Fit a power law relating FLOPs budget to optimal dataset size D_opt(C).

**Chinchilla's headline finding:** the exponent in both power laws is **~0.5**, meaning
model size and data size should scale at the *same rate* with compute — sharply different
from Kaplan's D ∼ C^0.27. Slide emphasis: Kaplan-style scaling implies models were
systematically **undertrained**, which the talk calls "obviously bad" given that bigger
models are more expensive to serve and use downstream.

A "slight refinements" slide mentions joint loss/eval fits, and notes that to validate any
change to a scaling law you compare a baseline law against a candidate law.

### "The End of Scaling?" / pushback section

- References "GPT-4.5 Doomers" and notes "LMSys is not the end-all-be-all" (i.e., chatbot
  arena rankings shouldn't be over-indexed on).
- Llama 4 Maverick is cited as a case where ranking volatility showed overfitting to human
  preference (presumably referring to its LMArena controversy).
- Counter-argument for continued scaling gains: (1) better NN design is still coming, (2)
  new data sources are being added.

### Better algorithms: MoE scaling

MoE (Mixture-of-Experts) scaling laws are described as "better" but with implications for
token hunger — "we're running out of internet." Notation introduced: **beta** is the
data-dependent exponent, **alpha** is the parameter-count exponent (figure not captured in
text extraction — flagged as sparse). Stated empirical result: "at same active param count
and fixed 100B token training, MoE 64E [64 experts] improves on dense." Source for this
section is implicitly Clark et al. 2022 "Unified Scaling Laws for Routed Language Models"
(arXiv:2202.01169), cited later in the references list.

### More data sources

States this is "where we spend most of our time, even as modelling people" — "probably half
my focus this year so far" (i.e., H1 2025). Two data-source axes:
1. **Multimodal data**: audio, visual, 3D, video, etc.
2. **Synthetic data**: without filtering, can help "in the Stein's paradox sense" (cites
   Jain et al. 2024 — exact title not captured in extraction). Tradeoff called out explicitly:
   generation quality vs. filtering effort, "for appropriate alpha" (the slide cuts off here;
   exact mathematical framing not recoverable from extracted text).

## Section 3: Small Model Customers

### Inference efficiency goals

Frames Google's internal motivation for small/efficient models ("Flash" and "Flash-lite")
around two demands: (1) higher-volume servicing, (2) real-time latency. Lists concrete
product surfaces requiring this: Free Tier Gemini App (chatbot), AIO, AIM, Vertex AI
(finetuning/deploying), AI Studio (generation API). (AIO/AIM are presumably internal Google
product codenames not expanded in the deck.)

### Real-time use cases cited by name

- **Astra** — credited to a video shout-out for Tara Sainath.
- **Mariner** — credited to a video shout-out for Anmol Gulati.

(Both are Google DeepMind real-time agent projects; the talk references demo videos that
aren't captured in text extraction.)

### Back-of-envelope latency budget for a web agent

Concrete worked example given on the slides:
- Web interaction agent: 128k-token prefill, but only 8k incremental prefill per step; 128
  decode tokens (assumed sufficient to produce one action).
- Latency constraint: no more than 1 second between agent actions.
- Of that 1 second, 250ms is assumed consumed by scaffolding, load balancing, request
  validation, KV-cache retrieval, etc. (explicitly called "optimistic").
- This leaves a tight remaining budget for the actual prefill+decode forward passes.

### Llama3-70B / TPU v5e roofline experiment

"An experiment with Llama3-70B and v5e chips" — assumes the system is fully **compute-bound
on prefill** and fully **HBM (memory)-bound on decode**. Headline number from this
napkin-math: **5.7 seconds for 1 chip** to serve the above workload — i.e., roughly 11x over
the latency budget. Conclusion drawn: to hit the 0.5-second API limit (note: the slide
narrows the target from 1s to 0.5s at this point) you already need a **4x4 topology of v5e
chips** (16 chips) just for the prefill station. An audience question is recorded inline:
"how would we shard on 4x4?" (left unanswered in the extracted text — likely discussed live).

This section's punchline: **this is why real-time use cases imply smaller models** — the
latency math forces you toward Flash-class models rather than frontier-scale ones, because
even a 70B model needs a substantial chip topology to clear real-time SLAs.

### Chinchilla-style scaling ignores inference cost

Transition slide: describes how changes get adopted in the "classical" scaling paradigm —
(1) derive a baseline L*(flops) loss-optimal curve, (2) derive a candidate L*(flops) curve
for the proposed change, compare via the IsoFlops-style methodology (or via L(N,D) fits and
the 6ND identity). Explicit claim: **Chinchilla-style scaling ignores inference cost** — it
only optimizes training FLOPs, not the cost of serving the resulting model.

## Section 4: Inference-Aware Scaling

Direct framing: "Most direct answer to previous question: globally optimize flops between
training and inference?" Primary citation: **Sardana et al. 2024, "Beyond Chinchilla-
Optimal: Accounting for Inference in Language Model Scaling Laws"** (arXiv:2401.00448) — this
paper's inference-aware laws are presented as the natural next step beyond Chinchilla.

### Challenges with inference-aware laws (two "deeper issues")

1. **Non-homogeneity of compute**: (a) inference-optimized chips exist and differ from
   training chips, and "global optimization is not how cross-org planning actually works" in
   practice; (b) in principle the cost formulas could be adjusted for "business cost" instead
   of raw FLOPs.
2. **Non-forecastability of D_inf** (inference-time token demand): invokes **Jevons's
   paradox** (cheaper/better service increases total demand, potentially offsetting
   efficiency gains) and "market expansion from quality improvements" — i.e., you can't
   easily predict how much inference volume a model will actually see once deployed, which
   undermines the inference-aware optimization's core input.
3. A second "deeper" issue is **badness of fit**, pointing to "Fig5/tbl1" in the Sardana et
   al. paper (figure/table contents not captured in text extraction — flagged as sparse).

### Addressing badness-of-fit #1: heavily overtrained models / data constraints

Cites **Muennighoff et al. 2023, "Scaling Data-Constrained Language Models"**
(arXiv:2305.16264). Key point: D (data) "was opaque and recipe-specific" in earlier work —
"you wouldn't be blamed for assuming iid [data]." This paper introduces a new dimension:
intentionally repeated/unique data, modeled as **L(N, U, R)** where U is presumably unique
tokens and R is the repeat factor. Upshot stated on slide: this implies "yet smaller models,
more resilient to repeats" than naive scaling would suggest.

### Addressing badness-of-fit #2: what about D_inf (actual inference demand)?

Cites **Llama 3 (arXiv:2407.21783)** as a case where D_inf was effectively treated as
infinite: direct quote pulled from the Llama 3 paper, "Both our 8B and 70B parameter models
continued to improve log-linearly after we trained them on up to 15T tokens." Framed as
plausibly valid strategy for open-source releases — "just pick sizes and train on all your
data."

Feinberg's own framing/critique: "We could be doing research with those flops! Use this
forecast to estimate how much regret we got." Describes an exercise (with a referenced Colab
notebook, not captured) assuming 5 epochs and the data-scarce law from the Muennighoff paper,
with the explicit task framed as "push the curves right." Using **L(U, N, R)** you can back
out "ideal shrunk datasets" that match the loss obtainable from 5 epochs over a larger
dataset — a way of quantifying the training-compute regret from training on far more unique
data than D_inf would have required.

### Distillation scaling laws

Cites **Busbridge et al. 2025, "Distillation Scaling Laws"** (arXiv:2502.08606). Framed as
adding "other axes" to everything discussed so far; the talk says it "will not say much here"
beyond noting the existence of these extra dimensions. Key question posed: "How to spend
flops with teacher?" — references "Fig4, eq8" from the paper (figure/equation contents not
captured — flagged as sparse, this is the single most figure-dependent part of the deck).

Three critical remarks Feinberg makes about the distillation-scaling framing:
1. There's only a "very weak effect from up-trend" and it's "not [the] typical regime" —
   i.e., the paper's headline trend is weak/narrow in applicability.
2. "Missing part of the story": teacher perplexity can be "arbitrarily weakened by just
   adding temperature." His pointed critique: "Take a really good teacher -> Eq8 predicts bad
   distill -> but add high temp and it will be good?" — implying Eq8's prediction is sensitive
   to a knob (temperature) the equation doesn't fully account for.
3. "In practice, you can James-Stein this away with weight tuning with [a] supervised
   objective" — invoking the James-Stein estimator (shrinkage/empirical Bayes) as a practical
   fix, echoing the earlier "Stein's paradox" reference for synthetic data.

Also poses: "Student Capacity Gap?" — answered as "Distill as variance reduction. Better
teacher will just reduce bias." (A "Distill blog post story" is referenced but not expanded
in extracted text.)

## Section 5: Closing Thoughts

### Two flavors of scaling work

1. Adding points to the Quality x Model-Size plot.
2. Increasing the *slope* of that plot.

Cites the **Gemini "tick-tock"** cadence: Flash's goal is to match Pro's quality from the
*previous* generation. Frames "Inference Efficiency Work" (compression-type work) as growing
along both axes: (1) developing better distillation recipes, (2) quantization, (3)
serving-friendly model design changes.

### Future pretraining research ideas ("without big costs")

- "Common refrain: pretraining is expensive, only can be researched in industry" — but
  hardware-focused kernel development is framed as the accessible "hot-loop for research
  now." Calls for more kernel programming languages, compiler tools, and developer tools —
  "or come up with the next FlashAttention."
- Quantization "entering a new frontier from vector quant[ization]."
- "Funsearch-style inference vs. quality tradeoffs" for LLM-in-the-loop search (referencing
  DeepMind's FunSearch methodology).
- Scaling laws are called "brittle, dataset dependent" — adding more dimensions to L(N, D,
  ...) can always improve fit, but **least squares vs. MLE & a formal statistical model**
  "imply different scaling recommendations" and need formalizing.
- Methodological question: "Rather than grid (N, D), where do we get max info gain? Active
  learn[ing]..." (sentence trails off in the slide).

## Key Resources (full reference list from final slide)

- Kaplan et al., "Scaling Laws for Neural Language Models," 2020 — arXiv:2001.08361
- Sardana et al., "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model
  Scaling Laws," 2024 — arXiv:2401.00448
- Muennighoff et al., "Scaling Data-Constrained Language Models," 2023 — arXiv:2305.16264
- Busbridge et al., "Distillation Scaling Laws," 2025 — arXiv:2502.08606
- Austin et al., "How to Scale Your Model" (JAX Scaling Book) — https://jax-ml.github.io/scaling-book/
- Pope et al., "Efficiently Scaling Transformer Inference," 2022 — arXiv:2211.05102
- Clark et al., "Unified Scaling Laws for Routed Language Models," 2022 — arXiv:2202.01169

(Also cited earlier in the deck but not repeated on the final references slide: DeepSeek-V3
Technical Report, arXiv:2412.19437; GSPMD, arXiv:2105.04663; Llama 3, arXiv:2407.21783.)

## Suggested Wiki Placement

This source spans several existing wiki pages rather than fitting one cleanly:
- **[[scaling-laws]]** — primary fit. Covers Kaplan/Chinchilla already; this talk adds the
  IsoFlops step-by-step walkthrough, the inference-aware scaling layer (Sardana et al.),
  Muennighoff data-constrained L(N,U,R), and the MoE alpha/beta scaling note.
- **[[inference-optimization]]** — the Section 3/4 material (real-time latency budgets,
  Llama3-70B/v5e roofline math, Chinchilla-ignores-inference-cost framing) is inference-
  serving-specific and roofline-adjacent.
- **[[how-to-scale-your-model]]** — Feinberg explicitly credits Jacob Austin's slides and
  cites the JAX Scaling Book directly; the "roofline methodology, but without sharding"
  framing is a direct callback to that book.
- **[[knowledge-distillation]]** — the distillation-scaling-laws critique (Busbridge et al.,
  the temperature/James-Stein argument, student capacity gap as variance reduction) fits
  here as a critical addendum.
- **[[mixture-of-experts]]** — the brief MoE scaling-law note (alpha/beta exponents, 64-expert
  vs. dense at fixed active params and 100B tokens) is a minor but relevant data point.
