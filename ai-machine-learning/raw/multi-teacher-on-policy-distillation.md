# Multi-Teacher On-Policy Distillation: A New Post-Training Primitive

**Source:** https://yumoxu.notion.site/ (Yumo Xu's blog)
**Author:** Yumo Xu
**Published:** April 29, 2026
**Reference models surveyed:** MiMo-V2-Flash (Jan 2026), GLM-5 (Feb 2026), Nemotron-Cascade 2 (Mar 2026), DeepSeek-V4 (Apr 2026)

## Key Concepts

Post-training has a *see-saw* problem: math RLVR shortens reasoning traces and hurts open-ended writing; RLHF trades off strict instruction following; tool-use RL drifts away from STEM benchmarks. When every specialization stage trades against the others, shipping one model that holds onto *everything* becomes difficult.

**On-policy distillation (OPD)** is the emerging fix: sample trajectories from the *student*, match a teacher's distribution along those rollouts via reverse KL. The natural extension is **multi-teacher OPD (MOPD)**: make each capability's strongest checkpoint a teacher and let the student absorb them all at once. Teachers usually share tokenizer + lineage with the student, so engineering overhead is small.

Four 2026 frontier reports all converge on MOPD but deploy it differently:
- **Final-stage consolidation** (MiMo-V2-Flash, GLM-5): MOPD as the last step of post-training.
- **Mid-pipeline stabilization** (Nemotron-Cascade 2): MOPD as a forgetting-recovery step *between* RL specialization stages.
- **Scaled-up training regime** (DeepSeek-V4): full-vocabulary logits, 10+ teachers, purpose-built infra.

## GRPO → OPD Primer

**Original GRPO loss** uses PPO-like clipped importance sampling, with per-token advantage from a group-relative reward:

```
Â_{i,t} = (R_i − mean(R_1..R_G)) / std(R_1..R_G)
```

**OPD** swaps the advantage for the stop-gradient log-ratio between teacher and student:

```
Â_{i,t} = sg[ log π_teacher(y_{i,t} | x, y_{i,<t}) − log π_student(y_{i,t} | x, y_{i,<t}) ]
```

When the teacher assigns higher probability than the student, advantage is positive → push student up on that token. When teacher is lower → push down. The teacher acts as a dense, per-token reward. Because advantage no longer depends on a group baseline, **group size G = 1 is both valid and throughput-optimal.**

**Why reverse KL?** Forward KL is *mean-seeking* (spreads student mass over all teacher modes, including unlikely ones); reverse KL is *mode-seeking* (concentrates on dominant modes). For multimodal real-text distributions, mode-seeking is the safer default. Forward KL is the standard objective for off-policy sequence distillation; reverse KL fits naturally with on-policy student sampling.

## IcePop: Mitigating Train/Inference Mismatch

Training and inference engines often produce different logits (different kernels, batch-invariance issues; MoE adds expert-routing nondeterminism). IcePop masks tokens whose train/infer probability ratio falls outside a tolerance band [α, β], dropping the noisiest updates entirely. Three of the four surveyed models (MiMo-V2-Flash, GLM-5, Nemotron-Cascade 2) explicitly adopt IcePop.

## MOPD Setup Comparison

| Model | Student init | Teachers | Prompts | Stage | Adv. + ORM? |
|---|---|---|---|---|---|
| **MiMo-V2-Flash (Jan 2026)** | General SFT | SFT + RL checkpoints + Self | Not specified | Final | Yes |
| **GLM-5 (Feb 2026)** | Post-RL | SFT + RL checkpoints | Each teacher's RL training set | Final | No |
| **Nemotron-Cascade 2 (Mar 2026)** | Post-Multi-domain-RL | SFT math + RLHF + multi-domain RL | RLHF / IF-RL / multi-domain pools + AceReason-Math | Intermediate | No |
| **DeepSeek-V4 (Apr 2026)** | Likely SFT | 10+ RL specialists | Not specified | Final | No |

## MiMo-V2-Flash: Final-Stage Capability Merging

Stage 1: general SFT. Stage 2: train domain-specialized teachers (search/code/general-tools + math/reasoning/safety). Stage 3 (MOPD): distill the SFT student against the full pool of teachers.

**Mixed-source teacher pool.** Three teacher *types*:
- Domain-specific SFT models
- RL-trained specialists
- **Self** — a snapshot of the student at the start of MOPD, a fixed stable reference. On tokens where SFT/RL teachers push the student into unfamiliar territory, distilling toward Self prevents catastrophic drift.

**Teacher routing.** Each prompt carries a domain label that deterministically selects *one* teacher. Per-token advantages are computed against that single teacher. Multi-teacher aggregation is therefore *implicit* — it happens through sample-level domain mixing, not per-token ensembling.

**Going beyond imitation: ORM interpolation.** Pure OPD is bounded by the teacher's distribution; it inherits the teacher's mistakes and style biases. MiMo combines the OPD advantage with an outcome-reward-model advantage:

```
Â_{i,t} = Â^OPD_{i,t} + α · Â^ORM_{i,t}
```

The paper doesn't report α or its sensitivity. The ablation (MOPD > MOPD-w/o-ORM > pure ORM) shows the ORM term contributes non-trivially. Empirically the combined advantage lets the student *exceed* teacher accuracy on several benchmarks.

**Connection to ExOPD reward extrapolation.** ExOPD adds a *reference* model (pre-RL checkpoint) and extrapolates beyond the teacher in the direction (teacher − reference). MiMo combines imitation with outcome reward; ExOPD combines imitation with the teacher-improvement direction. Same structural recipe: OPD as dense imitation core + a scaled corrective term. The choice reflects what you trust more (verifiable rewards vs. trajectory of teacher improvement).

**Who is the best teacher?** Across 12 benchmarks: RL wins 6, Self wins 5, SFT wins 1.
- RL teachers dominate verifiable-reward benchmarks (math, code, reasoning).
- Self wins on broad/open-ended tasks where SFT/RL specialists distort calibration.
- MOPD student exceeds best teacher on 8 of 12 tasks (largest gain: +4.1 on Arena-Hard Hard Prompt); underperforms on 4 (largest loss: −6.3 on BrowseComp).

**Important caveat:** the paper doesn't confirm that the best-per-benchmark teacher was actually used in MOPD. Best standalone teacher ≠ best distillation teacher. Higher benchmark score can come with worse calibration, worse log-prob support on student rollouts, worse token-level guidance, or distribution/style mismatch. OPD cares about the teacher's *conditional distribution over the student's on-policy samples*, not just the teacher's final task accuracy.

## GLM-5: Stage-Terminal Teachers for Capability Recovery

Starts from the final post-RL checkpoint of a sequential Reasoning → Agentic → General RL pipeline. Uses each *prior stage's terminal checkpoint* as a teacher. Prompts come from each teacher's own RL training set (so teacher routing is implicit via prompt source).

Unlike MiMo's mixed-type pool, GLM-5's teachers share lineage and differ only in *when* they were taken from the pipeline. MOPD's role here is **capability recovery** (against the regression incurred by later RL stages) rather than capability merging. Pure reverse-KL OPD, no ORM augmentation.

## Nemotron-Cascade 2: Mid-Pipeline Stabilization

Key observation: capability drift as the number of training environments grows.
- Non-math RLVR → Math Reasoning: RLVR shortens reasoning traces and reduces entropy, hurting math.
- RLHF → IF: RLHF-style optimization trades off against instruction following.

Where MiMo and GLM-5 use MOPD as a *final* consolidation, Nemotron inserts MOPD *between* specialization stages as a periodic re-anchor.

**Multi-domain RL before MOPD.** A blended stage covering STEM multiple-choice QA (55%), agentic tool calling (30%), structured-output IF (15%). Reasons for blending: (a) no degradation observed; (b) similar response lengths and verification times minimize inefficiency.

**Three teachers.** Math teacher = the SFT init itself (no math-RL specialist — paper's choice when SFT data quality is high enough that further math RL would risk shortening traces). RLHF teacher (helpfulness/safety alignment). Multi-domain teacher (IF-RL + Multi-domain RL checkpoint).

**Efficiency results.**
- AIME25 (math-only training): MOPD recovers teacher-level performance within ~30 steps (92.0% accuracy).
- ArenaHard v2: after ~52 steps, MOPD reaches 85.5 on Hard Prompt and 71.0 on Creative Writing. RLHF lags behind even with more steps.

## DeepSeek-V4: Scaling MOPD with Infra

DeepSeek-V4 trains domain experts independently with domain-specific SFT + RL (GRPO). Unique characteristics:
- **Full-vocabulary logit distillation.** Prior OPD often estimates KL only on the sampled token — cheap but high-variance, potentially unstable. DeepSeek preserves the full logit distribution when computing reverse KL.
- **10+ teachers** spanning at least 4 domains (math, coding, agent, instruction following), with some split across 3 reasoning-effort modes (Non-think, Think High, Think Max).
- **1.6T parameters** (49B activated in V4-Pro), **1M-token context**.

These force purpose-built infra:

**Efficient teacher scheduling.** Two memory challenges: many teachers (framework supports an effectively unbounded number); huge full logits (num_tokens × vocab_size, with vocab > 100K for long sequences). DeepSeek's infra improvements (not detailed in the post) address these.

**FP4 for inference-only forwards.** MXFP4 quantization applies to all inference-only forwards (teacher and reference). Training stays in FP8 via lossless FP4→FP8 dequantization. Critical when each batch fires 10+ teacher forwards.

**Token-granular Write-Ahead Log (WAL) for preemptible rollouts.** Naive recovery with fresh randomness introduces length bias (short completions finish before interruption; long ones get re-sampled). WAL persists each generated token immediately + saves the KV cache on preemption. On resume, decoding continues from the persisted WAL and saved KV cache. On fatal hardware error, prefill is rerun on persisted tokens to reconstruct KV cache. This preserves the identity of the original sample — mathematically required for correctness, not just speed.

**Agentic extension (DSec).** For agentic rollouts, environment transitions must also be reproducible or OPD data is corrupted. DeepSeek Elastic Compute keeps a globally ordered trajectory log per sandbox, recording every command + result, fast-forwarding by replaying cached results on resume — sandbox analogue of token WAL.

**Decoupling metadata from per-token payload.** Global planning (shuffling, packing, teacher assignment) only needs lightweight metadata (sample id, length, teacher id, domain, offsets, packing layout). Per-token payload (tokens, masks, logprobs, teacher hidden states) is heavy and loaded only for actual training, through a shared-memory data loader to avoid duplicate intra-node copies. Released immediately after mini-batch consumption.

## Convergence Across the Four Reports

Despite spanning 309B–1.6T parameters, 32k–1M context, agent-first to math-first emphasis:
- **All adopt reverse KL** over student rollouts as the central loss.
- **All use a multi-teacher framework.**
- **Three of four explicitly adopt IcePop-style** train/inference mismatch mitigation.
- Underlying motivations differed (capability merging vs. forgetting recovery), but reverse-KL OPD turned out to be the right primitive for both.

## Divergences

- **Teacher composition.** MiMo mixes *types* (SFT, RL specialist, Self); GLM-5 chains *stages* (each prior post-training stage's terminal); Nemotron mixes *capabilities* (math-as-SFT-init, RLHF, multi-domain RL); DeepSeek scales *count* (10+ RL specialists, no SFT-only, no Self).
- **Position.** Three at the end; Nemotron mid-pipeline.
- **Augmentation.** Only MiMo adds an ORM advantage on top of OPD.
- **Engineering scale.** Only DeepSeek-V4 pushed into the regime requiring purpose-built infra.

## Open Directions

- **Scaling teacher count/size.** DeepSeek shows 10+ trillion-param teachers are feasible. Does marginal benefit per additional teacher continue or saturate?
- **Black-box distillation.** All four rely on teacher logit access. Distilling from API-only teachers (sampled tokens only) opens a different design space.
- **Teacher-student co-evolution.** Distilled students re-enter specialist training to produce stronger teachers in an outer loop. Compute cost is substantial; whether gains compound or diminish across generations is an empirical question.
