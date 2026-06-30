# TMax: A Simple Recipe for Terminal Agents

**Source:** https://wai-org.com/blog/tmax/
**Authors:** Hamish Ivison*, Junjie Oscar Yin* (co-first authors), Rulin Shao, Teng Xiao, Nathan Lambert, Hannaneh Hajishirzi — Allen Institute for AI (Ai2) and University of Washington. Work on this post done while Ivison and Lambert were at Ai2.
**Published:** June 16, 2026
**Org:** WAI (wai-org.com) is the joint blog/site for this UW + Ai2 collaboration (footer: "WAI · University of Washington © 2026").
**Paper:** arXiv:2606.23321, "Tmax: A simple recipe for terminal agents" (cs.CL, 2026)
**Code:** https://github.com/hamishivi/tmax
**Models:** https://huggingface.co/collections/allenai/tmax (TMax-2B/4B/9B/27B, e.g. allenai/tmax-9b, allenai/tmax-27b)
**Datasets:** https://huggingface.co/datasets/allenai/tmax-15k, https://huggingface.co/datasets/allenai/tmax-sft-16.5k

## What "TMax" Is

TMax is **not** a sampling method or decoding trick — it's a full open recipe (dataset + RL training method + released model family) for training "terminal agents": LLMs that operate a shell/terminal environment by issuing commands, reading stdout/stderr, and iterating, the same interaction pattern used by tools like Claude Code. TMax has two deliverables:

1. **TMax-15k** — a dataset of 14,600 RL environments (executable, with Dockerfiles + verifiers), generated via a compositional/hierarchical sampling pipeline, over 2.5× larger than the next-largest open terminal dataset that releases full environment data.
2. **A simple outcome-only RL recipe** (GRPO + stability fixes, specifically a variant called DPPO) used to train open models from 2B to 27B parameters, evaluated mainly on Terminal Bench 2.0/2.1 and Terminal Bench Lite.

## Headline Results

- **TMax-9B** reaches **27.2%** on Terminal Bench 2.0 — the strongest open-weights model under 10B the authors are aware of under official Terminal Bench settings. It beats prior 32B terminal agents and approaches closed Claude Haiku 4.5 (29.8%).
- **TMax-27B** (built on Qwen3.6-27B) reaches **42.7%** on Terminal Bench 2.0, approaching the 1T-parameter Kimi K2.5 (43.2%) and the 230B MiniMax M2.7 (45.1%) — i.e. competitive with models 10–40× its size.
- TMax-2B and TMax-4B improve from 2.3%→2.9% and 16.6%→18.9% respectively over their un-RL'd base models.
- On Terminal Bench Lite, RL with TMax-15k on Qwen3.5-9B improves the base model from 41.9 → 57.2 (and from 16.1 → 28.8 on TB 2.1) — nearly a 13-point gain over the base model on TB 2.1.

## The Three Gaps TMax Targets

The authors argue open academic work on RL-training terminal agents lags because of:
1. **Hard benchmarks underused** — Terminal Bench captures genuinely long-horizon multi-step work, but most prior work sidesteps it for simpler bug-fixing / NL-to-bash tasks.
2. **Lack of data** — real terminal traces are scarce/proprietary; existing synthetic datasets are small, narrow (dominated by one domain, e.g. file manipulation), or don't release full RL environments.
3. **Lack of a simple baseline RL recipe** — most data papers stop at SFT; RL attempts in this space often report only ~1-point gains.

## Data Generation Pipeline (TMax-15k)

- Generator model: **Gemini-3-Pro** (chosen for strong Terminal Bench performance).
- Each task is composed by **hierarchically sampling across 9 structured axes**. The first two axes (domain, skills) are seeded from **Nemotron-Terminal**'s taxonomy (Pi et al., 2026). Six additional orthogonal axes target diversity and difficulty.
- After axis sampling, the generator instantiates: a Dockerfile, a unit-test verifier, source files, and task instructions — in a **single build step with no teacher-based validation** (this is the key cost-saving design choice).
- **Scalability via soft filtering**: rather than having a strong model attempt every task multiple times to validate it (expensive), TMax relies on RL training itself to filter — it drops any task where all rollouts get identical reward (no gradient signal). In practice fewer than 8 samples per batch get filtered this way. The only generation-time check is that the Docker environment successfully *builds* (one Docker image per task, sharing a pre-built base image per domain).
- **Diversity via hierarchical sampling** plus two extra mechanisms: (a) domain-conditioned **personas** (6–18 per domain) shape plausible task framing, e.g. a "red-team operator crafting an evasion payload"; (b) **multi-modal fixtures** embedded as task files — PNG images, audio, video, stripped binaries, vendored packages — which the (text-only) policy must inspect via ordinary terminal tooling (OCR, audio transcription, ffmpeg).
- **Difficulty via explicit calibration** along two new axes: *task complexity* (a handful of commands up to intricate 30–60-command workflows) and *command complexity* (bash-only up to bash + code + system services), sampled uniformly across buckets. Verification goes beyond exact string match to **graded verifiers**: metric thresholds (e.g. accuracy ≥ 0.95), adversarial corpora (accept clean / reject malicious), fuzz equivalence against an oracle, and multi-protocol service checks.

## Dataset Comparison (Gemini-3-Pro annotated, Gemini-3-Flash difficulty-scored, 250-task subsample, 8 rollouts each)

| Dataset | Size | Pass@1 | Pass@8 | Domain balance | Skill balance |
|---|---|---|---|---|---|
| TMax-15k (ours) | 15k | 42% | 53% | 0.998 | 0.732 |
| Endless Terminals | 2.4k | 92% | 95% | 0.481 | 0.284 |
| Open Thoughts Agents | 0.7k | 51% | 60% | 0.292 | 0.153 |
| TermiGen | 3k | 57% | 66% | 0.646 | 0.477 |
| TerminalTraj | 5.5k | 54% | 65% | 0.363 | 0.374 |
| CLI-Gym | 1.5k | 41% | 55% | 0.283 | 0.061 |
| SWE-smith | 59k | 54% | 72% | 0.146 | 0.042 |

TMax-15k has the highest domain balance (0.998) and skill balance (0.732) of any dataset compared — prior datasets concentrate 34–95% of mass on one domain (SWE-smith is 95% software engineering). TMax-15k also has the lowest Pass@1 (42%) and lowest Pass@8 (53%), i.e. it's the hardest, and the difficulty gap persists at higher rollout counts (so it's genuinely hard, not just high-variance). A 13-gram sliding-window contamination check against Terminal Bench tasks found **0% overlap**.

## SFT Data and Harness

- A small **2.2k-environment SFT set** (reusing the same generation pipeline) was built by sampling 8 trajectories per task from **Qwen3.6-27B**, yielding 16.5K trajectories (8K successful) — released as **TMax-SFT-16.5K**. Used only for the older, weaker Qwen3-8B experiments (not for the main Qwen3.5-9B/27B runs).
- Both data generation and RL rollouts run through a harness built on **mini-swe-agent** with a persistent shell. The authors found the default **Terminus-2** harness more brittle with small models because it expects the agent to send raw keystrokes.

## RL Training Recipe

- Base algorithm: **GRPO** (Shao et al., "DeepSeekMath," 2024), outcome-only, no learned reward model, plus stability modifications:
  - **DPPO** instead of vanilla GRPO (Qi et al., "Rethinking the Trust Region for LLM RL," 2026) — masks tokens where inference (vLLM) and training logprobs disagree, using a binary approximation of total-variation divergence. Described as "meaningfully taming training collapse."
  - **Token-level loss** (per DAPO, Yu et al. 2025), fully **asynchronous** training, filtering of zero-standard-deviation reward groups, and active sampling to keep batches full (following Olmo 3).
  - **FP32 LM head** — keeping the language-model head in full precision to minimize train/inference numeric mismatch; mattered most for the hybrid Qwen3.5 architecture.
- Infra: extends **open-instruct**, uses **vLLM** for rollouts, runs sandboxes via **Podman** or **Apptainer**. Typical run: 2 nodes training + 6 nodes inference on H100s, 2–3 days.
- Key hyperparameters: 500 training steps, group size 32, 8 prompts/batch, 65,536-token max context, 64 max tool calls per episode. Thinking is kept on intermediate turns ("interleaved thinking"). Evaluated on Terminal Bench 2.1 and Terminal Bench Lite, averaged over 3 runs.

## Dataset Ablation (fixed model: Qwen3.5-9B, varying RL dataset)

| RL dataset | TB Lite | TB 2.1 |
|---|---|---|
| None (base Qwen3.5-9B) | 41.9 | 16.1 |
| TermiGen | 49.4 | 25.1 |
| Endless Terminals | 52.6 | 25.5 |
| OpenThinker-Agent | 53.0 | 25.1 |
| TerminalTraj | 45.8 | 18.0 |
| CLI-Gym | 50.7 | 25.1 |
| SWE-Smith | 47.2 | 21.0 |
| **TMax-15k** | **57.2** | **28.8** |

TMax-15k beats every other tried dataset on the same base model and RL recipe. Training on TMax-15k also sustains **more steps per episode** throughout training (other datasets plateau / get solved faster), and assistant-turn token length (reasoning + tool-calling) climbs steadily over training — framed as "the agentic analogue of inference-time reasoning scaling."

## Generalization Results

**Across tasks** (TMax-9B vs. base Qwen3.5-9B, mean of 3 runs):
| Benchmark | Qwen3.5-9B | TMax-9B |
|---|---|---|
| SWE-Bench Verified | 44.0 | 53.5 |
| AIME'24/25 (terminal-agent) | 73.3 | 91.1 |

**Across harnesses** (Terminal Bench Lite, swapping prompts/tools never seen during RL):
| Harness | Qwen3.5-9B | TMax-9B |
|---|---|---|
| Ours (mini-swe-agent + persistent shell) | 41.9 | 57.2 |
| OpenHands | 36.0 | 46.9 |
| mini-swe-agent | 44.1 | 55.3 |
| Terminus-2 | 36.4 | 45.3 |

Gains of roughly 9–15 points hold in every harness tried, including unseen ones — used as evidence the model learns transferable terminal skills rather than overfitting to one harness or to Terminal Bench style tasks.

**Across model families** — applying the recipe to the older **Qwen3-8B** with a short SFT warm-start and shorter context:
| Model | TB Lite | TB 2.1 |
|---|---|---|
| Qwen3-8B | 7.3 | 1.1 |
| + SFT | 11.5 | 6.0 |
| + RL | 17.7 | 5.2 |

TB Lite improves substantially (7.3 → 17.7); TB 2.1 gains are smaller because the benchmark's difficulty makes improvement hard to detect at this scale.

## Key Finding 1: Strong models don't always want your SFT data

Common wisdom says warm-starting RL with SFT improves stability. TMax found the opposite for the strongly post-trained **Qwen3.5-9B**: existing SFT mixtures *degrade* its performance, even their own TMax SFT mixture distilled from a strong Qwen3.6-27B teacher. The older, less post-trained **Qwen3-8B** clearly benefits from the same SFT data.

| Model | TB Lite | TB 2.1 |
|---|---|---|
| Qwen3.5-9B | 41.9 | 16.1 |
| + TMax SFT | 35.5 | 15.0 |
| + large SFT | 31.3 | 16.9 |
| Qwen3-8B | 7.3 | 1.1 |
| + TMax SFT | 11.5 | 6.0 |
| + large SFT | 16.4 | 7.9 |

Hypothesis: larger SFT mixtures lean on relatively weak teacher models from prior work, and a heavily post-trained model has less to gain and more to lose from imitating them.

## Key Finding 2: Terminal-agent RL is hard to stabilize

Training frequently collapsed past 200–300 steps. Main culprit: numeric mismatch between training and inference logprobs, worsened by the hybrid Qwen3.5 architecture. Mitigations, in order of described impact:
- **FP32 LM head** removes the worst logprob spikes (Qwen3-8B doesn't show these spikes even without it, implicating the hybrid architecture specifically).
- Moving from **GRPO to DPPO**, plus a larger group size, further limits collapse.
- Long horizons (often 20+ steps) and infra load from running many concurrent sandboxes compound the instability.
- The same instabilities appeared on Qwen3-8B too, so this isn't unique to one model family.

## Future Directions Named by the Authors

Three directions identified as natural next steps: better training stability, more complex harnesses, and more complex data.

## Citation

```
Ivison, Hamish and Yin, Junjie Oscar and Shao, Rulin and Xiao, Teng and
Lambert, Nathan and Hajishirzi, Hannaneh, "Tmax: A simple recipe for terminal
agents", arXiv preprint arXiv:2606.23321, 2026.
```
