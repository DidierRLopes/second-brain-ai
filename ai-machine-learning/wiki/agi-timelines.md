# AGI Timelines

"When does AGI arrive?" is the central forecasting question of the field. There is no consensus — published views from credible researchers in 2024-2026 range from 2027 to "2045+ or never" — but the disagreement is highly structured and tracks specific underlying beliefs about scaling, RL, continual learning, and what counts as intelligence. This page synthesizes the main camps using the [[dwarkesh-podcast]] corpus as primary source material.

## The four camps (roughly)

**1. Imminent AGI (~2027-2028).** Argues current scaling trends, plus RL on verifiable tasks, plus "unhobblings" (longer context, tools, agency) will reach AGI within ~3 years. Holders: Leopold Aschenbrenner, John Schulman, Dario Amodei (in some moods), Shane Legg, Scott Alexander & Daniel Kokotajlo (AI 2027 scenario).

**2. Decade-scale AGI (~2030-2035).** Argues current architectures will get there but slowly; the bottleneck is research progress on continual learning, long-horizon reasoning, and reliable agency — not just compute. Holders: Andrej Karpathy ("decade of agents"), Ilya Sutskever (post-2024, "age of research"), Demis Hassabis, Sholto Douglas & Trenton Bricken (uncertain but lean shorter).

**3. Long-horizon AGI (~2045+).** Argues scaling will saturate, current architectures lack something fundamental, and there is no intelligence explosion in the cards. Holders: Ege Erdil & Tamay Besiroglu (Epoch), François Chollet (ARC-AGI), Tyler Cowen (the bottleneck is humans/institutions), Dwarkesh Patel (2025-onwards, leaning here).

**4. Wrong-path / never with current methods.** Argues LLM + RL is a dead end and that real AGI requires fundamentally different architectures rooted in continual experience-based learning. Holders: Richard Sutton (most articulately), Eliezer Yudkowsky (different concern — current path could kill us before reaching AGI), David Deutsch (universal explainers required), Adam Marblestone (brain has secrets we haven't learned).

## What the cruxes are

The disagreement isn't about the data — everyone sees the same benchmarks. It's about what scaling implies. Five cruxes recur across the corpus:

**Continual learning.** Karpathy, Sutton, Adam Marblestone, and (in his 2026 interview) even Dwarkesh argue that humans learn continually from a tiny stream of data, while LLMs are frozen after training and need to be re-trained from scratch. Whether you can RL-finetune your way around this is the open question. Dario Amodei pushes back: "we don't need continual learning."

**RL information efficiency.** RL gives the model one scalar reward per trajectory, while supervised learning gives a full distribution over tokens. Karpathy calls RL "terrible." Sutton says it's the only real learning. Dwarkesh's essay "RL is even more information inefficient than you thought" sides with the bears. Sholto Douglas is more optimistic on RL scaling.

**The exponential.** Are returns to scale still log-linear, or are we hitting the diminishing-returns part of the curve? Amodei (2026) says "we are near the end of the exponential" but means it differently from skeptics — he thinks RL adds a new exponential. Ege/Tamay think the data exponential is already broken.

**What "AGI" means.** Cowen and Karpathy both argue that AGI, when it arrives, will blend into the ~2% GDP growth trend — economically transformative but not a singular event. Aschenbrenner and Sutskever expect a sharp discontinuity. Hassabis is in between.

**Diffusion lag.** Even if frontier models become very capable, real-economy adoption takes years (compliance, integration, trust, organizational change). Cowen, Dwarkesh (2025), and Ege/Tamay all lean on this. Aschenbrenner and Amodei argue diffusion is "cope" — what matters is the frontier capability, not the long-tail rollout.

## How the field has updated

There is a real shift visible in the 2023-2026 corpus: in 2023, Aschenbrenner-style 2027 timelines were the default among lab insiders. By late 2025, Karpathy, Sutskever, and Dwarkesh have all moved toward longer timelines, citing continual-learning and RL-efficiency problems. The Anthropic team (Amodei, Sholto, Trenton) has held the shorter line. Sutton's intervention in late 2025 sharpened the architectural critique. Many people who were in Camp 1 in 2023 are now in Camp 2.

## Why this matters for downstream questions

Your timeline forecast determines what you think about [[alignment-methods]] (urgent or not?), [[scaling-laws]] (sufficient or not?), [[reasoning-models]] (the path forward or a detour?), AI policy, and compute investment. Most arguments downstream of "what should we do about AI?" are arguments about timelines in disguise.

## Related topics
- [[dwarkesh-podcast]] — Primary source material for this synthesis
- [[scaling-laws]] — The mechanism behind shorter timelines
- [[reasoning-models]] — Does test-time compute close the gap?
- [[alignment-methods]] — What you do depends on when you think AGI arrives
- [[neural-geometry]] — Interpretability evidence on what LLMs actually learn

## Primary sources in `raw/`
- `dwarkesh-leopold-aschenbrenner-2027-agi.md`
- `dwarkesh-andrej-karpathy-agi-decade-away.md`
- `dwarkesh-ilya-sutskever-age-of-research.md`
- `dwarkesh-ilya-sutskever-next-token-prediction.md`
- `dwarkesh-ege-tamay-agi-30-years.md`
- `dwarkesh-ai-2027-scott-alexander-daniel-kokotajlo.md`
- `dwarkesh-shane-legg-2028-agi-deepmind.md`
- `dwarkesh-demis-hassabis-scaling-alphazero-llm.md`
- `dwarkesh-john-schulman-openai-reasoning-rlhf-agi.md`
- `dwarkesh-richard-sutton-rl-father-llms-dead-end.md`
- `dwarkesh-thoughts-on-sutton.md`
- `dwarkesh-francois-chollet-arc-agi.md`
- `dwarkesh-dario-amodei-end-of-exponential.md`
- `dwarkesh-dario-amodei-anthropic-2023.md`
- `dwarkesh-sholto-trenton-rl-llms-agi-2025.md`
- `dwarkesh-sholto-trenton-how-llms-actually-think.md`
- `dwarkesh-why-agi-not-around-the-corner-timelines-june-2025.md`
- `dwarkesh-thoughts-on-ai-progress-dec-2025.md`
- `dwarkesh-will-scaling-work.md`
