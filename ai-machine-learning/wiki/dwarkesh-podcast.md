# Dwarkesh Podcast — AI/ML Coverage

The Dwarkesh Podcast (formerly The Lunar Society) is the most influential long-form interview show on AI in the current era. The host, Dwarkesh Patel, conducts 2-5 hour interviews with frontier-lab leaders, alignment researchers, economists, and AI skeptics — and his pre-reading and follow-up questions are unusually rigorous. This wiki page indexes the AI/ML-relevant episodes in this knowledge base and organizes them by theme.

The full source files are in `raw/` under filenames prefixed with `dwarkesh-`. Each contains the show notes, episode outline, and link back to the dwarkesh.com page where the full transcript lives.

## Why this corpus matters

Many of the most influential AI worldviews of the 2023-2026 era were articulated first or most clearly on this podcast: Aschenbrenner's "Situational Awareness" thesis, the Anthropic interpretability program (Sholto Douglas / Trenton Bricken), the AI 2027 scenario (Scott Alexander / Daniel Kokotajlo), Sutton's "LLMs are a dead end" critique, Karpathy's "decade of agents" thesis, Ilya's "age of research" framing, and the Ege/Tamay 30-year-timeline counter-narrative. Reading these chronologically is the closest thing to an oral history of how the frontier AI worldview evolved.

## Organized by theme

### AGI timelines & the scaling debate
The central question across the corpus: how far can scaling go, and when (if ever) does it yield AGI? Strong shorter-timeline voices: Aschenbrenner, Schulman, Hassabis, Amodei, Legg, Sutskever, Sholto/Trenton. Longer-timeline voices: Karpathy, Ege/Tamay, Sutton, Chollet, Cowen. Dwarkesh himself shifted from short to medium timelines over 2025. See [[agi-timelines]] for the synthesis.

Key sources:
- Leopold Aschenbrenner — 2027 AGI (2024) [[../raw/dwarkesh-leopold-aschenbrenner-2027-agi]]
- Andrej Karpathy — AGI is still a decade away (2025) [[../raw/dwarkesh-andrej-karpathy-agi-decade-away]]
- Ilya Sutskever — Why next-token prediction... (2023) and From scaling to research (2025)
- Dario Amodei — Hidden pattern (2023), End of the exponential (2026)
- Ege Erdil & Tamay Besiroglu — AGI is still 30 years away (2025)
- Scott Alexander & Daniel Kokotajlo — AI 2027 (2025)
- Shane Legg — 2028 AGI (2023)
- Demis Hassabis — Scaling, AlphaZero atop LLMs (2024)
- John Schulman — Plan for 2027 AGI (2024)
- Sholto Douglas & Trenton Bricken — How LLMs actually think (2024), Is RL+LLMs enough? (2025)
- Gwern — Predicted AI trajectory on $12K/yr (2024)
- Dwarkesh essays: Will scaling work? (2023); Why I don't think AGI is right around the corner (2025); Thoughts on AI progress Dec 2025

### Reinforcement learning & post-training
The RL-vs-LLM debate is the central methodological question of 2025-2026.

- Richard Sutton — Father of RL thinks LLMs are a dead end (2025) — see [[reasoning-models]]
- Dwarkesh's follow-up: Some thoughts on the Sutton interview
- Andrej Karpathy — RL is terrible (2025)
- John Schulman — RLHF deep dive (2024) — see [[alignment-methods]]
- Sholto Douglas & Trenton Bricken — Is RL + LLMs enough for AGI? (2025)
- Dwarkesh essay: RL is even more information inefficient than you thought (2025)

### Alignment, safety, and AI takeover
- Eliezer Yudkowsky — Why AI will kill us (2023)
- Paul Christiano — Preventing an AI takeover (2023) — invented RLHF; led OpenAI alignment
- Carl Shulman — 7-hour intelligence-explosion & takeover deep dive, Pt 1 & 2 (2023)
- Joe Carlsmith — Preventing an AI takeover (2024), Utopia, AI, & Infinite Ethics (2022)
- Dario Amodei — Anthropic CEO across multiple eras (2023, 2026)
- Dwarkesh essays: Give AIs a stake in the future (2025); Anthropic vs The Pentagon (2026); Contra Marc Andreessen on AI (2023)

### Interpretability & how LLMs think
- Sholto Douglas & Trenton Bricken — How LLMs actually think (2024) — mechanistic interpretability, superposition, dictionary learning
- Sholto & Trenton — Is RL + LLMs enough? (2025) — updates on interp findings
- See [[neural-geometry]] for the related Goodfire research program

### Compute, semiconductors, energy
- Dylan Patel — 3 big bottlenecks (compute, memory, power) (2026)
- @Asianometry & Dylan Patel — Semiconductor industry deep dive (2024)
- Satya Nadella — Microsoft's AGI plan & quantum (2025), Preparing for AGI / Fairwater 2 tour (2025)
- Casey Handmer — China is killing the US on energy (2025)
- Elon Musk — Orbital data centers (2026)
- Mark Zuckerberg — 1 GW datacenters (2024)
- Dwarkesh essay: Thoughts on the AI buildout (2025)

### AI economics & diffusion
- Tyler Cowen — Hayek/Keynes/Smith on AI (2024), AI bottleneck is humans (2025)
- Mark Zuckerberg — Llama 3 / 18-month code automation (2024, 2025)
- Dwarkesh essays: What fully automated firms will look like; Questions about the Future of AI

### Geopolitics & AI policy
- Leopold Aschenbrenner — China/US super-intelligence race (2024)
- Victor Shih — Xi Jinping's paranoid approach to AGI (2025)
- Casey Handmer — China energy lead (2025)

### Neuroscience, robotics, & alternative architectures
- Adam Marblestone — AI is missing something fundamental about the brain (2025)
- Sergey Levine — Fully autonomous robots are closer than you think (2025)
- François Chollet — Why the biggest models can't solve simple puzzles / ARC-AGI (2024)
- David Deutsch — AI, America, Fun, & Bayes (2022) + Dwarkesh's Contra (2022)
- Adam Brown — Bubble universes & AdS/CFT / DeepMind BlueShift (2024)

## How to use this corpus

When you compile or update a wiki entry, search `raw/dwarkesh-*` for relevant episodes — they almost always contain a quote, anecdote, or counterargument that sharpens the entry. The most worldview-shaping episodes are starred in the AGI-timelines section above.

## Related wiki pages
- [[agi-timelines]] — Synthesis of timeline views across the corpus
- [[scaling-laws]] — Where scaling fits in the AGI debate
- [[alignment-methods]] — Christiano, Yudkowsky, Schulman, Anthropic on alignment
- [[reasoning-models]] — Sutton, Karpathy, Schulman on RL & reasoning
- [[neural-geometry]] — Interpretability and representation research adjacent to the Sholto/Trenton episodes
