# A Mini Exercise on the Mismanaged Geniuses Hypothesis (RLMs on LongCoT)

**Authors:** Alex Zhang, Omar Khattab
**Published:** April 26, 2026
**Tags:** mismanaged, longcot
**Trajectories repo:** https://github.com/alexzhang13/longcot-mini-rlm-results
**LongCoT paper:** https://arxiv.org/abs/2604.14140
**LongCoT dataset:** https://huggingface.co/datasets/LongHorizonReasoning/longcot
**RLM paper:** https://arxiv.org/abs/2512.24601 (Recursive Language Models)
**Author's RLM impl:** https://github.com/alexzhang13/rlm
**Prime Intellect's RLM impl:** https://github.com/PrimeIntellect-ai/verifiers/blob/main/verifiers/envs/experimental/rlm_env.py
**Raymond Weitekamp's blog on RLMs:** https://raw.works/longcot-a-benchmark-worthy-of-a-rlms-attention/

## Key Concepts

The **Mismanaged Geniuses Hypothesis (MGH)**: we underestimate how good language models actually are, and they are inhibited by *how we use them*. Many "frontier models can't solve X" benchmark results turn out, on inspection, to reflect prompting and harness choices more than model capability. This post walks through a focused case study on the LongCoT-mini benchmark (Motwani et al. 2026), where small changes to RLM (Recursive Language Model) prompting more than double the reported score.

## Background: LongCoT-mini

LongCoT problems are graphs (often DAGs) where each node is a sub-problem. Each sub-problem relies on answers from incoming nodes, and produces an answer needed by outgoing nodes. LongCoT-mini has 500 problems (vs. 2500 in the full benchmark), each easier individually, but current frontier models still struggle.

Reported baselines (from the LongCoT paper, Figure 9):
- GPT-5.2 (base, strongest reported model): **38.7%** on LongCoT-mini.
- Most frontier models score <40%.
- Notably, RLMs reportedly *underperformed* their base models in most cases.

The authors of the LongCoT paper concluded (Appendix C): *"RLMs work well on problems with sequential or retrievable structure, but as soon as reasoning requires tracking graph-structured dependencies … context-folding becomes much harder."*

Alex Zhang's response: *"General method cannot do XYZ"* is a very strong statement. Nothing about RLM design *should* make tracking graph dependencies harder than map-reduce dependencies — both can be described in code.

## Raymond Weitekamp's First Result

A day after the benchmark released, Raymond Weitekamp ran DSPy.RLM on Claude Sonnet 4.5 on LongCoT-mini. Performance jumped from **13.0% → 45.4%** via a better-tuned RLM implementation. But MATH (6.3%) and CS (4.0%) stayed unsatisfactory.

The authors of LongCoT had already noted that an RLM's ability to use a coding environment inflates CHESS and LOGIC scores through solvers. So this looked compatible with the "RLMs just can't do graph reasoning" thesis.

## Alex Zhang's RLM(GPT-5.2) Experiment

| Method | Total | MATH | CHEM | CS | LOGIC | CHESS |
|---|---|---|---|---|---|---|
| Raymond's DSPy.RLM + Claude Sonnet 4.5 | 45.4% | 6.3% | 31.0% | 4.0% | 96.2% | 85.0% |
| RLM(GPT-5.2) | 50.6% | 5.6% | 50.0% | 11.0% | 86.7% | 93.0% |
| GPT-5.2 (base) | 38.7% | 26.0% | 37.0% | 40.4% | 53.6% | 36.6% |

Still terrible on MATH and CS.

**Manually examining traces** revealed the failure mode wasn't graph reasoning — it was *decision-making about decomposition*:
- The RLM was attempting to solve MATH/CS nodes with pure brute force, crashing the REPL, and failing the trajectory (a guardrail issue).
- When the RLM did recognize it could decompose a graph into sub-problems and launch sub-agents, it would rarely *verify* that the sub-agent got the sub-problem correct.

These looked like prompting issues, not capability issues.

## The Fix: Better Prompts via Claude Code

Overnight, Alex asked Claude Code to:
1. Examine the trajectories.
2. Write tips for the RLM to avoid the observed mistakes.
3. Restart the run on LongCoT-mini.

Results in the morning:
- Overall: **38.7% → 65.6%** (base → RLM with better prompts).
- With partial-credit scoring (many tasks ask for multiple answers, model sometimes gets one wrong): well above **70%**.

The prompt was the same across all tasks. It described the graph structure of LongCoT problems, an example of how to solve a fake problem, and tips for not brute-forcing problems.

**Ablation (same tips for the base LM, no RLM).** Alex iterated *more* on these prompts and still got worse performance than the base prompt. Even when the LM understood the right decomposition, a pure CoT reasoning model struggled to track and execute it. The RLM mechanism specifically is what enabled the prompts to pay off.

## Takeaways

1. **The MGH is real and operationally relevant.** A new benchmark showing "model X can't do Y" should be treated as a hypothesis about the harness, not (only) about the model. The most-cited frontier capability gaps may dissolve under better scaffolding.

2. **Best standalone teacher ≠ best harness setup.** Just as in OPD where "best benchmark score" doesn't predict "best distillation teacher," here "high base benchmark score" doesn't predict "easy to lift with RLMs." GPT-5.2 base was strongest, but the same RLM harness was weakest on the splits where GPT-5.2 base was strongest (MATH, CS).

3. **Steering models with better prompts has outsized effects on systems like RLMs.** Equipping models with more expressive capabilities (RLM-style decomposition, sub-agents) makes them more sensitive to prompting choices, not less.

4. **LMs can generate the prompts RLMs need.** The prompt for RLM(GPT-5.2) was generated by Claude Code from trace analysis. In some sense, the LM itself can recognize the decomposition an RLM needs to do — though it can't execute on that recognition without the RLM scaffolding.

5. **Post-training implication.** Naively bootstrapping RLM-like behavior from pure RL may be sub-optimal. A workable path: steer models through prompting while generating trajectories, then gradually remove these priors — i.e., curriculum-style priors for behavior the eventual RL run will need.

6. **The eventual goal.** A "Move 37" RLM — one that makes decompositions we don't understand but that are significantly better than the ones we come up with. For now, the practical short-term strategy is to avoid sparse rewards and steer.

## Author's Closing

> "Based on the MGH, I think our understanding of model capabilities is still quite poor. As someone who spent a lot of time building benchmarks, I have felt it extremely hard to curate novel problems that modern models truly cannot solve. Even without additional training, we can squeeze out a significant improvement in performance in harnesses like RLMs just by nudging it on the structure of a problem."
