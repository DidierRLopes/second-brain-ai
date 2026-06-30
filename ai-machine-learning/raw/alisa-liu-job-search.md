# Notes on the Industry Job Search

**Source:** https://alisawuffles.github.io/blog/job-search/
**Author:** Alisa Liu (NLP PhD, University of Washington; job search focused on tokenization research)
**Published:** June 20, 2026

## Background

Alisa Liu applied for Research Scientist / Member of Technical Staff roles at the end of a 6-year NLP PhD at the University of Washington. Her thesis work concentrated on tokenization in her last two years, which she credits with helping her stand out by establishing a clear area of expertise. Her timeline framing was inspired by [Nathan Lambert's AI PhD job hunt post](https://natolambert.com/writing/ai-phd-job-hunt).

## The numbers

- Interviewed at **11 companies** across **57 interviews** total.
- Plus **46 additional recruiter calls** and **16 post-offer chats**, on top of informal networking conversations before the search even began.
- Outcomes tracked as: offer, rejected, ghosted (recruiter never followed up), or withdrawn (she pulled out after already having offers she preferred).

## Company sequencing strategy

- Common wisdom: use a few companies for "practice" interviews, then time the rest so offers land around the same time for negotiation leverage.
- Caveats she adds:
  - Practice interviews help, but interview stamina is finite — don't burn out before reaching the companies you care most about.
  - External factors (whether a company currently has headcount, which teams are actively hiring) can matter more than your own preparation; get insight into this via friends and recruiters.
  - Offer deadlines have more flexibility than expected — recruiters know you're juggling multiple processes — but watch out for "exploding offers" with hard signing windows.

## Getting the first interview

Referrals matter: "sometimes you need to have someone inside the company vouching for you." She recommends being social at conferences, collaborating broadly during the PhD, and attending networking events as setup, then directly reaching out to your network (and extended network) during the search itself — including reconnecting with people you haven't spoken to in years.

## Interview format taxonomy

She breaks industry research-role interviews into seven categories, noting that **technical skills are evaluated more heavily than research experience**, though research experience is often what earns the interview in the first place:

1. **ML coding** — most common type. Implement an architecture, a decoding strategy, or a classic ML algorithm. Fluency in PyTorch is required; numpy-only tasks (e.g., writing a backward pass from scratch) appeared rarely and without an expectation of memorized numpy syntax.
2. **General coding** — LeetCode-style, sometimes with extra twists; foundational concepts here recur in ML coding interviews too.
3. **Technical discussion** — either (a) an extended deep-dive on one topic, e.g., designing experiments for a research question, with the interviewer pressing on design choices and hypothetical results, or (b) rapid-fire breadth questions, e.g., "What are different ways of encoding positional information?", "What is 5D parallelism?", "What is the difference between PPO and GRPO?"
4. **Research discussion** — interviewer asks about a past project and follows up, including questions about other CV papers. She recommends preparing your motivation, insights/opinions developed, and future directions, and tailoring the pitch's keywords to the specific role.
5. **Behavioral** — standard behavioral questions plus occasional AI safety / societal impact questions. She failed her first one by assuming she was inherently "well-behaved" and going in unprepared; recommends mapping memorable PhD anecdotes onto common behavioral prompts in advance for instant retrieval.
6. **Math** — ranges from logic puzzles to pen-and-paper derivations; recommends brushing up on probability, linear algebra, and calculus.
7. **Job talk** — shorter and more focused than an academic job talk, usually centered on a single paper/direction. Hers centered entirely on tokenizers, leading with a first-author paper and briefly covering related second-author/ongoing work.

## Preparation approach

- Treated the job search as a full-time job; kept running personal notes throughout — an [LLM notes doc](https://alisawuffles.notion.site/alisa-s-book-of-llms) and a [math notes doc](https://alisawuffles.notion.site/math-notes) built for one specific interview.
- Started by watching all lectures of Stanford's [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/) to map the breadth of topics needed and organize scattered knowledge into one coherent picture.
- Calls CS336 [Assignment 1](https://github.com/stanford-cs336/assignment1-basics) (implementing a transformer from scratch) "crucial" — this exact exercise recurs across interviews and is worth turning into muscle memory.
- Explicitly practiced coding **without** AI assistance to mimic interview conditions, noting people underestimate their reliance on AI tools otherwise — even though she used ChatGPT/Claude heavily during her actual study/research process.
- Tailored cramming per company using the job description, recruiter hints, and company reputation to scope each interview; describes the feeling as "a slightly different math or CS class, you never went to lectures, and now you have ~3 days to cram for the midterm."
- Sleep mattered concretely: did her first technical interview on 2 hours of sleep after cramming LLM inference details, none of which came up, and lost 10 minutes to an off-by-one error from impaired focus.
- Recommends recording notes immediately after each interview for future study/reflection.
- Reports a side benefit: the breadth of studying directly improved her confidence and research effectiveness, unlocking technical ideas she couldn't access before — and believes doing this earlier in the PhD would have expanded the range of problems she could engage with.

## Negotiation

- The work continues substantially after offers arrive: ongoing calls with future teammates/managers, lunch visits, recruiter calls — an "overwhelming amount of communication."
- Framing: you are structurally outmatched in market knowledge and negotiation skill relative to recruiters, so negotiate anyway even if you'd be content with the initial offer — initial offers are designed to leave negotiation room, and recruiters often say things like "I don't expect you to take our first offer."
- Claims a few weeks of negotiation effort can be "literally... equivalent to years of work at the initial offer."
- Tactics: lean on friends both for recruiter-interaction know-how and for compensation data points to calibrate asks; before every recruiter call, write down what you are and aren't willing to share plus exact quotes to recite; anticipate likely questions/pushback in advance and script comfortable responses.

## Emotional / psychological reflections

- Describes several months of being "stressed, miserable, and not functioning in other parts of my life."
- Notes the difficulty of managing social comparison to peers, unsolicited opinions from others, and decision-making under incomplete information where small choices (e.g., who to contact and when) can have outsized impact.
- Closing framing: cherishes the PhD as a unique period focused purely on having ideas and executing them, without the pressure of securing employment, and suggests the best work often happens when chasing genuine curiosity rather than career pressure.

## Appendix: learning resources she lists

- [LeetCode 75](https://leetcode.com/studyplan/leetcode-75/) / [NeetCode Blind 75](https://neetcode.io/practice/practice/blind75)
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/)
- [Self-Attention & Transformers (CS224N reading)](https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [Backpropagation (CS231n)](https://cs231n.github.io/optimization-2/)
- [Introduction to Policy Gradient for LMs](https://ivison.id.au/2026/02/09/policy-gradient.html)
- [Lightweight Guide to understanding GRPO and RL principles](https://gitlostmurali.com/blog/grpo-intro)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
