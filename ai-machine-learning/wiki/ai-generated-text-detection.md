# AI-Generated Text Detection

AI-text detection is an adversarial classification problem rather than a solved authorship test. Strong in-distribution results can coexist with false positives, domain shift, unseen model families, and paraphrase attacks; detector scores should therefore be treated as evidence for review, never as sole proof of misconduct or provenance.

## Hard-Negative Mining and Synthetic Mirrors

[Pangram Text (2402.14873)](../../papers/08-evaluation/analysis/Technical Report on the Pangram AI-Generated Text Classifier - 2402.14873.pdf) trains a Transformer classifier on human documents and synthetic LLM-written "mirrors" matched to the same content. Its key optimization insight is that most randomly sampled examples quickly become trivial and contribute almost no gradient. Pangram repeatedly mines the human examples the current detector misclassifies, adds those hard negatives, and asks LLMs to create matched synthetic mirrors, turning data selection into a curriculum.

On the report's 1,976-document benchmark spanning 10 domains and eight open- and closed-source LLMs, Pangram reports 99% accuracy and more than 97% recall for every evaluated generator at a 1% false-positive-rate operating point. Hard-negative mining reduced the domain-weighted held-out false-positive rate from 2.29% to 0.02%; individual domains improved by roughly 100-1,000x. The model also reported zero false positives on 91 TOEFL essays and 3,907 ELLIPSE essays, 0.09% on 5,600 ICNALE essays, and at least 99.61% recall on seven unseen open-source model families.

These results are promising but should be scoped to the authors' benchmark and 2024 model distribution. The report itself warns against using any detector as the sole arbiter of academic integrity, and notes that detection says nothing about the factual truth of a document.

## Detector Evasion as Reinforcement Learning

[AuthorMist (2503.08716)](../../papers/08-evaluation/safety/AuthorMist: Evading AI Text Detectors with Reinforcement Learning - 2503.08716.pdf) demonstrates why static detector accuracy is not enough. It fine-tunes Qwen2.5-3B-Instruct with GRPO, treating scores from six detector APIs as non-differentiable rewards. Six specialized paraphrasers are trained, one per detector, with a KL penalty to keep the policy near the base model and reduce semantic drift.

On 300 XSum-derived human/AI pairs, target-detector attack success rates were at least 92.33% in the reported cross-detector table, and the Originality-trained variant reached a 95.17% mean attack success rate across all six detectors. Median embedding similarity remained above 0.94. Evasion also transferred asymmetrically: the Originality-trained policy reduced mean AUROC across detectors to 0.49, while the GPTZero-trained and OpenAI-trained variants generalized less well.

The experiment exposes an arms-race dynamic. A detector can look strong on untouched model outputs yet fail after a small model optimizes directly against its score. The paper also has limitations: it relies on commercial APIs with rate limits and changing behavior, evaluates a small 300-pair corpus, uses embedding similarity rather than human semantic adjudication, and observes occasional semantic drift despite high average similarity.

## Practical Evaluation Rules

- Report false-positive and false-negative rates separately; class imbalance can make accuracy misleading.
- Test held-out domains, non-native writing, new model families, and multiple text lengths.
- Include adaptive paraphrase attacks, not only untouched generations.
- Calibrate thresholds for the deployment base rate and cost of false accusation.
- Require corroborating evidence and human review for high-stakes decisions.

## Related Topics

- [[llm-evaluation]] — benchmark design, distribution shift, and metric selection
- [[safety-misalignment]] — optimization against a proxy can produce behavior that defeats the intended safeguard
- [[reward-hacking-dynamics]] — AuthorMist is a direct example of optimizing a model against an imperfect external signal
- [[data-curation-mixtures]] — hard-negative mining changes the effective training distribution

## Sources

- [Technical Report on the Pangram AI-Generated Text Classifier (2402.14873)](../../papers/08-evaluation/analysis/Technical Report on the Pangram AI-Generated Text Classifier - 2402.14873.pdf) — synthetic mirrors, hard-negative mining, domain and unseen-model evaluation, ESL tests, and responsible-use warning.
- [AuthorMist: Evading AI Text Detectors with Reinforcement Learning (2503.08716)](../../papers/08-evaluation/safety/AuthorMist: Evading AI Text Detectors with Reinforcement Learning - 2503.08716.pdf) — detector-API rewards, GRPO paraphrasing, cross-detector transfer, semantic similarity, and dual-use limitations.
