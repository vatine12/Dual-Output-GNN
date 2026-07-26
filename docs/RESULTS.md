# Results

## Evaluation protocol

**Data.** WebNLG. The test split is evaluated as **2,510 unique ordered triple sets** — one row per distinct model input, with every human reference for that input pooled. Raw per-reference rows would score the same deterministic prediction several times and dilute corpus statistics.

**Split by predicate novelty.** 1,758 *seen* inputs (all predicates appear in training) and 752 *unseen* inputs (at least one predicate never seen in training), derived from the 372 training predicates.

**Metrics.**

| Metric | Definition |
|---|---|
| BLEU | corpus-level, SacreBLEU (`13a`, lowercased), pooled references |
| Entity recall | fraction of source-graph entities realized in the output |
| Hallucination | fraction of proper-noun/numeric mentions not traceable to the input |
| Corrupted names | examples containing a garbled realization of a graph entity |

The grounding metric normalizes before judging: diacritics (`Kovač`→`Kovac`), date verbalization (`1982-07-23` → "July 23rd"), name shortening (`Barkov, Jr.`→`Barkov`), leading articles, and acronym spacing (`C.D.`→`CD`). A mention that traces back to an input entity, literal or predicate is grounded; otherwise it counts as hallucinated.

**Statistics.** Paired bootstrap over the same 2,510 inputs, 2,000–3,000 resamples, 95% percentile intervals.

---

## Main table

| System | BLEU ↑ | Hallucination ↓ | Entity recall ↑ | Corrupted names ↓ |
|---|---:|---:|---:|---:|
| BART baseline | 47.40 | 5.01% | 78.53% | 288 |
| + graph fusion | 47.32 | 3.75% | 79.99% | 142 |
| + graph fusion + Hard-v2 | 47.68 | 1.88% | 85.17% | 23 |
| Adaptive selector (shared repr.) | 48.72 | 1.86% | 84.21% | 44 |
| Adaptive selector + Hard-v2 | **48.86** | 1.25% | **86.49%** | 21 |
| Dual-Head Adaptive GNN | 48.68 | 1.76% | 84.12% | 43 |
| Dual-Head Adaptive GNN + Hard-v2 | 48.82 | **1.19%** | 86.36% | **21** |

## Component contributions

**Graph fusion vs. baseline** — hallucination 5.01% → 3.75% (−1.23 pp, 95% CI [−1.66, −0.80]), recall +1.47 pp (CI [+0.83, +2.08]), BLEU unchanged. The reduction is class-specific: corrupted in-graph names fall by roughly half (288 → 142 examples), while fabrications of entities absent from the graph are unaffected. Fusion protects the integrity of entities the model already intends to mention.

**Adaptive selection vs. graph fusion** (dual-head, no Hard-v2) — recall +4.13 pp (CI [+3.61, +4.68]), hallucination −1.99 pp (CI [−2.33, −1.69]), BLEU +1.36, corrupted names 142 → 43. Improved on 302 examples, worsened on 36. Because BART, the R-GCN and fusion are all frozen during selector training, this is attributable to the decoding controller.

**Hard-v2 completion** — applied to graph fusion alone: hallucination 3.75% → 1.88%, recall +5.18 pp, corrupted names 142 → 23, BLEU +0.36. Applied on top of adaptive selection it still adds: hallucination 1.76% → 1.19%, recall +2.24 pp. The two mechanisms address different situations — the selector *initiates* mentions, Hard-v2 *completes* names the model started on its own — and their gains are complementary rather than overlapping.

**Constraint head (dual vs. shared representation)** — statistically tied end-to-end: recall −0.10 pp (CI [−0.30, +0.11]), hallucination −0.10 pp (CI [−0.23, +0.04]); with Hard-v2, recall −0.13 pp (CI [−0.30, +0.04]), hallucination −0.06 pp (CI [−0.18, +0.05]). Every interval includes zero. The dual-head attains the lowest hallucination in the table and one fewer corrupted-name example; the shared variant attains the highest recall and BLEU.

The constraint head does improve the selector as a classifier:

| Development intrinsic | Shared | Dual-head |
|---|---:|---:|
| Precision | 71.16% | **72.95%** |
| Recall | 89.41% | **90.12%** |
| F1 | 79.25% | **80.63%** |
| Entity-identity accuracy | 90.53% | **91.91%** |
| False triggers | 154 | **142** |
| Missed starts | 45 | **42** |

That the classifier improves while end-to-end output does not is informative: the confidence threshold, compatibility gate and coverage masking already absorb much of the error class the constraint head fixes. Once decoding rules are conservative, controller quality saturates.

---

## Seen vs. unseen predicates

| Subset | System | BLEU | Hallucination | Recall |
|---|---|---:|---:|---:|
| Seen (1,758) | graph fusion | 53.24 | 3.53% | 85.59% |
| Seen | fusion + Hard-v2 | 53.77 | 1.61% | 89.10% |
| Seen | Dual-Head | 54.49 | 1.22% | 89.73% |
| Seen | Dual-Head + Hard-v2 | **54.53** | **0.88%** | **90.44%** |
| Unseen (752) | graph fusion | 35.11 | 4.27% | 66.89% |
| Unseen | fusion + Hard-v2 | 35.69 | 2.54% | 76.01% |
| Unseen | Dual-Head | 36.14 | 3.02% | 71.00% |
| Unseen | Dual-Head + Hard-v2 | **36.63** | **1.92%** | **76.82%** |

Improvements hold on unseen predicates: hallucination 4.27% → 1.92% and recall 66.9% → 76.8% relative to graph fusion. The seen subset benefits more in absolute terms, as expected, but the mechanism does not depend on having observed the relation during training.

## Trigger behaviour

| System | Selector triggers | Examples with ≥1 trigger |
|---|---:|---:|
| Adaptive selector (shared) | 3,901 | 2,147 / 2,510 |
| Dual-Head | 3,820 | 2,100 / 2,510 |

Recall gain per 1,000 triggers is effectively identical (27.2 vs 27.1), i.e. the two representations convert activations into recovered entities at the same rate; residual differences in the totals reflect activation frequency, not decision quality. Hard-v2 operates at a different granularity — roughly 5.3 token-level completions per example — consistent with its role as a continuous backstop rather than an initiator.

---

## Scope and limitations

1. **Single seed** per configuration. Multi-seed replication is required before claiming robustness of the smaller differences (particularly dual vs. shared).
2. **Post-development evaluation.** Decoding parameters were frozen on development data before the test run, and the frozen protocol was registered in advance; however the test set informed earlier design decisions, so these results are descriptive rather than a clean held-out confirmation.
3. **Entity-level metric.** Hallucination and recall count entity mentions. They do not detect relation-direction errors, subject/object reversal, or unsupported propositions built from valid graph entities. A triple-level factuality audit (human-labeled, blinded, with an explicit role-reversal category) is in progress.
4. **Coverage is binary and permanent**, which suits short KG-to-text generation. Longer-form or dialogue settings would need mention counts, recency, or discourse state instead.
5. **External validity.** Results are WebNLG-only. Replication on DART and a KGQA benchmark is the natural next step.
