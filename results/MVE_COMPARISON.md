# MVE comparison — MIMIC-IV 3.1 (DATA_ACCESS_SPEC section 11, step 7)

Three MVE variants, each: random stay draw (seed 42) -> Challenge-rule
Sepsis-3 labels -> section-2 cohort rules -> grouped subject split (seed 42)
-> Strategy B (19 channels) -> SepsisTransformer -> corrected patient-level
evaluation, training seeds {42,123,456}. Operating point = threshold set for
70% PATIENT recall, matching how the PhysioNet baseline is quoted.

| | A | B | C | PhysioNet Config I |
|---|---|---|---|---|
| Stays drawn | 1,000 | 5,000 | 5,000 | ~40k patients |
| Episodes (septic) | 693 (111) | 3,458 (440) | 3,458 (440) | |
| Post-onset truncation | +24h | +24h | **+3h** | records end near onset |
| Patient prevalence | 16.0% | 12.7% | 12.7% | 8.8% |
| Timestep prevalence | 9.0% | 7.0% | **2.3%** | 2.2% |
| Test AUROC | 0.679 +/- 0.016 | 0.718 +/- 0.013 | 0.630 +/- 0.011 | 0.814 +/- 0.004 |
| Test AUPRC | 0.156 +/- 0.014 | 0.197 +/- 0.012 | 0.063 +/- 0.011 | 0.144 |
| Timestep precision | 0.187 | 0.208 | 0.059 | 0.093 |
| False alerts / patient-day | 3.90 | 2.45 | 2.50 | 1.7 |
| **Median lead time (h)** | **-0.8** | **6.1** | **19.0** | **23.5** |
| Capture >=3h before onset | 0.31 | 0.43 | 0.57 | - |
| Capture >=6h before onset | 0.24 | 0.35 | 0.49 | - |
| Capture >=12h before onset | 0.22 | 0.27 | 0.41 | - |

## What the two comparisons isolate

**A -> B (cohort size, truncation held at +24h).** The AUROC acceptance
failure at 1,000 stays was small-n: 693 episodes with 111 septic (78 in
train) is not enough to fit the model. Five times the data moves AUROC
0.679 -> 0.718 (into the section-11 band) and median lead -0.8h -> 6.1h,
with no pipeline change at all.

**B -> C (truncation, cohort size held at 5,000).** Keeping 24h of
post-onset hours makes 68% of a septic episode positive. Those hours are
established sepsis: easy to classify and worth nothing as early warning.
They inflate AUROC (0.718 vs 0.630), AUPRC (0.197 vs 0.063) and precision
(0.208 vs 0.059) while pushing the alarm to onset (6.1h lead vs 19.0h) and
halving capture at 12h (0.27 vs 0.41).

Truncating at +3h makes the task genuinely prospective — almost every
positive hour is now pre-onset — and the lead-time/capture numbers move
toward the PhysioNet baseline while discrimination drops. **The AUROCs are
not comparable across truncation settings: they are different label mixes,
not different models.** C is the setting comparable to PhysioNet (2.3% vs
2.2% timestep prevalence); B is not.

## Why +3h: it reconstructs the PhysioNet convention

Overall timestep prevalence as a function of the truncation window,
computed over all 8,542 usable septic stays in MIMIC-IV 3.1:

| Post-onset truncation | Positive fraction of a septic episode | Timestep prevalence |
|---|---|---|
| +0h | 30% | 1.5% |
| **+3h** | **39%** | **2.3%** |
| +6h | 46% | 3.1% |
| +12h | 56% | 4.5% |
| +24h (previous default) | 68% | 7.2% |
| none | 87% | 25.4% |

PhysioNet 2019 sits at 2.2%, i.e. its septic records end at or just after
onset. DATA_ACCESS_SPEC section 14.1 lists the exact PhysioNet cohort
filters as an open uncertainty; this is evidence for one of them, and it
means `POST_ONSET_TRUNCATE_H = 24` was never PhysioNet-comparable.

## Recommendation for the full training run (HANDOFF work item 4)

1. Extract all qualifying stays at **truncation +3h** as the primary
   configuration, and at +24h as the secondary (spec section 2 asks for both
   variants to be recorded).
2. Judge the run on the section-1 objective — lead time and alert burden at
   70% patient recall — not on AUROC alone. The section-11 AUROC band of
   0.70-0.85 was set before this interaction was known; it is only meaningful
   for a stated truncation setting, and at +3h it should be re-derived rather
   than treated as a pass/fail gate.
3. Expect discrimination to rise substantially: C trains on 2,431 episodes
   (308 septic) against PhysioNet's ~40k patients, and the A->B step showed
   this pipeline is still firmly data-limited at this scale. The full cohort
   is ~20x larger.

## Acceptance criteria status (section 11.5)

| Criterion | A | B | C |
|---|---|---|---|
| Pipeline runs end to end | PASS | PASS | PASS |
| No subject leakage | PASS | PASS | PASS |
| Patient prevalence within 2x of 8.8% | PASS | PASS | PASS |
| Timestep prevalence within 2x of 2.2% | FAIL | FAIL | **PASS** |
| Mean test AUROC in [0.70, 0.85] | FAIL | **PASS** | FAIL (see 2 above) |
| NaN densities logged and compared | PASS | PASS | PASS |

NaN densities (variant A, all three are the same extraction settings) track
PhysioNet closely: HR 6.3% (7.7%), SpO2 9.8% (12.0%), resp 8.1% (9.8%),
temp 71.3% (66.0%), lactate 96.9% (97.3%), WBC 93.7% (93.6%), platelets
93.6% (94.0%), bilirubin 98.4% (98.5%). pH is the one outlier: 95.8% vs
89.3% — MIMIC pH here is arterial blood gas only (50820), while PhysioNet
appears to pool a wider set of pH sources. Worth revisiting if pH earns its
place in the ablation.
