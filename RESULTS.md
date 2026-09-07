# SepSentinel Model B — results

Two datasets, in the order they were worked: PhysioNet/CinC 2019 (part 1,
2026-08 and earlier) and MIMIC-IV 3.1 (part 2, 2026-09-03 to 2026-09-06).

**Read part 2 first if you want current numbers.** Part 1 is kept because the
PhysioNet baseline is still the continuity reference, but one of its
conclusions is superseded — see "Reconciling the two parts" at the end.

**Current best operating point** (MIMIC-IV 3.1, full cohort, at <=1.0 false
alerts per nonseptic patient-day): flat XGBoost on 18 features with a
pre-onset target — patient recall 0.64, median lead **20.6 h**, capture >=6h
**0.53**, AUROC 0.736. Against 0.64 / 13.5 h / 0.43 for where this session
started, at identical burden and recall.

---

# Part 1 — PhysioNet/CinC 2019 (stage 1)

Per-timestep sepsis prediction using only physiological signals.

## Dataset

**PhysioNet/CinC 2019 Sepsis Challenge** (Reyna et al.)
- 14,057 patients (12,818 healthy, 1,239 septic; 8.8% sepsis prevalence)
- Hourly ICU time-series, variable length (8-336 hours, mean 39)
- Downloaded via kagglehub

### Selected features (stage 1)

| Feature | PhysioNet Column | NaN Density |
|---------|-----------------|-------------|
| Heart Rate | HR | 7.7% |
| SpO2 | O2Sat | 12.0% |
| Temperature | Temp | 66.2% |
| Respiratory Rate | Resp | 9.8% |

Stage 1 uses only physiological signals available across all patients.
Biomarkers (Lactate, pH, IL-6) are reserved for stages 2-3.

> **Trap, fixed 2026-09-06.** `STAGES[3]` lists `il6`, which PhysioNet does
> not measure. `load_physionet` used to initialise its array with `np.zeros`
> and fill only mappable columns, so `stage=3` silently produced an all-zero
> IL-6 channel reading 0.0% missing — a phantom perfectly-measured variable.
> It now raises. No published result was affected: experiments 2-6 all pass
> an explicit ten-feature list of real mapped columns.

## Data split

Patient-level stratified split (70/15/15), ensuring all timesteps from one
patient stay in exactly one partition.

| Split | Patients | Healthy | Septic | Sepsis % | Timesteps |
|-------|----------|---------|--------|----------|-----------|
| Train | 9,839 | 8,972 | 867 | 8.8% | 382,661 |
| Val | 2,109 | 1,923 | 186 | 8.8% | 82,478 |
| Test | 2,109 | 1,923 | 186 | 8.8% | 80,983 |

Random seed 42, stratified on patient-level sepsis label.

## Held-out test metrics (stage 1, raw normalised values only)

| Model | AUROC | AUPRC | F1 | Recall | Precision | Specificity | Time |
|-------|-------|-------|-----|--------|-----------|-------------|------|
| **Transformer** | **0.7926** | **0.1180** | **0.1295** | 0.6212 | 0.0723 | 0.8215 | 10.8m |
| TCN | 0.7867 | 0.1092 | 0.1093 | 0.6770 | 0.0594 | 0.7601 | 9.1m |
| GRU | 0.7815 | 0.1087 | 0.1259 | 0.6026 | 0.0703 | 0.8215 | 14.3m |
| XGBoost | 0.6380 | 0.0384 | 0.0706 | 0.4605 | 0.0382 | 0.7448 | 1s |
| Random Forest | 0.5762 | 0.0272 | 0.0038 | 0.0023 | 0.0118 | 0.9998 | 2.2m |

All sequential models trained on CPU.

Random Forest reaches 99.98% specificity at 0.2% recall — it predicts
"healthy" almost everywhere, confirming that accuracy alone is misleading on
imbalanced clinical data.

## Production baseline (Config I, corrected metrics)

9 features (creatinine excluded), Strategy B preprocessing (causal ffill +
train-median + lab masks + lab deltas = 19 channels), causal Transformer
(d_model 64, 2 layers, 4 heads).

- Test AUROC **0.814 +/- 0.004**, AUPRC 0.144 (3 seeds)
- At 70% patient recall: precision 9.3%, 1.7 false alerts/patient-day,
  median lead **23.5 h**

A patient-level pairing bug was found and fixed 2026-08-19; distrust any
patient-level number from before that date. Key negative results: loss
reweighting does not move AUROC (exp5), MAE pretraining does not improve
discrimination at this scale (exp6).

---

# Part 2 — MIMIC-IV 3.1

**Headline:** the MIMIC upgrade did not raise discrimination — and
discrimination turned out never to be the binding constraint. Two changes to
the *problem definition*, none to the architecture, bought nearly nine hours
of additional warning at identical alert burden.

| Full cohort, at <=1.0 false alerts per nonseptic patient-day | AUROC | AUPRC | Patient recall | Timestep precision | Patient precision | Median lead | Capture >=6h |
|---|---|---|---|---|---|---|---|
| Config I features, standard target, Transformer | 0.736 | 0.072 | 0.59 | 0.081 | - | 11.9 h | 0.36 |
| Config I features, standard target, XGBoost | 0.722 | 0.068 | 0.64 | 0.083 | - | 13.5 h | 0.43 |
| **Extended features, pre-onset target, XGBoost** | 0.736 | 0.072 | **0.64** | 0.097 | 0.225 | **20.6 h** | **0.53** |
| Extended features, pre-onset target, Transformer (3-seed ensemble) | **0.756** | **0.088** | 0.50 | **0.116** | **0.272** | 19.0 h | 0.41 |
| Extended features, pre-onset target, logreg | 0.705 | 0.057 | 0.49 | 0.079 | 0.218 | 22.0 h | 0.43 |
| *PhysioNet Config I (different dataset, not comparable)* | *0.814* | *0.144* | *0.70* | *0.093* | *-* | *23.5 h* | *-* |

**Two precisions are reported and both matter.** *Timestep precision* is the
fraction of alarm-HOURS that were labelled positive — the hour-by-hour burden.
*Patient precision* is the fraction of ALARMED PATIENTS who were genuinely
septic — what a clinician means by "when it fires, how often is it right".
Patient precision is ~2.5x higher; quoting only one would be misleading, so
`scripts/operating_curves.py` carries both at every threshold.

**Two Transformer AUROCs appear in this document and both are correct.**
0.751 +/- 0.001 is the mean of three independently-seeded models (0.7502,
0.7527, 0.7514) and is the right number for "how well does this architecture
do". 0.756 is the AUROC of the seed-AVERAGED probabilities, i.e. a 3-model
ensemble, and is the right number for the operating-curve tables where a
single probability vector is needed. Ensembling is worth +0.004.

Full operating curves with precision at every burden:
`results/operating_curves_full_ext/`.

## 1. Cohort and labels

Sepsis-3 with PhysioNet/CinC 2019 Challenge timing rules
(`sepsentinel/data/sepsis3.py`, a DuckDB port of the MIT-LCP mimic-code
concepts; every deviation is enumerated in `sepsis3.DEVIATIONS`).

| | |
|---|---|
| ICU stays in MIMIC-IV 3.1 | 94,458 |
| Qualifying (age >=18, LOS >=6h) | 93,224 (98.7%) |
| With a suspicion-of-infection pair | 57,601 (61.8%) |
| Septic by the Challenge rule | 36,118 (38.7%) |
| ... onset at or before ICU hour 4 (excluded, spec section 2) | 27,576 |
| Episodes after all section-2 exclusions | 63,672-64,236 |
| Patient prevalence | 11.5-12.5% (PhysioNet 8.8%) |
| Timestep prevalence | 2.2-2.6% (PhysioNet 2.2%) |

The 98.7% retention is expected, not a filter that failed to bite: MIMIC-IV's
ICU module is adults-only (minimum `anchor_age` is 18, so the age criterion
drops nothing), and only 1,220 stays are shorter than 6 h (1st percentile of
LOS is 5.0 h, median 47.2 h).

Onset among usable septic stays: median 20 h from ICU admission, p75 46 h,
p95 142 h — ample pre-onset history. All 38 extraction and SOFA itemids
verified against the live 3.1 dictionaries; interleukin and procalcitonin are
absent from both `d_labitems` and `d_items`.

## 2. Post-onset truncation reconstructs a PhysioNet convention

Timestep prevalence by how many post-onset hours are retained, over all 8,542
usable septic stays:

| Truncation | Positive fraction of a septic episode | Timestep prevalence |
|---|---|---|
| +0h | 30% | 1.5% |
| **+3h** | **39%** | **2.3%** |
| +6h | 46% | 3.1% |
| +24h (our original default) | 68% | 7.2% |
| none | 87% | 25.4% |

PhysioNet sits at 2.2%, i.e. its septic records end at or just after onset.
DATA_ACCESS_SPEC section 14.1 lists the exact PhysioNet cohort filters as an
open uncertainty; this is evidence for one of them. `POST_ONSET_TRUNCATE_H =
24` was never PhysioNet-comparable.

> **Status:** this table is current (computed over all 8,542 usable septic
> stays from the label table) but it describes the *intermediate*
> configuration. The current best configuration supersedes it: truncation at
> +0h combined with the pre-onset target of section 5, which gives 2.6%
> timestep prevalence. The +3h row was the best available choice before the
> pre-onset target existed, and it is what the section 3, 4 and 6 analyses
> were run on.

## 3. Why performance plateaued (`scripts/diagnose_ceiling.py`)

**Detectability decays smoothly; there is no cliff.** Septic hours in each
window against all control hours:

| Window before onset | AUROC |
|---|---|
| 0-6h | 0.724 |
| 6-12h | 0.692 |
| 12-24h | 0.660 |
| 24-48h | 0.634 |
| >48h | 0.607 |

Even in the final six hours before onset, discrimination is 0.724 —
deterioration tasks with a strong physiological signal reach 0.85-0.90 near
the event. The signal is weak everywhere, not merely early.

**Representation is worth about six times the architecture:**

| | AUROC |
|---|---|
| Raw current-hour values (no ffill, mask or delta) | 0.640 |
| + Strategy B 19 channels, same flat model | 0.722 |
| + full causal Transformer over that history | 0.736 |

**Because in the pre-onset window there are effectively three inputs.**
Measured-hour percentage in septic episodes:

| HR | SpO2 | Resp | Temp | Lactate | WBC | Bilirubin |
|---|---|---|---|---|---|---|
| 95% | 94% | 95% | 31% | 4% | 7% | 2% |

Labs — the variables carrying organ dysfunction — are measured in 2-8% of
pre-onset hours. Everything else is a forward-filled staircase, and the
mask/delta channels encode clinician ordering behaviour as much as
physiology. On top of that `t_sepsis` is partly a treatment-decision
timestamp (culture drawn, antibiotics started), so part of the target is when
a clinician acted.

## 4. AUROC is anti-correlated with early warning here

Observed five times independently:

| Change | AUROC | Median lead |
|---|---|---|
| logreg -> XGBoost -> Transformer (Config I) | 0.682 -> 0.722 -> 0.736 | 16.0 -> 14.7 -> 13.3 h |
| Truncation +3h -> +24h | 0.630 -> 0.718 | 19.0 -> 6.1 h |
| 9 features -> 18 features (standard target) | 0.627 -> 0.691 | 22.7 -> 11.2 h |
| Injected HR prodrome 0.25 -> 4.0 SD | 0.836 -> 0.992 | 11.0 -> 7.0 h |
| Extended + pre-onset: XGBoost -> Transformer | 0.736 -> 0.751 | 20.6 -> 19.3 h |

The synthetic row is the tell: with the signal controlled exactly, a 16x
stronger prodrome gives higher AUROC and *shorter* warning, because the
injected ramp peaks at onset. AUROC rewards recognising the developed state.

**Consequence for reporting:** no fixed operating point is quoted any more.
Results are threshold-free (AUROC, AUPRC) plus equal-ALERT-BURDEN tables —
the burden is the constraint a unit actually imposes and, unlike recall, it
forces nothing. `scripts/operating_curves.py` emits full curves.

## 5. What fixed it: the pre-onset target

Confine positives to `[t_sepsis - 12h, t_sepsis)` and truncate at onset, so
established-sepsis hours leave the dataset entirely and early detection is
the only way to score (`gridding.make_labels(prodrome_window_h=...)`).

Full cohort, extended features, at <=1.0 alerts/patient-day: recall 0.64,
median lead **20.6 h**, capture >=6h **0.53** — against 0.64 / 13.5 h / 0.43
for the original Config I standard-target run at the same burden and recall.

**Was the PhysioNet baseline suffering the same flaw?** Largely no. The
Challenge truncated its records at about `t_sepsis + 4h`, so a septic
PhysioNet episode carries a median of 4 post-onset hours against the 24 our
extraction kept. Applying the same fix there gives about +4 h (XGBoost,
<=1.0 alerts/day: 24.0 -> 28.0 h lead, capture >=6h 0.55 -> 0.61) versus +7 h
on MIMIC. **The old 23.5 h baseline was legitimate; our MIMIC numbers were
depressed by a choice we introduced.**

## 6. Positive control: measurement frequency beats effect size

`scripts/positive_control.py` injects a synthetic prodrome ramping to *d*
standard deviations over the 12 h before onset, into septic episodes only,
applied only where a real measurement exists so a sparse channel stays
sparse. Effect sizes are in units of the SD of the CLIPPED distribution the
model sees, not the raw SD (see section 8).

> **Provenance:** run on `mimic31_full_trunc3.pkl` — the full cohort, but the
> Config I / standard-target configuration, not the current best one. The
> control arm therefore reproduces 0.722 (that configuration's XGBoost
> baseline) rather than 0.736. The comparison between the dense and sparse
> arms is internally valid because both arms share that configuration; it has
> not been repeated on the extended/pre-onset dataset.

| Injected effect | Heart rate (93% of hours, SD 17.9 bpm) | Lactate (2.6% of hours, SD 2.21) |
|---|---|---|
| 0 (control) | 0.722 — reproduces the real baseline exactly | 0.722 |
| 0.25 SD | **0.972** | 0.797 |
| 1.0 SD | 0.969 | 0.794 |
| 4.0 SD | 0.990 | **0.813** |

A **0.25 SD drift (4.5 bpm) in a densely-measured channel beats a 4 SD shift
in a sparsely-measured one** — 16x the biological effect, and it still loses.
Lactate saturates near 0.80 regardless of magnitude: signal size cannot
compensate for being measured in 2.6% of hours.

Two conclusions. The pipeline is demonstrably capable — the control arm
reproduces 0.722 and a quarter-SD signal takes it to 0.97, so the flat
detectability curve in section 3 reflects the data, not the machinery. And
this is a **quantitative design specification for Model A**: continuous
sensing of a modest signal dominates intermittent sampling of a dramatic one.

## 7. Comparators

Bedside scores computed on the **same full cohort, same test split and same
metrics** as the models above (`scripts/clinical_scores.py`, rule-based so
nothing is fitted). Recomputed on the full 63,672-episode cohort on
2026-09-07; an earlier version of this table was computed on a 5,000-stay
subset and reported slightly different numbers (SIRS 0.551, qSOFA 0.587,
NEWS2 0.626).

| Score | AUROC | AUPRC | Recall @<=1.0 alerts/pt-day | Median lead | Capture >=6h |
|---|---|---|---|---|---|
| SIRS | 0.617 | 0.037 | 0.10 | 18.1 h | 0.08 |
| qSOFA | 0.617 | 0.036 | 0.25 | 22.9 h | 0.21 |
| NEWS2 (deployed UK standard) | 0.638 | 0.040 | 0.42 | 21.5 h | 0.35 |
| *our XGBoost, same cohort* | *0.736* | *0.072* | *0.64* | *20.6 h* | *0.53* |

The bedside scores are coarse-grained (integer point totals), so their curves
are step functions and several alert budgets map to the same operating point.

**SOFA is deliberately excluded.** Our label is defined as a >=2-point SOFA
rise, so scoring SOFA against it is near-tautological. Any comparison of a
sepsis model against SOFA on Sepsis-3 labels largely measures its own label
definition.

## 8. Data quality

MIMIC-IV contains impossible charted values: heart rates to 11,337 bpm,
respiratory rates to 7,000,400/min, SpO2 to 9,765,430%. These are
data-entry errors (several look like slipped keystrokes — 102102, 86101).
Out-of-range rates in the extracted grid ran 0.01-0.16% of observed cells,
worst in respiratory rate, touching 5.95% of episodes.

Extraction now applies the section-5 plausibility filters
(`scripts/extract_mimic.py`); `scripts/apply_plausibility_filter.py`
retrofits existing pickles. Measured cost of having omitted them initially:
+0.003 to +0.007 AUROC, no change to any deployment metric or conclusion —
contained because `CLIP_RANGES` caught the values downstream. It did corrupt
the effect-size units in a first run of the section-6 positive control, which
is how it was found.

### Cross-field violations: what a range filter cannot see

A per-channel filter cannot catch a value that is impossible only in context.
`scripts/crossfield_audit.py` on the full extended cohort (3,185,273
patient-hours):

| Check | Violations | Evaluable hours | Rate |
|---|---|---|---|
| SBP < MAP (impossible ordering) | 1,890 | 2,698,830 | 0.070% |
| MAP > 0.95 x SBP (implausibly narrow pulse pressure) | 5,993 | 2,698,830 | 0.222% |
| SpO2 <=90% with PaO2 >=200 mmHg (contradictory pair) | 129 | 86,172 | 0.150% |
| Urine output >1000 mL in one hour | 5,916 | 1,392,752 | 0.425% |
| Respiratory rate > heart rate | 25 | 2,902,190 | 0.001% |
| GCS outside 3-15 | 0 | 930,628 | 0% |
| FiO2 < 21% (below room air) | 0 | 192,238 | 0% |
| Heart rate frozen for >=24 consecutive readings (episodes) | 185 | 41,877 | 0.442% |

GCS and FiO2 are clean only because the extraction range filter already
removes them. **Cross-field violations are roughly 5x more common than
single-channel ones** (~0.44% of relevant cells against 0.086%), and no range
filter would find them.

**Does it matter? Measured, not assumed.** Blanking every violating cell
(both members of an inconsistent pair, since which one is wrong is
unknowable) and refitting: 26,239 cells removed, logreg AUROC 0.705 -> 0.706,
XGBoost 0.736 -> 0.738, and the deployment metrics move by less than the
seed-to-seed noise (XGBoost at <=1.0 alerts/day: recall 0.64 -> 0.63, lead
20.6 -> 20.0 h, capture>=6h 0.53 -> 0.52). So: real, worth fixing for
correctness and for anyone reusing the extraction, but not load-bearing for
any conclusion here. `scripts/crossfield_impact.py` reproduces this.

## 9. Reproducing part 2

```bash
python scripts/verify_mimic_itemids.py --data-root <mimic-iv-3.1> --out results/itemids.md
python scripts/build_sepsis3_labels.py --data-root <mimic-iv-3.1> --out results/sepsis3_labels.csv
python scripts/extract_mimic.py --data-root <mimic-iv-3.1> --feature-set extended \
    --post-onset-truncate-h 0 --prodrome-window-h 12 \
    --sepsis3-labels results/sepsis3_labels.csv --out results/episodes.pkl
python scripts/run_baselines.py --episodes results/episodes.pkl --out-dir results/baselines
python scripts/run_mve.py --episodes results/episodes.pkl --out-dir results/transformer
python scripts/operating_curves.py --episodes results/episodes.pkl \
    --checkpoints results/transformer --out-dir results/curves
```

Labeller rule tests: `python scripts/test_sepsis3_rules.py` (6 cases on a
synthetic mini-MIMIC, no data access needed).

Reproducing part 1: `pip install -r requirements.txt && python train_stage1.py`
(~40 min on CPU). Requires the PhysioNet CSV in the kagglehub cache.

---

# Reconciling the two parts

Part 1 concluded that "sequential models vastly outperform flat models —
temporal context is critical" (Transformer 0.793 vs XGBoost 0.638). Part 2
finds flat XGBoost matching the Transformer's AUROC while beating its early
warning by nearly nine hours. Both are correct, and the reconciliation is the
useful part:

**The part-1 comparison gave the flat models raw normalised values only.**
The sequential models could integrate history; the flat models could not see
it at all. Part 2 measured that gap directly on MIMIC — raw current-hour
values give 0.640, and the same flat model on Strategy B channels (causal
forward-fill + observation mask + time-since-last-measurement delta) gives
0.722, while the Transformer on top of those adds 0.014.

So the part-1 gap was **representation, not architecture**. Once the temporal
information is compressed into per-hour features — last value, whether it was
measured, how long ago — a sequence model has little left to add. The
practical lesson is that "sequence model beats GBDT" comparisons are only
meaningful when both get the same temporal features.
