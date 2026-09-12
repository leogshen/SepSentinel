# Prodrome window: which target definition is the useful one?

Episodes: `results/mimic31_full_ext_prodrome.pkl`  (63633 after dropping 39 with no positive hour at some candidate window)
Grouped subject-level split, seed 42. Design matrix built once and shared across windows; only the label vector changes.
Total runtime 34.7 min.

**Selected W\* = 3 h** -- max validation patient recall at <=1.0 alerts per nonseptic patient-day.

Lead time and capture are measured against the raw clinical onset `t_sepsis_hour`, so they mean the same thing at every W. Timestep metrics are not comparable across W without care: prevalence runs 0.69% to 4.32% across the sweep, and AUPRC scales with it, which is why AUPRC lift over the prevalence floor is shown too.

## Stage 1 -- XGBoost sweep, VALIDATION patients

Selection was made here and nowhere else.

| Window | Prevalence | AUROC | AUPRC | AUPRC lift | Pos. rows |
|---|---|---|---|---|---|
| W=3 h | 0.0069 | 0.7674 | 0.0278 | 4.0x | 3324 |
| W=6 h | 0.0139 | 0.7440 | 0.0440 | 3.2x | 6648 |
| W=12 h | 0.0259 | 0.7352 | 0.0740 | 2.9x | 12385 |
| W=24 h | 0.0432 | 0.7283 | 0.1169 | 2.7x | 20683 |

### Window geometry

A septic episode with no negative hours contributes nothing to *when* the alarm should fire, only to *whether*. Where that fraction is large the target has drifted from early warning toward patient-level classification.

| Window | Septic episodes | Fully positive | Median positive hours | Median fraction of record |
|---|---|---|---|---|
| W=3 h | 7343 | 0.0% | 3 | 0.13 |
| W=6 h | 7343 | 5.1% | 6 | 0.26 |
| W=12 h | 7343 | 23.7% | 12 | 0.52 |
| W=24 h | 7343 | 53.0% | 23 | 1.00 |

At <=1.0 false alerts per nonseptic patient-day:

| Window | TS prec. | TS recall | TS F1 | Patient recall | Patient prec. | Median lead (h) | Cap >=6h | Cap >=12h | Thr |
|---|---|---|---|---|---|---|---|---|---|
| W=3 h | 0.033 | 0.234 | 0.058 | 0.640 | 0.218 | 18.0 | 0.491 | 0.400 | 0.72 |
| W=6 h | 0.061 | 0.208 | 0.094 | 0.619 | 0.224 | 17.8 | 0.490 | 0.384 | 0.73 |
| W=12 h | 0.098 | 0.197 | 0.130 | 0.622 | 0.222 | 19.7 | 0.514 | 0.409 | 0.73 |
| W=24 h | 0.153 | 0.177 | 0.164 | 0.587 | 0.228 | 20.3 | 0.516 | 0.402 | 0.74 |

Under other objectives the choice would have been: capture_6h -> W=24 h, capture_12h -> W=12 h, timestep_f1 -> W=24 h, auprc_lift -> W=3 h.

## Stage 1 (reference) -- the same XGBoost fits on TEST

Shown for completeness; nothing was selected on these.

| Window | Prevalence | AUROC | AUPRC | AUPRC lift | Pos. rows |
|---|---|---|---|---|---|
| W=3 h | 0.0068 | 0.7679 | 0.0266 | 3.9x | 3249 |
| W=6 h | 0.0136 | 0.7470 | 0.0433 | 3.2x | 6498 |
| W=12 h | 0.0254 | 0.7399 | 0.0720 | 2.8x | 12140 |
| W=24 h | 0.0417 | 0.7370 | 0.1101 | 2.6x | 19948 |

| Window | TS prec. | TS recall | TS F1 | Patient recall | Patient prec. | Median lead (h) | Cap >=6h | Cap >=12h | Thr |
|---|---|---|---|---|---|---|---|---|---|
| W=3 h | 0.032 | 0.241 | 0.056 | 0.624 | 0.205 | 16.8 | 0.477 | 0.367 | 0.72 |
| W=6 h | 0.057 | 0.206 | 0.089 | 0.595 | 0.214 | 17.2 | 0.464 | 0.361 | 0.73 |
| W=12 h | 0.097 | 0.185 | 0.127 | 0.596 | 0.224 | 18.0 | 0.482 | 0.375 | 0.74 |
| W=24 h | 0.151 | 0.190 | 0.168 | 0.591 | 0.224 | 18.8 | 0.504 | 0.386 | 0.74 |

## Stage 2 -- does attention add value at W\* = 3 h?

Transformer trained only at the selected window, 3 seeds, reported mean +/- sd on TEST. XGBoost is the single fit from stage 1, same split, same channels.

| Model | AUROC | AUPRC | AUPRC lift |
|---|---|---|---|
| XGBoost (flat) | 0.7679 | 0.0266 | 3.9x |
| Transformer | 0.7869 +/- 0.0018 | 0.0313 +/- 0.0003 | 4.6x |

At <=1.0 false alerts per nonseptic patient-day:

| Model | TS prec. | TS recall | TS F1 | Patient recall | Patient prec. | Median lead (h) | Cap >=6h | Cap >=12h |
|---|---|---|---|---|---|---|---|---|
| XGBoost (flat) | 0.032 | 0.241 | 0.056 | 0.624 | 0.205 | 16.8 | 0.477 | 0.367 |
| Transformer | 0.037 +/- 0.000 | 0.261 +/- 0.008 | 0.065 +/- 0.001 | 0.529 +/- 0.007 | 0.242 +/- 0.007 | 14.4 +/- 0.5 | 0.390 +/- 0.015 | 0.293 +/- 0.010 |
