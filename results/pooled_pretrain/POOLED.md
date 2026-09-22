# Pooled pretraining: four arms

Supervised task, split and evaluation identical across arms. Arms 3 and 4 are compute-matched at 4000 pretraining steps, so the contrast isolates SICdb rather than extra compute. Thresholds are frozen on VALIDATION (`spend_burden`) at <=1.0 alerts per nonseptic patient-day. Capture >=6h is the governing metric: the alert goes to a nurse or response team, whose action needs hours to matter.

MIMIC 63672 episodes (7345 septic). SICdb 13951 episodes pretrained on, **5948 sealed** and never seen, so SICdb survives as external validation. SICdb labels are never used anywhere: they are rung 1 and over-call by 3-7x.

Pretext: contiguous 6-hour block masking, loss restricted to positions genuinely OBSERVED in the raw record. Both fix measured defects in the experiment-6 recipe.

| | xgboost | scratch | mae_mimic | mae_pooled |
|---|---|---|---|---|
| **Capture >=6h** | 0.524 | 0.389 +/- 0.047 | 0.368 +/- 0.009 | 0.371 +/- 0.020 |
| Capture >=12h | 0.425 | 0.307 +/- 0.033 | 0.285 +/- 0.006 | 0.292 +/- 0.014 |
| Capture >=3h | 0.580 | 0.432 +/- 0.051 | 0.402 +/- 0.012 | 0.409 +/- 0.024 |
| Patient recall | 0.623 | 0.463 +/- 0.054 | 0.425 +/- 0.013 | 0.435 +/- 0.031 |
| Patient precision | 0.219 | 0.251 +/- 0.007 | 0.256 +/- 0.006 | 0.258 +/- 0.009 |
| Timestep precision | 0.092 | 0.102 +/- 0.003 | 0.100 +/- 0.002 | 0.100 +/- 0.002 |
| Timestep recall | 0.196 | 0.199 +/- 0.020 | 0.208 +/- 0.004 | 0.197 +/- 0.013 |
| Timestep F1 | 0.125 | 0.134 +/- 0.004 | 0.135 +/- 0.002 | 0.133 +/- 0.004 |
| Median lead (h) | 20.95 | 19.60 +/- 0.36 | 20.07 +/- 0.65 | 20.01 +/- 0.82 |
| Realised burden (alerts/pt-day) | 1.02 | 0.91 +/- 0.12 | 0.97 +/- 0.02 | 0.92 +/- 0.06 |
| auroc | 0.720 | 0.733 +/- 0.001 | 0.729 +/- 0.000 | 0.735 +/- 0.003 |
| auprc | 0.071 | 0.076 +/- 0.002 | 0.073 +/- 0.001 | 0.075 +/- 0.002 |

## Paired differences (95% CI, patient bootstrap)

| Contrast | **Capture >=6h** | Patient recall | Median lead (h) |
|---|---|---|---|
| mae_mimic_vs_scratch | -0.087 [-0.109, -0.066] ** | -0.117 [-0.139, -0.094] ** | +0.942 [-0.779, +2.909] |
| mae_pooled_vs_scratch | -0.093 [-0.113, -0.073] ** | -0.103 [-0.124, -0.083] ** | +0.545 [-1.187, +2.279] |
| mae_pooled_vs_mae_mimic | -0.006 [-0.025, +0.012] | +0.014 [-0.006, +0.033] | -0.397 [-2.075, +1.000] |
| scratch_vs_xgboost | -0.069 [-0.111, -0.028] ** | -0.084 [-0.124, -0.042] ** | -1.081 [-4.000, +1.731] |

## Hospital-shortcut probe

Logistic probe on frozen encoder features, predicting which database a window came from. Read every row against the RANDOM-ENCODER CONTROL: the two databases differ by up to 10x in blood-gas measurement density, so their inputs are already separable and a high AUROC on an untrained encoder is a property of the data, not of pretraining. Only the GAP above the control is evidence that pretraining spent capacity on site identity.

- **random encoder (control): AUROC 0.993**
- mae_mimic: AUROC 0.989 (-0.004 vs control)
- mae_pooled: AUROC 0.989 (-0.003 vs control)

Built in 90.8 min.
