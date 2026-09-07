# Flat baselines (no sequence model)

Episodes: `results/mimic31_yaib_aligned.pkl`
Same grouped split (seed 42), same Strategy B 19 channels, same metrics as the Transformer runs. Every timestep is an independent sample.
Built in 4.8 min.

## Threshold-free discrimination

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.728 | 0.035 |
| xgboost | 0.753 | 0.048 |

## At equal alert burden

No recall target is imposed: each row is the best patient recall reachable inside the stated false-alert budget.

**Budget: 0.5 false alerts per nonseptic patient-day**

| Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| logreg | 0.34 | 20.0 | 0.28 | 0.22 |
| xgboost | 0.50 | 17.0 | 0.38 | 0.29 |

**Budget: 1.0 false alerts per nonseptic patient-day**

| Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| logreg | 0.51 | 20.0 | 0.44 | 0.35 |
| xgboost | 0.63 | 19.0 | 0.51 | 0.40 |

**Budget: 2.0 false alerts per nonseptic patient-day**

| Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| logreg | 0.71 | 20.0 | 0.62 | 0.48 |
| xgboost | 0.78 | 19.0 | 0.65 | 0.50 |

**Budget: 4.0 false alerts per nonseptic patient-day**

| Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| logreg | 0.84 | 22.0 | 0.77 | 0.60 |
| xgboost | 0.91 | 20.0 | 0.79 | 0.62 |
