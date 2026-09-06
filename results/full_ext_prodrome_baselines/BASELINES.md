# Flat baselines (no sequence model)

Episodes: `results/mimic31_full_ext_prodrome.pkl`
Same grouped split (seed 42), same Strategy B 19 channels, same metrics as the Transformer runs. Every timestep is an independent sample.
Built in 4.5 min.

## Threshold-free discrimination

| Model | AUROC | AUPRC |
|---|---|---|
| logreg | 0.705 | 0.057 |
| xgboost | 0.736 | 0.072 |

## At equal alert burden

No recall target is imposed: each row is the best patient recall reachable inside the stated false-alert budget.

**Budget: 0.5 false alerts per nonseptic patient-day**

| Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| logreg | 0.31 | 22.5 | 0.27 | 0.21 |
| xgboost | 0.49 | 20.0 | 0.39 | 0.31 |

**Budget: 1.0 false alerts per nonseptic patient-day**

| Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| logreg | 0.49 | 22.0 | 0.43 | 0.34 |
| xgboost | 0.64 | 20.6 | 0.53 | 0.42 |

**Budget: 2.0 false alerts per nonseptic patient-day**

| Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| logreg | 0.66 | 22.0 | 0.60 | 0.47 |
| xgboost | 0.78 | 21.0 | 0.67 | 0.54 |

**Budget: 4.0 false alerts per nonseptic patient-day**

| Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| logreg | 0.81 | 22.3 | 0.75 | 0.60 |
| xgboost | 0.91 | 21.7 | 0.83 | 0.66 |
