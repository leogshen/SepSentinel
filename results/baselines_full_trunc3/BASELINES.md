# Flat baselines (no sequence model)

Episodes: `results/mimic31_full_trunc3.pkl`
Same grouped split (seed 42), same Strategy B 19 channels, same metrics as the Transformer runs. Every timestep is an independent sample.
Built in 2.6 min.

| Model | AUROC | AUPRC | Precision @70% pt recall | Alerts/patient-day | Median lead (h) |
|---|---|---|---|---|---|
| logreg | 0.682 | 0.050 | 0.070 | 1.22 | 16.0 |
| xgboost | 0.722 | 0.068 | 0.083 | 1.39 | 14.7 |
| PhysioNet Transformer (Config I) | 0.814 +/- 0.004 | 0.144 | 0.093 | 1.7 | 23.5 |

Capture rates:

| Model | >=3h | >=6h | >=12h |
|---|---|---|---|
| logreg | 0.63 | 0.54 | 0.42 |
| xgboost | 0.59 | 0.49 | 0.39 |
