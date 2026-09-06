# PhysioNet 2019: standard vs pre-onset target

Same episodes, same split (seed 42), same metrics. `standard` is the label as shipped by the Challenge (1 from t_sepsis-6 onward); `prodrome` truncates each septic episode at t_sepsis and keeps positives only in the 12 h before it.

Lead time and capture are anchored to the same unshifted t_sepsis = onset_step + 6 in both, so they are comparable; AUROC is not (different labels).

**Budget: 0.5 false alerts per nonseptic patient-day**

| Target | Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|---|
| standard | logreg | 0.51 | 23.5 | 0.36 | 0.33 |
| standard | xgboost | 0.59 | 22.0 | 0.46 | 0.39 |
| prodrome | logreg | 0.43 | 30.5 | 0.35 | 0.32 |
| prodrome | xgboost | 0.57 | 23.0 | 0.48 | 0.39 |

**Budget: 1.0 false alerts per nonseptic patient-day**

| Target | Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|---|
| standard | logreg | 0.66 | 23.0 | 0.48 | 0.42 |
| standard | xgboost | 0.69 | 24.0 | 0.55 | 0.46 |
| prodrome | logreg | 0.54 | 31.0 | 0.45 | 0.41 |
| prodrome | xgboost | 0.71 | 28.0 | 0.61 | 0.54 |

**Budget: 2.0 false alerts per nonseptic patient-day**

| Target | Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|---|
| standard | logreg | 0.78 | 28.0 | 0.61 | 0.51 |
| standard | xgboost | 0.84 | 25.0 | 0.68 | 0.56 |
| prodrome | logreg | 0.70 | 36.0 | 0.59 | 0.50 |
| prodrome | xgboost | 0.87 | 26.0 | 0.75 | 0.63 |

**Budget: 4.0 false alerts per nonseptic patient-day**

| Target | Model | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|---|
| standard | logreg | 0.88 | 29.0 | 0.73 | 0.61 |
| standard | xgboost | 0.95 | 31.0 | 0.81 | 0.67 |
| prodrome | logreg | 0.82 | 36.5 | 0.69 | 0.60 |
| prodrome | xgboost | 0.98 | 32.0 | 0.96 | 0.72 |
