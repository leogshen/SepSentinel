# Synthetic positive control

Episodes: `results/mimic31_full_trunc3.pkl`
A prodrome is injected into septic episodes only, ramping linearly to `effect` SD over the 12 h before onset. Controls are untouched. The injection is applied only where a real measurement exists, so a sparse channel stays sparse.

| Channel | Measured in | SD |
|---|---|---|
| heart_rate | 93.3% of hours | 17.86 |
| lactate | 2.6% of hours | 2.21 |

## Injected into `heart_rate`

| Effect (SD) | AUROC | Recall @<=1.0 alerts/day | Median lead (h) | Capture >=6h |
|---|---|---|---|---|
| 0.00 | 0.722 | 0.64 | 13.5 | 0.43 |
| 0.25 | 0.972 | 0.99 | 11.1 | 0.85 |
| 0.50 | 0.972 | 0.99 | 11.0 | 0.84 |
| 1.00 | 0.969 | 0.99 | 11.0 | 0.79 |
| 2.00 | 0.973 | 0.99 | 9.0 | 0.74 |
| 4.00 | 0.990 | 1.00 | 8.9 | 0.79 |

## Injected into `lactate`

| Effect (SD) | AUROC | Recall @<=1.0 alerts/day | Median lead (h) | Capture >=6h |
|---|---|---|---|---|
| 0.00 | 0.722 | 0.64 | 13.5 | 0.43 |
| 0.25 | 0.797 | 0.72 | 9.0 | 0.46 |
| 0.50 | 0.799 | 0.70 | 9.0 | 0.44 |
| 1.00 | 0.794 | 0.70 | 8.2 | 0.43 |
| 2.00 | 0.801 | 0.73 | 6.9 | 0.41 |
| 4.00 | 0.813 | 0.77 | 6.7 | 0.44 |
