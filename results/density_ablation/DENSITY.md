# Density ablation: is measurement the binding constraint?

Same patients throughout, observations DELETED at random. Severity cannot confound this the way stratifying by observed density would, because the cohort is identical in every arm and only what the model sees changes. Flat XGBoost; thresholds solved for exact equal burden (1.00 alerts/nonseptic patient-day) so no arm can buy capture by alarming more; paired patient-bootstrap intervals against full density.

Episodes: `results/mimic31_pooled17.pkl`, 63672 episodes, 17 features.

Baseline observed fraction of each target's own channels: labs 0.048, vitals 0.654, all 0.333.

| Arm | Observed frac. of thinned channels | Capture >=6h | Capture >=12h | Patient recall | Median lead | AUROC | vs full: capture >=6h |
|---|---|---|---|---|---|---|---|
| full | nan | 0.523 | 0.421 | 0.619 | 20.9 | 0.7203 | - |
| labs_keep0.5 | 0.024 | 0.537 | 0.427 | 0.617 | 21.0 | 0.6997 | +0.014 [-0.010, +0.037] |
| labs_keep0.25 | 0.012 | 0.559 | 0.444 | 0.645 | 20.3 | 0.6844 | +0.036 [+0.012, +0.061] ** |
| labs_keep0 | 0.000 | 0.564 | 0.448 | 0.664 | 20.0 | 0.6677 | +0.041 [+0.015, +0.068] ** |
| vitals_keep0.5 | 0.327 | 0.497 | 0.394 | 0.595 | 20.0 | 0.7142 | -0.025 [-0.047, -0.004] ** |
| vitals_keep0.25 | 0.163 | 0.477 | 0.378 | 0.570 | 19.7 | 0.7077 | -0.046 [-0.072, -0.021] ** |
| vitals_keep0 | 0.000 | 0.387 | 0.309 | 0.470 | 20.0 | 0.6778 | -0.136 [-0.165, -0.108] ** |
| all_keep0.5 | 0.166 | 0.537 | 0.432 | 0.622 | 20.4 | 0.6919 | +0.014 [-0.010, +0.038] |
| all_keep0.25 | 0.083 | 0.490 | 0.376 | 0.561 | 19.0 | 0.6763 | -0.033 [-0.060, -0.005] ** |
| all_keep0 | 0.000 | 0.838 | 0.680 | 1.000 | 20.0 | 0.5694 | +0.315 [+0.284, +0.346] ** |

`**` marks a paired 95% interval excluding zero.

Built in 40.9 min.
