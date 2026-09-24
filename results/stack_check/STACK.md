# Do the two wins stack?

Utility weighting (+0.042 capture >=6h) and dropping the nine routine labs (+0.041) were measured separately against the same baseline and never run together. They attack different things -- which HOURS are weighted versus which CHANNELS are seen -- so they may add. Or they may be two fixes for one defect: if the labs hurt BECAUSE they teach late firing, and the utility ramp already penalises late firing, the second buys nothing on top of the first.

Same split, same XGBoost settings, thresholds solved for exact equal burden, paired bootstrap across all four arms.

| | hard | utility | hard_nolabs | utility_nolabs |
|---|---|---|---|---|
| **Capture >=6h** | 0.532 | 0.574 | 0.557 | 0.607 |
| Capture >=12h | 0.424 | 0.456 | 0.440 | 0.485 |
| Capture >=3h | 0.586 | 0.616 | 0.617 | 0.654 |
| Patient recall | 0.616 | 0.642 | 0.653 | 0.681 |
| Patient precision | 0.214 | 0.204 | 0.176 | 0.172 |
| Timestep precision | 0.094 | 0.090 | 0.078 | 0.077 |
| Timestep recall | 0.201 | 0.189 | 0.150 | 0.148 |
| Timestep F1 | 0.129 | 0.122 | 0.103 | 0.101 |
| Median lead (h) | 21.00 | 21.44 | 20.30 | 22.00 |
| Realised burden (alerts/pt-day) | 1.00 | 1.00 | 1.00 | 1.00 |
| Channels | 17 | 17 | 8 | 8 |
| AUROC | 0.7202 | 0.7140 | 0.6676 | 0.6628 |

## Paired differences (95% CI)

| Contrast | Capture >=6h | Capture >=12h | Patient recall | Median lead (h) |
|---|---|---|---|---|
| utility | +0.042 [+0.024, +0.061] ** | +0.031 [+0.015, +0.048] ** | +0.026 [+0.007, +0.046] ** | +0.386 [-0.694, +1.677] |
| hard_nolabs | +0.025 [-0.002, +0.051] | +0.016 [-0.008, +0.038] | +0.037 [+0.008, +0.066] ** | -0.451 [-1.969, +1.000] |
| utility_nolabs | +0.075 [+0.049, +0.102] ** | +0.061 [+0.039, +0.084] ** | +0.066 [+0.037, +0.094] ** | +0.834 [-0.623, +2.517] |
| interaction_nolabs_given_utility | +0.033 [+0.008, +0.058] ** | +0.030 [+0.007, +0.052] ** | +0.040 [+0.014, +0.065] ** | +0.448 [-1.000, +2.000] |
| interaction_utility_given_nolabs | +0.050 [+0.030, +0.071] ** | +0.045 [+0.028, +0.064] ** | +0.029 [+0.009, +0.049] ** | +1.285 [+0.000, +2.704] |

`**` marks a paired 95% interval excluding zero.

Built in 8.4 min.
