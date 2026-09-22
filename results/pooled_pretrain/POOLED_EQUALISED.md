# Four arms, re-scored at EQUAL test burden

pooled_pretrain.py froze thresholds on validation, which is unbiased but let realised burden drift: the pretrained arms landed 0.099 and 0.182 alerts/patient-day BELOW scratch, both intervals excluding zero. An arm that alarms less catches less, so those capture deficits were partly an artifact of the operating point. Here every arm is re-pinned to spend the same test burden. Nothing retrained.

| | xgboost | scratch | mae_mimic | mae_pooled |
|---|---|---|---|---|
| **Capture >=6h** | 0.523 | 0.408 | 0.375 | 0.392 |
| Capture >=12h | 0.421 | 0.323 | 0.290 | 0.306 |
| Capture >=3h | 0.577 | 0.454 | 0.411 | 0.431 |
| Patient recall | 0.619 | 0.488 | 0.436 | 0.457 |
| Patient precision | 0.220 | 0.251 | 0.257 | 0.255 |
| Timestep precision | 0.092 | 0.099 | 0.099 | 0.098 |
| Timestep recall | 0.194 | 0.211 | 0.212 | 0.210 |
| Timestep F1 | 0.125 | 0.135 | 0.135 | 0.134 |
| Median lead (h) | 20.95 | 19.61 | 19.74 | 20.29 |
| Realised burden (alerts/pt-day) | 1.00 | 1.00 | 1.00 | 1.00 |

## Paired differences at equal burden (95% CI)

| Contrast | Capture >=6h | Patient recall | Median lead (h) | Realised burden |
|---|---|---|---|---|
| mae_mimic_vs_scratch | -0.020 [-0.037, -0.002] ** | -0.025 [-0.045, -0.005] ** | +0.208 [-1.250, +1.917] | +0.001 [-0.037, +0.038] |
| mae_pooled_vs_scratch | -0.015 [-0.033, +0.002] | -0.034 [-0.054, -0.015] ** | +1.827 [+0.223, +3.398] ** | +0.000 [-0.039, +0.037] |
| mae_pooled_vs_mae_mimic | +0.005 [-0.010, +0.020] | -0.009 [-0.027, +0.008] | +1.619 [+0.141, +3.219] ** | -0.000 [-0.034, +0.035] |
| scratch_vs_xgboost | -0.124 [-0.166, -0.083] ** | -0.152 [-0.192, -0.111] ** | -0.899 [-4.000, +2.000] | +0.001 [-0.103, +0.106] |

`**` marks a paired 95% interval excluding zero.

Built in 4.4 min.
