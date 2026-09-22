# Training toward capture, not toward the label

The target the project reports is capture >=6h at fixed alert burden, but the target it TRAINS on gives every pre-onset hour the same label -- an alarm 1 h before onset scores like one 11 h before. The utility arms weight each hour by the warning it actually delivers: B(l) = 0 below 1 h, rising to full credit at 6 h.

TLS is the same machinery pointed the other way: it peaks AT onset and decays backwards, weighting the least useful hours most. If ramp direction is what matters, utility should beat TLS on capture while TLS looks relatively better on plain recall, which credits any lead above zero.

Positive mass: hard 57935, tls 16040 (28%), utility 41094 (71%). The mass-matched arm rescales spw by 1.41x so the ramp's SHAPE is isolated from simply carrying less positive weight.

Flat XGBoost, thresholds solved for exact equal burden, paired patient bootstrap (2000 resamples).

| | hard | tls | utility | utility_massmatched |
|---|---|---|---|---|
| **Capture >=6h** | 0.532 | 0.504 | 0.558 | 0.574 |
| Capture >=12h | 0.424 | 0.390 | 0.442 | 0.456 |
| Capture >=3h | 0.586 | 0.574 | 0.599 | 0.616 |
| Patient recall | 0.616 | 0.628 | 0.630 | 0.642 |
| Patient precision | 0.214 | 0.221 | 0.208 | 0.204 |
| Timestep precision | 0.094 | 0.097 | 0.090 | 0.090 |
| Timestep recall | 0.201 | 0.206 | 0.188 | 0.189 |
| Timestep F1 | 0.129 | 0.132 | 0.122 | 0.122 |
| Median lead (h) | 21.00 | 18.75 | 21.26 | 21.44 |
| Realised burden (alerts/pt-day) | 1.00 | 1.00 | 1.00 | 1.00 |
| AUROC | 0.7202 | 0.7249 | 0.7153 | 0.7140 |

## Paired differences (95% CI)

| Contrast | Capture >=6h | Capture >=12h | Patient recall | Median lead (h) |
|---|---|---|---|---|
| tls | -0.029 [-0.050, -0.007] ** | -0.034 [-0.055, -0.015] ** | +0.012 [-0.009, +0.034] | -2.600 [-4.027, -1.070] ** |
| utility | +0.026 [+0.007, +0.046] ** | +0.018 [+0.001, +0.035] ** | +0.014 [-0.006, +0.034] | +0.362 [-0.740, +1.564] |
| utility_massmatched | +0.042 [+0.024, +0.061] ** | +0.031 [+0.015, +0.048] ** | +0.026 [+0.007, +0.046] ** | +0.386 [-0.694, +1.677] |
| utility_vs_tls | +0.055 [+0.030, +0.080] ** | +0.052 [+0.030, +0.074] ** | +0.002 [-0.022, +0.027] | +2.962 [+1.359, +4.800] ** |

`**` marks a paired 95% interval excluding zero.

Built in 5.3 min.
