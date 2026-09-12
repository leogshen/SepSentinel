# Re-reported with honest thresholds

Both source experiments chose their operating point with `at_burden()`, i.e. max(patient recall) over thresholds -- computed on test. That optimises the quantity being reported. Nothing is retrained here; the same checkpoints and the same deterministic XGBoost fits are re-scored under two rules:

- **frozen** -- threshold pinned on VALIDATION to spend the budget (`spend_burden`), applied unchanged to test. No test optimisation.
- **burden-eq** -- threshold re-pinned so all arms spend the same TEST burden. Uses only control patients' alarm rate.

Paired patient-bootstrap intervals throughout (2000 resamples, all arms on the same resampled patients).

## Recipe experiment (W=12), validation-frozen thresholds

Columns are means over 3 seeds. Intervals contrast one matched seed per arm against the same seed of the baseline (checkpoints are globbed in a fixed order, so the pairing is consistent), resampled on patients.

| | baseline_posw | no_posweight | tls_posw | tls_no_posweight | no_posweight - baseline (95% CI) | tls_posw - baseline (95% CI) | tls_no_posweight - baseline (95% CI) |
|---|---|---|---|---|---|---|---|
| **Patient recall** | 0.505 | 0.484 | 0.576 | 0.438 | -0.005 [-0.025, +0.015] | +0.090 [+0.066, +0.114] ** | -0.151 [-0.180, -0.124] ** |
| Patient precision | 0.261 | 0.265 | 0.244 | 0.276 | +0.001 [-0.009, +0.011] | -0.039 [-0.050, -0.027] ** | +0.011 [-0.007, +0.029] |
| Timestep precision | 0.110 | 0.111 | 0.109 | 0.134 | +0.002 [-0.005, +0.008] | -0.006 [-0.013, +0.001] | +0.044 [+0.026, +0.064] ** |
| Timestep recall | 0.230 | 0.206 | 0.229 | 0.155 | -0.020 [-0.031, -0.009] ** | -0.003 [-0.017, +0.011] | -0.140 [-0.158, -0.123] ** |
| Timestep F1 | 0.149 | 0.144 | 0.148 | 0.134 | -0.003 [-0.011, +0.004] | -0.006 [-0.015, +0.002] | -0.044 [-0.059, -0.029] ** |
| Median lead (h) | 18.65 | 18.67 | 17.32 | 16.22 | -0.296 [-2.000, +1.484] | -0.123 [-1.743, +1.496] | -2.655 [-4.950, -0.326] ** |
| Capture >=3h | 0.467 | 0.445 | 0.515 | 0.393 | -0.008 [-0.027, +0.012] | +0.077 [+0.054, +0.100] ** | -0.139 [-0.167, -0.112] ** |
| Capture >=6h | 0.419 | 0.398 | 0.444 | 0.337 | -0.017 [-0.036, +0.002] | +0.047 [+0.027, +0.068] ** | -0.144 [-0.169, -0.118] ** |
| Capture >=12h | 0.330 | 0.312 | 0.353 | 0.265 | -0.012 [-0.028, +0.005] | +0.046 [+0.028, +0.065] ** | -0.120 [-0.144, -0.096] ** |
| Realised burden (alerts/pt-day) | 0.96 | 0.85 | 0.98 | 0.55 | -0.099 [-0.138, -0.060] ** | +0.045 [+0.003, +0.086] ** | -0.655 [-0.728, -0.589] ** |

## Recipe experiment (W=12), burden-equalised thresholds

Columns are means over 3 seeds. Intervals contrast one matched seed per arm against the same seed of the baseline (checkpoints are globbed in a fixed order, so the pairing is consistent), resampled on patients.

| | baseline_posw | no_posweight | tls_posw | tls_no_posweight | no_posweight - baseline (95% CI) | tls_posw - baseline (95% CI) | tls_no_posweight - baseline (95% CI) |
|---|---|---|---|---|---|---|---|
| **Patient recall** | 0.494 | 0.484 | 0.566 | 0.438 | -0.005 [-0.025, +0.015] | +0.090 [+0.066, +0.114] ** | -0.151 [-0.180, -0.124] ** |
| Patient precision | 0.263 | 0.265 | 0.245 | 0.276 | +0.001 [-0.009, +0.011] | -0.039 [-0.050, -0.027] ** | +0.011 [-0.007, +0.029] |
| Timestep precision | 0.112 | 0.111 | 0.111 | 0.134 | +0.002 [-0.005, +0.008] | -0.006 [-0.013, +0.001] | +0.044 [+0.026, +0.064] ** |
| Timestep recall | 0.223 | 0.206 | 0.224 | 0.155 | -0.020 [-0.031, -0.009] ** | -0.003 [-0.017, +0.011] | -0.140 [-0.158, -0.123] ** |
| Timestep F1 | 0.149 | 0.144 | 0.148 | 0.134 | -0.003 [-0.011, +0.004] | -0.006 [-0.015, +0.002] | -0.044 [-0.059, -0.029] ** |
| Median lead (h) | 18.65 | 18.67 | 17.34 | 16.22 | -0.296 [-2.000, +1.484] | -0.123 [-1.743, +1.496] | -2.655 [-4.950, -0.326] ** |
| Capture >=3h | 0.457 | 0.445 | 0.506 | 0.393 | -0.008 [-0.027, +0.012] | +0.077 [+0.054, +0.100] ** | -0.139 [-0.167, -0.112] ** |
| Capture >=6h | 0.411 | 0.398 | 0.436 | 0.337 | -0.017 [-0.036, +0.002] | +0.047 [+0.027, +0.068] ** | -0.144 [-0.169, -0.118] ** |
| Capture >=12h | 0.323 | 0.312 | 0.346 | 0.265 | -0.012 [-0.028, +0.005] | +0.046 [+0.028, +0.065] ** | -0.120 [-0.144, -0.096] ** |
| Realised burden (alerts/pt-day) | 0.91 | 0.85 | 0.94 | 0.55 | -0.099 [-0.138, -0.060] ** | +0.045 [+0.003, +0.086] ** | -0.655 [-0.728, -0.589] ** |

## Window sweep (XGBoost), validation-frozen thresholds

| | W=3 h | W=6 h | W=12 h | W=24 h | W=6 h - W=3 h (95% CI) | W=12 h - W=3 h (95% CI) | W=24 h - W=3 h (95% CI) |
|---|---|---|---|---|---|---|---|
| **Patient recall** | 0.624 | 0.595 | 0.630 | 0.591 | -0.030 [-0.050, -0.008] ** | +0.005 [-0.020, +0.030] | -0.033 [-0.058, -0.006] ** |
| Patient precision | 0.205 | 0.214 | 0.218 | 0.224 | +0.009 [+0.002, +0.016] ** | +0.013 [+0.005, +0.021] ** | +0.019 [+0.011, +0.029] ** |
| Timestep precision | 0.032 | 0.057 | 0.093 | 0.151 | +0.025 [+0.021, +0.029] ** | +0.061 [+0.054, +0.069] ** | +0.119 [+0.105, +0.133] ** |
| Timestep recall | 0.241 | 0.206 | 0.199 | 0.190 | -0.035 [-0.051, -0.018] ** | -0.042 [-0.060, -0.023] ** | -0.051 [-0.072, -0.031] ** |
| Timestep F1 | 0.056 | 0.089 | 0.127 | 0.168 | +0.033 [+0.027, +0.039] ** | +0.071 [+0.062, +0.080] ** | +0.112 [+0.099, +0.125] ** |
| Median lead (h) | 16.85 | 17.21 | 17.85 | 18.81 | +0.722 [-0.564, +2.000] | +1.264 [-0.041, +2.847] | +2.245 [+0.791, +3.943] ** |
| Capture >=3h | 0.546 | 0.535 | 0.585 | 0.551 | -0.011 [-0.033, +0.011] | +0.039 [+0.013, +0.066] ** | +0.005 [-0.021, +0.033] |
| Capture >=6h | 0.477 | 0.464 | 0.516 | 0.504 | -0.014 [-0.034, +0.007] | +0.038 [+0.014, +0.062] ** | +0.027 [+0.001, +0.053] ** |
| Capture >=12h | 0.367 | 0.361 | 0.407 | 0.386 | -0.006 [-0.025, +0.012] | +0.040 [+0.017, +0.061] ** | +0.019 [-0.006, +0.041] |
| Realised burden (alerts/pt-day) | 0.98 | 0.92 | 1.00 | 0.95 | -0.054 [-0.078, -0.030] ** | +0.025 [-0.008, +0.058] | -0.031 [-0.069, +0.007] |

### Attention at W=3, validation-frozen thresholds

| | XGBoost | Transformer | Transformer - XGBoost (95% CI) |
|---|---|---|---|
| **Patient recall** | 0.624 | 0.538 | -0.086 [-0.126, -0.043] ** |
| Patient precision | 0.205 | 0.241 | +0.029 [+0.008, +0.051] ** |
| Timestep precision | 0.032 | 0.037 | +0.006 [+0.000, +0.011] ** |
| Timestep recall | 0.241 | 0.271 | +0.031 [-0.002, +0.064] |
| Timestep F1 | 0.056 | 0.065 | +0.010 [+0.000, +0.019] ** |
| Median lead (h) | 16.85 | 14.23 | -1.881 [-4.618, +0.922] |
| Capture >=3h | 0.546 | 0.469 | -0.064 [-0.105, -0.021] ** |
| Capture >=6h | 0.477 | 0.396 | -0.068 [-0.110, -0.027] ** |
| Capture >=12h | 0.367 | 0.297 | -0.062 [-0.102, -0.022] ** |
| Realised burden (alerts/pt-day) | 0.98 | 0.94 | -0.058 [-0.146, +0.032] |

## Window sweep (XGBoost), burden-equalised thresholds

| | W=3 h | W=6 h | W=12 h | W=24 h | W=6 h - W=3 h (95% CI) | W=12 h - W=3 h (95% CI) | W=24 h - W=3 h (95% CI) |
|---|---|---|---|---|---|---|---|
| **Patient recall** | 0.624 | 0.595 | 0.596 | 0.591 | -0.030 [-0.050, -0.008] ** | -0.029 [-0.055, -0.005] ** | -0.033 [-0.058, -0.006] ** |
| Patient precision | 0.205 | 0.214 | 0.224 | 0.224 | +0.009 [+0.002, +0.016] ** | +0.019 [+0.011, +0.028] ** | +0.019 [+0.011, +0.029] ** |
| Timestep precision | 0.032 | 0.057 | 0.097 | 0.151 | +0.025 [+0.021, +0.029] ** | +0.065 [+0.057, +0.074] ** | +0.119 [+0.105, +0.133] ** |
| Timestep recall | 0.241 | 0.206 | 0.185 | 0.190 | -0.035 [-0.051, -0.018] ** | -0.056 [-0.075, -0.039] ** | -0.051 [-0.072, -0.031] ** |
| Timestep F1 | 0.056 | 0.089 | 0.127 | 0.168 | +0.033 [+0.027, +0.039] ** | +0.071 [+0.061, +0.081] ** | +0.112 [+0.099, +0.125] ** |
| Median lead (h) | 16.85 | 17.21 | 18.00 | 18.81 | +0.722 [-0.564, +2.000] | +1.500 [+0.000, +3.000] | +2.245 [+0.791, +3.943] ** |
| Capture >=3h | 0.546 | 0.535 | 0.546 | 0.551 | -0.011 [-0.033, +0.011] | -0.001 [-0.025, +0.026] | +0.005 [-0.021, +0.033] |
| Capture >=6h | 0.477 | 0.464 | 0.482 | 0.504 | -0.014 [-0.034, +0.007] | +0.004 [-0.019, +0.028] | +0.027 [+0.001, +0.053] ** |
| Capture >=12h | 0.367 | 0.361 | 0.375 | 0.386 | -0.006 [-0.025, +0.012] | +0.007 [-0.014, +0.029] | +0.019 [-0.006, +0.041] |
| Realised burden (alerts/pt-day) | 0.98 | 0.92 | 0.89 | 0.95 | -0.054 [-0.078, -0.030] ** | -0.093 [-0.126, -0.062] ** | -0.031 [-0.069, +0.007] |

### Attention at W=3, burden-equalised thresholds

| | XGBoost | Transformer | Transformer - XGBoost (95% CI) |
|---|---|---|---|
| **Patient recall** | 0.624 | 0.529 | -0.086 [-0.126, -0.043] ** |
| Patient precision | 0.205 | 0.242 | +0.029 [+0.008, +0.051] ** |
| Timestep precision | 0.032 | 0.037 | +0.006 [+0.000, +0.011] ** |
| Timestep recall | 0.241 | 0.261 | +0.031 [-0.002, +0.064] |
| Timestep F1 | 0.056 | 0.065 | +0.010 [+0.000, +0.019] ** |
| Median lead (h) | 16.85 | 14.39 | -1.881 [-4.618, +0.922] |
| Capture >=3h | 0.546 | 0.463 | -0.064 [-0.105, -0.021] ** |
| Capture >=6h | 0.477 | 0.390 | -0.068 [-0.110, -0.027] ** |
| Capture >=12h | 0.367 | 0.293 | -0.062 [-0.102, -0.022] ** |
| Realised burden (alerts/pt-day) | 0.98 | 0.89 | -0.058 [-0.146, +0.032] |


`**` marks a paired 95% interval excluding zero.

Built in 17.5 min.
