# Temporal Label Smoothing on flat XGBoost (W=12 h)

Episodes: `results/mimic31_full_ext_prodrome.pkl`. Grouped subject split, seed 42. Only the training targets differ between arms -- same channels, same XGBoost settings (native API, 200 rounds, depth 6, eta 0.1), same seed.
Built in 5.8 min.

Soft-label weighting is the exact weighted-CE reduction `w = spw*q + (1-q)`, `y = spw*q/w`, which reduces to `scale_pos_weight` semantics at q in {0,1}, so both arms are weighted on one scale. TLS is keyed to `t_sepsis_hour`, not to the last positive row. **Operating thresholds are frozen on validation** and applied unchanged to test, so no test-set maximum enters these numbers; the realised test burden is reported rather than assumed.

The TLS ramp carries only **27.7%** of the hard-label positive mass, so a plain TLS-vs-hard contrast confounds the *shape* of the target with *less total positive weight*. The third arm rescales `spw` by 3.61x to match total mass, isolating the shape effect.

## Test set, threshold frozen on validation

| | Hard | TLS | TLS mass-matched | TLS - hard (95% CI) | TLS-mm - hard (95% CI) |
|---|---|---|---|---|---|
| **Patient recall** | 0.623 | 0.651 | 0.629 | +0.028 [+0.006, +0.050] ** | +0.005 [-0.016, +0.027] |
| Patient precision | 0.225 | 0.226 | 0.221 | +0.001 [-0.006, +0.008] | -0.003 [-0.011, +0.004] |
| Timestep precision | 0.095 | 0.098 | 0.099 | +0.002 [-0.003, +0.008] | +0.004 [-0.002, +0.009] |
| Timestep recall | 0.188 | 0.208 | 0.198 | +0.019 [+0.008, +0.031] ** | +0.010 [-0.001, +0.021] |
| Timestep F1 | 0.127 | 0.133 | 0.132 | +0.006 [-0.001, +0.013] | +0.005 [-0.001, +0.013] |
| Median lead (h) | 20.04 | 18.58 | 19.22 | -1.770 [-3.067, -0.095] ** | -0.778 [-2.036, +0.953] |
| Capture >=3h | 0.581 | 0.589 | 0.578 | +0.007 [-0.015, +0.030] | -0.004 [-0.024, +0.018] |
| Capture >=6h | 0.518 | 0.512 | 0.508 | -0.006 [-0.027, +0.015] | -0.009 [-0.030, +0.011] |
| Capture >=12h | 0.412 | 0.401 | 0.393 | -0.011 [-0.030, +0.008] | -0.020 [-0.039, +0.000] |
| Realised burden (alerts/pt-day) | 0.92 | 1.01 | 0.95 | +0.087 [+0.055, +0.119] ** | +0.029 [-0.002, +0.059] |

`**` marks a paired 95% interval excluding zero. Intervals are on the difference, from 2000 patient-level resamples in which every arm sees the same resampled patients.

## Test set, burden equalised across arms

Same models, threshold re-pinned so every arm spends the same test burden. This touches test data, but only the control patients' alarm rate -- never the septic outcomes being compared -- so it is far weaker than choosing a threshold to maximise test recall. Read it as the equal-burden view that the frozen-threshold table above cannot quite deliver.

| | Hard | TLS | TLS mass-matched | TLS - hard (95% CI) | TLS-mm - hard (95% CI) |
|---|---|---|---|---|---|
| **Patient recall** | 0.623 | 0.624 | 0.629 | +0.001 [-0.021, +0.024] | +0.005 [-0.016, +0.027] |
| Patient precision | 0.225 | 0.229 | 0.221 | +0.004 [-0.004, +0.012] | -0.003 [-0.011, +0.004] |
| Timestep precision | 0.095 | 0.100 | 0.099 | +0.005 [-0.001, +0.010] | +0.004 [-0.002, +0.009] |
| Timestep recall | 0.188 | 0.195 | 0.198 | +0.006 [-0.005, +0.018] | +0.010 [-0.001, +0.021] |
| Timestep F1 | 0.127 | 0.132 | 0.132 | +0.006 [-0.002, +0.013] | +0.005 [-0.001, +0.013] |
| Median lead (h) | 20.04 | 18.75 | 19.22 | -1.641 [-3.032, -0.025] ** | -0.778 [-2.036, +0.953] |
| Capture >=3h | 0.581 | 0.561 | 0.578 | -0.020 [-0.042, +0.002] | -0.004 [-0.024, +0.018] |
| Capture >=6h | 0.518 | 0.487 | 0.508 | -0.031 [-0.052, -0.010] ** | -0.009 [-0.030, +0.011] |
| Capture >=12h | 0.412 | 0.381 | 0.393 | -0.032 [-0.051, -0.012] ** | -0.020 [-0.039, +0.000] |
| Realised burden (alerts/pt-day) | 0.92 | 0.92 | 0.95 | -0.007 [-0.038, +0.025] | +0.029 [-0.002, +0.059] |

| Arm | Thr (val) | Thr (burden-eq) | spw | AUROC | AUPRC |
|---|---|---|---|---|---|
| hard | 0.73 | 0.73 | 37.5 | 0.7360 | 0.0712 |
| tls | 0.43 | 0.44 | 37.5 | 0.7390 | 0.0761 |
| tls_massmatched | 0.72 | 0.72 | 135.4 | 0.7349 | 0.0746 |
