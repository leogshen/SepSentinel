# Full MIMIC-IV, extended features, pre-onset target report (DATA_ACCESS_SPEC sections 10-11)

Episodes: `results/mimic31_full_ext_prodrome.pkl`
Seeds 42,123,456, device cuda, 32.1 min.

## Acceptance criteria

| Criterion | Value | Pass |
|---|---|---|
| patient prevalence within 2x of PhysioNet 8.8% | 11.5% | PASS |
| timestep prevalence within 2x of PhysioNet 2.2% | 2.6% | PASS |
| mean test AUROC in [0.70, 0.85] | 0.751 +/- 0.001 | PASS |
| no subject leakage across splits | verified | PASS |
| pipeline runs end to end | yes | PASS |

## Metrics (3 seeds, mean +/- sd)

| Metric | This run | PhysioNet Config I |
|---|---|---|
| Test AUROC | 0.751 +/- 0.001 | 0.814 +/- 0.004 |
| Test AUPRC | 0.085 +/- 0.000 | 0.144 |

No recall target is imposed. Each row below is the best patient recall reachable inside the stated false-alert budget.

| Alerts/patient-day budget | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| <= 0.5 | 0.33 +/- 0.01 | 17.8 +/- 0.6 | 0.27 | 0.21 |
| <= 1.0 | 0.49 +/- 0.01 | 18.7 +/- 0.5 | 0.41 | 0.32 |
| <= 2.0 | 0.66 +/- 0.02 | 20.1 +/- 0.3 | 0.56 | 0.45 |
| <= 4.0 | 0.81 +/- 0.02 | 21.5 +/- 0.2 | 0.72 | 0.58 |

## Split

| Split | Episodes | Subjects | Septic |
|---|---|---|---|
| train | 44664 | 33705 | 5174 |
| val | 9496 | 7222 | 1091 |
| test | 9512 | 7223 | 1080 |
