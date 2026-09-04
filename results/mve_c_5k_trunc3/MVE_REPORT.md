# MVE run report (DATA_ACCESS_SPEC section 11)

Episodes: `results/mve_c_5k_trunc3.pkl`
Seeds 42,123,456, device cuda, 1.3 min.

## Acceptance criteria

| Criterion | Value | Pass |
|---|---|---|
| patient prevalence within 2x of PhysioNet 8.8% | 12.7% | PASS |
| timestep prevalence within 2x of PhysioNet 2.2% | 2.3% | PASS |
| mean test AUROC in [0.70, 0.85] | 0.630 +/- 0.011 | FAIL |
| no subject leakage across splits | verified | PASS |
| pipeline runs end to end | yes | PASS |

## Metrics (3 seeds, mean +/- sd)

| Metric | MIMIC-IV MVE | PhysioNet Config I |
|---|---|---|
| Test AUROC | 0.630 +/- 0.011 | 0.814 +/- 0.004 |
| Test AUPRC | 0.063 +/- 0.011 | 0.144 |

At the primary operating point (threshold set for 70% patient recall; achieved 72%):

| Metric | MIMIC-IV MVE | PhysioNet Config I |
|---|---|---|
| Timestep precision | 0.059 +/- 0.009 | 0.093 |
| False alerts / patient-day | 2.50 +/- 0.57 | 1.7 |
| Median lead time (h) | 19.0 +/- 5.8 | 23.5 |
| Capture >=3h before onset | 0.57 | - |
| Capture >=6h before onset | 0.49 | - |
| Capture >=12h before onset | 0.41 | - |

## Split

| Split | Episodes | Subjects | Septic |
|---|---|---|---|
| train | 2431 | 2352 | 308 |
| val | 514 | 504 | 67 |
| test | 513 | 504 | 65 |
