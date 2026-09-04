# MVE run report (DATA_ACCESS_SPEC section 11)

Episodes: `results/mve_b_5k_trunc24.pkl`
Seeds 42,123,456, device cuda, 1.4 min.

## Acceptance criteria

| Criterion | Value | Pass |
|---|---|---|
| patient prevalence within 2x of PhysioNet 8.8% | 12.7% | PASS |
| timestep prevalence within 2x of PhysioNet 2.2% | 7.0% | FAIL |
| mean test AUROC in [0.70, 0.85] | 0.718 +/- 0.013 | PASS |
| no subject leakage across splits | verified | PASS |
| pipeline runs end to end | yes | PASS |

## Metrics (3 seeds, mean +/- sd)

| Metric | MIMIC-IV MVE | PhysioNet Config I |
|---|---|---|
| Test AUROC | 0.718 +/- 0.013 | 0.814 +/- 0.004 |
| Test AUPRC | 0.197 +/- 0.012 | 0.144 |

At the primary operating point (threshold set for 70% patient recall; achieved 71%):

| Metric | MIMIC-IV MVE | PhysioNet Config I |
|---|---|---|
| Timestep precision | 0.208 +/- 0.007 | 0.093 |
| False alerts / patient-day | 2.45 +/- 0.17 | 1.7 |
| Median lead time (h) | 6.1 +/- 0.3 | 23.5 |
| Capture >=3h before onset | 0.43 | - |
| Capture >=6h before onset | 0.35 | - |
| Capture >=12h before onset | 0.27 | - |

## Split

| Split | Episodes | Subjects | Septic |
|---|---|---|---|
| train | 2431 | 2352 | 308 |
| val | 514 | 504 | 67 |
| test | 513 | 504 | 65 |
