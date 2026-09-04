# MVE run report (DATA_ACCESS_SPEC section 11)

Episodes: `results/mimic_mve_episodes.pkl`
Seeds 42,123,456, device cuda, 0.3 min.

## Acceptance criteria

| Criterion | Value | Pass |
|---|---|---|
| patient prevalence within 2x of PhysioNet 8.8% | 16.0% | PASS |
| timestep prevalence within 2x of PhysioNet 2.2% | 9.0% | FAIL |
| mean test AUROC in [0.70, 0.85] | 0.679 +/- 0.016 | FAIL |
| no subject leakage across splits | verified | PASS |
| pipeline runs end to end | yes | PASS |

## Metrics (3 seeds, mean +/- sd)

| Metric | MIMIC-IV MVE | PhysioNet Config I |
|---|---|---|
| Test AUROC | 0.679 +/- 0.016 | 0.814 +/- 0.004 |
| Test AUPRC | 0.156 +/- 0.014 | 0.144 |

At the primary operating point (threshold set for 70% patient recall; achieved 71%):

| Metric | MIMIC-IV MVE | PhysioNet Config I |
|---|---|---|
| Timestep precision | 0.187 +/- 0.019 | 0.093 |
| False alerts / patient-day | 3.90 +/- 0.79 | 1.7 |
| Median lead time (h) | -0.8 +/- 4.2 | 23.5 |
| Capture >=3h before onset | 0.31 | - |
| Capture >=6h before onset | 0.24 | - |
| Capture >=12h before onset | 0.22 | - |

## Split

| Split | Episodes | Subjects | Septic |
|---|---|---|---|
| train | 483 | 483 | 78 |
| val | 106 | 103 | 16 |
| test | 104 | 104 | 17 |
