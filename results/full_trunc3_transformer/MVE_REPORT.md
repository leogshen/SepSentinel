# Full MIMIC-IV 3.1 run (truncation +3h) report (DATA_ACCESS_SPEC sections 10-11)

Episodes: `results/mimic31_full_trunc3.pkl`
Seeds 42,123,456, device cuda, 41.8 min.

## Acceptance criteria

| Criterion | Value | Pass |
|---|---|---|
| patient prevalence within 2x of PhysioNet 8.8% | 12.5% | PASS |
| timestep prevalence within 2x of PhysioNet 2.2% | 2.2% | PASS |
| mean test AUROC in [0.70, 0.85] | 0.736 +/- 0.000 | PASS |
| no subject leakage across splits | verified | PASS |
| pipeline runs end to end | yes | PASS |

## Metrics (3 seeds, mean +/- sd)

| Metric | MIMIC-IV MVE | PhysioNet Config I |
|---|---|---|
| Test AUROC | 0.736 +/- 0.000 | 0.814 +/- 0.004 |
| Test AUPRC | 0.072 +/- 0.001 | 0.144 |

At the primary operating point (threshold set for 70% patient recall; achieved 70%):

| Metric | MIMIC-IV MVE | PhysioNet Config I |
|---|---|---|
| Timestep precision | 0.081 +/- 0.003 | 0.093 |
| False alerts / patient-day | 1.65 +/- 0.18 | 1.7 |
| Median lead time (h) | 13.3 +/- 0.4 | 23.5 |
| Capture >=3h before onset | 0.57 | - |
| Capture >=6h before onset | 0.47 | - |
| Capture >=12h before onset | 0.37 | - |

## Split

| Split | Episodes | Subjects | Septic |
|---|---|---|---|
| train | 45076 | 33967 | 5634 |
| val | 9567 | 7279 | 1213 |
| test | 9593 | 7279 | 1186 |
