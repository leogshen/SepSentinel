# Bedside clinical scores on our cohort

Episodes: `results/mimic31_full_ext_prodrome.pkl`; test split only (these are rule-based, nothing is fitted).

SOFA is deliberately excluded: our label IS a >=2-point SOFA rise, so scoring SOFA against it would be near-tautological.

## Threshold-free discrimination

| Score | AUROC | AUPRC |
|---|---|---|
| sirs | 0.617 | 0.037 |
| qsofa | 0.617 | 0.036 |
| news2 | 0.638 | 0.040 |

## At equal alert burden

**Budget: 0.5 false alerts per nonseptic patient-day**

| Score | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| sirs | 0.10 | 18.1 | 0.08 | 0.06 |
| qsofa | 0.25 | 22.9 | 0.21 | 0.17 |
| news2 | 0.30 | 22.2 | 0.24 | 0.20 |

**Budget: 1.0 false alerts per nonseptic patient-day**

| Score | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| sirs | 0.10 | 18.1 | 0.08 | 0.06 |
| qsofa | 0.25 | 22.9 | 0.21 | 0.17 |
| news2 | 0.42 | 21.5 | 0.35 | 0.29 |

**Budget: 2.0 false alerts per nonseptic patient-day**

| Score | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| sirs | 0.47 | 21.0 | 0.39 | 0.33 |
| qsofa | 0.25 | 22.9 | 0.21 | 0.17 |
| news2 | 0.54 | 21.2 | 0.47 | 0.37 |

**Budget: 4.0 false alerts per nonseptic patient-day**

| Score | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| sirs | 0.47 | 21.0 | 0.39 | 0.33 |
| qsofa | 0.75 | 24.0 | 0.68 | 0.54 |
| news2 | 0.68 | 22.0 | 0.59 | 0.47 |
