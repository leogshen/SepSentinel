# Bedside clinical scores on our cohort

Episodes: `results/mve_ext_5k.pkl`; test split only (these are rule-based, nothing is fitted).

SOFA is deliberately excluded: our label IS a >=2-point SOFA rise, so scoring SOFA against it would be near-tautological.

## Threshold-free discrimination

| Score | AUROC | AUPRC |
|---|---|---|
| sirs | 0.551 | 0.028 |
| qsofa | 0.587 | 0.030 |
| news2 | 0.626 | 0.037 |

## At equal alert burden

**Budget: 0.5 false alerts per nonseptic patient-day**

| Score | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| sirs | 0.07 | 0.0 | 0.01 | 0.01 |
| qsofa | 0.24 | 12.4 | 0.18 | 0.15 |
| news2 | 0.24 | 18.5 | 0.19 | 0.15 |

**Budget: 1.0 false alerts per nonseptic patient-day**

| Score | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| sirs | 0.07 | 0.0 | 0.01 | 0.01 |
| qsofa | 0.24 | 12.4 | 0.18 | 0.15 |
| news2 | 0.40 | 14.0 | 0.33 | 0.24 |

**Budget: 2.0 false alerts per nonseptic patient-day**

| Score | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| sirs | 0.33 | 11.4 | 0.24 | 0.15 |
| qsofa | 0.24 | 12.4 | 0.18 | 0.15 |
| news2 | 0.57 | 15.0 | 0.46 | 0.36 |

**Budget: 4.0 false alerts per nonseptic patient-day**

| Score | Patient recall | Median lead (h) | Capture >=6h | Capture >=12h |
|---|---|---|---|---|
| sirs | 0.33 | 11.4 | 0.24 | 0.15 |
| qsofa | 0.78 | 16.3 | 0.61 | 0.48 |
| news2 | 0.72 | 14.0 | 0.55 | 0.40 |
