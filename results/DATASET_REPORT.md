# Dataset characterisation

## Cohort size and composition

| Dataset | Episodes (ICU stays) | Unique patients | Stays/patient | Total patient-hours | Septic | Control | Patient-level positive | Timestep-level positive |
|---|---|---|---|---|---|---|---|---|
| `mimic31_full_ext_prodrome.pkl` | 63672 | 48150 | 1.32 | 3185273 | 7345 | 56327 | 11.5% | 2.58% |
| `mimic31_full_trunc3.pkl` | 64236 | 48525 | 1.32 | 3206113 | 8033 | 56203 | 12.5% | 2.23% |

## Record length (hours per ICU stay)

| Dataset | Mean | Median | p25 | p75 | Min | Max | Median septic | Median control |
|---|---|---|---|---|---|---|---|---|
| `mimic31_full_ext_prodrome.pkl` | 50.0 | 35 | 21 | 61 | 6 | 336 | 23 | 37 |
| `mimic31_full_trunc3.pkl` | 49.9 | 35 | 21 | 61 | 6 | 336 | 24 | 37 |

## Missingness per input channel (% of patient-hours with no measurement)

| Channel | mimic31_full_ext_prodrome.pkl | mimic31_full_trunc3.pkl |
|---|---|---|
| bilirubin | 98.4% | 98.4% |
| bun | 93.4% | n/a |
| creatinine | 93.4% | 93.3% |
| fio2 | 94.0% | n/a |
| gcs | 70.8% | n/a |
| glucose | 93.6% | n/a |
| heart_rate | 6.8% | 6.7% |
| lactate | 97.5% | 97.4% |
| map | 13.7% | n/a |
| pao2 | 96.8% | n/a |
| ph | 96.3% | 96.2% |
| platelets | 93.8% | 93.8% |
| respiratory_rate | 8.7% | 8.4% |
| sbp | 14.9% | n/a |
| spo2 | 9.4% | 9.3% |
| temperature | 73.0% | 73.0% |
| urine_output | 56.3% | n/a |
| wbc | 93.9% | 93.8% |

## Static / admission attributes NOT currently used as model inputs

Completeness over the 93224 qualifying ICU stays.

| Attribute | Available | Kind |
|---|---|---|
| age (patients.anchor_age) | 100.0% | static |
| sex (patients.gender) | 100.0% | static |
| first ICU care unit | 100.0% | static |
| admission type (admissions) | 100.0% | static |
| insurance (admissions) | 98.4% | static |
| race (admissions) | 100.0% | static |
| marital status (admissions) | 91.8% | static |
| weight (chartevents 226512/224639) | 97.1% | static-ish (charted once) |
| height (chartevents 226730) | 46.6% | static-ish (charted once) |
