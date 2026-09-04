# Sepsis-3 label build report

Source: `F:/Claude/Sepsentinel/data_local/mimic-iv-clinical-database-demo-2.2`
Built in 4.2s by `scripts/build_sepsis3_labels.py`.

## Cohort and labels

| Quantity | Value |
|---|---|
| Qualifying ICU stays (age >= 18, LOS >= 6h) | 137 |
| Stays with a suspicion-of-infection pair | 89 (65.0%) |
| Septic stays (Challenge rule) | 64 (46.7%) |
| ... onset at or before ICU hour 4 (excluded by spec section 2) | 53 |
| ... usable septic stays after that exclusion | 11 (8.0%) |
| Septic stays under the mimic-code variant | 81 (59.1%) |

## Onset time (hours from ICU intime, unshifted)

min -10.5 | p25 -2.1 | median 0.7 | p75 2.5 | max 117.5

SOFA at onset: median 6.0 (rise of median 3.0 points vs the preceding 24h minimum).

Onset driven by suspicion vs by the SOFA rise: 48 vs 16 stays.

## Challenge rule vs mimic-code rule

Both rules fire on 64 stays; the mimic-code onset is a median -0.2 h later (p25 -4.6, p75 0.0). Agreement within 1h: 64%.

## Deviations from mimic-code

1. t_SOFA uses the Challenge rule (>=2-point RISE vs the minimum of the preceding 24h). mimic-code's sepsis3.sql instead requires an absolute SOFA >=2 in the suspicion window (baseline-0 assumption). Both are computed; the absolute variant is reported as t_sepsis_hour_mimiccode.
2. Sepsis window is [t_susp-24h, t_susp+12h] (Challenge). mimic-code uses [t_susp-48h, t_susp+24h]; that variant is also reported.
3. Events are binned with floor(hours_from_intime) so hour bins are [t, t+1), matching gridding.py. mimic-code's sofa.sql anchors its hourly windows on the bin end time; the difference is sub-hour.
4. PaO2/FiO2: PaO2 from chartevents 220224 (arterial) and labevents 50821; FiO2 from chartevents 223835 carried forward at most 6h (ASOF join). mimic-code pairs them inside its `bg` concept using specimen linkage.
5. GCS requires all three components (220739/223900/223901) at the same charttime; mimic-code carries components forward up to 6h. In the demo 3244/3279 GCS charttimes are complete triples.
6. The urine-output renal criterion is applied only from hour 24 onward, where the 24h window is full; before that renal scores on creatinine alone. mimic-code scores a partial-window uo_24hr, which inflates the renal component early in a stay.
7. Hourly SOFA is scored only from ICU hour 0 onward; labs drawn up to 24h before intime are clamped into hour 0 rather than scored in pre-ICU hours. Scoring empty pre-ICU hours gives SOFA 0 there (missing data read as normal organ function), which made admission itself look like a >=2-point rise for nearly every stay. mimic-code's sofa.sql does emit hr < 0 rows, but its sepsis3.sql uses an absolute threshold, so it never differences against them.
8. Vasopressin (222315) is extracted but not scored: classical SOFA and mimic-code's sofa.sql both score only dopamine/dobutamine/epinephrine/norepinephrine.
9. Antibiotic times come from hosp.prescriptions.starttime UNION the administration times in hosp.emar (event_txt like 'Administered'). mimic-code's antibiotic.sql uses prescriptions only.
