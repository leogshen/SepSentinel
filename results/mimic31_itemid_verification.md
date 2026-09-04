# MIMIC-IV itemid verification and IL-6 census

Source: `F:/Claude/Sepsentinel/data_local/mimic-iv-3.1`

## Scale

| ICU stays | Patients | d_items rows | d_labitems rows |
|---|---|---|---|
| 94458 | 364627 | 4095 | 1650 |

## Itemids

| itemid | table | used for | label in DB |
|---|---|---|---|
| 50813 | d_labitems | extract: lactate | Lactate |
| 50820 | d_labitems | extract: ph | pH |
| 50821 | d_labitems | SOFA: PaO2 | pO2 |
| 50885 | d_labitems | SOFA: bilirubin; extract: bilirubin | Bilirubin, Total |
| 50912 | d_labitems | SOFA: creatinine; extract: creatinine | Creatinine |
| 51265 | d_labitems | SOFA: platelets; extract: platelets | Platelet Count |
| 51301 | d_labitems | extract: wbc | White Blood Cells |
| 220045 | d_items | extract: heart_rate | Heart Rate |
| 220052 | d_items | SOFA: MAP | Arterial Blood Pressure mean |
| 220181 | d_items | SOFA: MAP | Non Invasive Blood Pressure mean |
| 220210 | d_items | extract: respiratory_rate | Respiratory Rate |
| 220224 | d_items | SOFA: PaO2 (arterial) | Arterial O2 pressure |
| 220277 | d_items | extract: spo2 | O2 saturation pulseoxymetry |
| 220739 | d_items | SOFA: GCS eye | GCS - Eye Opening |
| 221289 | d_items | SOFA: epinephrine | Epinephrine |
| 221653 | d_items | SOFA: dobutamine | Dobutamine |
| 221662 | d_items | SOFA: dopamine | Dopamine |
| 221906 | d_items | SOFA: norepinephrine | Norepinephrine |
| 222315 | d_items | SOFA: vasopressin | Vasopressin |
| 223761 | d_items | extract: temperature_f | Temperature Fahrenheit |
| 223762 | d_items | extract: temperature | Temperature Celsius |
| 223835 | d_items | SOFA: FiO2 | Inspired O2 Fraction |
| 223900 | d_items | SOFA: GCS verbal | GCS - Verbal Response |
| 223901 | d_items | SOFA: GCS motor | GCS - Motor Response |
| 225312 | d_items | SOFA: MAP | ART BP Mean |
| 225792 | d_items | SOFA: ventilation | Invasive Ventilation |
| 225794 | d_items | SOFA: ventilation | Non-invasive Ventilation |
| 226557 | d_items | SOFA: urine output | R Ureteral Stent |
| 226558 | d_items | SOFA: urine output | L Ureteral Stent |
| 226559 | d_items | SOFA: urine output | Foley |
| 226560 | d_items | SOFA: urine output | Void |
| 226561 | d_items | SOFA: urine output | Condom Cath |
| 226563 | d_items | SOFA: urine output | Suprapubic |
| 226564 | d_items | SOFA: urine output | R Nephrostomy |
| 226565 | d_items | SOFA: urine output | L Nephrostomy |
| 226567 | d_items | SOFA: urine output | Straight Cath |
| 226584 | d_items | SOFA: urine output | Ileoconduit |
| 227489 | d_items | SOFA: urine output | GU Irrigant/Urine Volume Out |

## Interleukin / procalcitonin census

- `d_labitems`: no matching items.
- `d_items`: no matching items.

**Confirmed: MIMIC-IV carries no interleukin items.** The IL-6 bridge must come from ImmPort SDY1662.
