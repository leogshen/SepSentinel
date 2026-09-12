#Desk research only: nothing downloaded, no accounts created. Resolves several
"verify on access" items in DATA_ACCESS_SPEC.md section 12 **without** access,
because SICdb publishes its full schema openly (unlike the data).

Sources: `github.com/nrodemund/sicdb` (esp. `Data/DatasetConfiguration.conf`,
which carries the real CREATE TABLE DDL), `sicdb.com/Documentation` wiki,
`physionet.org/content/sicdb/1.0.8/`, the Sci Data 2024 descriptor
(s41597-024-03164-9), and the third-party `github.com/yueritian/sicdb-derived`.

## 1. Access — harder than MIMIC-IV, and not immediate

**RESOLVED 2026-09-08: access was granted and the full 1.0.8 archive
downloaded (2.38 GB, all 8 tables plus Documentation.pdf and
d_references). The gates below were cleared; they are left in place as
the record of what it took. Data lives outside the repo at
`F:/Claude/Sepsentinel/data_local/sicdb-1.0.8`, same DUA hygiene as
MIMIC-IV.**

PhysioNet 1.0.8, verbatim: "Only credentialed users who sign the DUA can
access the files. **In addition, users must have individual studies reviewed
by the contributor.**" Required training: CITI *Data or Specimens Only
Research*. Three gates: credentialed account, CITI certificate, and a
per-project request approved by the Salzburg team.

- **No demo or open subset exists** (PhysioNet lists only 1.0.5/1.0.6/1.0.8;
  the `SICdb_MEDS` ETL tests against synthetic data, not a real subset).
- Same PI-sponsorship blocker as MIMIC-IV, plus a human review step whose
  turnaround is author-dependent — the maintainer himself noted v1.0.8 sat in
  PhysioNet review for months.
- What IS free: the schema, DDL, item-ID conventions and example queries.
  A SICdb loader can be written and unit-tested against synthetic events
  before access exists — exactly how `scripts/test_sepsis3_rules.py` works.

## 2. The critical risk in spec section 12 is now CONFIRMED, not hypothetical

**SICdb contains no microbiology, culture, specimen or organism data at all.**
Not in any of the 8 distributed tables, not in the wiki file list. The
third-party derived project reached the same conclusion independently:

> "no reliable culture timestamp has been identified in the reviewed tables
> ... suspected infection is represented by systemic antibacterial exposure"

So the Challenge-rule `t_suspicion` we implemented in
`sepsentinel/data/sepsis3.py` — antibiotics PAIRED with cultures — **cannot be
reproduced in SICdb**. The only other infection signal is
`cases.AdmissionFormHasSepsis`, which the maintainer describes as the
admitting physician's SAPS3 checkbox impression, with acknowledged observer
bias and no onset timing.

This selects **rung 1** of the section-12 fallback ladder (antibiotic-only
suspicion, documented as a *different label*, never silently pooled with
MIMIC labels), with **rung 3** (control-only external set for false-alarm
burden, which needs no labels at all) as the honest floor.

## 3. Schema (verified from the public DDL)

8 gzipped CSVs. `cases` 27,386 rows (1 per ICU admission) | `laboratory`
17.6M | `medication` 5.1M | `data_float_h` 36.8M | `data_ref` 354k |
`data_range` 183k | `unitlog` 140k | `d_references` 1,608 (the dictionary).

| Need | SICdb | Verdict |
|---|---|---|
| Time base | integer `Offset` = **seconds since PDMS admission** | maps to our hours-from-intime — but subtract `cases.ICUOffset`, offset 0 is pre-surgery admission, NOT ICU admission |
| Vitals | `data_float_h(CaseID, DataID, Offset, Val, cnt, rawdata)`; `Val` = hourly mean, `rawdata` = up to 60 per-minute floats | far denser than MIMIC (HR 56.8 measurements/h vs 1.09); grids to 1h trivially, keeps a high-resolution path open |
| Labs | `laboratory(...)`, 426 parameters, `LaboratoryType` flags arterial BGA | **v1.0.8 adds LOINC codes** to lab references — a real advantage for harmonising with MIMIC-IV |
| Antibiotics | `medication.DrugID` -> `d_references`, INN-unified names, not free text; bolus vs infusion distinguished; `AmountPerMinute` gives true rates | good |
| Cultures | **absent** | **blocker, see above** |
| Vasopressors | `AmountPerMinute` + `WeightOnAdmission`; norepinephrine 1562, epinephrine 1502, dopamine 1618, dobutamine 1559 | usable |
| Urine output | hourly, `DataID=725` | usable (official KDIGO script warns it overestimates AKI) |
| Ventilation | respirator settings as signals (FiO2, PEEP, vent RR) + airway device items; **no explicit ventilated yes/no flag** | derivable |
| GCS | added 1.0.7 but the maintainer says: "it is not mandatory on our ICU and therefore not valid enough" | **unusable**; the derived project substitutes a RASS-based neuro surrogate |
| Units | creatinine **mg/dl**, Hb g/dl, PO2 mmHg — US-style, NOT the SI we assumed in the spec table. Bilirubin unverified | spec section 12 row "likely SI (umol/L)" is **wrong** — read `d_references.ReferenceUnit` per analyte |

## 4. Scale and case mix

27,386 admissions / 21,583 patients, 2013-2021, University Hospital Salzburg,
4 ICUs. Roughly half MIMIC-IV's case count but far denser per case.

**Selection bias the authors state plainly:** the non-surgical/internal
medicine ICU is excluded for technical reasons, so the cohort is heavily
perioperative and cardiac-surgery weighted with correspondingly low
mortality, and two of the four wards are intermediate care without invasive
ventilation, CRRT or ECMO. For a *sepsis* cohort this matters: it is not a
representative medical-ICU population.

## 5. IL-6 — RESOLVED 2026-09-12: present, in quantity

Counted directly from `d_references.csv.gz` and `laboratory.csv.gz`. The
guessed LOINC codes were both right.

| DataID | Name | Unit | LOINC |
|---|---|---|---|
| 569 | Interleukin 6 (ZL) | pg/ml | 26881-3 |
| 570 | Interleukin 6 / Cordalblut | pg/ml | 26881-3 |
| 568 | Interleukin 10 (ZL) | pg/ml | 26848-2 |
| 263 | Procalcitonin (ZL) | ug/l | 33959-8 |
| 341 / 351 | C-reactive protein / hs-CRP | mg/dl | 1988-5 / 30522-7 |

Coverage over 27,350 cases:

| Analyte | Cases | % | Measurements | Median |
|---|---|---|---|---|
| CRP | 26,577 | 97.2% | 164,795 | 7.2 mg/dl |
| Procalcitonin | 5,448 | 19.9% | 15,160 | 0.5 ug/l |
| **Interleukin-6** | **4,225** | **15.4%** | 11,677 | 46.0 pg/ml |

Serial IL-6, which is what Model A actually needs:

| >=2 measurements | >=3 | >=5 | >=10 | max | mean |
|---|---|---|---|---|---|
| 1,965 cases | 1,278 | 657 | 198 | 42 | 2.76 |

For scale, the external leads in `outreach/DATA_REQUESTS.md` section 3 were
VASST ancillary (363 patients, 2 timepoints) and a medRxiv longitudinal
study (98 adults). SICdb has **1,965 patients with repeated IL-6 and 657
with five or more**, already on disk.

**The confound this creates.** Section 4's selection bias now cuts against
us specifically: the cohort is heavily perioperative and cardiac-surgery
weighted, and IL-6 is a routine post-CPB marker in German-speaking ICUs.
That is very likely *why* coverage is 15% rather than 2%. So a large share
of those 4,225 cases are post-bypass inflammation, where IL-6 rises without
infection, and the median 46 pg/ml (normal <7) is consistent with either.
Separating "IL-6 in sepsis" from "IL-6 in surgical inflammation" is now the
central Model-A design problem in this dataset. `cases` carries the
admission and surgical fields to stratify on. This is a better problem to
have than "no serial IL-6 anywhere", but it is not the clean cohort the
plan assumed.

Note also that the cultures blocker in section 2 is unchanged: these are ICU
patients with IL-6 trajectories, not confirmed-septic patients with IL-6
trajectories.

## 6. What this changes in the plan

1. SICdb is **not** a drop-in external validation set for our labels. Spec
   section 8 protocol (c) "MIMIC-trained -> SICdb test (the honest external
   validation)" survives only for *transfer discrimination* under a
   relabelled, antibiotic-only definition, or as a control-only alarm-burden
   set. Any lead-time claim against SICdb would be measuring a different
   clock.
2. It stays valuable for two things MIMIC cannot give: **cross-site alarm
   burden** on a genuinely different population, and **high-resolution
   dynamics** (per-minute vitals).

   Two corrections to this point, 2026-09-12. First, the parenthetical about
   `signals.py` carrying `DEFAULT_SAMPLING_INTERVAL_MIN = 5` implied the
   codebase is ready for sub-hourly data. It is not: that constant's only
   consumer is `sepsentinel/data/sequences.py`, which nothing in the repo
   imports — it is vestigial from the synthetic-wearable era. The live MIMIC
   path is hour-based throughout (`gridding.grid_stay(events, n_hours, ...)`,
   hourly `make_labels`, alerts per patient-day, capture by lead hour), so
   adopting 5-minute data is a new pipeline, not a config change.

   Second, the high-resolution motivation is weaker than written. Flat
   XGBoost matches or beats the causal Transformer on every deployment
   metric, and the 2026-09-12 window experiment reconfirmed it at the window
   most favourable to early detection; HiRID's history ablation finds
   sequence models extract almost nothing beyond ~12 h. If attention over
   1-hour steps does not earn its keep, 5-minute steps multiply sequence
   length 12x to chase temporal structure the hourly analysis says goes
   unused. And per RESULTS.md the ceiling is *lab* sparsity — per-minute
   resolution improves only HR/SpO2/RR, already at ~95% hourly coverage, so
   it adds resolution where coverage is already good and none where it is
   not. The good framing for the per-minute data is not "revisit the 1-hour
   grid" but "continuous vitals with no labs is the closest available proxy
   for what a wearable actually produces" — a Model A argument.
3. ~~Access effort should not start before the IL-6 question is answered~~
   **DONE.** Access obtained, IL-6 confirmed present in quantity (section 5).
   SICdb is now strategically important to Model A, not merely useful to
   Model B — but the question it answers has changed: not "are there serial
   IL-6 trajectories" (there are, 1,965 of them) but "can IL-6 in sepsis be
   separated from IL-6 in post-surgical inflammation in this cohort".
