# SICdb reconnaissance — 2026-09-06

Desk research only: nothing downloaded, no accounts created. Resolves several
"verify on access" items in DATA_ACCESS_SPEC.md section 12 **without** access,
because SICdb publishes its full schema openly (unlike the data).

Sources: `github.com/nrodemund/sicdb` (esp. `Data/DatasetConfiguration.conf`,
which carries the real CREATE TABLE DDL), `sicdb.com/Documentation` wiki,
`physionet.org/content/sicdb/1.0.8/`, the Sci Data 2024 descriptor
(s41597-024-03164-9), and the third-party `github.com/yueritian/sicdb-derived`.

## 1. Access — harder than MIMIC-IV, and not immediate

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

## 5. IL-6 — still open, and cheaply resolvable

Neither interleukin-6 nor procalcitonin appears in any public SICdb material,
but `d_references` (the 426 lab names) ships only inside the restricted
download, so absence of public mention is not evidence of absence. The
descriptor lists only the highest-volume labs; CRP is not named either.

An Austrian tertiary ICU very likely records procalcitonin; IL-6 is plausible
(routine post-CPB marker in German-speaking ICUs) but unconfirmed. Since
1.0.8 attaches LOINC codes, they would be immediately identifiable if present
(IL-6 = LOINC 26881-3, PCT = 33959-8/75241-0).

**Action, no DUA required:** ask the maintainer directly — a GitHub issue on
`nrodemund/sicdb` or an email. He answers issues promptly and substantively.
One question: "does `d_references` contain Interleukin-6 and Procalcitonin
lab items, and roughly how many cases have them?" See
outreach/DATA_REQUESTS.md.

## 6. What this changes in the plan

1. SICdb is **not** a drop-in external validation set for our labels. Spec
   section 8 protocol (c) "MIMIC-trained -> SICdb test (the honest external
   validation)" survives only for *transfer discrimination* under a
   relabelled, antibiotic-only definition, or as a control-only alarm-burden
   set. Any lead-time claim against SICdb would be measuring a different
   clock.
2. It stays valuable for two things MIMIC cannot give: **cross-site alarm
   burden** on a genuinely different population, and **high-resolution
   dynamics** (per-minute vitals) if the project ever revisits the 1-hour
   grid — `signals.py` already carries `DEFAULT_SAMPLING_INTERVAL_MIN = 5`.
3. Access effort should not start before the IL-6 question is answered, since
   that is the one thing that would make SICdb strategically important to
   Model A rather than merely useful to Model B.
