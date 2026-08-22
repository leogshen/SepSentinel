# DATA_ACCESS_SPEC — MIMIC-IV & SICdb Extraction for SepSentinel

Status: pre-credentialing specification (2026-08-22). Execute on access.
Grounded in the current codebase; every "verify" tag marks something that must
be checked against the live database before extraction, not guessed.

---

## 0. The contract the extraction must satisfy

Everything downstream of the loader is dataset-agnostic. The entire existing
pipeline (Strategy-B preprocessing, trajectory features, Transformer, training,
corrected patient-level evaluation) consumes one structure, produced today by
`sepsentinel/data/physionet.py::load_physionet()`:

```python
episode = {
    "patient_id": str,          # unique per ICU stay
    "time":       (T,) float32, # hours since ICU admission, hourly grid
    "signals":    (T, F) float32 with NaN where unobserved,
    "labels":     (T,) float32, # per-hour 0/1 sepsis label (pre-shifted; see §3)
    "label":      int,          # patient-level (any timestep positive)
    "onset_step": int|None,     # first positive label index
    "features":   [str],        # feature names, vitals first
}
```

New loaders (`sepsentinel/data/mimic.py`, `sepsentinel/data/sicdb.py`) must emit
exactly this, **plus** new fields required by §3 and §8:

```python
    "subject_id":   str,        # person identity (NOT stay) — for splitting
    "stay_id":      str,
    "t_sepsis_hour": float|None, # raw clinical onset, UNSHIFTED, hours from intime
    "dataset":      "mimic4" | "sicdb",
}
```

Everything else in this document describes how to fill that structure.

---

## 1. Research question

Early prediction of sepsis onset in ICU patients from longitudinal
vital-sign and lab trajectories, optimizing not raw discrimination but the
**deployable alarm trade-off**: maximize septic patients captured >=3h/6h/12h
before clinical onset while minimizing false-alert episodes per nonseptic
patient-day. Current PhysioNet baseline (Config I, corrected metrics): at 70%
patient recall — precision 9.3%, 1.7 false alerts/patient-day, median lead
23.5h. MIMIC-IV/SICdb must beat or externally validate that operating point.

---

## 2. Cohort definition

### Inclusion (both datasets)
- Adult ICU stays: age >= 18 at admission (MIMIC: `patients.anchor_age`;
  SICdb: age field in `cases` — verify name/coding).
- Stay duration >= 6h of charted data (matches `MIN_LENGTH = 6` used by every
  experiment).
- At least one heart-rate measurement in the first 6h (guards against
  administrative/empty stays; new rule — PhysioNet data was pre-cleaned).

### Exclusion
- Sepsis onset (t_sepsis, §3) at or before ICU hour 4: no pre-onset history to
  learn from, and label ambiguity with community-acquired sepsis on arrival.
  **Record how many stays this drops** — if it's a large share of septic stays,
  revisit (PhysioNet's cohort implicitly made a similar cut; exact rule
  unpublished — flagged uncertainty).
- Stays with > 50% of hours having zero observations of any kind.
- Do NOT exclude suspected-infection-without-organ-dysfunction stays from
  controls. They are the realistic hard negatives for alarm-burden estimation;
  excluding them inflates precision.

### Septic vs nonseptic
- Septic: t_sepsis defined per §3, occurring during the ICU stay
  (t_sepsis <= outtime; onset after ICU discharge = control for this model).
- Nonseptic control: everything else that passes inclusion.

### Repeated admissions
- Unit of modeling = ICU stay (one episode per stay_id).
- Unit of splitting = person (subject_id). All stays of one subject go to the
  same split (§8). Multiple ICU stays within one hospitalization each become
  separate episodes.
- Time is re-zeroed per stay (hours since that stay's intime), matching current
  ICULOS semantics.
- Sensitivity analysis (later, not primary): first-stay-per-subject-only cohort.

### Truncation
- Cap episodes at 336 hours (14 days) — this is the observed PhysioNet maximum,
  keeps the O(T^2) causal attention tractable, and stays inside the positional
  encoding's `max_len=2000`. Post-onset hours beyond t_sepsis + 24h may also be
  truncated (they contribute label-1 timesteps but no early-warning signal);
  make this a loader flag, default ON, and record both variants' label
  prevalence.

---

## 3. Sepsis onset definition and label mapping — the highest-risk section

### What the current code assumes (verified in source)
1. `physionet.py` uses `SepsisLabel` verbatim; `onset_step` = first label-1 hour.
2. CinC 2019 labels are **pre-shifted**: label turns 1 at t_sepsis − 6h.
3. `experiment5_recall_study.py::compute_early_warning_metrics` **hard-codes**
   `t_sepsis = onset_step + 6`. All lead-time and capture metrics depend on
   this constant.

### Required definition (replicate the Challenge, not vanilla Sepsis-3)
To keep continuity with everything trained so far, reconstruct t_sepsis the way
Reyna et al. 2020 (the Challenge organizers) did — which is Sepsis-3 with
specific timing rules:

- **t_suspicion**: earlier of (antibiotic start, culture draw) when the pair
  co-occurs: culture within 24h AFTER antibiotics, or antibiotics within 72h
  AFTER culture.
- **t_SOFA**: first hour where SOFA rises >= 2 points relative to the lowest
  value in the preceding 24h window.
- **t_sepsis = min(t_suspicion, t_SOFA)** provided t_SOFA falls within
  [t_suspicion − 24h, t_suspicion + 12h]; otherwise the stay is not septic by
  this pairing.

MIMIC-IV: do NOT hand-roll this. The MIT-LCP `mimic-code` repository ships
audited SQL for `suspicion_of_infection`, hourly `sofa`, and `sepsis3`
concepts (antibiotics from `hosp.prescriptions`/`hosp.emar`, cultures from
`hosp.microbiologyevents`). Adapt its sepsis3 output to the Challenge timing
rules above (mimic-code's default windows differ slightly — diff them
explicitly and document the delta). SICdb: see §12 — suspicion may be
under-determined there; that is the single biggest SICdb risk.

### Label generation — do it in the loader, not the extraction
Store **unshifted** `t_sepsis_hour` in the episode and generate the training
label at load time:

```
labels[t] = 1  if  t >= t_sepsis_hour - SHIFT   (septic stays; SHIFT default 6)
```

Rationale: (a) PhysioNet compatibility falls out at SHIFT=6; (b) the 3h/6h/12h
early-warning targets in §4 become one parameter instead of three extractions;
(c) the extraction stays label-agnostic, so a label-definition bug never forces
a re-extraction.

### Leakage rules for label construction
- t_suspicion/t_SOFA computation may use any clinical data, but **none of the
  inputs to t_sepsis may enter the model's feature set at finer timing than the
  hourly grid** (e.g., do not add an "antibiotics started" feature — it IS the
  label for suspicion-driven onsets).
- SOFA is computed for labeling only. Its components (bilirubin, platelets,
  creatinine, MAP...) may still be model features as raw hourly values — that
  is legitimate (the model may learn organ dysfunction) — but the *aggregated
  SOFA score itself* must not be a feature: it encodes the label definition.
- Never impute across the onset boundary differently from elsewhere; the
  forward-fill must not know where t_sepsis is.

### Required code change (flagged)
`compute_early_warning_metrics` must take the shift as a parameter (or read
`t_sepsis_hour` from the episode) instead of the hard-coded `+ 6`. Without
this, any SHIFT != 6 run silently reports wrong lead times.

---

## 4. Prediction windows

- **History**: full causal past within the (truncated) stay — the Transformer
  already attends over all prior hours; no fixed lookback window. Minimum
  history before a prediction is scored: none currently; keep as-is but report
  metrics stratified by history length in evaluation (predictions at hour 2
  are near-blind).
- **Targets**: per-timestep "within SHIFT hours of onset or after", SHIFT in
  {3, 6, 12} as three training variants. SHIFT=6 is the PhysioNet-compatible
  primary; 3 and 12 are the candidate early-warning re-targets. Larger SHIFT
  raises label prevalence and rewards earlier alarms at precision cost — that
  trade-off is precisely what the §10 evaluation measures.
- Evaluation lead times remain anchored to **unshifted t_sepsis** in all
  variants, so capture@3h/6h/12h numbers are comparable across SHIFT settings.

---

## 5. Variables to extract

Priority P0 = required (current Config I + creatinine for ablation continuity),
P1 = high-value additions to test, P2 = context. For each: MIMIC-IV source and
itemid (all itemids **verify against d_items/d_labitems on access** — they are
from the standard mimic-code mappings), expected frequency, plausibility filter
(extraction-time; the pipeline additionally clips to `CLIP_RANGES` from
`preprocessing.py`), and missingness treatment (all P0/P1 use Strategy B:
causal ffill -> train-median for leading NaN -> mask + time-since-last delta
channels for labs; vitals value-only).

| # | Variable | P | MIMIC-IV source (verify) | Unit | Expected freq | Extraction filter |
|---|----------|---|--------------------------|------|---------------|-------------------|
| 1 | Heart rate | P0 | icu.chartevents 220045 | bpm | ~1/h charted | 20–250 |
| 2 | Respiratory rate | P0 | icu.chartevents 220210 | /min | ~1/h | 2–60 |
| 3 | SpO2 | P0 | icu.chartevents 220277 | % | ~1/h | 50–100 |
| 4 | Temperature | P0 | icu.chartevents 223762 (°C), 223761 (°F -> convert) | °C | q2–4h | 30–43 |
| 5 | Lactate | P0 | hosp.labevents 50813 (blood gas) | mmol/L | sporadic (ABG-driven) | 0–30 |
| 6 | pH | P0 | hosp.labevents 50820; chartevents 223830 as secondary (verify overlap) | — | sporadic | 6.5–7.8 |
| 7 | Creatinine | P0 | hosp.labevents 50912 | mg/dL | q12–24h | 0–25 |
| 8 | WBC | P0 | hosp.labevents 51301 (also 51300 — reconcile) | 10^3/µL | q12–24h | 0–100 |
| 9 | Platelets | P0 | hosp.labevents 51265 | 10^3/µL | q12–24h | 0–1200 |
| 10 | Bilirubin (total) | P0 | hosp.labevents 50885 | mg/dL | q24h+ | 0–60 |
| 11 | MAP | P1 | icu.chartevents 220052 (art) / 220181 (NIBP) | mmHg | ~1/h | 20–200 |
| 12 | FiO2 | P1 | icu.chartevents 223835 | % | vent-dependent | 21–100 |
| 13 | PaO2 | P1 | hosp.labevents 50821 | mmHg | ABG-driven | 30–600 |
| 14 | Glucose | P1 | hosp.labevents 50931 | mg/dL | q4–12h | 20–1000 |
| 15 | BUN | P1 | hosp.labevents 51006 | mg/dL | q24h | 0–200 |
| 16 | CRP | P1 | hosp.labevents 50889 | mg/L | rare in US ICUs (verify count) | 0–500 |
| 17 | Age | P2 | hosp.patients.anchor_age | years | static | 18–91 (89+ capped) |
| 18 | Sex | P2 | hosp.patients.gender | — | static | — |
| 19 | Admission type | P2 | hosp.admissions | — | static | — |

Notes:
- pH from labevents is specimen-dependent (arterial vs venous). Extract
  specimen type; primary = arterial; record venous separately (verify itemids
  50820 vs 50831).
- P2 statics are justified only as constant channels IF an ablation shows
  gain; the current architecture has no static pathway — do not block
  extraction on this, extract them cheaply and decide later.
- Lactate selection bias is documented in PhysioNet (measured in 38% of
  patients, 65% of septic); expect the same in MIMIC — the mask channel is the
  mitigation, extract order-time faithfully.

### IL-6 — investigate, do not assume
Run on day one, before any pipeline work:

```sql
SELECT itemid, label, fluid, category FROM mimiciv_hosp.d_labitems
 WHERE LOWER(label) LIKE '%interleukin%' OR LOWER(label) LIKE '%il-6%' OR LOWER(label) LIKE '%il6%';
-- then, for each hit:
SELECT COUNT(*), COUNT(DISTINCT subject_id) FROM mimiciv_hosp.labevents WHERE itemid = <hit>;
```

Expectation: absent or measured in a vanishing fraction of stays (it is not a
routine US ICU lab). Whatever the count, record it in the extraction report;
if >0, extract all rows (cheap) — even a few hundred IL-6-annotated stays are
a bridge-validation asset (see CODA discussion in project memory). Repeat the
same investigation against SICdb's laboratory reference list (§12) — European
ICUs measure IL-6 more often than US ones (verify; plausible but unconfirmed).

### Hourly gridding (new code, both datasets)
MIMIC/SICdb are event streams, not grids. Gridding rule, per stay, hour bins
[t, t+1) from intime:
- Vitals: median of in-bin measurements (robust to charting bursts).
- Labs: last in-bin value (charttime, NOT storetime — storetime is a leakage
  vector, values can be back-dated).
- Observation mask for Strategy B = "any measurement in bin" (before filling).
- SICdb 1-min vitals: same hourly grid for phase 1 (continuity); retain raw
  minute data in the extraction so a finer grid (`signals.py` already carries
  `DEFAULT_SAMPLING_INTERVAL_MIN = 5`) is a re-grid, not a re-extract.

---

## 6. Trajectory construction

Reuse the existing implementations unchanged; they are already causal:
- Strategy B (`AblationPreprocessor`): causal forward-fill, train-only medians
  for leading NaN, clip to `CLIP_RANGES`, z-score with train-only mean/std;
  labs get mask (raw, unnormalized) + time-since-last-observation delta
  (z-scored with train-only stats). 9 features -> 19 channels.
- Experiment 4 features (`sepsentinel/data/trajectory.py`): per feature
  diff_1h, causal rolling mean_6h, dev_6h (deviation from own recent
  baseline — the "patient-relative baseline" requirement), computed
  post-imputation pre-normalization; Config I -> 46 channels. `ChannelGate`
  (channel_gate.py) groups channels per raw feature for gating.
- The delta channel's "hours since last observation" is where MIMIC's
  irregular lab timing becomes signal — extraction must preserve true order
  times for this to work (hence charttime, §5).
- One adaptation flag: PhysioNet deltas are small integers (hourly grid,
  short stays); MIMIC deltas can reach 100+ h. The z-scoring absorbs scale,
  but re-check the delta distribution before trusting trained-on-PhysioNet
  checkpoints in transfer experiments.

---

## 7. Data leakage protections (audit checklist)

Existing protections to preserve: patient-level splits; preprocessor stats
(medians, mean/std, delta stats) fit on train only; causal-only forward fill
(no back-fill — Strategy D showed backfill inflates AUROC +0.006 and was
rejected for deployment); causal attention mask in the model.

New rules for MIMIC/SICdb:
1. Split by **subject_id**, not stay_id (§8) — the current `patient_split`
   would leak the same person across splits.
2. Use charttime (clinical event time), never storetime/verified-time.
3. No feature derived from label machinery: no antibiotic/culture indicators,
   no SOFA score as input (§3).
4. Grid bins are [t, t+1) — a value timestamped t+0.5h appears in the bin
   *ending* at t+1, i.e., the model sees it from hour t+1 onward. Never round
   down across the bin edge.
5. pos_weight, thresholds, and all normalization recomputed per dataset from
   its own train split — no PhysioNet statistics reused in MIMIC training runs
   (transfer experiments that deliberately reuse them are separate and must be
   labeled as such).
6. Truncation (§2) decided by rules independent of model outputs.
7. Chronology caveat, flagged not enforced: MIMIC-IV spans ~2008–2019+ with
   shifted dates; true temporal external validation is impossible within
   MIMIC. SICdb serves as the cross-site external set.
8. **Overlap warning (flagged uncertainty)**: PhysioNet/CinC 2019 hospital
   system A is Beth Israel Deaconess — the same hospital as MIMIC. A
   PhysioNet-trained model evaluated on MIMIC-IV is NOT clean external
   validation; some patients may appear in both. Treat MIMIC as a *bigger
   training set*, and SICdb as the *independent validation site*. Do not claim
   external validity from MIMIC alone.

---

## 8. Train/validation/test strategy

- 70/15/15 grouped by subject_id, stratified by subject-level "ever septic
  in any stay". Implement as `GroupShuffleSplit`-style logic replacing
  `sepsentinel/data/splitting.py::patient_split` (episode-level stratified
  today; correct for PhysioNet's one-stay-per-patient, wrong for MIMIC).
- Fixed split seed 42 (continuity with all experiments), 3 training seeds
  {42, 123, 456} (frozen convention from experiment 2).
- All stays of a subject inherit the subject's split.
- Cross-dataset protocol: (a) MIMIC-only train/val/test; (b) PhysioNet-trained
  -> MIMIC test (transfer, with §7.8 caveat); (c) MIMIC-trained -> SICdb test
  (the honest external validation); (d) pooled training with dataset tag —
  later, only after (a)–(c) baselines exist.

---

## 9. What changes in the current codebase (complete list)

| Component | Status | Required change |
|-----------|--------|-----------------|
| `data/physionet.py` | keep | untouched; PhysioNet remains runnable |
| `data/mimic.py`, `data/sicdb.py` | NEW | event->hourly-grid loaders emitting §0 schema |
| `data/splitting.py` | CHANGE | subject-level grouped split (§8) |
| Label generation | NEW | in-loader shifted labels from `t_sepsis_hour`, SHIFT param (§3) |
| `experiment5_recall_study.py::compute_early_warning_metrics` | CHANGE | parameterize the hard-coded `t_sepsis = onset_step + 6` |
| `AblationPreprocessor` (exp3) | keep | works for any feature list; promote from experiment file into `sepsentinel/data/` (it currently lives in an experiment script and is imported cross-experiment — fragile) |
| `data/trajectory.py`, `model_b/channel_gate.py` | keep | unchanged (Experiment 4 still unrun — run it on PhysioNet before or in parallel with MIMIC work) |
| `model_b/transformer.py` | keep | 336h cap keeps T^2 attention fine; `max_len=2000` sufficient |
| `model_b/training.py` | minor | move `lengths` to device (latent CUDA crash, found in audit); pos_weight recomputed per dataset |
| Alarm cooldown/aggregation | NEW | not implemented anywhere; required by §10 |
| `evaluation.py` | keep | fixed-threshold summary only; patient-level metrics live in exp5 code (post-bugfix) |

---

## 10. Primary evaluation plan

Timestep level (existing): AUROC, AUPRC, precision/recall/F1 at best-F1 and at
target-recall thresholds.

Patient level (existing, post-2026-08-19 bugfix — use ONLY the corrected
`collect_patient_predictions`): patient recall / capture at >=3h, >=6h, >=12h
before unshifted t_sepsis; median + IQR lead time; precision at 70% patient
recall (primary operating point); false-alert rate and alerts per nonseptic
patient-day.

New — alarm aggregation/cooldown (build once, evaluate everywhere):
- Merge consecutive above-threshold hours into **alert episodes**; after an
  episode fires, suppress re-alerts for a refractory period R.
- Grid: R in {2, 6, 12} h; report per R: alert episodes per nonseptic
  patient-day (the burden number clinicians feel), septic capture and lead
  time under episode semantics (an alert during suppression still counts as
  the earlier episode's detection).
- Report alongside raw per-hour alarms so PhysioNet-era numbers stay
  comparable.

Stratifications to report (cheap, high-value): by history length at alarm
time, by availability class (the 270-pattern analysis showed measurement
patterns carry signal), by dataset, and MIMIC-vs-SICdb calibration curves.

---

## 11. Minimum viable extraction (run first, ~day 1–3 of access)

Purpose: verify pipeline end-to-end before bulk extraction.

1. MIMIC-IV, 1,000 stays: every septic stay from a random 1,000-stay draw plus
   top-up random controls (expect ~5–15% septic depending on definition —
   recording this prevalence IS one of the MVE's outputs).
2. Variables: P0 rows 1–10 only. Hourly grid per §5.
3. Labels: t_sepsis via adapted mimic-code sepsis3 (§3); SHIFT=6.
4. Run: loader -> subject split -> Strategy B -> existing
   `SepsisTransformer` -> corrected evaluation. Single seed.
5. Acceptance: pipeline runs; label prevalence within 2x of PhysioNet's
   (2.2% timestep / 8.8% patient); NaN densities per variable logged and
   compared to PhysioNet's (HR 7.7%, Temp 66%, labs 89–99%); AUROC on the
   1,000-stay subsample anywhere in 0.70–0.85 (sanity, not a target).
6. Also in MVE: the IL-6 census query (§5), and the d_items/d_labitems
   verification of every itemid in the §5 table.

---

## 12. MIMIC-IV vs SICdb mapping

**Confidence asymmetry, stated plainly**: the MIMIC-IV schema knowledge in
this spec is solid (standard mimic-code mappings, verify itemids on access).
SICdb knowledge is thin — table/field names below are from documentation
memory and must ALL be verified against the SICdb data dictionary on access.

| Aspect | MIMIC-IV | SICdb (verify everything) | Harmonizable? |
|---|---|---|---|
| Scale | ~70–90K ICU stays | ~27K admissions (Salzburg, 2013–2021) | yes |
| Vitals | ~hourly charted | 1-min monitor feed (`data_float_h`?) | yes — grid both to 1h (SICdb keeps raw for later high-res work) |
| Labs | labevents + d_labitems | `laboratory` + reference table | yes, with unit conversion |
| Units | US conventional (creatinine mg/dL, bilirubin mg/dL) | likely SI (µmol/L) | yes — explicit conversion table required per analyte; NEVER trust column names for units |
| Time | absolute (shifted) datetimes | offsets in seconds from admission | yes — both reduce to hours-from-intime |
| Antibiotics | prescriptions/emar | `medication` table | probably |
| Cultures | microbiologyevents | **unknown — may be absent** | THE critical risk: without culture times, Challenge t_suspicion is unreconstructable |
| SOFA inputs | complete | vent/vasopressor data likely present; GCS/urine verify | mostly |
| IL-6 | probably absent (census in §5) | plausibly present sometimes (European practice) — census on access | n/a |

SICdb fallback ladder if cultures are missing (decide only after inspecting
the real schema): (1) antibiotic-initiation + SOFA-rise definition without the
culture pairing (document as a *different* label, never silently pooled with
MIMIC labels); (2) ICD-coded sepsis for patient-level labels only (loses onset
timing — usable for transfer AUROC, not for lead-time metrics); (3) SICdb as
control-only external set for false-alarm-burden validation, which needs no
sepsis labels at all — this weakest fallback is still genuinely useful for the
research question.

Harmonized feature set for cross-dataset runs = the intersection: expect the
10 P0 variables to survive; P1 partially. The pipeline's feature-list
parameterization (`AblationPreprocessor(all_features, selected)`) handles
per-dataset subsets without code changes.

---

## 13. Extraction checklist (execute in order, on credentialing day)

MIMIC-IV:
- [ ] 1. Confirm accessible version (2.2 vs 3.1); record it in every artifact.
- [ ] 2. Verify every itemid in §5 against d_items / d_labitems; fix table.
- [ ] 3. Run IL-6 census (§5); record counts.
- [ ] 4. Pull mimic-code concepts: suspicion_of_infection, hourly SOFA,
        sepsis3; adapt timing windows to Challenge rules (§3); diff and
        document deviations.
- [ ] 5. Build stay-level cohort table (inclusion/exclusion §2) with counts at
        each filter step (CONSORT-style attrition log).
- [ ] 6. MVE extraction (§11): 1,000 stays, P0 variables, hourly grid ->
        parquet/CSV in the §0 episode schema + t_sepsis_hour.
- [ ] 7. Run pipeline on MVE; check acceptance criteria; fix before scaling.
- [ ] 8. Full extraction: all qualifying stays, P0+P1+P2, same schema.
- [ ] 9. Freeze an extraction manifest: query SHAs, row counts, per-variable
        NaN densities, label prevalence, cohort attrition — committed to repo.

SICdb:
- [ ] 10. Download data dictionary; map every §5 variable to real
        table/field/reference IDs; write the unit conversion table.
- [ ] 11. Determine culture-data existence -> pick label rung on the §12
        ladder; document the choice.
- [ ] 12. IL-6 census.
- [ ] 13. MVE-equivalent (500 stays) -> pipeline run.
- [ ] 14. Full extraction + manifest.

Code (can start before access, in order of blocking):
- [ ] 15. Grouped subject-level split (blocks everything).
- [ ] 16. Parameterize the +6 in early-warning metrics (blocks any SHIFT!=6).
- [ ] 17. Grid-and-load module for the §0 schema (test on synthetic events).
- [ ] 18. Alarm-episode/cooldown evaluator (§10).
- [ ] 19. Promote AblationPreprocessor out of experiment3 into the package.

---

## 14. Open uncertainties (flagged, not guessed)

1. Exact PhysioNet cohort filters (Challenge paper under-specifies exclusions)
   — our early-onset exclusion (§2) approximates, prevalence comparison in the
   MVE is the check.
2. All MIMIC itemids until verified on the live d_items/d_labitems.
3. Entire SICdb schema, units, and whether culture times exist.
4. IL-6 presence/frequency in both datasets.
5. BIDMC patient overlap between PhysioNet 2019 and MIMIC-IV (§7.8) — extent
   unknowable from our side; affects claims, not training.
6. mimic-code sepsis3 vs Challenge timing-rule deltas — quantify in step 4.
7. Whether post-onset truncation (§2) shifts pos_weight enough to need
   re-tuned thresholds — measure in MVE.
