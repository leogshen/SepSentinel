# HANDOFF — Continue SepSentinel on this machine

You are picking up an in-progress project. This file is your complete context;
the previous machine's session memory does not transfer. Read this fully, then
skim: DATA_ACCESS_SPEC.md, TODO.md, MIGRATION.md, DATASETS.md,
outreach/DATA_REQUESTS.md.

## Project in one paragraph

SepSentinel: wearable multimodal sepsis early-warning platform. Model A (future):
electrochemical sensor signals -> biomarker concentrations (IL-6/lactate/pH).
Model B (active work): physiological + lab time series -> per-hour sepsis risk.
Trained so far on PhysioNet/CinC 2019; now upgrading to MIMIC-IV with SICdb as
external validation. The researcher (Leo) is a high-school student — gated data
applications need a sponsoring PI; open-tier data is fine.

## Current production model and the numbers to beat

- Config I: 9 features (HR, SpO2, Resp, Temp, Lactate, pH, WBC, Platelets,
  Bilirubin — creatinine excluded, it hurts), Strategy B preprocessing
  (causal ffill + train-median + lab masks + lab deltas = 19 channels),
  causal Transformer (d_model=64, 2 layers, 4 heads).
- Test AUROC 0.814 +/- 0.004, AUPRC 0.144 (PhysioNet, 3 seeds).
- CORRECTED patient-level baseline (a pairing bug was found+fixed 2026-08-19;
  distrust any patient-level numbers from before that date): at 70% patient
  recall -> precision 9.3%, 1.7 false alerts/patient-day, median lead 23.5h.
- Key negative results: loss reweighting doesn't move AUROC (exp5); MAE
  pretraining doesn't improve discrimination at this scale (exp6) — the
  PhysioNet ceiling is a DATA limitation. Hence the MIMIC upgrade.
- Frozen conventions: split seed 42, training seeds {42,123,456}, epochs 50,
  batch 32, lr 1e-3, patience 7 (defined in experiment2_imputation.py).

## State of the MIMIC-IV pipeline (all committed, all tested)

DONE on 2026-09-03 against MIMIC-IV 3.1 + the open 100-patient demo:
- `sepsentinel/data/gridding.py` — event->hourly-grid episode builder
  (vitals=median-in-bin, labs=last-in-bin, 336h cap, post-onset truncation,
  labels generated from unshifted t_sepsis_hour with label_shift_hours param).
- `sepsentinel/data/splitting.py::grouped_patient_split` — person-level split
  (subject_id), leak-tested. ALWAYS use this for MIMIC, never patient_split.
- `scripts/extract_mimic.py` — DuckDB extraction (works on demo + full 3.1).
  All itemids verified against real 3.1 dictionaries. Temp F->C handled.
- `compute_early_warning_metrics(..., label_shift_hours=6)` — episode
  t_sepsis_hour takes precedence; PhysioNet default unchanged.
- End-to-end verified: demo -> grid -> grouped split -> Strategy B ->
  Transformer forward. Gridded-MIMIC NaN profile matches PhysioNet closely.
- IL-6 census: MIMIC-IV has ZERO interleukin items (definitive; d_labitems
  and d_items both). The IL-6 bridge comes from ImmPort SDY1662 (open,
  downloadable) — see outreach/DATA_REQUESTS.md.

## Machine setup (do first)

1. Clone: `git clone https://github.com/leogshen/SepSentinel` (this file is in it).
2. venv + `pip install -r requirements.txt`, then CUDA torch:
   `pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu124`
   Verify: `python -c "import torch; print(torch.cuda.is_available())"` -> True.
   Also: `pip install duckdb`.
3. PhysioNet 2019 (for regression baselines):
   `python -c "import kagglehub; print(kagglehub.dataset_download('tea340yashjoshi/sepsis-prediction-dataset'))"`
4. MIMIC-IV 3.1: Leo downloads it (credentialed) — likely to Downloads.
   DO NOT unzip fully. Selective extract (13 tables, ~7.6GB, stays .csv.gz):
   `cd C:/data && unzip -o -q <zip> mimic-iv-3.1/hosp/patients.csv.gz mimic-iv-3.1/hosp/admissions.csv.gz mimic-iv-3.1/hosp/labevents.csv.gz mimic-iv-3.1/hosp/d_labitems.csv.gz mimic-iv-3.1/hosp/prescriptions.csv.gz mimic-iv-3.1/hosp/microbiologyevents.csv.gz mimic-iv-3.1/hosp/emar.csv.gz mimic-iv-3.1/icu/icustays.csv.gz mimic-iv-3.1/icu/chartevents.csv.gz mimic-iv-3.1/icu/d_items.csv.gz mimic-iv-3.1/icu/inputevents.csv.gz mimic-iv-3.1/icu/outputevents.csv.gz mimic-iv-3.1/icu/procedureevents.csv.gz -d .`
   DUA hygiene: keep outside any cloud-synced folder (OneDrive!), on Leo's
   account only, encrypted disk preferred. Never in the git repo.

## Work queue, in order

1. **Sepsis-3 Challenge-rule labels — THE blocker.** Everything extracts as
   controls until this exists. Build t_suspicion (antibiotics from
   prescriptions/emar paired with cultures from microbiologyevents:
   culture <=24h after ABX, or ABX <=72h after culture), hourly SOFA
   (inputevents vasopressors, outputevents urine, labevents, chartevents GCS/
   MAP/FiO2+PaO2), t_SOFA (>=2-pt rise vs min of prior 24h), then
   t_sepsis = min(t_suspicion, t_SOFA) if t_SOFA in [t_susp-24h, t_susp+12h].
   Spec section 3 has the exact rules + leakage constraints (no SOFA/ABX as
   model features!). Port from MIT-LCP mimic-code repo (Postgres/BigQuery
   SQL) to DuckDB rather than writing from scratch; document any timing-rule
   deltas. Wire the result into scripts/extract_mimic.py as t_sepsis_hour.
2. **MVE (spec section 11)**: 1,000-stay extraction with labels, run the
   existing pipeline, check acceptance criteria (prevalence within 2x of
   PhysioNet's 2.2%/8.8%, NaN densities logged, AUROC 0.70-0.85 sanity).
3. **TODO items 4-5**: alarm-episode/cooldown evaluator (spec section 10);
   promote AblationPreprocessor from experiment3_feature_ablation.py into
   sepsentinel/data/.
4. **Full training runs** on all qualifying stays, 3 seeds, evaluated with
   the corrected patient-level metrics at the 0.70-patient-recall operating
   point. Compare against the PhysioNet baseline above.
5. Later: SICdb (spec section 12 — verify schema first, culture data may not
   exist there), experiment 4 (trajectory+gating, built but never run),
   ImmPort SDY1662 analysis when Leo downloads it.

## Gotchas that cost us time once already

- collate_fn SORTS batches by length: any per-patient pairing must replicate
  that sort (see fixed collect_patient_predictions in experiment5). Never
  pair by dataset order.
- labevents has no stay_id — join subject_id + charttime within stay window.
  Use charttime, never storetime (leakage).
- PhysioNet 2019 labels are PRE-SHIFTED 6h; MIMIC labels are generated in the
  loader from unshifted t_sepsis_hour. Don't double-shift.
- PhysioNet <-> MIMIC share a hospital (BIDMC): MIMIC is NOT clean external
  validation for PhysioNet-trained models. SICdb is.
- Windows console is cp1252: no unicode symbols in logger/print output.
- Checkpoints (*.pt) are gitignored; retrain (minutes on GPU) rather than
  hunting for them.

Start by confirming the environment (step-by-step above), then begin work
item 1. Ask Leo before anything gated, outward-facing, or destructive.
