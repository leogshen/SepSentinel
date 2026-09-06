# HANDOFF — Continue SepSentinel

You are picking up an in-progress project. This file is your complete
context; previous session memory does not transfer. Read this fully, then
skim RESULTS.md (part 2 first), DATA_ACCESS_SPEC.md, TODO.md, SICDB_RECON.md.

Last updated 2026-09-06.

## Project in one paragraph

SepSentinel: wearable multimodal sepsis early-warning platform. Model A
(future): electrochemical sensor signals -> biomarker concentrations
(IL-6/lactate/pH). Model B (active work): physiological + lab time series ->
per-hour sepsis risk. Trained on PhysioNet/CinC 2019, now running on MIMIC-IV
3.1. The researcher (Leo) is a high-school student — gated data applications
need a sponsoring PI; open-tier data is fine.

## Where things actually stand

**The MIMIC-IV pipeline is complete and running end to end.** Labels,
extraction, cohort rules, baselines, Transformer, evaluation, diagnostics.
Full numbers in RESULTS.md part 2. Current best operating point (full cohort,
at <=1.0 false alerts per nonseptic patient-day): flat XGBoost, 18 features,
pre-onset target — recall 0.64, median lead 20.6 h, capture >=6h 0.53,
AUROC 0.736.

**The three findings that matter, if you read nothing else:**

1. **AUROC is anti-correlated with early warning on this task.** Five
   independent observations, including a synthetic control where a 16x
   stronger injected signal gave higher AUROC and shorter warning. Never
   select a model or a setting on AUROC here. Report threshold-free metrics
   plus equal-ALERT-BURDEN tables (`scripts/operating_curves.py`). The old
   70%-patient-recall operating point has been removed from the codebase.

2. **The sequence model is not earning its complexity.** Flat XGBoost matches
   or beats the causal Transformer on every deployment metric. The temporal
   information that matters is already in the Strategy B channels (last
   value, was it measured, how long ago); attention over the trajectory adds
   ~0.014 AUROC and costs lead time. Do not spend effort on architecture.

3. **The ceiling is measurement sparsity, not model capacity.** In the
   pre-onset window the model effectively sees three signals (HR, SpO2,
   respiratory rate at ~95% of hours); every lab is 2-8%. A synthetic 0.25 SD
   drift in a dense channel beats a 4 SD shift in a sparse one. This is the
   quantitative case for Model A, and it is the most publishable thing here.

## Machine setup

1. `git clone https://github.com/leogshen/SepSentinel`
2. venv + `pip install -r requirements.txt` (duckdb and xgboost are in it),
   then CUDA torch:
   `pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu124`
   Verify `python -c "import torch; print(torch.cuda.is_available())"` -> True.
3. PhysioNet 2019: `python -c "import kagglehub; print(kagglehub.dataset_download('tea340yashjoshi/sepsis-prediction-dataset'))"`
4. MIMIC-IV 3.1: Leo downloads the zip (credentialed). Selective extract of
   13 tables, staying `.csv.gz`, ~7.6 GB:
   `cd <datadir> && unzip -o -q <zip> mimic-iv-3.1/hosp/{patients,admissions,labevents,d_labitems,prescriptions,microbiologyevents,emar}.csv.gz mimic-iv-3.1/icu/{icustays,chartevents,d_items,inputevents,outputevents,procedureevents}.csv.gz -d .`
   DUA hygiene: outside any cloud-synced folder, never in the git repo.
   On the current machine it lives at
   `F:/Claude/Sepsentinel/data_local/mimic-iv-3.1`.

Runtimes on a 2080 Ti / 32 GB box: labels 4.7 min, full extraction ~10 min,
flat baselines <1 min, Transformer 3 seeds ~42 min.

## Work queue, in order

1. **Sweep the prodrome window.** It is fixed at 12 h and was never tuned;
   3/6/12/24 h are one flag each (`--prodrome-window-h`). This is the
   cheapest remaining win and directly sets the clinical claim.
2. **Grouped bootstrap CIs on the operating-burden metrics.** Everything is
   currently a point estimate from one split. The headline claim (+7 h lead)
   needs an interval before it goes in a paper.
3. **Alarm-episode/cooldown evaluator** (spec section 10, TODO item 4) —
   merge consecutive alarm hours into episodes with a refractory period
   R in {2,6,12} h. Still not built; the alert-burden numbers currently count
   raw per-hour alarms, which overstates burden.
4. **Promote AblationPreprocessor into the package** (TODO item 5). Note it
   drops every key except signals/labels/length/patient_id/label, so
   `subject_id` and `t_sepsis_hour` do not survive preprocessing.
5. **Feature ablation on the extended set.** 18 features went in as a block;
   nobody has checked which ones earn their place. MAP and urine output are
   the dense ones and the likely winners.
6. Later: SICdb (read SICDB_RECON.md first — cultures are confirmed ABSENT,
   so Challenge-rule labels are impossible there), experiment 4
   (trajectory+gating, built but never run), ImmPort SDY1662.

## Gotchas that cost time once already

- **Do not score SOFA in hours with no data.** Scoring empty pre-ICU hours
  gave SOFA 0 there, so ICU admission itself read as a >=2-point rise and 58%
  of demo stays came out septic at hour 0. SOFA now starts at hour 0 with
  pre-ICU labs clamped into it.
- **MIMIC contains impossible values** — 11,337 bpm heart rates, 7,000,400
  breaths/min. Extraction filters them now; `CLIP_RANGES` is the second line
  of defence, not the first.
- **DuckDB `read_csv_auto` sniffs types per file**: a table with no usable
  rows comes back VARCHAR and every comparison fails at bind time.
  `sepsis3.register_sources` pins the types it needs.
- **`load_physionet` used to zero-fill unmappable features**, so `stage=3`
  produced an all-zero IL-6 channel reading 0.0% missing. It raises now.
  Always pass an explicit `features` list.
- **collate_fn SORTS batches by length**: any per-patient pairing must
  replicate that sort (see `collect_patient_predictions`). Never pair by
  dataset order.
- **labevents has no stay_id** — join subject_id + charttime within the stay
  window, and use charttime, never storetime (leakage).
- **AblationPreprocessor indexes into the feature list positionally** and had
  the four PhysioNet vitals hardcoded; it now takes an explicit `vitals`.
- **PhysioNet <-> MIMIC share a hospital (BIDMC)**: MIMIC is NOT clean
  external validation for PhysioNet-trained models.
- Windows console is cp1252: no unicode in logger/print output.
- Checkpoints (`*.pt`) and episode pickles are gitignored; regenerate them.

## Open questions worth an email, not an experiment

- **SICdb IL-6/procalcitonin census.** `d_references` ships only inside the
  restricted download, so this cannot be settled from outside. One GitHub
  issue on `nrodemund/sicdb` answers it, no DUA needed. Worth doing before
  any SICdb access effort, since IL-6 presence is the only thing that would
  make SICdb strategically important to Model A. See outreach/DATA_REQUESTS.md.
