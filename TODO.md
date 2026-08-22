# Pre-Credentialing TODO

Code fixes required before MIMIC-IV/SICdb data arrives, ordered by priority.
Details and rationale: DATA_ACCESS_SPEC.md §9 and §13.

- [ ] **1. Subject-level grouped splitting** — replace episode-level
  `patient_split()` (sepsentinel/data/splitting.py) with a grouped split by
  `subject_id`, stratified by subject-level sepsis. Blocks ALL multi-stay
  datasets; current code would leak the same person across train/test.

- [ ] **2. Parameterize the +6 label shift** — `compute_early_warning_metrics`
  (experiment5_recall_study.py) hard-codes `t_sepsis = onset_step + 6`. Read
  shift from episode `t_sepsis_hour` / a SHIFT param instead. Blocks any
  3h/12h label-shift experiment; silently wrong lead times otherwise.

- [ ] **3. Event-to-grid loader skeleton** — new `sepsentinel/data/gridding.py`
  producing the §0 episode schema (hourly bins [t,t+1), vitals=median,
  labs=last, mask=any-in-bin) from a generic (time, variable, value) event
  table. Unit-test on synthetic events now; MIMIC/SICdb loaders become thin
  wrappers later.

- [ ] **4. Alarm-episode/cooldown evaluator** — merge consecutive alarms into
  episodes with refractory period R ∈ {2,6,12}h; report alert episodes per
  nonseptic patient-day + capture/lead under episode semantics. Doesn't exist
  anywhere yet; required for the §10 evaluation plan. Can be validated on
  existing PhysioNet checkpoints immediately.

- [ ] **5. Promote AblationPreprocessor into the package** — move from
  experiment3_feature_ablation.py to `sepsentinel/data/` (cross-experiment
  imports from an experiment script are fragile). Pure refactor, no behavior
  change.

- [ ] **6. Move `lengths` to device in Trainer** — latent crash on CUDA
  (training.py builds the padding mask on `x.device` against CPU `lengths`).
  One-line fix; must land before first GPU-machine run.

Optional while waiting: run Experiment 4 (trajectory + gating, built but
never run) on PhysioNet — its features are part of the MIMIC plan (§6) and
knowing whether they help is cheap now.
