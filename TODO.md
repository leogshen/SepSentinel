# Pre-Credentialing TODO

Code fixes required before MIMIC-IV/SICdb data arrives, ordered by priority.
Details and rationale: DATA_ACCESS_SPEC.md §9 and §13.

- [x] **1. Subject-level grouped splitting** — DONE 2026-09-03:
  `grouped_patient_split()` added to splitting.py; leak-tested (300 synthetic
  subjects, multi-stay) + PhysioNet fallback verified.

- [x] **2. Parameterize the +6 label shift** — DONE 2026-09-03:
  `compute_early_warning_metrics(..., label_shift_hours=6)`; episode
  `t_sepsis_hour` takes precedence when present; regression-tested at
  shifts 3/6 and with explicit t_sepsis.

- [x] **3. Event-to-grid loader** — DONE 2026-09-03:
  `sepsentinel/data/gridding.py` (unit-tested) + `scripts/extract_mimic.py`
  (DuckDB, works on demo AND full 3.1; itemids verified against real
  dictionaries; IL-6 census: ZERO interleukin items in MIMIC-IV).
  End-to-end verified: demo -> grid -> grouped split -> Strategy B ->
  Transformer forward. REMAINING: Challenge-rule sepsis3 label SQL
  (t_suspicion/t_SOFA) — episodes are all-control until then.

- [ ] **4. Alarm-episode/cooldown evaluator** — merge consecutive alarms into
  episodes with refractory period R ∈ {2,6,12}h; report alert episodes per
  nonseptic patient-day + capture/lead under episode semantics. Doesn't exist
  anywhere yet; required for the §10 evaluation plan. Can be validated on
  existing PhysioNet checkpoints immediately.

- [ ] **5. Promote AblationPreprocessor into the package** — move from
  experiment3_feature_ablation.py to `sepsentinel/data/` (cross-experiment
  imports from an experiment script are fragile). Pure refactor, no behavior
  change.

- [x] **6. Move `lengths` to device** — DONE 2026-09-03: fixed centrally in
  TransformerEncoder.forward (covers plain/gated/MAE models).

Optional while waiting: run Experiment 4 (trajectory + gating, built but
never run) on PhysioNet — its features are part of the MIMIC plan (§6) and
knowing whether they help is cheap now.
