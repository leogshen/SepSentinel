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
  Transformer forward.

- [x] **3b. Challenge-rule Sepsis-3 labels** — DONE 2026-09-03:
  `sepsentinel/data/sepsis3.py` (DuckDB port of mimic-code
  suspicion_of_infection + hourly SOFA + sepsis3, with the Challenge timing
  rules), `scripts/build_sepsis3_labels.py` (labels CSV + deviation report),
  wired into `scripts/extract_mimic.py` as `t_sepsis_hour` together with the
  §2 cohort exclusions and a CONSORT attrition log. Rules regression-tested
  on a synthetic mini-MIMIC (`scripts/test_sepsis3_rules.py`, 6/6) and run on
  the open demo. Still to check on real 3.1: onset-time spread (see HANDOFF).

- [x] **3c. Pre-onset target + extended features** — DONE 2026-09-06:
  `gridding.make_labels(prodrome_window_h=...)` confines positives to
  [t_sepsis-W, t_sepsis); `extract_mimic.py --feature-set extended` adds MAP,
  SBP, GCS, FiO2, urine output, PaO2, BUN, glucose. Together: +7h median lead
  and capture>=6h 0.43 -> 0.53 at equal alert burden. See RESULTS.md part 2.

- [ ] **4. Alarm-episode/cooldown evaluator** — merge consecutive alarms into
  episodes with refractory period R ∈ {2,6,12}h; report alert episodes per
  nonseptic patient-day + capture/lead under episode semantics. Doesn't exist
  anywhere yet; required for the §10 evaluation plan. Can be validated on
  existing PhysioNet checkpoints immediately.

- [ ] **5. Promote AblationPreprocessor into the package** — move from
  experiment3_feature_ablation.py to `sepsentinel/data/` (cross-experiment
  imports from an experiment script are fragile). Not a pure refactor any
  more: its `transform()` drops every key except signals/labels/length/
  patient_id/label, so `subject_id` and `t_sepsis_hour` do not survive
  preprocessing — the MIMIC patient-level metrics need both. Carry them
  through.

- [x] **6. Move `lengths` to device** — DONE 2026-09-03: fixed centrally in
  TransformerEncoder.forward (covers plain/gated/MAE models).

Optional while waiting: run Experiment 4 (trajectory + gating, built but
never run) on PhysioNet — its features are part of the MIMIC plan (§6) and
knowing whether they help is cheap now.
