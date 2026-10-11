# Richer-input sepsis missions

Update 2026-10-11: the separately authorized sequential queue completed all 27
fits on 2026-10-10. Results, limitations and reproduction details are recorded
in [dev.md](dev.md#richer-measurements-and-transfer-completed-results-2026-10-10-documented-2026-10-11).
The sections below describe the original staged workflow; the original run
directories now contain frozen results. Use a new run name for retraining.

No jobs were launched while adding this workflow. Existing five-hour experiment
code and outputs are preserved. The new workflow uses a separate output directory.

The configuration is `configs/sepsis_missions.json`: 27 arms, comprising three
seeds, three input sets (10 shared, 13 including BP, 34 measurement channels), and
scratch/source/source-plus-target-MAE initialization. All train the next-12-hour
endpoint with the same architecture and epoch/patience budget. Source models use
lower encoder learning rates, as specified in the existing fitter. Adaptation
adds 1,000 target-training MAE updates: extra compute is explicit, not equal-total-
compute evidence. Each arm's input mask also applies to its adaptation task.

This reuses existing ten-channel source pretraining. New target feature embeddings
are learned during adaptation/fine-tuning; it does not newly pretrain on 34-channel
MIMIC. A scratch-plus-target-MAE arm would be needed to isolate source pretraining
from target self-supervision in a further study.

## Original explicit stage commands

From `F:\Claude\Sepsentinel\Code`, using its Conda Python:

```powershell
& '..\.conda\python.exe' scripts/sepsis_missions.py plan
& '..\.conda\python.exe' scripts/sepsis_missions.py prepare
& '..\.conda\python.exe' scripts/sepsis_missions.py train
& '..\.conda\python.exe' scripts/sepsis_missions.py calibrate
& '..\.conda\python.exe' scripts/sepsis_missions.py evaluate
& '..\.conda\python.exe' scripts/sepsis_missions.py summarize
```

Train one arm with `train --arm source_34f_12h_seed42`. There is no default stage
and no automatic training-to-test chain. `plan` prints the matrix without creating
outputs. `prepare` reuses the previous record split and exclusions and reads raw
data to expand channels. `train` loads fitting/tuning records, not test records.
`calibrate` requires every configured fit to be complete, freezes model hashes,
and reads only calibration records for predictions and policy selection.
`evaluate` is the only model-based test stage and requires all frozen policies.
Changing code/config/data after work starts requires a new run name. Completed
fits are resumable; interrupted fits restart, rather than resume optimizer state.
This full matrix has no inherited five-hour timer: schedule its budget explicitly
when execution is authorized. Do not execute the old all-in-one runner by mistake.

## Evaluation and interpretation

Windows: 1–6 hours before onset, 1–12 hours before onset, and six hours before
through three hours after onset (recognition, not early warning). All use causal
predictions and six-hour alarm suppression. Existing cohort exclusions remain;
the recognition results do not cover admission-onset cases previously excluded.

For every window, gates are: all; BP within two hours; BP plus creatinine/WBC/
platelets within 24 hours; those plus lactate or pH within 24 hours. Gates and
record alignment are identical across feature/model arms. Report population
recall, subgroup recall, coverage, hourly metrics, patient metrics, alarm precision,
alarms/day, and counts of too-early/duplicate/nonseptic/other unmatched alarms.
Patient metrics mean a flagged record with an eligible event anywhere in its
evaluated history, irrespective of the alarm's timing; they are explicitly not
timely capture. Time-window comparisons can have different resolved decisions
and event denominators; they are not pure timing-uncertainty experiments.

Thresholds are frozen on calibration data for target recall 20/30/50/60%, using
a 501-quantile grid with endpoints. Report calibration curves, infeasibility, and
achieved test metrics rather than claiming exact matched test recall. The grid is
approximate and may miss narrow feasible regions because refractory alarm recall
need not be monotone. No test threshold optimization is performed.

Uncertainty: 2,000 released-record bootstrap resamples with thresholds fixed.
Paired comparisons resample identical records for transfer versus scratch and
richer versus ten-channel inputs, within seed/window/gate/target. Summaries show
seed mean and range separately; seeds on the same patients are not independent
validation cohorts. Repeated-person identity is unavailable. Multiple comparisons
are exploratory, with no multiplicity correction or selected-best claim.

All test data have previously been inspected. These results cannot establish
independent confirmation or the causal benefit of ordering additional labs.
The 30% precision / 60% recall objective is an assessment target, not a guaranteed
outcome. Optional XGBoost reference and the fallback tasks in dev.md are not part
of this training matrix.

## Lightweight verification

`scripts/test_sepsis_mission_metrics.py` tests only synthetic CPU arrays: matching,
duplicate penalties, subgroup denominators, ordering, and paired bootstrap.
It does not load clinical records, initialize GPU models, or train anything.
