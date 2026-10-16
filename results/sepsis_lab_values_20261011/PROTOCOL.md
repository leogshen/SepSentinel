# Do numeric lab values help beyond measurement availability?

Locked before fitting new ablated models, 11 October 2026. Exploratory same-target
comparison; no independent confirmation claim. The previous control queue is
complete and its full-input fits are reused as prespecified comparators.

## Design

Compare numeric lab values present versus withheld, on identical records,
decision times, outcome labels and calibration/test rules. Both arms retain the
observation flags, observation ages and missingness/count information for all
measurements, all vital-sign values, BP, EtCO2, FiO2, static features and time.
FiO2 is treated as an oxygen-support setting and remains numeric in both arms.
Twenty-five lab/blood-gas channels are ablated (indices and names follow the
34-channel measurement manifest; channels 4–9, 14–15, 17–33).

For the transformer, set only the normalized lab-value input to zero after
causal collation. For trees, set lab last/mean/std/min/max/slope/change summaries
to zero in every 6/12/24/72-hour window, retaining count, age and seen features.
Train the ablated model anew; do not just remove values at test time.

Families: XGBoost, RF and scratch transformer at LR 3e-4. Seeds 42/123/456.
Nine new fits versus nine reused full-input fits. Tree tuning choices, neural
training budget, sampled tuning decisions and preprocessing match the completed
control run. The transformer LR is the original scratch rate, not selected by
new test results. Both tree arms have the same two-candidate tuning budget.

Primary evaluation uses all eligible decisions: no patient or time is excluded
because labs are unavailable. Secondary evaluation applies the same recent-lab
gate to both arms, with subgroup and population denominators explicitly reported.
The primary contrast is timely alarm precision at calibration recall target .6;
also report .3/.5, achieved recall, patient precision/recall, AUROC/AP, alarm
burden and coverage. These targets match calibration recall, not necessarily
test recall. Compare paired differences within family/seed/gate at frozen
thresholds with 2,000 record-bootstrap replicates, including patient metrics.

All ablated fits finish before policy calibration and evaluation. Full-input
test predictions are already available from prior work: this experiment remains
exploratory. No model is selected based on new test performance.

## Original recall-drop audit

`scripts/explain_lab_gate_recall.py` separately toggles the lab gate at an
unchanged score threshold and then applies the recalibrated gate threshold.
In original scratch-34 seed 42, 79 events were detected without the gate and
44 with it at the **same threshold**. The 35 lost detections are all events
without any qualifying measurement time in the warning window. No lab values
or model scores changed. Seeds 123/456 lost 33 detections each at fixed threshold;
recalibration recovered one/two detections respectively. This establishes that
the reported drop to 29.3% was an alert-eligibility-policy effect, not evidence
that extra lab information worsened recall.

## Execution

`python scripts/test_lab_value_ablation.py` validates that arbitrary lab-value
changes cannot alter the blinded inputs and that availability metadata and
nonlab values remain identical. Existing causal-feature tests also apply.

`python scripts/run_lab_value_experiment.py queue` runs all jobs sequentially,
reusing full models after checking their frozen hashes. It depends on the
completed `sepsis_controls_20261011/local` caches and the original target data.
Code/data/source-model provenance is locked. Private arrays, models and scores
remain under ignored local/. Results are written to evaluation.json,
seed_summary.json, and paired_lab_value_comparisons.json.
