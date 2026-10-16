# Five-fold development cross-validation

Objective: test split stability of neural versus XGBoost alarms, numerical lab
information versus observation availability, and source plus longer target MAE.
This is exploratory internal validation, not a fresh external confirmation.

Pool the previous fit/tune/calibration records; exclude the previous test records.
Use five stratified outer folds (eligible sepsis event, seed 20261013). Within each
outer-training partition, use 70% for fitting, 15% tuning and 15% calibration,
stratified by event. Keep every released record's hours together. The public
identifiers do not establish distinct persons across multiple admissions; this
limitation remains. Each development record is evaluated exactly once out of fold.

Five models per fold, all with 34 channels:
1. XGBoost, full values.
2. XGBoost, numeric labs withheld but measurement availability retained.
3. Scratch neural model, full values.
4. Scratch neural model, numeric labs withheld but availability retained.
5. Source-initialized neural model plus fold-local longer target MAE, full values.

Neural settings: fixed learning rate 0.0001, seed 42, maximum 100 epochs, minimum
20, patience 15, tuning AP selection. Existing causal 72-hour pipeline and masked
lab-value controls are reused. One sampled decision per fit record per epoch.
XGBoost uses all eligible fit hours with equal total record weights and the
existing two candidate configurations selected on tuning AP. Training exposure
therefore differs by model family, as in the existing baseline experiment.

All extra-channel normalization is fitted separately on each fold's fit records;
the ten shared channels keep the original source-training scaler. Reuse only
the source (MIMIC/SIC) encoder. No existing CinC-pretrained encoder is reused.
Each fold performs source-initialized 34-channel target MAE only on its fit
records, with fixed-mask validation on its tuning records: at most 20,000 steps,
at least 5,000, check every 500, patience eight checks. This tests one pretraining
recipe; it does not separately compare target-only MAE or pretraining without labs.

Before any outer evaluation, finish all five folds' training and calibration.
Calibrate thresholds for 30%, 50% and 60% timely recall, using the same existing
12-hour warning window, six-hour refractory period, and all/recent-lab gates.
No threshold is chosen using outer-fold outcomes. Paired full/withheld models
use identical partitions, decision eligibility, and seeds.

Report every fold and pooled out-of-fold record statistics, separately for
patient precision/recall, timely alarm precision/recall and alarms per patient-day.
Paired record-bootstrap intervals condition on fitted models and policies;
they do not include training-set/model-selection uncertainty or justify treating
overlapping-fold fits as independent. Show fold variability alongside intervals.
Cross-validation here addresses split sensitivity; it is not repeated-seed CV.

Execution: `..\.conda\python.exe -u scripts/run_sepsis_cv.py queue`
The queue waits for the current fine-tuning queue, then serializes all fitting.
Fold datasets, scalers, IDs and implementation hashes are frozen before fitting.
Checkpoints and record IDs remain in ignored local folders.
