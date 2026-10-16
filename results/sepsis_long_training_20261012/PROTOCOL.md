# Longer matched supervised training — 12 October 2026

Motivation: eight of 24 neural controls reached the prior 20-epoch ceiling,
and three selected their best validation AP at epoch 20. Four-epoch patience
could also stop a model before recovery from an early validation plateau.
The earlier lab-value queue completed before this run was started.

## Prespecified matrix and budget

Twenty-seven fresh supervised fits, sequentially on the RTX 2080 Ti:

- 34-input scratch/source-MAE/target-MAE/source-plus-target-MAE, three seeds,
  encoder-and-head LR 3e-4 or 1e-4: 24 fits.
- Scratch with lab numeric values withheld but availability retained, LR 3e-4,
  three seeds: three additional fits. These pair with the corresponding full
  models above; the lab-information comparison gets the same longer budget.

Maximum **100 sampled epochs**, minimum **20 epochs** before stopping, and
**15 consecutive epochs** without tuning AP improvement. Each epoch samples
one decision per eligible fitting record: 10,888 examples and 341 optimizer
updates. Thus the maximum is 34,100 updates and 1,088,800 sampled prefixes per
fit, five times the earlier cap. This is not 100 passes over every hourly window.
Patient-balanced sampling is retained deliberately to isolate training budget.

Keep the original architecture, AdamW optimizer, batch 32, fixed learning rate,
weighted BCE, normalization, dropout RNG reset and sampled tuning decisions.
Fresh supervised optimization starts from exactly the original random/source/
target-MAE initializations. Target MAE checkpoints are reused byte-for-byte;
source MAE and the 1,000-step target reconstruction budget are not extended in
this experiment. A learning-rate schedule is not added as a second confound.

Checkpoint selection uses tuning record-weighted AP only. Record per-epoch AP,
training loss, optimizer steps, selected epoch and whether the new cap binds.
Save model/optimizer/sampling/CUDA RNG state each epoch for interrupted-fit
recovery. No test-based stopping or selection. Models running the full 100
epochs are reported as budget-limited, not automatically converged.

## Evaluation

Same target records and decision rows, 1–12-hour warning window, six-hour alarm
suppression, all-decisions primary view and identical recent-lab secondary gate.
Finish all planned fits before calibrating policies; freeze policies before
new test predictions. Calibration recall targets .3/.5/.6; report achieved
patient and alarm precision/recall, coverage, alarm burden and hourly AUROC/AP.

Compare pretrained to scratch at the same learning rate/seed, full versus lab
values withheld at the same longer budget, and long versus earlier short-budget
models. Use paired record bootstrap at separately calibration-frozen policies.
Separate short-budget and long-budget test results; do not select a winning
budget after looking at test performance. The reused target test remains
exploratory; independent confirmation is still unavailable.

## Execution and outputs

`python scripts/run_long_sepsis_training.py queue`

Configuration: `configs/sepsis_long_training_20261012.json`. Depends on the
completed control and lab-value experiments and their local checkpoint caches.
Implementation, data and initialization hashes reject in-place changes.
Private weights, optimizer recovery, and predictions remain in ignored local/.
The queue writes status.json and per-arm logs; final outputs include
evaluation.json, seed_summary.json, paired_comparisons.json,
budget_and_lab_comparisons.json, and convergence.json.
