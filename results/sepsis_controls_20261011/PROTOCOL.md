# Matched sepsis controls: locked before new fitting

Question: does source MAE help after controlling fine-tuning learning rate, and
does target-only MAE explain any benefit? How do strong causal tree baselines
compare on the identical released records and hourly decisions?

This is an exploratory follow-up to the completed measurement-expansion study.
The existing target test set has been inspected. No result from this run will be
called independent confirmation. See CONFIRMATION_AUDIT.md for available cohorts.

## Planned comparisons

- XGBoost and random forest, each with 10 and 34 channels and seeds 42/123/456:
  12 final tree models, each selected from two prespecified configurations using
  tuning data only. XGBoost uses depths 3/6, 500-tree limit and 40-round early
  stopping; RF uses 300 trees, depths 14/22 and minimum leaves 20/8. Four CPU
  threads; all fits run sequentially.
- Neural 34-channel controls: scratch, source MAE, target-only MAE, and source
  plus target MAE, each at learning rates 1e-4 and 3e-4, each with three seeds:
  24 models. The same rate applies to encoder and prediction head. Each rate
  is reported, rather than selecting the best rate on test data.
- Every neural arm has the same architecture, classifier initialization, batch
  size 32, 20-epoch maximum, patience four, tuning AP criterion, and seed-specific
  sampled supervised decisions. The supervised dropout RNG is reset identically
  after initialization/adaptation. Target MAE runs 1,000 steps, with identical
  corruption sampling and initial decoder RNG across its two arms. Its encoder
  is reused across the two learning rates. Extra pretraining compute is explicit.

Source MAE checkpoints remain the original three ten-channel models. This does
not test richer source pretraining, alternate self-supervised objectives, or a
claim that all possible pretraining methods fail.

## Data and features

Use byte-identical fit/tune/cal/test records and normalization from
`results/sepsis_missions_adapt_20261010/local`. Each model predicts onset in the
next 12 hours using up to 72 hours of causal observations, with the previously
audited exclusions and assigned onset. Extra labs are never added from the future.

Trees use 6/12/24/72-hour last value, mean, standard deviation, min, max, linear
slope, first-to-last change, count, age, and seen indicator per channel, plus
the same four static inputs and elapsed ICU hour. Values use the neural arm's
normalization/clipping; unobserved summaries are zero with explicit counts/age.
Both trees and neural models have access to the same raw prefixes and labels.
Trees fit all eligible fitting hours, weighted equally per record; neural models
sample one fitting hour per record per epoch as in the original experiment.
This training-sampling difference is reported, not hidden as identical training.
Tuning AP uses the same up-to-eight decisions per record for both model families.

## Evaluation

All planned fits finish before calibration. All thresholds are then frozen before
test prediction. The primary window is 1–12 hours before onset with six-hour
alarm suppression and one event match. Report ungated and the predeclared recent
BP+routine+perfusion-lab gate. Calibration targets event/subgroup recall .3/.5/.6
on a 501-quantile grid plus endpoints. Achieved test recall may differ.

Report patient precision and recall, timely alarm precision and event recall,
subgroup coverage, alarms/day, hourly AUROC/AP, and all unmatched-alarm categories.
Bootstrap 2,000 released-record samples at frozen policies; compare each MAE arm
against scratch at the same seed and learning rate. No multiplicity-corrected or
selected-best claim. Source-vs-target-only MAE is an additional descriptive contrast.
Strong null conclusions require confidence bounds and external confirmation.

`scripts/run_sepsis_controls.py queue` is resumable per completed fit and writes
status/logs. Code/config/data/source-checkpoint hashes reject in-place changes
after fitting begins. Arrays, scores and model weights stay under ignored local/.
The run does not fit or normalize on a potential confirmation cohort.
