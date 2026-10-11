# Five-hour exploratory experiment, 2026-10-09

Objective: test richer-input sequence transfer, wider timing windows, and
measurement-defined subgroups. All previous results remain unchanged. This is
exploratory reuse of the previously inspected set-B test split, not independent
confirmation. No COVID task or alternative endpoint is introduced.

Inputs: 34 CinC measurement channels, with age/gender and their presence flags.
The first ten channels retain earlier QC and source scaling. New channel scaling
is fit on target fitting records only. Blood pressures receive broad physical
screening; no patients are excluded for unfavorable outcomes or performance.
The existing record split and prior mirror exclusions are preserved.

Architecture: existing causal temporal encoder, expanded feature embeddings for
34 measurements. Source initialization copies all temporal/value parameters and
the ten shared feature IDs; new feature IDs are initialized randomly. Scratch,
source-only transfer, and transfer plus 1,000 target-training MAE updates are
compared using identical target inputs. All encoder parameters are fine-tuned;
source encoders use a lower learning rate. Source scaling is retained on shared
channels so transferred input semantics remain unchanged.

Fit priority: 34-channel next-12-hour models (three initializations, seed 42),
10/13-channel scratch/source controls at 12 hours, full-input seed 123 repeats,
then 34-channel next-six-hour models. Up to 20 epochs, four-epoch patience,
selection by record-weighted tuning AP on up to eight fixed times per record.
Each fitting epoch samples one eligible decision per record. Positive BCE weight
is inverse prevalence capped at 20. No model is selected using test results.

Evaluation: strict 1–6h warning, 1–12h warning, and recognition from six hours
before through three hours after recorded onset. Recognition is not early-warning
performance. Predictions remain causal. Unknown negative horizons are excluded.
Six-hour alarm suppression, one event credit per event. Too-early, nonseptic,
duplicate and other unmatched alarms are reported explicitly. Calibration uses
151 quantile thresholds plus endpoints, targeting 30% and 60% event recall;
grid infeasibility is not a proof of impossibility. Every arm is reported.

Gates: all decisions; at least one BP measurement within two hours; BP plus
creatinine/WBC/platelets within 24 hours; those plus lactate or pH within 24 hours.
All gates depend only on currently available history. Report subgroup recall,
event coverage, population recall and interruption burden. Compare input arms
on identical gated decisions; subgroup membership alone cannot show that ordering
additional tests causes improved performance. A definitive clinical utility claim
would require independent confirmation and review of measurement availability.

Compute budget: 4.7 hours from runner start, leaving operational margin within
the user's five hours. Do not start new fits in the last 90 minutes; reserve time
for scoring and reporting. Fixed priority governs omitted arms, never test scores.
Weights and patient-level predictions remain in ignored local/ directories.
