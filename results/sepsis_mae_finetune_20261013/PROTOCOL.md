# Fine-tuning longer MAE: matched seed-42 pilot

Four new fits: target-only MAE and source-initialized target MAE, each at
learning rates 0.0001 and 0.0003. Both encoders were selected at 20,000 updates
using fixed-mask tuning reconstruction. Reuse the two exact seed-42 scratch
controls from the completed longer supervised experiment. Compare against its
1,000-update MAE arms at the same learning rates as well.

All 34 inputs, causal 72-hour histories, unchanged fit/tune/calibration/test
records, normalization, model architecture, fresh classifier initialization,
sampled training decisions, dropout seed, weighted BCE, and AdamW settings.
Maximum 100 epochs, minimum 20, patience 15; select tuning record-weighted AP.
Reuse the existing epoch recovery implementation. Verify encoder loading exactly.

Freeze all fits before selecting alarm policies on calibration data; then evaluate
the previously used exploratory test cohort, with paired record bootstrap intervals.
Report patient precision/recall separately from alarm precision/timely event recall.
This is one seed, not a three-seed replication or untouched confirmation cohort.
Improved reconstruction is not evidence of improved sepsis prediction by itself.

Run: `..\.conda\python.exe -u scripts/run_mae_finetune.py queue`
Checkpoint hashes, source code hashes and data hashes are locked before fitting.
