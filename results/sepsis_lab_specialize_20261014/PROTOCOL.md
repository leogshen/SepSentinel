# Broad learning followed by lab-rich specialization

Reuse the five development folds from sepsis_cv_20261013. These outer folds have
already been inspected in earlier experiments: this is exploratory internal
validation, not untouched confirmation. Do not use the original test set.

The fixed gate at each decision is: any SBP/MAP/DBP in the preceding <2 hours;
creatinine, WBC and platelets each <24 hours old; and lactate or pH <24 hours old.
Only information available at that decision is used. Do not select visits based
on whether they eventually receive labs. Histories retain all 34 channels and
the preceding causal 72-hour window. Fit/tune supervised decisions are restricted
to this gate; calibration/evaluation keep the full original event denominator
and permit alarms only when the gate is true.

Three equal-search-budget approaches:
1. Random initialization, supervised training on lab-rich decisions.
2. The exact fold-local broad supervised scratch checkpoint, including its head,
   then lab-rich fine-tuning.
3. MIMIC/SIC source MAE plus fold-local broad CinC MAE, then lab-rich fine-tuning
   with a freshly initialized classifier.

Reuse broad-training fit-only normalization identically in all approaches. Extend
each fold's saved CinC MAE optimizer/RNG state to at most 100,000 total updates.
Minimum 20,000 updates, validation every 500 on the same 512 fixed-mask tuning
examples, patience 12 checks (6,000 updates), best reconstruction checkpoint.
This extends target MAE, not the original ten-channel source MAE. Cap binding is
reported as unresolved convergence, not declared sufficient training.

Each supervised method searches learning rates 0.00003, 0.0001 and 0.0003 inside
the same fold. AdamW weight decay 0.0001, maximum 200 epochs, minimum 30, patience
30. Reduce LR by half after eight tuning-AP plateau epochs, minimum 0.000001.
An epoch samples one lab-eligible decision per eligible fit record, batch 32.
All methods share sample/dropout seeds. Record-weighted tuning AP selects both
checkpoint and candidate; retain the initial checkpoint if adaptation worsens it.
Thus there are 45 new candidate fits and five extended MAE runs, sequentially.

Retain unmodified broad scratch and XGBoost models as reference policies on the
exact same lab-rich decisions. Freeze all folds' selected fits before evaluation;
choose thresholds separately on calibration for 30/50/60% subgroup timely recall.
Report subgroup timely recall AND population timely recall/event coverage,
patient precision/recall, alarm precision, interruptions, fold variability and
paired conditional record-bootstrap intervals. Do not call a gated subset's
recall population recall. Bootstrap intervals condition on trained models.

Code and original fold data/checkpoint hashes are frozen. Existing CV outputs
are not changed. Models and recovery states remain local.

Run: `..\.conda\python.exe -u scripts/run_lab_specialization.py queue`
