# Pretraining checks before the control run

Eight CPU contract checks passed on 11 October 2026: four existing alarm-metric
checks and four new feature/checkpoint checks. New checks compare rolling
summaries with direct observation calculations, verify that future-value changes
cannot change earlier features, check input-set isolation, and confirm the
loaded source weights exactly match the saved source encoder (including its
first ten feature embeddings). They also verify identical initial classifier
weights and causal/padding invariance after source loading. These checks do not
prove the absence of all implementation errors.

The old experiment changed encoder/head learning rates from 3e-4/3e-4 in scratch
to 1e-4/1e-3 after pretraining. It also omitted target-only MAE. The new matrix
controls these differences. Supervised dropout randomness is explicitly reset
after MAE so adaptation's extra random draws do not change that control.

Remaining hypotheses, not established explanations:

- Reconstruction can learn stable physiology without learning impending sepsis.
- Only ten channels were pretrained. In the 34-channel model the added channel
  embeddings are random initially, and pooling includes all 34 active channels
  even when some have missing-value representations. Expanding the input space
  therefore changes the encoder's input distribution. Target MAE may help adapt
  it, but improvement is not guaranteed.
- Source/target measurement practices and cohort distributions differ.
- The supervised BCE/tuning-AP objective does not directly optimize finite-alarm
  matching. Alarm-policy performance can differ from ranking performance.
- The original source MAE beat carry-forward reconstruction on its validation
  task (seed 42: best Huber 0.1588 versus 0.1807), but that is not evidence of
  improved downstream event prediction.

The new run tests one architecture, two matched supervised learning rates and
one target MAE budget. It cannot justify the general claim that pretraining is
ineffective. Broader source-channel pretraining, alternate objectives and a
separately validated confirmation cohort remain outside this completed audit.
