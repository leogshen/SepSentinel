# Experiment report: do additional measurements improve sepsis prediction?

Completed 10 October 2026 on one NVIDIA RTX 2080 Ti. This document reports one
experiment comprising 27 fits: three input sets, three initializations, and three
seeds (42, 123, 456). It is separate from the development log.

**Main result:** the scratch 34-input model achieved mean **40.6% patient-level
precision**, **52.0% patient-level recall**, and **51.1% timely event recall**.
Its timely alarm precision was **6.15%**. Additional measurements modestly improved
risk ranking but did not consistently improve timely alarm performance.

## Data, inputs, and prediction target

Target: official PhysioNet/CinC 2019 training set B. Of 20,000 released records,
19,503 remained after excluding 223 left-censored onsets and 274 records without
a resolved eligible decision. Fit/tuning/calibration/test splits contain
11,759/2,991/1,934/2,819 records. Test includes **150 eligible sepsis events**.
Splits are deterministic by released-record identity; repeat-person identity is
unavailable. The historical 14,057-record mirror matches set A, not set B.

Input at each decision: preceding hourly measurements, capped at 72 hours, plus
age/gender values and their knownness indicators. The measurement sets are:

| Set | Measurements |
| --- | --- |
| 10 | HR, O2Sat, Resp, Temp, Lactate, pH, Creatinine, WBC, Platelets, Bilirubin_total |
| 13 | The ten above plus SBP, MAP, DBP |
| 34 | The thirteen above plus EtCO2, BaseExcess, HCO3, FiO2, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Bilirubin_direct, Glucose, Magnesium, Phosphate, Potassium, TroponinI, Hct, Hgb, PTT, Fibrinogen |

The ten-input baseline already includes six labs. This is an additional-measurement
experiment, not a vitals-only versus labs experiment.

Output: a sigmoid risk score for assigned sepsis onset **within the next 12 hours**.
Assigned onset is the first positive challenge label's ICULOS plus six hours,
undoing its early label shift. Decisions start at hour six; early-warning inputs
precede onset. Negative decisions without sufficient follow-up are excluded.
The reported 12-hour test view contains 63,738 decisions, 1,622 positive (2.54%).

## Model and fitting

`RichModel` in [scripts/train_rich_transfer.py](scripts/train_rich_transfer.py)
uses the encoder in [scripts/transfer_sequence.py](scripts/transfer_sequence.py):
two causal transformer layers, width 96, four attention heads, feature embeddings,
and a 100-to-64-to-1 prediction head with GELU and dropout 0.1. The 100 head inputs
combine a 96-dimensional history embedding with four static inputs.

Each channel supplies normalized value, current-observation flag, ever-observed
flag, and observation age. Carry-forward uses preceding observations only;
never-observed values use the normalization median with explicit missingness.
Feature/value interactions occur before feature pooling. Attention masks future
steps and padding; missing labs remain represented rather than deleting the hour.
The first ten channels use source-fitting normalization; added channels use
target-fitting normalization. Broad physical QC is applied before normalization.

Initializations: **scratch**; **source MAE** trained on ten shared MIMIC/SICdb
channels; and **source MAE + target MAE**. Source fitting contains 58,844 MIMIC
and 13,645 SICdb episodes, with 6,565/1,543 validation episodes and previous
patient holdouts preserved. Source histories are outcome-independent and capped
at 336 hours. Source MAE budget is 20,000 steps per seed; target MAE adds 1,000
steps using target fitting records and each arm's active input mask. Added target
channels are not pretrained on the source datasets.

All arms fine-tune with weighted BCE, batch 32, one sampled eligible decision per
record per epoch, up to 20 epochs, patience four, and tuning record-weighted AP
selection. AdamW encoder/head learning rates are 3e-4/3e-4 for scratch and
1e-4/1e-3 for source-initialized arms. Transfer therefore includes a different
learning-rate configuration; adaptation also receives additional compute.

## Evaluation and results

Calibration selects thresholds for target **60% timely alarm event recall** using
a 501-quantile grid plus endpoints. Thresholds are frozen before test evaluation.
The reported view accepts one matched alarm **1–12 hours before onset**, with
six-hour alarm suppression. Actual test recall need not equal calibration recall.

**Patient-level precision** = flagged records with an eligible sepsis event /
all flagged records, regardless of warning timing. **Patient-level recall** =
flagged event records / all eligible event records. **Timely event recall** =
events receiving a matched alarm / eligible events. **Timely alarm precision** =
matched alarms / all alarm episodes. Patient precision and timely recall have
different units; their juxtaposition is not an ordinary precision–recall pair.

| Initialization | Inputs | Patient precision | Timely event recall | Patient recall | Timely alarm precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scratch | 10 | 34.3% | 51.8% | 52.7% | 5.90% |
| Scratch | 13 | 30.2% | 52.2% | 52.4% | 5.82% |
| **Scratch** | **34** | **40.6%** | **51.1%** | **52.0%** | **6.15%** |
| Source MAE | 10 | 36.7% | 52.7% | 55.3% | 6.49% |
| Source MAE | 13 | 30.9% | 52.4% | 57.1% | 6.18% |
| Source MAE | 34 | 36.1% | 51.3% | 54.9% | 6.37% |
| Source + target MAE | 10 | 32.4% | 55.8% | 59.3% | 6.35% |
| Source + target MAE | 13 | 33.1% | 54.4% | 58.0% | 6.50% |
| Source + target MAE | 34 | 31.8% | 49.6% | 55.8% | 6.03% |

Values are arithmetic means of three seeds, not ensemble predictions. Scratch-34
patient precision ranges from 26.2% to 51.0%. Scratch hourly AUROC increases from
0.752 to 0.770 between ten and 34 inputs; AP changes from 0.090 to 0.093.
Paired comparisons do not show a consistent alarm improvement across seeds.

The previously quoted **35.2% patient precision / 50.7% timely recall** describes
**source-34, seed 42**, not the scratch three-seed mean. Its patient recall is
54.0% and timely alarm precision 6.30%: 81/230 flagged records have events,
76/150 events have timely alarms, and 76/1,207 alarms match. Unmatched alarms
comprise 648 more than 12 hours early, 60 duplicates, and 423 in nonseptic stays.

### Subset with recent labs

The causal availability gate requires BP less than two hours old, all
creatinine/WBC/platelets less than 24 hours old, and lactate or pH less than
24 hours old. It covers **80/150 events (53.3%)** and **22.0% of decisions**.
It is based on availability at the decision, not measurements collected later.

| Model | Patient precision | Timely alarm precision | Timely recall among covered events | Timely recall among all events |
| --- | ---: | ---: | ---: | ---: |
| Scratch / 10 | 59.0% | 10.77% | 55.0% | 29.3% |
| Scratch / 34 | 57.0% | 10.18% | 55.0% | 29.3% |
| Source MAE / 10 | 53.6% | 10.78% | 54.6% | 29.1% |
| Source MAE / 34 | 52.5% | 10.50% | 55.4% | 29.6% |

The availability subgroup is more favorable, but added channels do not consistently
help on the same gated decisions. Lab ordering may reflect clinical suspicion and
severity; this does not show that ordering additional tests causes improvement.
The gate does not require every added lab and does not isolate individual assays.

## Scripts, saved evidence, and reproduction

The entry point is [scripts/sepsis_missions.py](scripts/sepsis_missions.py).
Matching/bootstrap code is [scripts/sepsis_mission_metrics.py](scripts/sepsis_mission_metrics.py);
the original single-GPU queue is [scripts/run_sepsis_priority_queue.py](scripts/run_sepsis_priority_queue.py).
The complete matrix is specified in
[configs/sepsis_missions_adapt_20261010.json](configs/sepsis_missions_adapt_20261010.json).

Aggregate results, per-arm training histories, frozen policies, paired comparisons,
and original code/data/model hash locks are in
[results/sepsis_missions_adapt_20261010](results/sepsis_missions_adapt_20261010).
Source and target cohort manifests and source MAE histories are in
[results/transfer_protocol](results/transfer_protocol).
Private raw records, predictions, and weights remain local and are not on GitHub.

From the repository root, reproduce the aggregate numbers without training:

```powershell
python scripts/report_measurement_findings.py
python scripts/analyze_sepsis_priority_compact.py  # also needs saved local predictions
python scripts/test_sepsis_mission_metrics.py     # synthetic CPU checks
```

For full retraining, source preparation requires the local MIMIC event database
`results/prospective_qc/events.duckdb`, `results/mimic31_pooled17.pkl`, SICdb raw
tables/previous holdout inputs, and the cached historical Kaggle mirror expected
by the preparation scripts. Their current paths must match your local data layout.
These prerequisites are not supplied by the GitHub commit. Prepare them and run:

```powershell
python scripts/download_cinc_transfer.py
python scripts/prepare_transfer_sources.py
python scripts/prepare_transfer_cinc.py
python scripts/pretrain_transfer_mae.py --seeds 42,123,456 --steps 20000
```

Copy the final matrix config to `configs/measurement_reproduction.json`, change
`run_name` to a new name, and run the following stages sequentially on one GPU.
The original results are frozen and should not be used as a fresh training target.

```powershell
python scripts/sepsis_missions.py --config configs/measurement_reproduction.json prepare
python scripts/sepsis_missions.py --config configs/measurement_reproduction.json train
python scripts/sepsis_missions.py --config configs/measurement_reproduction.json calibrate
python scripts/sepsis_missions.py --config configs/measurement_reproduction.json evaluate
python scripts/sepsis_missions.py --config configs/measurement_reproduction.json summarize
```

On the original machine, use `..\.conda\python.exe` in place of `python`.
Recorded post-run environment: Python 3.13.15, torch 2.6.0+cu124, NumPy 2.5.3,
pandas 3.0.5, scikit-learn 1.9.1, DuckDB 1.5.5. Byte-level hash guards and CUDA
behavior can depend on platform; exact cross-machine reproducibility is not promised.

## Interpretation and limitations

The principal alarm limitation is temporal specificity and repeated false alarms.
Patient-level discrimination is more promising, but it is a retrospective
flag-anywhere metric, not a validated fixed-landmark predictor. A longer horizon
would require a separately defined endpoint. These results are exploratory:
the test records were previously inspected, only 150 test events are available,
seeds reuse the same cohort, onset/receipt timestamps are uncertain, and lab
availability is selective. Bootstrap intervals use 2,000 record resamples at
frozen policies; multiple comparisons are not multiplicity-corrected.

The experiment supports investigating measurement availability and evaluation
definitions. It does not establish clinical readiness, consistent transfer benefit,
or a confirmed gain from additional measurements. A manuscript claiming improvement
needs matched controls and a fresh independent confirmation set.
