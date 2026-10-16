"""Build the paper-style report from aggregate results, without loading patient data."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
INPUTS={}

def read(relative):
    path=RESULTS/relative
    INPUTS[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def pct(x): return f'{100*x:.1f}%'

def table(rows,subgroup=False):
    keys=['patient_precision','patient_recall','precision','recall','alarms_per_patient_day']
    headers=['Approach','Patient precision','Patient recall (all events)','Timely alarm precision','Timely recall (all events)','Alarms/patient-day']
    if subgroup:
        keys.insert(3,'subgroup_recall'); headers.insert(4,'Timely recall (lab-covered events)')
    lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    for name,m in rows:
        values=[f'{m[k]:.3f}' if k=='alarms_per_patient_day' else f'{100*m[k]:.2f}%' for k in keys]
        lines.append('| '+' | '.join([name]+values)+' |')
    return '\n'.join(lines)

def seeded(run):
    rows=read(f'{run}/seed_summary.json')
    return [(r['model'],{k:v['mean'] for k,v in r['metrics'].items()}) for r in rows if r['gate']=='all' and r['target_recall']==.5]

def main():
    final=read('sepsis_lab_specialize_20261014/out_of_fold_summary.json')
    final_rows=[(r['name'],r['metrics']) for r in final if r['target_recall']==.5]
    labels={'rich_scratch':'Scratch, lab-rich training','rich_supervised':'Broad supervised → lab-rich',
        'rich_mae':'Extended MAE → lab-rich','broad_scratch':'Broad neural, no specialization','broad_xgb':'Broad XGBoost, no specialization'}
    main_table=table([(labels[n],m) for n,m in final_rows],True)
    cv=read('sepsis_cv_20261013/out_of_fold_summary.json')
    cv_rows=[(r['name'],r['out_of_fold_metrics']) for r in cv if r['gate']=='all' and r['target_recall']==.5]
    convergence=read('sepsis_lab_specialize_20261014/convergence.json')
    pairs=read('sepsis_lab_specialize_20261014/paired_comparisons.json')
    random=read('sepsis_lab_specialize_20261014/random_baselines.json')
    audit=read('sepsis_cv_20261013/patient_precision_audit.json')
    original=read('sepsis_missions_adapt_20261010/evaluation.json')
    original_groups={}
    for r in original:
        if r['window']==[12,0] and r['gate']=='all' and r['target_recall']==.5 and r['test'] is not None:
            original_groups.setdefault(r['name'].rsplit('_seed',1)[0],[]).append(r['test'])
    original_rows=[]
    for name,rr in original_groups.items():
        original_rows.append((name,{k:sum(x[k] for x in rr)/len(rr) for k in ['patient_precision','patient_recall','precision','recall','alarms_per_patient_day']}))
    controls=seeded('sepsis_controls_20261011')
    long=seeded('sepsis_long_training_20261012')
    lab=seeded('sepsis_lab_values_20261011')
    fine=seeded('sepsis_mae_finetune_20261013')
    mae=read('sepsis_long_mae_20261012/targetmae_34f_seed42.json')
    source_mae=read('sepsis_long_mae_20261012/sourcetargetmae_34f_seed42.json')
    source=read('transfer_protocol/source_manifest.json')
    # Reconcile headline percentages with exact pooled integer counts.
    for _,m in final_rows+cv_rows:
        c=m['counts']
        assert abs(m['patient_precision']-c['flagged_event_records']/c['flagged_records'])<1e-12
        assert abs(m['precision']-c['matched']/c['alarms'])<1e-12
        assert abs(m['recall']-c['matched']/c['event'])<1e-12
        assert c['event']==769
    assert all(m['counts']['eligible_event']==396 for _,m in final_rows)
    assert len([r for r in convergence if r['name']!='extended_mae'])==45
    assert not any(r['hit_cap'] for r in convergence)
    body='''# Do richer measurements and pretraining improve sepsis warning?

## A reproducible exploratory study of causal histories, transfer learning, and lab-rich specialization

**Working paper; results frozen through 16 October 2026.** This report synthesizes
the measurement-expansion, matched-control, training-budget, cross-validation,
and lab-rich specialization experiments. It is a research summary, not a clinical
validation or a claim of readiness for deployment. Earlier feasibility studies
with different cohorts/endpoints are outside its quantitative synthesis.

## Abstract

**Purpose.** Determine whether richer measurements, broad pretraining, and
specialization to lab-rich decision times improve sepsis early warning beyond
causal tree baselines, and distinguish identifying high-risk visits from issuing
usefully timed alarms. **Methods.** We used official CinC 2019 set B, source MAE
on MIMIC/SICdb histories, matched controls, and five-fold development-record
cross-validation. Final comparisons used identical lab-rich decision times and
fold-local preprocessing/pretraining. Three specialization approaches each
searched three learning rates. Target MAE was extended to validation plateaus,
and supervised training allowed 200 epochs with learning-rate reductions.
**Results.** The CV pool contained 16,684 records and 769 eligible sepsis events;
396 events had an eligible lab-rich warning opportunity. Broad supervised
training followed by lab-rich specialization achieved 46.3% patient precision,
53.5% timely recall among lab-covered events, 27.6% timely recall across all
events, and 9.57% timely alarm precision. Extended MAE achieved 46.2%, 47.5%,
24.4%, and 9.87%, respectively. Broad XGBoost evaluated at the same lab-rich
times achieved 50.7% patient precision and 50.3% subgroup timely recall, with
9.67% alarm precision. **Conclusion.** Supervised specialization produced modest
trade-offs, but extended MAE did not consistently outperform strong controls.
Lab-rich eligibility increases background risk and limits event coverage;
subgroup precision is not evidence that obtaining additional tests causes better
predictions. Patient-level metrics substantially overstate the precision of
correctly timed alarm episodes if those endpoints are conflated.

## 1. Purpose and questions

The central question is whether broad physiological representations can be
adapted to a smaller set of richly monitored ICU decision times to improve sepsis
warning. We test three distinct ideas: numerical lab values add information
beyond their measurement patterns; pretraining improves the downstream task;
and supervised specialization improves a broad model within a lab-rich subgroup.
These questions require different controls. Selecting a subgroup is not the
same intervention as adding measurements or fine-tuning a model for it.

The contribution is a controlled empirical comparison and an accounting of
event coverage and alarm burden. The study does not introduce a new architecture
or establish that a neural network is superior to XGBoost.

## 2. Data and endpoint

| Cohort | Records/episodes | Role |
| --- | ---: | --- |
| Official CinC set B, downloaded | 20,000 | Target source files |
| Retained CinC cohort | 19,503 | After 223 left-censored-onset and 274 decision-eligibility exclusions |
| Original fit / tune / calibration / test | 11,759 / 2,991 / 1,934 / 2,819 | Initial experiments |
| Five-fold development pool | 16,684 | Original fit+tune+calibration; 769 eligible sepsis events |
| Original test, excluded from CV | 2,819 | 150 eligible events; previously inspected |
| MIMIC source train / validation | 58,844 / 6,565 | Unsupervised source reconstruction |
| SICdb source train / validation | 13,645 / 1,543 | Unsupervised source reconstruction |

There are 919 eligible events in the retained target cohort, not 19,503 events.
These counts describe the analysis cohort, not raw sepsis-positive totals across
all downloaded databases. A visit supplies many hourly decisions but at most
one onset event here. Released-record identifiers do not permit verification
that separate admissions belong to distinct people. Source histories are capped
at 336 hours; target neural prediction histories are capped at 72 hourly rows.
Source histories are outcome-independent and preserve prior source holdouts.

At hour t, the binary target is assigned onset in [t+1, t+12] hours. Assigned
onset is the first positive challenge-label hour plus six hours, reversing the
dataset's early label shift. Predictions begin at ICU hour six and precede
assigned onset. Negative decisions require 12 hours of available follow-up;
positive decisions within the window remain eligible. These choices define a
retrospective evaluable task and are not equivalent to unrestricted prospective
monitoring. Onset is operational rather than independently adjudicated.

Dataset resource: [PhysioNet/CinC 2019](https://physionet.org/content/challenge-2019/1.0.0/).
Exact local cohort construction and source counts are documented in the
[original experiment report](MEASUREMENT_EXPANSION_REPORT.md) and
[source manifest](results/transfer_protocol/source_manifest.json).

## 3. Inputs, temporal structure, and missingness

The ten shared channels are HR, O2Sat, Resp, Temp, Lactate, pH, Creatinine, WBC,
Platelets and Bilirubin_total. The 13-channel version adds SBP, MAP and DBP.
The 34-channel version also includes EtCO2, BaseExcess, HCO3, FiO2, PaCO2, SaO2,
AST, BUN, Alkalinephos, Calcium, Chloride, Bilirubin_direct, Glucose, Magnesium,
Phosphate, Potassium, TroponinI, Hct, Hgb, PTT and Fibrinogen. Four static inputs
encode age, gender and their knownness. The ten-channel baseline already has
six labs: it is not a vitals-only model.

The neural model is a two-layer causal transformer, width 96, four attention
heads, with a 100→64→1 classifier combining the last history embedding and
static inputs. Each measurement supplies value, current-observation indicator,
ever-observed indicator and observation age. Values are causally carried forward;
never-observed values use a median with explicit flags. Feature/value binding
occurs before pooling. Attention masks future steps and padding; it does not
simply ignore all missing labs. Histories are encoded in temporal order.

XGBoost/RF use causal summaries over 6, 12, 24 and 72 hours: last, mean, SD,
minimum, maximum, slope, change, count, observation age and seen indicator per
channel, plus static inputs and ICU time. They are not current-value-only
comparators. Added-channel normalization is fit-only within each fold; shared
channels retain source-training normalization. Broad physical screening precedes
normalization. Missingness and test-ordering behavior may themselves carry risk
information and may vary by institution.

Our MAE is masked time-series reconstruction with a causal encoder and Huber
loss, not an exact implementation of the image-patch architecture in
[He et al., Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).
It hides 40% of observed targets using spans, channel masks and points, recomputes
carry-forward after masking, and balances reconstruction across patients/channels.
Source MAE uses ten channels; target MAE uses 34. Improved reconstruction need
not improve the discriminative objective or the timing of alarms.

## 4. Design and definitions

Five outer folds stratify by eligible sepsis event. Each outer-training partition
is split 70% fit, 15% tuning and 15% calibration. Each development record is
evaluated exactly once out of fold. The original test set is excluded, but the
development data and CV results have been used iteratively: this remains
exploratory internal validation. Fold models share training data; their results
are not independent replications. CV uses one training seed, 42.

The fixed lab-rich gate requires any blood pressure channel <2 hours old;
creatinine, WBC and platelets each <24 hours old; and lactate or pH <24 hours old.
The gate is evaluated at each decision using only available history. We do not
select a visit because it will eventually receive labs. It permits 85,758 of
375,037 evaluable decision hours (22.9%). There are 4,075 visits with any eligible
lab-rich decision, including 497 event-positive visits. Only 396 of 769 events
have a lab-rich decision during the warning window. The 497 and 396 counts answer
different questions: any opportunity in a septic visit versus a timely opportunity.

Every alarm suppresses further alarms for six hours. At most one timely alarm
is credited per onset. Additional alarms in the warning window are duplicates;
alarms earlier than the window are too early, even in a visit that later becomes
septic. These categories describe this endpoint, not adjudicated clinical utility.

| Metric | Numerator / denominator |
| --- | --- |
| Patient precision | Event-positive visits with any eligible alert / all alerted visits |
| Patient recall | Event-positive visits with any eligible alert / all 769 event-positive visits |
| Timely subgroup recall | Events with a matched 1–12-hour warning / 396 lab-covered events |
| Timely population recall | Events with a matched warning / all 769 events |
| Timely alarm precision | Matched events / all alarm episodes after six-hour suppression |
| Alarms/patient-day | 24 × alarm episodes / evaluable hours across the original cohort |

Here “patient” is shorthand for a released visit record. Patient precision does
not require the alert to be timely. It must not be paired with subgroup timely
recall as if the two came from one binary-classification confusion matrix.

Policies are calibrated for 30%, 50% or 60% recall; the final lab-rich experiment
targets subgroup recall. Held-out recall can differ from its calibration target.
The primary comparison below uses the 50% calibration target, not test-selected
thresholds. All compared models see identical decision eligibility at evaluation.

## 5. What each approach means

| Approach | Initialization and supervised training |
| --- | --- |
| Scratch, lab-rich | Random neural initialization; train only at lab-rich decisions |
| Broad supervised → lab-rich | Start from the fold's broad supervised CinC scratch model, including its classifier; continue training on lab-rich decisions |
| Extended MAE → lab-rich | MIMIC/SIC MAE, then broad fold-local CinC reconstruction, then a fresh classifier and supervised lab-rich fine-tuning |
| Broad neural, no specialization | The original broad supervised neural model, evaluated only when the same lab gate holds |
| Broad XGBoost, no specialization | Broad supervised XGBoost, evaluated at those same lab-rich decisions |

“Broad supervised” learns sepsis labels on the broad **CinC** fit partition; it
does not mean supervised MIMIC/SIC sepsis training. “No specialization” refers
to training, not to a different evaluation population.

Each specialization method searches learning rates 0.00003, 0.0001 and 0.0003
inside each fold. Training allows 200 epochs, a minimum of 30, patience 30, and
halves the rate after eight tuning-AP plateau epochs, down to 0.000001. It uses
weighted BCE and AdamW, weight decay 0.0001. Each epoch samples one eligible
decision per training record, not every hourly row. Tuning record-weighted AP
selects both the candidate and checkpoint; the initial checkpoint is retained if
fine-tuning worsens it. There are 45 candidate fits. The unchanged broad controls
do not receive this extra three-rate specialization search budget.

Each fold's target MAE resumes its saved optimizer/RNG state, with a 100,000-step
cap, minimum 20,000 steps, checks every 500 updates, and patience 12 checks on
512 fixed-mask tuning examples. All target pretraining remains within its fold;
no globally CinC-pretrained checkpoint is reused. The original ten-channel
MIMIC/SIC pretraining is not extended by this intervention.

## 6. Results

### 6.1 Primary lab-rich comparison: pooled out-of-fold performance

'''+main_table+'''

These are ratios of pooled held-out counts, not averages of fold percentages.
Broad supervised specialization has the highest subgroup timely recall here;
XGBoost has the highest patient precision. Extended MAE has the highest alarm
precision by a small margin, alongside lower timely recall. None dominates every
metric, and these different realized operating points do not establish a winner
at exactly equal recall or alarm burden.

### 6.2 Why the three headline metrics differ

For broad supervised → lab-rich, 637 visits trigger alerts: 295 event-positive
and 342 nonseptic visits. Patient precision is 295/637 = 46.3%. Among 396 events
with a lab-rich warning opportunity, 212 receive a timely alarm: 212/396 = 53.5%.
Across all 769 events, the same 212 matches imply 27.6% population timely recall.

Those 637 visits generate 2,215 alarm episodes: 212 timely matches, 1,190 too-early
alarms, 100 duplicate timely-window alarms, and 713 alarms in nonseptic visits.
Thus alarm precision is 212/2,215 = 9.57%. Patient precision can be much higher
because it counts a visit once and credits a septic visit even when its alerts
are too early. A model can improve alarm precision yet reduce patient precision
by reducing repeat alarms while distributing remaining false alerts across more
different nonseptic visits.

### 6.3 Paired differences and uncertainty

Broad supervised specialization versus lab-rich scratch increased subgroup
timely recall by 3.28 percentage points (conditional paired 95% bootstrap interval
0.26 to 6.65 points). Alarm precision increased only 0.12 points (−0.47 to +0.78).
Against the unchanged broad neural model, specialization increased alarm
precision by 0.65 points (+0.15 to +1.13), while its subgroup recall difference
was uncertain (−0.50 to +4.50 points). Its patient precision was lower, not higher.

Extended MAE versus lab-rich scratch changed alarm precision by +0.43 points
(−0.48 to +1.39) and subgroup timely recall by −2.78 points (−6.85 to +1.49).
These intervals do not establish a consistent MAE advantage. They condition on
fitted models/policies, omit training-selection uncertainty, and are not adjusted
for multiple comparisons. No patient-precision confidence interval is implied
by these alarm-metric intervals. Full policies and intervals are linked below.

### 6.4 Was training sufficient to reach the tested stopping rules?

All 45 supervised candidates stopped before the 200-epoch cap. All five extended
MAEs stopped under the validation-patience rule, before 100,000 steps:

| Fold (1-based) | MAE stopped at | Selected MAE checkpoint |
| --- | ---: | ---: |
'''
    for r in convergence:
        if r['name']=='extended_mae': body+=f"| {r['fold']+1} | {r['steps']:,} | {r['selected_step']:,} |\n"
    body+='''
This substantially reduces the short-budget explanation for these runs. It
does not prove a global optimum, adequacy of the architecture/objective, or
convergence of the original source pretraining. Earlier 20,000-update target-only
MAE reduced validation Huber loss from 0.309 at 1,000 updates to 0.214 at 20,000;
source-initialized MAE reached 0.200. Better reconstruction did not consistently
translate into better alarms.

### 6.5 Broad-population CV and the numerical value of labs

'''+table(cv_rows)+'''

`values` retains numerical labs; `availability` removes their numerical content
but keeps observation/age indicators, vital signs and FiO2. All rows retain the
same visits and eligible decisions. For scratch networks, lab values improved
pooled patient precision from 30.2% to 35.2%, while timely recall decreased from
53.3% to 50.7%. XGBoost showed a smaller patient-precision change, 35.6% to 36.7%,
also with lower timely recall. This is a mixed incremental-value result, not
evidence that labs are useless or that more measurement would cause improvement.

The previously quoted 51.1% patient precision for the longer scratch model was
a three-seed mean on one earlier test split. CV pooled precision was 35.2%.
An independent reconstruction from saved scores verified 426 septic and 783
nonseptic flagged visits: 426/1,209 = 35.2%. Fold precision ranged 29.2–50.6%;
the unweighted fold mean was 37.7%. CV had fewer fit records per model, a lower
event prevalence, higher realized timely recall, and more nonseptic stays alerted.
This is evidence of split/operating-point sensitivity, not an isolated causal
estimate of the effect of cross-validation.

### 6.6 Random-guessing reference

For one independent binary guess per visit, with event prevalence p and positive
guess probability q, population precision is p and recall is q. In the full CV
cohort p = 769/16,684 = 4.61%. At q = 0.5, random recall is 50% and precision
remains 4.61%. Among visits with any eligible lab-rich opportunity, prevalence is
497/4,075 = 12.20%. This latter denominator is not the 396 timely-covered events.

Repeated random hourly guesses differ: longer/denser histories get more chances
to alert. We computed exact expected counts under independent Bernoulli decisions,
the same causal gate, six-hour suppression and one matched alarm/event. Guess
probabilities were fitted from training prevalence or matched to each model's
alarm count on calibration data, never selected using held-out outcomes.

For broad supervised → lab-rich, the matched random policy yielded approximately
15.49% patient precision, 18.84% subgroup timely recall and 3.45% alarm precision,
at 0.138 alarms/patient-day. The model achieved 46.31%, 53.54% and 9.57%, at 0.142
alarms/patient-day. This is meaningful signal above the matched random baseline,
but most model alarm episodes still do not meet the specified timing criterion.
Random results are ratios of expected counts, not Monte Carlo averages of ratios.
The analytic recurrence was checked against exhaustive enumeration through the
production evaluator on irregular timelines.

## 7. Discussion and limitations

The strongest supported finding is that useful risk signal exists, while the
timing and repetition requirements substantially reduce alarm precision. Rich
measurements, a gated evaluation population, and specialized training must be
reported separately. The lab-rich gate covers only 51.5% of events during their
warning windows; withholding alerts elsewhere limits whole-population recall.

Supervised broad-to-subgroup adaptation is a reasonable candidate for further
study, but its gains are modest and metric-dependent. Extended MAE cannot be
declared generally ineffective: the tested reconstruction objective may favor
stable physiology rather than transition timing; source inputs cover only ten
channels; added feature embeddings and source/target measurement practices differ.
These are hypotheses, not demonstrated explanations. Other architectures,
pretraining targets and layer-specific learning rates were not systematically
searched. The final search varied three common learning rates, not all relevant
hyperparameters.

Key limitations are operational onset uncertainty; retrospective eligibility
requiring follow-up; a restricted public variable set without full clinical context;
missingness and measurement-ordering confounding; unverified cross-admission
person identity; one training seed in CV; repeated inspection of development and
test results; and conditional rather than full model-selection uncertainty.
Tree and neural training also differ in hourly sampling exposure. No currently
verified untouched external cohort was available for the same endpoint, as
documented in the [confirmation audit](results/sepsis_controls_20261011/CONFIRMATION_AUDIT.md).
No causal recommendation to increase particular measurements follows from these
retrospective subgroup results. No comparison to published AUROC or official
challenge utility should be treated as an apples-to-apples alarm-precision claim.

## 8. Conclusion

Across these experiments, neural models and causal tree baselines performed
well above matched random alarms but retained low timely alarm precision.
Longer pretraining and lab-rich specialization did not produce a consistent
dominant model. The defensible contribution is a reproducible analysis of
measurement value, transfer-learning limits, and the gap between visit-level
discrimination and timely warning—not a clinically usable sepsis alarm system.

## 9. Reproducibility and artifact map

The report is generated from aggregate JSON by
[`scripts/build_working_report.py`](scripts/build_working_report.py). Run
`python scripts/build_working_report.py` from the repository root to regenerate
it without patient-level data or a GPU. The accompanying
[input-hash manifest](results/paper_snapshot_20261016/report_inputs.json) identifies
the exact source reports. [Runtime versions](results/paper_snapshot_20261016/runtime.json)
record the reporting machine; they are not a guarantee of bitwise GPU reproducibility.

Training ran locally on one RTX 2080 Ti. Raw clinical data, fold pickles, per-record
scores, feature caches, model weights and optimizer recovery states remain local
and are not distributed in this commit. Repository code and aggregate reports
are sufficient to inspect the methods and regenerate this paper, but retraining
requires authorized data access and rebuilding the local prepared artifacts.

| Stage | Entry point / protocol | Results |
| --- | --- | --- |
| Original measurement expansion | [sepsis_missions.py](scripts/sepsis_missions.py), [report](MEASUREMENT_EXPANSION_REPORT.md) | [evaluation](results/sepsis_missions_adapt_20261010/evaluation.json) |
| Matched XGB/RF/MAE controls | [run_sepsis_controls.py](scripts/run_sepsis_controls.py) | [summary](results/sepsis_controls_20261011/seed_summary.json) |
| Numerical lab ablation | [run_lab_value_experiment.py](scripts/run_lab_value_experiment.py) | [summary](results/sepsis_lab_values_20261011/seed_summary.json) |
| Longer supervised training | [run_long_sepsis_training.py](scripts/run_long_sepsis_training.py) | [summary](results/sepsis_long_training_20261012/seed_summary.json) |
| Longer target MAE / pilot fine-tuning | [run_long_target_mae.py](scripts/run_long_target_mae.py), [run_mae_finetune.py](scripts/run_mae_finetune.py) | [pilot summary](results/sepsis_mae_finetune_20261013/seed_summary.json) |
| Five-fold broad CV | [run_sepsis_cv.py](scripts/run_sepsis_cv.py), [protocol](results/sepsis_cv_20261013/PROTOCOL.md) | [pooled results](results/sepsis_cv_20261013/out_of_fold_summary.json) |
| Lab-rich specialization | [run_lab_specialization.py](scripts/run_lab_specialization.py), [protocol](results/sepsis_lab_specialize_20261014/PROTOCOL.md) | [pooled results](results/sepsis_lab_specialize_20261014/out_of_fold_summary.json), [folds](results/sepsis_lab_specialize_20261014/fold_metrics.json), [paired intervals](results/sepsis_lab_specialize_20261014/paired_comparisons.json) |
| Random baseline / patient-count audit | [calculate_random_baselines.py](scripts/calculate_random_baselines.py), [audit_cv_patient_precision.py](scripts/audit_cv_patient_precision.py) | [random baselines](results/sepsis_lab_specialize_20261014/random_baselines.json), [count audit](results/sepsis_cv_20261013/patient_precision_audit.json) |

### Local training prerequisites and commands

Install compatible packages from `requirements.txt`; use the recorded CUDA-enabled
PyTorch build for GPU training. Existing data-access paths are local-project
conventions, not bundled datasets. Source preparation uses
`prepare_transfer_sources.py`, `pretrain_transfer_mae.py --seeds 42,123,456`, and
`prepare_transfer_cinc.py`; inspect their input paths and existing manifests before
running. The source-preparation script also expects the existing extracted MIMIC
cache and SICdb access. This is not a fresh-download, one-command reproduction.

The later runners expect these local artifacts:

- `results/transfer_protocol/local/source_scaler.npz` and `mae_seed42.pt`
  (also seeds 123/456 for the earlier matrices).
- `results/sepsis_missions_adapt_20261010/local/{fit,tune,cal,test}.pkl` and
  `scaler.npz`, prepared with `python scripts/sepsis_missions.py --config
  configs/sepsis_missions_adapt_20261010.json prepare` after target preparation.
- CV creates its own fit/tune/cal/test files, scaler and fold-local MAE states;
  specialization requires those completed fold artifacts.

```powershell
# After authorized data preparation, in a separate reproduction checkout:
python scripts/run_sepsis_controls.py queue
python scripts/run_lab_value_experiment.py queue
python scripts/run_long_sepsis_training.py queue
python scripts/run_long_target_mae.py
python scripts/run_mae_finetune.py queue
python scripts/run_sepsis_cv.py prepare
python scripts/run_sepsis_cv.py queue
# Frozen original CV runner has a reporting-only aggregation bug; recover with:
python scripts/summarize_sepsis_cv.py
python scripts/run_lab_specialization.py prepare
python scripts/run_lab_specialization.py queue
python scripts/calculate_random_baselines.py
python scripts/audit_cv_patient_precision.py
python scripts/build_working_report.py
```

Existing JSON reports are archived results, not reusable completion markers for
new training without their matching local model files. Use separate fresh result
directories/checkout for reruns and preserve the published archive; runners lock
code/data hashes and intentionally reject mismatches. Hard-coded result roots in
wrapper scripts must be kept consistent if renamed. Do not silently replace lock
files to mix changed data with published results. The original CV aggregation
failed on subtraction of nested count dictionaries; `summarize_sepsis_cv.py`
repairs only reporting, verifies frozen models/policies, and records its recovery.
The independent patient-count audit and the report generator reconcile headline
ratios with their numerators and denominators.

## Appendix: earlier comparison matrices

These tables preserve the preceding results rather than selecting only favorable
runs. Unless specified otherwise, they show three-seed means on the original
2,819-record test split at the 50% calibration target, 12-hour warning window,
and no lab gate. They must not be pooled with five-fold out-of-fold counts.
Additional 30/60% policies and lab-gated curves are available in each result set.

### A. Original measurement expansion (27 fits)

'''+table(original_rows)+'''

### B. Matched controls (12 tree fits, 24 neural fits)

'''+table(controls)+'''

### C. Numeric lab values versus availability (nine new fits, nine reused)

'''+table(lab)+'''

### D. Longer supervised training (27 fits)

'''+table(long)+'''

Twenty-three of 27 fits selected checkpoints after epoch 20; two reached the
100-epoch cap. The earlier stopping rule was too restrictive for several fits,
but validation gains did not guarantee improvements in the frozen alarm policy.

### E. Fine-tuning 20,000-update MAE (seed 42 only; four new fits, two reused)

'''+table(fine)+'''

This one-seed pilot did not establish a robust transfer advantage and motivated
fold-local validation. The final lab-rich study supersedes it for the specific
broad-to-lab-rich adaptation question, not for every possible pretraining recipe.
'''
    (ROOT/'working.md').write_text(body,encoding='utf-8')
    destination=RESULTS/'paper_snapshot_20261016'; destination.mkdir(exist_ok=True)
    (destination/'report_inputs.json').write_text(json.dumps(INPUTS,indent=2))
    runtime=dict(python=platform.python_version(),platform=platform.platform(),packages={})
    for name in ['numpy','pandas','torch','scikit-learn','xgboost','duckdb']:
        try: runtime['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: runtime['packages'][name]=None
    (destination/'runtime.json').write_text(json.dumps(runtime,indent=2))
    print(f'Built working.md from {len(INPUTS)} aggregate reports; pooled count checks passed.')

if __name__=='__main__': main()
