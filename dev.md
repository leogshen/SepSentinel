# SepSentinel Local Development

## Python environment

The project Conda environment is located at:

```text
F:\Claude\Sepsentinel\.conda\python.exe
```

Codex can invoke this interpreter directly, so activating the environment is
not required when Codex runs local jobs.

For interactive work in PowerShell or the PyCharm terminal, activate it with:

```powershell
conda activate F:\Claude\Sepsentinel\.conda
cd F:\Claude\Sepsentinel\Code
```

## GPU configuration

Verified from Codex:

- PyTorch: `2.10.0+cu130` (dev.md previously said 2.6.0+cu124; re-verified 2026-09-12)
- CUDA available: `True`
- GPU: NVIDIA GeForce RTX 2080 Ti

To verify the environment again:

```powershell
F:\Claude\Sepsentinel\.conda\python.exe -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Running project jobs

Run scripts from the `Code` directory so project imports and relative output
paths resolve correctly:

```powershell
cd F:\Claude\Sepsentinel\Code
F:\Claude\Sepsentinel\.conda\python.exe <script.py> [arguments]
```

Generated outputs should be written beneath `F:\Claude\Sepsentinel`, normally
under `Code\results` for experiments and evaluations.

---

# Experiment record and conclusions

Compiled 2026-09-19 from the git history, `Code/RESULTS.md`, `Code/results/`
and `Code/TECHNIQUES.md`. This is the "what did we learn" companion to the
environment notes above. Numbers marked **(re-scored)** come from
`Code/results/refrozen/REFROZEN.md`, which uses validation-frozen thresholds
and paired patient-bootstrap intervals; earlier point estimates elsewhere in
the repo were produced with a threshold maximised on test and are optimistic.

## The one-paragraph state

Model B (physiology + labs to hourly sepsis risk) is complete end to end on
MIMIC-IV 3.1, 63,672 episodes. The best deployable configuration is **flat
XGBoost**, 18 features, pre-onset target: patient recall 0.64, median lead
20.6 h, capture >=6h 0.53, at <=1.0 false alerts per nonseptic patient-day.
Nine months of architecture work has produced no durable gain; every gain
came from changing the problem definition or the evaluation. The binding
constraint is measurement sparsity, not model capacity, which is the
quantitative case for Model A (wearable biomarker sensing).

## Timeline of experiments

### Part 1 - PhysioNet/CinC 2019 (Jun-Aug 2026)

| # | Experiment | Conclusion |
|---|---|---|
| 1 | Five architectures | Transformer 0.793 AUROC > TCN 0.787 > GRU 0.782 >> XGBoost 0.638 > RF 0.576. Concluded "sequence models vastly outperform flat". **Later overturned - see Part 2.** |
| 2 | Imputation / Strategy B | Causal forward-fill + train-median + lab masks + time-since-last deltas. Became the standard representation. |
| 3 | Feature ablation, 15 configs | Config I (all minus creatinine), 9 features, confirmed over 3 seeds. |
| 4 | Trajectory + gating | **Built 2026-08-19, never run.** Still outstanding. |
| 5 | Recall study | Loss reweighting does not move AUROC. |
| 6 | MAE self-supervised pretraining | No discrimination gain: scratch 0.814, fine-tuned 0.816, frozen 0.776. **And the pretext was defective - see below.** |
| - | Production baseline | AUROC 0.814 +/- 0.004, AUPRC 0.144; at 70% patient recall: precision 9.3%, 1.7 alerts/pt-day, median lead 23.5 h. |

**The 2026-08-19 pairing bug.** `collect_patient_predictions` paired patient
metadata by dataset order while `collate_fn` sorts each batch by length,
scrambling attribution for 97% of test episodes. Every patient-level number
before that date is invalid. Timestep metrics, AUROC and AUPRC were
unaffected.

### Part 2 - MIMIC-IV 3.1 (Sep 2026)

| # | Experiment | Conclusion |
|---|---|---|
| 7 | Sepsis-3 labels (DuckDB port of mimic-code) | Culture + antibiotic suspicion, hourly SOFA, Challenge timing rules. 63,672 episodes, 7,345 septic. |
| 8 | Post-onset truncation | Post-onset hours were hiding the early warning: the easy positives dominate and the model alarms late. |
| 9 | **Pre-onset target** | Positives confined to the window before onset. **+7 h median lead, capture >=6h 0.43 to 0.53** at equal burden. The single largest gain in the project. |
| 10 | Ceiling diagnosis | The model effectively sees three signals - HR, SpO2, respiratory rate at ~95% of hours. Every lab is 2-8%. |
| 11 | **AUROC is anti-correlated with early warning** | Five independent observations, including a synthetic control where a 16x stronger injected signal gave higher AUROC and *shorter* warning. |
| 12 | Selection-metric study (n=20) | Selecting on AUROC instead of capture costs a quarter of the early warning; **capture moves by 0.162**. |
| 13 | Missingness encoding | Four arms span **0.007 AUROC**. Hand-built mask/delta worth +0.0045 over forward-fill alone; GRU-D decay and learned embeddings add nothing. Not the bottleneck. |
| 14 | Positive control (synthetic ramp) | **Measurement frequency beats effect size**: a 0.25 SD drift in a dense channel beats a 4 SD shift in a sparse one. The quantitative case for Model A. |
| 15 | Architecture x trajectory sweep | Six configs span 0.010 AUROC; TCN's AUPRC deficit (1.6 points) is the only real difference. |
| 16 | Literature review (47 papers) | External anchors; YAIB MIMIC-IV sepsis GRU 83.6 AUROC / 9.1 AUPRC, but label definition alone moves their AUPRC 6.1 to 17.7. |
| 17 | Foundation models | No FM worth building on. TimesFM installed and rejected (no classification head, wrong context geometry, padding-mask not observation-mask, no clinical pretraining). MIRA architecturally right but weights unreleased. |

### Part 3 - 2026-09-12 to 09-19

| # | Experiment | Conclusion |
|---|---|---|
| 18 | Recipe factorial (4 arms x 3 seeds) | **`pos_weight` + TLS is the best recipe.** Re-scored: recall **+0.090 [+0.066, +0.114]**, capture >=6h **+0.047 [+0.027, +0.068]**, >=12h **+0.046**. Lead-time change is null (-0.12 h). Cost: -0.039 patient precision. |
| 19 | ...`pos_weight` alone | Neutral: -0.005 recall, CI spans zero. TECHNIQUES ranked this #1 at "up to +4.2 AUPRC". It is worth 22% faster training and nothing else. |
| 20 | ...TLS *without* `pos_weight` | **Catastrophic**: -0.151 recall. Applying the review's two top actions together gives the worst of four arms. |
| 21 | Prodrome window sweep (3/6/12/24 h) | A real but small dial. W=3 maximises recall (+0.029 to +0.033 over others); W=24 maximises lead (+2.25 h) and capture >=6h (+0.027); patient precision rises monotonically with W. W=12 vs W=3 trades -2.9 pts recall for +1.9 pts precision and ~1.5 h lead. |
| 22 | Attention at W=3 | Transformer **loses** recall (-0.086), capture >=6h (-0.068), >=12h (-0.062); **wins** patient precision (+0.029), AUROC (+0.019), AUPRC (+18%). The "alarms later" explanation is *not* measurable (-1.88 h, CI spans zero). |
| 23 | **Threshold-on-test defect** | `at_burden()` maximises patient recall on the split it reports. Measured cost: a **+0.028 recall gain with a CI excluding zero that fell to +0.001** at equal burden. Replaced by `spend_burden()`. |
| 24 | TLS on XGBoost | **Null to harmful.** At equal burden recall +0.001; capture >=6h **-0.031** and >=12h **-0.032**, both excluding zero. A mass-matched third arm is null on everything, so the ramp shape contributes nothing on a tree ensemble. |
| 25 | SICdb IL-6 census | 4,225 cases with IL-6 (15.4%), **1,965 with >=2 draws**, 657 with >=5. PCT 5,448, CRP 26,577. |
| 26 | **Alarm episodes + refractory period** | An episode-*rate* budget does not constrain. At R=12 h capture >=6h appears to gain +0.394 while timestep precision falls 0.097 to 0.052 and nonseptic patients alarmed rises **28% to 87%**. Fix: constrain on patients-disturbed (R-invariant), report episode rate descriptively. |

## What has NOT worked - every model-side change

Ordered by how much effort it consumed.

| Intervention | Result |
|---|---|
| MAE self-supervised pretraining | Null (0.814 scratch, 0.816 fine-tuned, 0.776 frozen). **And the recipe was defective**: reconstruction targets are post-preprocessing values, so forward-filled and median-imputed values are targets; the observation masks sit in the same tensor and never restrict the loss; masking is i.i.d. per timestep, so on a staircase the unmasked neighbour is the answer. The reported 72% MSE reduction may largely measure copy-the-neighbour. |
| Architecture (GRU / TCN / Transformer / gating) | Sweep spans 0.010 AUROC. Attention adds +0.014 AUROC over flat XGBoost on matched features and *loses* every early-warning metric. |
| Missingness encoding (GRU-D decay, learned embedding) | Four arms span 0.007 AUROC. Closed the question of adopting an irregular-time-series encoder. |
| `pos_weight` / class reweighting | Neutral. Tuned weighted-CE and focal loss reduce to plain CE (Yeche et al.), and we measured the same. |
| Temporal Label Smoothing **on XGBoost** | Null to harmful. Works on the Transformer only. |
| Foundation models | TimesFM rejected on four verified grounds; MIRA blocked on unreleased weights; no FM has a published head-to-head win over a tuned GRU or LGBM on ICU sepsis. |

**The pattern.** No model-side change has produced a durable win. Every gain
came from the problem definition or the evaluation: the pre-onset target
(+7 h lead), the selection metric (capture +/-0.162), TLS+pos_weight on the
Transformer (+0.090 recall), and the Strategy B representation (+0.082 AUROC
over raw current-hour values, most of which is the forward-fill).

**The recurring failure mode**, eight instances and counting: *a quantity
optimised or compared on the same axis it is then reported on.* The
patient-pairing bug, SOFA scored in empty pre-ICU hours, the phantom
zero-filled IL-6 channel, AUROC-based model selection, the
imputation/indicator conflation, the test-selected threshold, the TLS anchor
offset, cross-window F1 comparison, and the episode-rate budget. Assume the
next surprising result is this before believing it.

## What could improve the model - ranked by evidence

1. **Conformal abstention (COMPOSER-style).** Reported -78 to -86% false
   alarms per patient-hour and +13-24 specificity points at near-zero AUROC
   change; prospectively a 17% relative reduction in sepsis mortality. The
   largest untested lever, and it acts on alert burden, the axis where we are
   already competitive. Caveat: a rejected prediction is a delayed or
   withheld alarm, not a free one - rejected septic patients must stay in the
   denominator with their delay counted.
2. **Multi-source pretraining (MIMIC + SICdb).** The only SSL setting with a
   surviving positive result (GenHPF: -0.001 AUROC single-source, **+0.9
   multi-source**). Unblocked now that both datasets are in hand; pretraining
   is self-supervised so SICdb's missing culture labels do not block it.
   Design around hospital-shortcut learning, negative transfer from a
   perioperative case mix, a ~50x density mismatch, and the fact that SICdb
   patients used for pretraining cannot also serve as external validation -
   split and seal a hold-out first.
3. **Optimise the utility surrogate directly.** The CinC 2019 winner's
   largest single lever: regress on the per-timestep utility differential
   rather than the binary label, then tune the threshold by direct utility
   maximisation. Never tried here, and it attacks the exact metric-mismatch
   problem that has dominated this project.
4. **Feature ablation on the extended 18.** They went in as a block and
   nobody has checked which earn their place. MAP and urine output are the
   dense ones and the likely winners. Cheap.
5. **Experiment 4 (trajectory + gating).** Built 2026-08-19, never run.
6. **Seed averaging / ensembling.** Measured at +0.004 AUROC. Real but small.
7. **Calibration.** Necessary if a cost-derived threshold is ever to be
   computed, but **bounded**: it is a monotone remap of the score and
   therefore cannot move the recall-burden frontier at all. It buys
   interpretability of the number, not performance. Isotonic is
   rank-preserving only up to the ties it creates.
8. **More measurement - i.e. Model A.** The ceiling is lab sparsity, and the
   positive control quantified it: frequency beats effect size by a wide
   margin. This is the project thesis and the only intervention that attacks
   the actual binding constraint.

## Open decisions blocking further work

1. **Who receives the alert, and what action does it trigger?** This governs
   three things at once: which metric selects models (recall vs capture >=6h
   vs patient precision), which cost unit a utility formula charges in (per
   alarm-hour, per alarm episode, or per alarmed patient), and what the
   refractory period R should be. At 20-24% patient precision, four of five
   alerted patients are not septic - defensible for a clinician, probably not
   for a consumer device.
2. **Episode-rate or patients-disturbed as the alert budget?** Only the
   second actually constrains.
3. **W=12 or W=3?** W=3 won the pre-registered rule; W=12 keeps more lead and
   more precision. A trade, not a dominance.

## Datasets in hand

| Dataset | Scale | Sepsis labels | Serial IL-6 | Role |
|---|---|---|---|---|
| MIMIC-IV 3.1 | 63,672 episodes | Culture-based Sepsis-3 | none | Primary training and evaluation |
| PhysioNet/CinC 2019 | 14,057 patients | Challenge rules | none | Continuity reference; shares BIDMC with MIMIC, so NOT clean external validation |
| SICdb 1.0.8 | 27,350 cases | **none** (no microbiology) | 1,965 patients | Pretraining pool; cross-site alarm burden; perioperative case mix |
| Davoudian / Humanitas (PMID 36189302) | 135 patients | Sepsis-3, with 28d/90d/1y outcomes | **79 patients, d1 + d5** | Labelled serial IL-6 + PTX3; complements SICdb's scale-without-labels |
| Zigong Fourth People's Hospital | downloaded 2026-09-07 | unexamined | unexamined | Unexamined |

---

# Update 2026-09-22

Everything above was written 2026-09-19. Three things changed since: SICdb is
loaded end to end, the multi-source pretraining question is answered, and the
product decision is made. Where this update contradicts the record above,
this update wins.

## The product decision is settled

**Inpatient hospital monitoring. The alert goes to a nurse or clinical
response team, not to the patient.** That single choice closes four open
questions at once:

| Question | Resolved to | Why |
|---|---|---|
| Governing metric | **Capture >=6h at constrained burden** | The team's action (cultures, fluids, escalation) needs hours to matter. Plain recall over-credits an alarm 1 h before onset that changes nothing. |
| Cost unit | **Episodes per patient-day**, with patient precision as a credibility floor | A nurse covering 20 beds experiences total interruptions. But precision is what stops alarm fatigue: the R=12 h config failed at precision 0.126, not at workload. |
| Refractory R | **2-6 h, set not tuned** | A ward number. 12 h was never clinical. |
| Prodrome window | **W=12 stands** | Under capture >=6h the ordering flips from the recall ordering. W=24 is nominally best but 53% degenerate. |

Two consequences: **XGBoost wins decisively** (capture >=6h 0.523 vs the
Transformer's 0.408), and **TLS becomes moot** -- it helps the Transformer and
harms XGBoost, so choosing XGBoost retires the best model-side result we had.

## Experiments 27-30

| # | Experiment | Conclusion |
|---|---|---|
| 27 | SICdb rung-1 labels (`build_sicdb_labels.py`) | Built, then **measured unfit for training**: 27.6% sepsis rate in a low-mortality perioperative cohort, 7x more cases than the admitting physician flagged, and 22.7% of ELECTIVE surgical patients called septic at 0.9% mortality. They do track severity (3.2x mortality, +9 SAPS3), so they are kept for stratification only. |
| 28 | SICdb episodes (`extract_sicdb.py`) | 19,899 episodes on the pooled 17-feature layout. |
| 29 | **Four arms: pooled pretraining** | **Adding SICdb buys nothing.** Compute-matched, +0.005 capture >=6h [-0.010, +0.020]. MAE still does not help even with the pretext repaired (-0.020 vs scratch). XGBoost beats every encoder arm by 0.124 capture >=6h, ~6 CI widths. |
| 30 | Density ablation + utility surrogate | IN FLIGHT as of this update. |

## The exact-burden threshold, and why it matters more than it sounds

Burden depends only on NON-SEPTIC alarm-hours, so the threshold that hits a
target burden is a quantile, not a search:

    k = budget * nonseptic_hours / 24
    threshold = k-th largest nonseptic hourly probability

`operating_curves.threshold_for_exact_burden()`. Without it, **two burden
confounds nearly produced two different wrong answers in one afternoon**:

- Validation-frozen thresholds said pretraining *hurts badly* (-0.087
  capture >=6h) -- but the pretrained arms had drifted to 0.099-0.182
  alerts/patient-day BELOW scratch, and an arm that alarms less catches less.
- Re-pinning on the 0.01 threshold grid flipped the sign to **+0.021**,
  because that grid is far too coarse at 2.6% prevalence and left arms spread
  across 0.86-0.97.
- Solving exactly gives the real answer: **-0.020**.

This is the same failure that manufactured a CI-backed +0.028 TLS "gain" that
was really +0.001. It is now the ninth instance of the project's recurring bug
class, and the fix is mechanical rather than a matter of care.

## Harmonisation: five findings, all from checking rather than assuming

1. **`Harnstoff` (355) is UREA, not BUN.** MIMIC's BUN is urea nitrogen, so
   the channel needs /2.14. A definition mismatch, not a unit one -- no
   amount of normalisation would have caught it.
2. **FiO2 is DataID 2283**, not the SignalFloat 727, which has zero rows.
3. **Urine output is per-hour, not cumulative** (cnt ~1.0/bin, median 50 mL),
   verified rather than inferred from the "(c)" in the item name.
4. **The "50x density mismatch" is wrong at the level the model sees.** Raw
   measurements are ~58/h vs ~1/h, but after hourly binning vitals match at
   1.0-1.1x and routine labs at 0.9-1.0x. The gap is confined to the
   blood-gas panel: lactate 10x, PaO2 7.6x, pH 6.7x.
5. **GCS is dropped from BOTH sources** -- 17 pooled features, not 18. A
   channel 100% absent at one site and present at the other is a free
   hospital identifier, readable straight off the observation mask.

Plus one bug that would have poisoned the experiment silently: `data_float_h`
spans the HOSPITAL stay, not the ICU stay. Gridding to max(hr) gave every
case a 144-hour record with empty post-ICU tails and discarded **63% of SICdb
as "too sparse"**. Bounding by ICU LOS took the cohort from 3,157 to 19,899.

## The hospital-shortcut alarm was a false positive

The probe reads **AUROC 0.989** on pretrained encoders, which looks damning.
The **random-encoder control reads 0.993**. The two databases are separable
from the raw inputs alone -- those blood-gas mask channels -- and pretraining
moves it **-0.004**. Reporting the probe without the control would have
invented a shortcut that does not exist.

## Revised: what could improve the model

Reordered, because item 2 has now been tested.

1. **Optimise the reported metric directly.** The product decision makes this
   concrete: weight each pre-onset hour by the warning it actually delivers
   rather than labelling all 12 identically. IN FLIGHT.
2. ~~Multi-source pretraining~~ **TESTED AND NULL** (experiment 29). Two
   hospitals, 17 channels harmonised, compute-matched: +0.005 [-0.010,
   +0.020]. This was the highest-evidence open item and it is now closed.
3. **Conformal abstention (COMPOSER-style).** Now the largest untested lever:
   -78 to -86% false alarms, acting on the axis the product actually
   constrains. Caveat: a rejected prediction is a delayed or withheld alarm,
   so rejected septic patients must stay in the denominator with their delay
   counted -- zero cost is an assumption, not a property.
4. **Clinical derived features.** Shock index, BUN/creatinine ratio, partial
   SOFA, SOFA deterioration deltas. NOT reformatting -- encoding
   physiological relationships BETWEEN channels that a tree must otherwise
   rediscover. The CinC 2019 winner got 0.389 -> 0.418 from exactly this, and
   it is the one item in TECHNIQUES.md Topic C never attempted here.
5. **Feature ablation on the extended set.** Still never done.
6. **More measurement, i.e. Model A** -- but see the density ablation, which
   is the first causal test of whether that premise holds.

## Is data standardisation a dead end? Four nulls, one open case

| What was standardised | Gain |
|---|---|
| Raw values -> Strategy B (ffill + mask + delta) | **+0.082 AUROC** |
| ...but the mask/delta indicators ALONE | +0.0045 |
| GRU-D decay / learned (mask, delta) embedding | null (4 arms span 0.007) |
| MIMIC + SICdb harmonisation and pooled pretraining | +0.005, null |

Representation delivered the single largest win in the project -- and it was
almost entirely the **forward-fill**, which is already in every model. Every
further attempt to represent irregularity better has returned nothing.

The honest split: **reformatting and harmonising the data is a closed
question. Deriving new physiological features from it is open and untested.**
Those are different claims and the evidence separates them cleanly.

## Datasets, revised

| Dataset | Scale | Sepsis labels | Role now |
|---|---|---|---|
| MIMIC-IV 3.1 | 63,672 episodes (17-feature pooled: `mimic31_pooled17.pkl`) | Culture-based Sepsis-3 | Primary training and evaluation |
| SICdb 1.0.8 | **19,899 episodes loaded**; 13,951 pretrained on, **5,948 SEALED** | rung 1, unfit for training | Pretraining pool (tested, null); sealed hold-out preserves external validation |
| PhysioNet/CinC 2019 | 14,057 patients | Challenge rules | Continuity reference; shares BIDMC with MIMIC so NOT clean external validation |
| Davoudian / Humanitas (PMID 36189302) | 135 patients, 79 with d1+d5 | Sepsis-3 with outcomes | Serial IL-6 + PTX3. **Immortal-time trap: 0 of the 79 with a day-5 sample died by day 5**, and 28-day mortality is 9.4% vs 43.1%. Any trajectory-vs-outcome analysis needs a day-5 landmark, which leaves 6 deaths at 28d and 12 at 90d -- one predictor, not nine. |
| Zigong Fourth People's Hospital | downloaded 2026-09-07 | unexamined | Unexamined |
## Deferred alternative tasks — 2026-10-09

These are fallback research directions, to consider only if the sepsis-focused
work on richer-input transfer learning, timing tolerance, and measurement-defined
subgroups is unsuccessful. They are not currently authorized training jobs or
replacements for the primary sepsis objective. Simpler means a clearer endpoint
or decision time, not merely an easier denominator.

| Candidate task | Exact setup | Dataset fit and limitation |
| --- | --- | --- |
| Recognize current sepsis | At a scheduled assessment, identify whether operational sepsis criteria have already been met. | CinC supports a retrospective label-based version after accounting for its six-hour label shift. Potentially circular if inputs reconstruct the label; this is recognition, not early warning. |
| Predict deterioration after sepsis recognition | Among patients already identified as septic, predict a subsequent shock or organ-support event within a fixed horizon. | MIMIC is a candidate; medication and outcome timelines need extraction and audit. Preserves a sepsis-focused objective. Published precedent: [TREWScore](https://pubmed.ncbi.nlm.nih.gov/26246167/). |
| Predict circulatory failure | Among currently stable patients, predict a clearly defined future circulatory-failure event. | MIMIC/SICdb are candidates because richer physiological and medication data exist. Requires a validated label pipeline; HiRID results do not guarantee transfer. [circEWS](https://www.nature.com/articles/s41591-020-0789-4). |
| Predict mortality from a fixed early window | Use the first 24 or 48 hours to predict subsequent in-hospital death among patients alive at that landmark. | MIMIC is suitable; SICdb mortality endpoints need harmonization. CinC 2019 does not provide mortality outcomes. [Established first-48-hour benchmark](https://www.nature.com/articles/s41597-019-0103-9). |
| Forecast a future measurement | Predict the next observed lactate or creatinine value from preceding history. | Tests whether temporal representations beat carry-forward. Evaluate timing and observation-selection effects: results apply to patients remeasured. A supporting experiment, not a substitute for clinical event prediction. |

Primary work before considering these fallbacks: retain pretrain–fine-tune;
expand beyond the ten shared channels using feature-specific adapters and masks;
compare scratch and pretrained encoders with identical target inputs. Evaluate
predeclared early-warning windows separately from near-onset recognition, and
test measurement groups using the same patients, decision times, and endpoint.
Higher performance in a well-measured subgroup alone does not establish that
ordering more tests would cause an improvement. Report subgroup coverage and
population event capture in addition to subgroup precision/recall.

## Richer measurements and transfer: completed results (2026-10-10; documented 2026-10-11)

The sequential RTX 2080 Ti queue completed 27 fits: seeds 42/123/456, inputs
10/13/34, and scratch/source-MAE/source-MAE-plus-target-MAE initialization.
These are exploratory results on the same previously inspected target records,
not three independent confirmation cohorts. Aggregate evidence is in
`results/sepsis_missions_adapt_20261010/`, including `evaluation.json`,
`seed_summary.json`, `paired_comparisons.json`, and `measurement_findings.json`.

### Inputs, output, and time handling

The target is official PhysioNet/CinC 2019 **set B**: 20,000 released records,
19,503 retained; fit/tune/calibration/test = 11,759/2,991/1,934/2,819.
Exclusions: 223 left-censored onsets and 274 records without a resolved eligible
decision. Test contains 150 eligible sepsis events. Record identity is available;
repeated-person identity is not. The old 14,057-record mirror matches set A,
not B. This supersedes the older target description above for these experiments.

At each eligible hourly decision, output is a risk score for assigned sepsis onset
within the **next 12 hours**, using at most 72 hours of causal history and known
age/gender. Assigned onset is the first positive CinC label's ICULOS plus six
hours, undoing the challenge's early label shift. No post-onset input is used for
the early-warning views. Unresolved negative horizons are excluded. The 12-hour
test view has 63,738 decisions, 1,622 positive (2.54%).

| Arm | Measurement channels |
| --- | --- |
| 10 | HR, O2Sat, Resp, Temp; **Lactate, pH, Creatinine, WBC, Platelets, Bilirubin_total** |
| 13 | Those ten plus SBP, MAP, DBP |
| 34 | Those thirteen plus EtCO2, BaseExcess, HCO3, FiO2, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Bilirubin_direct, Glucose, Magnesium, Phosphate, Potassium, TroponinI, Hct, Hgb, PTT, Fibrinogen |

Thus ten versus 34 is **additional measurements**, not a vitals-only versus labs
comparison. A vitals-only ablation has not been run in this matrix.
The encoder is a two-layer causal transformer, width 96, four attention heads.
Each feature supplies normalized value, current observation flag, ever-observed
flag, and observation age. Values carry forward from preceding observations;
never-observed features use the normalization median with explicit missingness.
Padding and future time steps are attention-masked. Missing individual labs are
represented explicitly rather than simply deleting the corresponding hour.
Feature/value interactions precede feature pooling. Sparse lab trajectories may
still be hard to learn with this small pooled architecture; this is a hypothesis,
not evidence that the data lack predictive lab information.

### Keep patient and alarm metrics separate

Each threshold is chosen on calibration data for target **alarm event recall 60%**,
then frozen. The achieved test recall need not be 60%. Early-warning matching here
accepts one alarm 1–12 hours before onset per event, with six-hour alarm suppression.
Patient precision counts flagged released records that have an eligible sepsis event
anywhere in the evaluated stay, regardless of warning timing. Patient recall counts
event records flagged anywhere. These are retrospective record-level metrics;
they do not establish a fixed-landmark admission classifier's performance.

| Initialization / inputs | Patient precision | Patient recall | Timely alarm precision | Timely event recall |
| --- | ---: | ---: | ---: | ---: |
| Scratch / 10 | 34.3% | 52.7% | 5.90% | 51.8% |
| Scratch / 13 | 30.2% | 52.4% | 5.82% | 52.2% |
| **Scratch / 34** | **40.6%** | **52.0%** | **6.15%** | **51.1%** |
| Source MAE / 10 | 36.7% | 55.3% | 6.49% | 52.7% |
| Source MAE / 34 | 36.1% | 54.9% | 6.37% | 51.3% |
| Source + target MAE / 34 | 31.8% | 55.8% | 6.03% | 49.6% |

All entries are arithmetic means over three seeds, not a pooled or ensemble result.
Scratch-34 patient precision varies from **26.2% to 51.0%** across seeds; the 40.6%
mean is not yet stable evidence of a robust gain. Scratch hourly AUROC rises from
0.752 (10) to 0.770 (34), whereas AP changes only from 0.090 to 0.093.
Paired alarm comparisons do not show a consistent precision/recall gain across seeds.
Pretraining and target MAE also do not consistently improve the richer-input arm.

The earlier **35.2% patient precision / 50.7% timely recall** example is
source-34 seed 42. Its matching patient recall is **54.0%**; its matching alarm
precision is **6.30%**. Do not present the mixed-unit pair as ordinary precision/recall.
Here 81/230 flagged records develop sepsis; 76/150 events receive timely alarms.

### Do labs help when available?

The predeclared strongest availability gate requires a BP observation less than
two hours old, all creatinine/WBC/platelets less than 24 hours old, and lactate
or pH less than 24 hours old, evaluated using preceding observations only.
It covers **80/150 events (53.3%)** and **22.0% of eligible decisions** in this view.
This is a decision-time gate, not a cohort selected by ever having labs later.

| Inputs / initialization, same gate | Alarm precision | Recall among gate-covered events | Recall among all events |
| --- | ---: | ---: | ---: |
| Scratch / 10 | 10.77% | 55.0% | 29.3% |
| Scratch / 13 | 10.23% | 55.0% | 29.3% |
| Scratch / 34 | 10.18% | 55.0% | 29.3% |
| Source MAE / 10 | 10.78% | 54.6% | 29.1% |
| Source MAE / 34 | 10.50% | 55.4% | 29.6% |

Availability identifies a more favorable setting, but there is no consistent
incremental benefit of the additional channels within that setting. Availability
may reflect clinician suspicion, illness severity, and practice patterns. These
results cannot establish that ordering more labs causes better prediction.
This gate does not require every added lab and does not answer the value of each
individual additional assay. A future study should predeclare assay-specific
availability gates and remove/add those assay inputs on identical decisions.

### Main limitation and defensible research claim

The principal deployment limitation is **temporal specificity and repeated alarm
burden**, along with false warnings in nonseptic stays. It is not merely that a
warning lasts too long. In source-34 seed 42, 1,207 alarm episodes comprise 76 timely
matches, **648 more than 12 hours early**, 60 duplicates, and 423 alarms in nonseptic
records. Thus patient discrimination can be useful while individual alarms remain
poorly actionable. Too-early warnings are not proof of clinically useful long-range
prediction: that endpoint would need a separately specified horizon and evaluation.
Allowing warnings after onset changes the task toward recognition; it must not be
reported as early prediction. Six-hour suppression and one-match-per-event rules
explain part of the patient/alarm discrepancy, but nonseptic false alarms persist.

These results could support a methodological study of measurement availability,
transfer, and evaluation definitions, but performance alone establishes neither
publishability nor novelty. A stronger manuscript needs a clinically justified
decision time/horizon, strong matched baselines, uncertainty, and a fresh independent
confirmation set. Current limits include only 150 test events, seed instability,
repeated test inspection, missing receipt timestamps, uncertain operational onset,
selection by lab ordering, and unequal scratch/source encoder learning rates.
The source MAE uses only ten shared channels: it does not pretrain the additional
target labs. Target adaptation adds compute and lacks a scratch-plus-target-MAE
control, so source and target self-supervision effects are not fully isolated.
No claim of clinical readiness, superiority to published systems, or a causal
benefit of extra monitoring is supported by these experiments.

### Reproduction and provenance

Committed configs are `configs/sepsis_missions_{measurements,source,adapt}_20261010.json`;
`SEPSIS_MISSIONS.md` defines the explicit stages. Original code/config hashes,
prepared-data hashes, frozen model hashes and policies are retained in
`implementation_lock.json`, `data_sha256.json`, `fit_lock.json`, `policy_lock.json`,
and `policies.json`. Per-arm JSON files retain training histories and selected epochs.
Private records, predictions and weights remain in ignored `local/` directories.
Source preparation used 58,844 MIMIC fitting episodes and 13,645 SICdb fitting
episodes, with 6,565/1,543 validation episodes; previous patient holdouts were
preserved. Source histories are outcome-independent and capped at 336 hours.
MAE pretraining has a 20,000-step budget per seed; target adaptation adds 1,000 steps.
Fine-tuning uses batch 32, up to 20 epochs, patience four, tune AP selection;
source encoder LR 1e-4 versus scratch 3e-4. Bootstrap uses 2,000 record resamples
at frozen policies; seed replicates reuse the same test records.

Recorded reporting environment: Python 3.13.15, torch 2.6.0+cu124, NumPy 2.5.3,
pandas 3.0.5, scikit-learn 1.9.1, DuckDB 1.5.5; one RTX 2080 Ti.
These are the locally installed versions recorded after completion, not a
guarantee of bit-identical CUDA reproducibility on other systems.

From `F:\Claude\Sepsentinel\Code`, aggregate reporting and synthetic CPU verification:

```powershell
& '..\.conda\python.exe' scripts/report_measurement_findings.py
& '..\.conda\python.exe' scripts/test_sepsis_mission_metrics.py
```

For training reproduction, first obtain the permitted raw datasets and existing
local provenance inputs: `results/prospective_qc/events.duckdb`,
`results/mimic31_pooled17.pkl`, SICdb raw tables/previous holdout inputs required
by `prepare_transfer_sources.py`, and the historical Kaggle mirror required by
`prepare_transfer_cinc.py`. Those private inputs are not supplied by this commit;
the scripts currently contain local paths. Then run the following prerequisites:

```powershell
& '..\.conda\python.exe' scripts/download_cinc_transfer.py
& '..\.conda\python.exe' scripts/prepare_transfer_sources.py
& '..\.conda\python.exe' scripts/prepare_transfer_cinc.py
& '..\.conda\python.exe' scripts/pretrain_transfer_mae.py --seeds 42,123,456 --steps 20000
```

Use a copy of the final config with a **new run_name** for a fresh training run;
the original directory contains frozen results and is not a clean retraining target.
Replace `NEW_CONFIG.json` below with that copied configuration. Stages run sequentially:

```powershell
& '..\.conda\python.exe' scripts/sepsis_missions.py --config NEW_CONFIG.json prepare
& '..\.conda\python.exe' scripts/sepsis_missions.py --config NEW_CONFIG.json train
& '..\.conda\python.exe' scripts/sepsis_missions.py --config NEW_CONFIG.json calibrate
& '..\.conda\python.exe' scripts/sepsis_missions.py --config NEW_CONFIG.json evaluate
& '..\.conda\python.exe' scripts/sepsis_missions.py --config NEW_CONFIG.json summarize
```

The original queue (`scripts/run_sepsis_priority_queue.py`) ran scratch first,
source transfer second and target adaptation third, copying identical earlier
fits into the later phase. It is resumable against the original local artifacts;
it is not a command for an independent confirmation study. Reproduction commands
are documentation only: no new training was launched while recording these findings.
