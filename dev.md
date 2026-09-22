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
