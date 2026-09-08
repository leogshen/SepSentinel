# Techniques review — what the strong ICU sepsis models actually do

Compiled 2026-09-08 from primary sources (released config files, benchmark
code, and paper appendices — not abstracts). Companion to `Literature.md`,
which catalogues *results*; this file is about *method*.

Two agents were tasked; one hit the session rate limit mid-run, so topics
A/B/C below are complete and the "clinical derived features" topic is
partial. Tags: **[R]** read at source, **[S]** snippet only, **[NF]** not found.

---

## The bar, stated precisely

**YAIB (van de Water et al., ICLR 2024), MIMIC-IV hourly sepsis** — the
closest published setup to ours, ~1% positive hourly bins, AUROC×100 /
AUPRC×100 **[R]**:

| Model | MIMIC-IV |
|---|---|
| LR | 77.1 / 4.6 |
| LGBM | 77.5 / 5.9 |
| **GRU** | **83.6 / 9.1** |
| LSTM | 82.0 / 8.0 |
| TCN | 82.7 / 8.8 |
| Transformer | 80.0 / 6.6 |

**Our best sequence model: 75.5 / 8.2 at 2.6% prevalence.** AUPRC is within
a point of YAIB's GRU, but at 2.6x their prevalence — so on AUPRC *lift over
baseline* they are ahead (9.1x vs our 3.1x). AUROC we are 8 points behind.

**Critical caveat before treating 83.6 as a target:** YAIB reports that the
MIMIC-IV sepsis *label definition alone* moves GRU AUPRC from **6.1**
(Seymour-adapted) to **9.3** (Calvert Sepsis-2) to **17.7** (Moor Sepsis-3).
Cohort exclusions move HiRID mortality AUPRC by 6-20 points. **Label and
cohort choices dominate every architecture, SSL and ensembling delta in this
review by an order of magnitude.** Comparing our number to theirs across
different label definitions is close to meaningless.

---

## Ranked actions for this project

| # | Action | Expected | Effort | Evidence |
|---|---|---|---|---|
| 1 | **Drop `pos_weight`** (we use ~37) | up to **+4.2 AUPRC** | one line | HiRID Table 5: balanced weighting helps multiclass but **hurts every binary task**; released configs set `LOSS_WEIGHT = None` **[R]** |
| 2 | **Temporal Label Smoothing** | **+1.5 AUPRC, +9.7 event recall** | label transform only | Yeche et al. ICML 2023, p=0.002 **[R]** |
| 3 | **Conformal abstention layer** | **-78 to -86% false alarms/patient-hour**, +13-24 specificity points | moderate, no retraining | COMPOSER, npj Digit Med 2021 **[R]** |
| 4 | **Early-stop on val AUPRC** (we use AUROC) | unquantified | one line | Loss/AUROC and AUPRC diverge badly at ~2% prevalence |
| 5 | **Average 5-10 seeds** | ~ one architecture change | cheap | HiRID 10-seed std is 0.3-0.8 AUPRC; our "architecture differences" are inside that |
| 6 | **Grad clip 1.0 + LR schedule** | stability | one line each | **Absent from both HiRID and YAIB codebases [R]** |
| 7 | **Pre-norm Transformer or add warmup** | up to several AUPRC | small | Ours is PyTorch-default **post-norm with no warmup, no clipping, no schedule** — the exact configuration Xiong et al. ICML 2020 show is unstable |
| 8 | **Optimise the utility surrogate directly** | won CinC 2019 | moderate | Morrill regressed on per-timestep U1-U0 rather than the binary label |
| 9 | **Per-site isotonic recalibration** | **0 AUROC cost**, large net-benefit gain | cheap | Monotone transforms are rank-preserving; Huang JAMIA 2020 Table 2: AUROC 0.870 -> 0.870, ECE 0.109 -> 0.011 **[R]** |
| 10 | **Do NOT retry MAE pretraining** | saves weeks | — | See Topic B |

---

## Topic A — training recipe

**What HiRID and YAIB actually ship [R], from the released `.gin` files:**
Adam (never AdamW), LR 1e-4 to 3e-4, weight decay 1e-6 universally and
unsearched, batch 64 for RNN/TCN but **8-16 for Transformers** (a
batch-size confound in every published RNN-vs-Transformer comparison),
patience 10, min_delta 1e-4. **No LR scheduler. No gradient clipping. No
warmup.** Early stopping on validation *loss*; hyperparameter *selection* on
validation AUPRC.

**Dropout trap:** HiRID's released GRU/LSTM pass no dropout at all.
`nn.GRU(dropout=p)` applies dropout only *between stacked layers*, so on a
1-layer GRU it is silently a no-op. Apply it manually to inputs/outputs.

**Class weighting — the strongest single finding.** Yeche et al. state that
tuned weighted-CE and focal loss **reduce exactly to plain CE**, and report
identical numbers to three decimals for all three. HiRID reports balanced
weighting costing up to **-4.2 AUPRC** on binary tasks. We currently train
with `pos_weight ~= 37`.

**Temporal Label Smoothing** (arXiv:2208.13764): replace the hard label with
a smooth function of time-to-event. Exponential variant, `gamma`
grid-searched, window (0, 2h):

```
q(t) = 0                          if t <= t_e - h_max
     = exp(-gamma(t_e - t - d)) + A   in between
     = 1                          if t >= t_e - h_min
```

HiRID circulatory failure: AUPRC 39.1 -> **40.6**, event-recall 82.8 ->
**92.5**. Nothing else changes. **This is the best value-per-line item in
the review, and it fits our pre-onset target naturally.**

**Sequence length:** HiRID's history ablation shows sequence models extract
essentially nothing beyond ~12h of context; their Transformer *improves*
when history is shortened. Chunk to 24-72h rather than feeding whole stays.

**Augmentation:** Window warping and window slicing are the only reliably
positive classical transforms; **permutation and time warping are
consistently harmful** (Iwana & Uchida, 12 methods x 128 datasets).
Mix-based methods (manifold mixup, CutMix) win 16/18 trials on physiological
data, +1-2pp. **[NF]** No evaluation exists on hourly gridded tabular ICU
data with an early-warning label — all of it is dense waveform/sensor data.
None of HiRID, YAIB or TLS uses any input augmentation.

## Topic B — self-supervised pretraining: don't

**Our MAE null result is the modal published outcome, not a bug.** The decay
curve, from NCL on MIMIC-III decompensation (hourly, per-timestep, ~50k
stays — nearly our setup), SSL minus supervised, AUPRC points **[R]**:

| Labeled fraction | 1% | 10% | 50% | **100%** |
|---|---|---|---|---|
| Delta AUPRC | +6.7 | +4.0 | +1.8 | **+0.9** |

Full-label ablations elsewhere: GenHPF **-0.001 AUROC** single-source;
PrimeNet's masked-reconstruction-only arm **-0.012 AUROC** (worse than no
pretraining); NCL's Seq2Seq-AE-forecast **-5.5 AUPRC** on decompensation.
Labrador (100M MIMIC lab results) "does not consistently outperform
XGBoost". Newell & Deng, CVPR 2020: *"utility approaches zero as labeled
data grows... for all pretraining methods."*

**Reconstruction pretexts rank last in every paper that compares them to a
contrastive objective.** The likely mechanism: a pretext reconstructing long
masked histories optimises for structure the downstream task does not use.

**The one surviving setting is cross-dataset.** GenHPF: single-source SSL
-0.001 AUROC, multi-source (MIMIC-III + eICU + MIMIC-IV pooled) **+0.9**.
If SSL is retried, pretrain on eICU/HiRID/AUMCdb and fine-tune on MIMIC-IV,
use NCL-style patient-temporal contrastive (alpha 0.3-0.4, w 12-16h, history
cutout/crop never touching the last timestep), and **evaluate with an MLP
head, never linear probing** — the ranking reverses between the two.

**[NF]** No matched head-to-head exists in which any foundation model beats
a tuned GRU or LGBM on ICU sepsis or per-timestep deterioration.

## Topic C — abstention, ensembling, calibration

**COMPOSER (npj Digit Med 2021) is the most actionable system in the
review.** Its conformal layer is *not* split conformal on the risk score. It
is a two-sided typicality test in latent space:

1. Build two trust sets of encoder representations, septic and non-septic,
   chosen for low loss and low missingness.
2. Nonconformity = summed negative cosine similarity to set members.
3. Rank-based p-value per set.
4. **Abstain iff BOTH p-values < 0.05** — the point is atypical of septic
   *and* non-septic patients, so say nothing.

Effect: AUROC moved +0.015 to +0.028 — nearly nothing. **Specificity moved
+13 to +24 points and false alarms per patient-hour fell 78-86%** (0.296 ->
0.043 ICU). Abstention lands ~2x more often on non-septic windows.
Patient-level safety: median 1.1% of septic patients had *all* windows
rejected. Prospectively (Boussina 2024) this produced a **17% relative
reduction in sepsis mortality** at ~1.65 alerts per nurse per month.

**This is a decision-layer intervention, orthogonal to discrimination — and
alert burden is the axis where we are already competitive.**

**Seed noise:** HiRID's 10-seed std is 0.3-0.8 AUPRC. **Our entire
architecture sweep spans 0.010 AUROC, which is inside that.** Averaging
5-10 seeds is worth about as much as an architecture change, at lower risk.

**Calibration costs exactly zero AUROC.** Platt, temperature and isotonic
are monotone, hence rank-preserving; Huang JAMIA 2020 Table 2 shows AUROC
0.870 -> 0.870 under both, with ECE falling 10x. Justify it by net benefit
(Van Calster & Vickers: miscalibration *always* reduced NB) and by threshold
portability — Zabihi's hospital C held AUROC at 0.793 while utility
collapsed to -0.146, a pure calibration failure under site shift. With
>1000 calibration positives use isotonic; below that, Platt. Apply per site.

**PhysioNet/CinC 2019 — every top-5 finisher was a tree ensemble.** No pure
deep sequence model won. The winner's largest single lever: **regress on the
per-timestep utility differential U1 - U0 instead of the binary label**,
making the loss a surrogate for the actual metric, then tune the threshold
by direct utility maximisation. Feature ablation: raw 40 features 0.389 ->
+ hand-crafted clinical (shock index, BUN/CR, partial SOFA, SOFA
deterioration) **0.418** -> + path signatures 0.430.

**Beware evaluation granularity.** Same Epic ESM v2, same data: patient-level
AUROC 0.86 vs prediction-level **0.62**; PPV 14.5% vs 4%. Never compare a
per-timestep number to a patient-level one from the literature.

---

## What could not be determined

- **[NF]** Augmentation evaluated on hourly gridded tabular ICU data.
- **[NF]** Any peer-reviewed sepsis paper isolating the gain from stacking a
  GBDT with a sequence model.
- **[NF]** A published seed-averaging delta on ICU time series (only the
  per-seed std).
- The "clinical derived features" agent died on a rate limit mid-task; the
  Nemati/Moor feature lists were not retrieved. Morrill's list above is the
  verified one.
