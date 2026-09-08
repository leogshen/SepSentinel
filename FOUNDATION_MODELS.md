# Foundation models and SepSentinel — everything we found, and what we could do

Compiled 2026-09-08 from three investigations this project ran: a TimesFM
novelty/feasibility review, a 47-paper literature review (`Literature.md`), a
techniques review of training and SSL practice (`TECHNIQUES.md`), plus our
own hands-on tests. Detail lives in those files; this is the synthesis and
the decision.

**Bottom line: no foundation model is currently worth building on for Model
B, and the evidence for that is unusually direct. The two exceptions worth a
day each are TabPFN-style TABULAR models and a deliberately-designed
small-data experiment. MIRA is the one time-series model architecturally
suited to us and it is blocked on unreleased weights.**

---

## 1. What we tested ourselves

### TimesFM — installed, run, and rejected

`pip install timesfm` -> 3.0.1. The `google/timesfm-2.5-200m-pytorch`
checkpoint downloads and loads in **42 s** on our 2080 Ti; forecasting runs
and returns correct shapes. It works. It is still the wrong tool, for four
reasons we verified rather than inferred:

| Finding | How we know |
|---|---|
| **No classification head, no exposed embeddings.** `classification/embedding entry points on the model object: NONE` | `scripts/timesfm_smoke.py`, run on this machine |
| **Context geometry is wrong.** It patches input at length 32; our median record is **35 hours**, so a 200-330M-parameter model sees **one or two tokens** | Arithmetic from published patch lengths |
| **Its masking is padding-masking, not observation-masking.** Our 38-channel mask+delta scheme has nowhere to go in its input format | ICML 2024 paper; feeding it forward-filled staircases means feeding it our imputation artifacts |
| **No clinical data in pretraining.** Google Trends, Wikipedia pageviews, synthetic series, public forecasting archives | Paper + README |

Version notes: 1.0/2.0 archived; **2.5** = 200M params, 16k context,
**Apache-2.0**; **3.0** (31 Aug 2026) = 330M, first version with native
multivariate input and covariates, but licensed
`timesfm-non-commercial-license-v1.0` — fine for a paper, a constraint to
state. LoRA/PEFT fine-tuning example ships in the README.

**Verdict: do not pursue.** Correcting an earlier claim of ours: we
initially relayed that Turing GPUs lack bf16 — `torch.cuda.is_bf16_supported()`
returns True here, so that was wrong and is not a reason against it. The
reasons above are.

### MIRA — architecturally right, blocked on weights

`microsoft/MIRA`, MIT licence. Pretrained on **454B time points including
ICU physiological signals**, with **Continuous-Time RoPE** for irregular
timestamps, frequency-specialised mixture-of-experts, and Neural-ODE latent
dynamics. Its data format has a **`mask` field**. Its own example uses
**context 12 / horizon 6** — short context, which fits our 35-hour records
far better than TimesFM's 32-length patches.

On every axis that disqualified TimesFM, MIRA is the answer.

**But no pretrained weights are released.** Inference examples load from
checkpoint paths with no download links; nothing on HuggingFace. Without
weights there is no zero-shot and no fine-tuning, and pretraining on 454B
time points is not happening on our hardware.

**Verdict: blocked, not rejected.** Cost of unblocking: one GitHub issue
asking whether weights will be released. Worth sending. It is also
forecasting-only, so a classification head remains our work even if weights
appear.

---

## 2. The landscape, and why each one does not apply

### Generic time-series foundation models

| Model | Status for us |
|---|---|
| **Chronos / Chronos-Bolt** (Amazon) | Forecasting. One benchmark reports fine-tuned Chronos-2 at +107% weighted-F1 over frozen Chronos-2, so *fine-tuning matters far more than frozen probing* — a useful general lesson |
| **MOMENT** | **The only one with a native classification path**, and the one Gen-P-Tuning validated on MIMIC. But its own classification evaluation is UCR univariate via an SVM on frozen representations. Best candidate if we probe anything |
| **Moirai** (Salesforce), **Lag-Llama**, **TinyTimeMixers** (IBM), **Time-MoE**, **UniTS**, **Timer** | Forecasting-only, no clinical evaluation found |

**[NF]** MOMENT, Chronos and Moirai have **no published head-to-head against
a tuned GRU or LGBM on ICU sepsis or per-timestep deterioration.** That
absence is itself informative.

### Clinical / EHR foundation models — a different lineage

These are event-sequence models over diagnosis codes, not hourly physiology,
so they do not slot into our pipeline. They matter because they compete for
the same headline and a reviewer will ask.

- **ETHOS** — GPT-style over tokenised MIMIC-IV timelines, does **zero-shot
  sepsis risk**. ICU mortality AUC 0.927 vs the best prior XGBoost 0.918 —
  **+0.009, cross-study, unmatched cohorts, stay-level not per-timestep.** It
  *loses* on ICU readmission.
- **MOTOR**, **CLMBR**, **Med-BERT**, **Foresight** — same family.
- **EHRSHOT**: CLMBR-T-base, **141M params pretrained on 2.57M patients**,
  vs a count-based GBM across 15 tasks — *"as k increases the advantage of
  the pretrained model tends to shrink"*, and the **GBM exceeds it** on some
  tasks at k > 64.
- **Labrador** (best paper, ML4H 2024), continuous-token MLM on **100M MIMIC
  lab results**: *"neither Labrador nor BERT consistently outperform
  XGBoost."* This is the closest published analogue to "we did big masked
  pretraining on EHR and got nothing."

### ICU-specific pretrained models

- **BAT** (NeurIPS 2025 TS4H) — ICU-pretrained Bi-Axial Transformer.
  **Its gains concentrate on datasets under 5,000.** We have 63,672
  episodes — the regime where pretraining helps least.
- **PULSE-ICU** — "Pretrained Unified Long-Sequence Encoder for Multi-task
  Prediction in ICUs". Competes directly for our claim; its task list was
  not verified.

### Tabular foundation models — the one family with a positive result

**TabPFN v2** is the first foundation model to beat tuned GBDTs on
small/medium tabular benchmarks, and there is EHR-specific follow-up work
(retrieval-aligned tabular FMs for clinical risk prediction).

**This matters to us specifically because our own results say the task
behaves tabular, not temporal**: flat XGBoost matches or beats the causal
Transformer on every deployment metric, and the missingness experiment found
architecture worth <=0.03 capture. If any foundation model helps here, the
prior should be on a tabular one, not a time-series one.

---

## 3. The evidence against pretraining at our scale

From `TECHNIQUES.md`, this is the strongest and most decision-relevant block.

**The decay curve** (NCL on MIMIC-III decompensation — hourly, per-timestep,
~50k stays, nearly our setup), SSL minus supervised, AUPRC points:

| Labeled fraction | 1% | 10% | 50% | **100%** |
|---|---|---|---|---|
| Delta AUPRC | +6.7 | +4.0 | +1.8 | **+0.9** |

**Full-label ablations, every one:**

| Study | Setting | Delta at full labels |
|---|---|---|
| GenHPF | MIMIC-III, single-source SSL | **-0.001 AUROC** |
| GenHPF | multi-source (MIMIC-III + eICU + MIMIC-IV) | **+0.9 AUROC** |
| PrimeNet | masked reconstruction only — the MAE analogue | **-0.012 AUROC** |
| NCL | Seq2Seq-AE-forecast, decompensation | **-5.5 AUPRC** |
| NCL | best contrastive (NCL n_w) | +0.9 AUPRC |
| Newell & Deng, CVPR 2020 | general | *"utility approaches zero as labeled data grows... for all pretraining methods"* |

**Our own MAE null result (experiment 6) is the modal published outcome, not
a bug in our code.** Reconstruction pretexts rank last in every paper that
compares them against a contrastive objective. The likely mechanism: HiRID's
history ablation shows sequence models extract almost nothing beyond ~12h of
context, so a pretext that reconstructs long masked histories optimises for
structure the downstream task does not use.

**The one setting that survives is cross-dataset.** GenHPF: pooling
MIMIC-III + eICU + MIMIC-IV gives +0.9 AUROC where single-source gives zero.
Their explanation: *"when the quantity of pretraining data exceeds that of
the fine-tuning data to a larger extent, pretraining significantly affects
the downstream tasks."*

**And two clinical-domain verdicts:** Gu et al. (ML4H 2024) on TSFMs for
vital-sign forecasting — *"the 'ChatGPT moment' for time series foundation
models, in the typical clinical domain, is yet to come."* And
arXiv:2601.16516 on irregular ICU classification — **encoder design matters
far more than the pretrained backbone** (+12.8% AUPRC from irregular-aware
encoders vs +2.9% from alignment), with LLM-based methods needing **10x
longer training** for comparable performance.

---

## 4. What we could actually do — ranked

### A. TabPFN-style tabular foundation model **(recommended, ~1 day)**

The only FM family with published evidence of beating tuned GBDTs on
small/medium data, applied to the one framing our own results support.
Our design matrix is already the right shape: 2.2M rows x 38 channels.

Caveat to design around: TabPFN v2 has a sample-size ceiling (order 10k
rows), so this needs subsampling or a per-patient aggregation, which changes
the task. Honest framing: test it on a **patient-level** or
**window-level** version of our task rather than per-hour.

### B. The small-data experiment **(recommended, ~1 day, and genuinely novel)**

Every FM paper claims gains concentrate in the low-label regime (BAT:
n<5,000; NCL: +6.7 AUPRC at 1% labels decaying to +0.9 at 100%). **Nobody
has tested that claim on sepsis onset prediction with a proper deployment
metric.** We can, because we have the full cohort and can subsample it.

Design: train at n = 500 / 2k / 10k / 63k patients, with and without a
frozen-FM feature extractor, and report recall-at-burden as well as AUROC.
Two outcomes, both publishable: either FM gains appear at the small end and
vanish by 10k — replicating the literature on a new task with a deployment
metric nobody else uses — or they never appear, which is a sharper negative
result than any single-point comparison.

This also directly serves the project's thesis: a wearable deployed at a
small hospital *is* the low-data regime.

### C. Unblock MIRA **(one GitHub issue, zero effort)**

Ask whether pretrained weights will be released. It is the only TS foundation
model whose architecture matches our data. Costs nothing to ask; if weights
appear, it becomes option A's main competitor.

### D. Frozen-embedding probe on MOMENT **(~1 day, low expected value)**

The minimal credible FM experiment: frozen embeddings + linear/MLP probe. If
it does not clear XGBoost's 0.736, LoRA fine-tuning probably will not rescue
it. Two design requirements from the literature: **evaluate with an MLP
head, never linear probing** (the ranking reverses between them in NCL, and
Newell & Deng confirm linear eval does not correlate with fine-tuning), and
make sure the supervised baseline is genuinely tuned.

### E. Cross-dataset SSL, not a foundation model **(~1 week, best evidence)**

Not an off-the-shelf FM but the setting with the only surviving positive
result. Pretrain on eICU (~200k stays) + HiRID + AmsterdamUMCdb, fine-tune
on MIMIC-IV, using **NCL-style patient-temporal contrastive** (alpha 0.3-0.4,
neighbourhood w 12-16h, augmentations = channel dropout + Gaussian noise +
history cutout/crop that never touch the last timestep). Expected: ~+0.9
AUPRC. Blocked on acquiring eICU/HiRID access.

### F. Do nothing **(the honest default)**

Everything in section 3 says the expected gain at 63k labeled episodes is
approximately zero, and every cheap non-FM action in `TECHNIQUES.md` — drop
pos_weight, temporal label smoothing, conformal abstention, seed averaging —
has better-evidenced expected value than any FM option here.

---

## 5. What to say in a paper

Do **not** frame anything as "we fine-tuned a foundation model for sepsis."
That cell is empty in the literature partly because people looked and moved
on, and TimesFM specifically is the worst-fitting member of the family.

The defensible framings, in order:

1. **"Do time-series foundation models help for hourly sepsis early
   warning? A controlled comparison."** Fixed cohort, fixed representation,
   fixed label, one evaluation protocol, arms from logistic regression up to
   frozen and fine-tuned FMs — **evaluated on recall at fixed alert burden,
   not AUROC.** A rigorous null result here is publishable and nobody has
   produced this table for sepsis.
2. **The low-data curve** (option B) — tests the field's central claim about
   when FMs help, on a task where it has never been tested.
3. **"How do you feed 93%-missing irregular clinical series to a model
   pretrained on dense regular data?"** — the genuinely unsolved methods
   question, and the part TimesFM's API does not answer.

**Most likely reviewer objection to any of these:** *"Your Transformer beats
XGBoost by 0.014 AUROC, so any foundation-model gain you report is within
noise, and you have not shown the FM helps where it should — small data,
cross-hospital transfer."* Pre-empt with grouped bootstrap CIs, the
learning-curve ablation of option B, and an external-validation arm.
