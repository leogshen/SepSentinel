# Literature review — machine learning for sepsis prediction / early warning on MIMIC-IV

**Prepared for advisor review.** Compiled 2026-09-07.

**Scope.** ML studies that predict *sepsis onset* (early warning) on MIMIC-IV, with MIMIC-III
and PhysioNet/CinC 2019 included where directly relevant. Deliberately excluded: the very
large literature that predicts *mortality or complications among patients who are already
septic* (see note in §1.4) — that is a different task and is often conflated with ours in
citation lists.

**Evidence grading used throughout.** Every factual claim is tagged:

| Tag | Meaning |
|---|---|
| **[V]** | Verified by me from the primary source in this session (fetched full text, abstract, or the actual source file) |
| **[V-sub]** | Verified from a primary source by a delegated search agent, which reported reading it |
| **[S]** | Secondary — from a search snippet, a citing paper, or a listing page; **not** read at source |
| **[?]** | Could not verify; recorded as a lead only |

Where I could not find something, I say so explicitly. **Absence of a finding in this review
is not evidence of novelty** — §6 separates "I searched and found nothing" from "this is
unclaimed".

---

## 0. Our work, for positioning

Stated here so the comparisons below are legible.

| | SepSentinel (this project) |
|---|---|
| Task | Per-hour sepsis risk, ICU vitals + labs, hourly grid |
| Dataset | MIMIC-IV **v3.1** |
| Labels | Sepsis-3 with **PhysioNet/CinC 2019 timing rules** — `t_suspicion` from antibiotic–culture pairing; `t_SOFA` = ≥2-point SOFA rise vs prior-24h minimum; `t_sepsis` = min of the two inside a window. DuckDB port of MIT-LCP mimic-code concepts |
| Cohort | 93,224 qualifying ICU stays → 63,672 episodes after exclusions; 48,150 unique patients |
| Prevalence | 11.5% patient-level; ~2.6% timestep-level |
| Results | Causal Transformer AUROC **0.751**; flat XGBoost **0.736**; logistic regression 0.682–0.705; NEWS2 **0.626** |
| Deployment metric | Median lead **~20 h** at ≤1.0 false alerts per nonseptic patient-day |
| Distinctive findings | (a) retaining post-onset hours inflates AUROC ~0.09 while worsening clinical utility; (b) AUROC anti-correlated with lead time across five independent manipulations; (c) measurement-frequency-vs-effect-size synthetic control; (d) equal-alert-burden model comparison |

**One sanity check to raise with the advisor before submission.** MIMIC-IV v3.1 contains
**94,458 ICU stays in total** ([V] — PhysioNet v3.1 landing page, via delegated verification
[V-sub]). Our "93,224 qualifying ICU stays (age ≥18, LOS ≥6 h)" is 98.7% of that. MIMIC-IV
has been adult-only since v2.0, so the age filter removes almost nothing, but a LOS ≥6 h
filter should plausibly remove more than 1.3% of stays. Worth re-deriving that number — a
reviewer will check it against the published total.

---

## 1. Summary table — sepsis *onset* prediction studies

Sorted roughly by relevance to our setup. **"Post-onset?"** records whether the paper states
what it does with hours after `t_sepsis` — the single most important undocumented choice in
this literature (see §2).

### 1.1 MIMIC-IV, hourly / per-timestep onset prediction — the direct comparators

| # | Study | Dataset & version | Cohort | Sepsis definition | Target / horizon | Post-onset? | Model | AUROC | AUPRC | Lead time / alert burden | Validation | Code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **van de Water et al., "Yet Another ICU Benchmark" (YAIB), ICLR 2024.** arXiv:2306.05109 · https://arxiv.org/abs/2306.05109 | MIMIC-III, **MIMIC-IV (v2.x)**, eICU, HiRID, AUMCdb | MIMIC-IV ~73k stays; **~1% positive hourly bins** | Sepsis-3 via **ricu `sep3_alt`**; excludes onset <6 h after ICU admission | **Sepsis within next 6 h**, hourly | **Yes — explicit.** Features truncated 6 h before onset (`stop_obs_at`, offset 6 h) | LR, LGBM, GRU, LSTM, TCN, Transformer | MIMIC-IV: LR **77.1±0.4**, LGBM **77.5±0.3**, **GRU 83.6±0.3**. eICU 71.8/69.1/77.4; HiRID 76.5/76.1/80.6; AUMCdb 74.7/74.0/79.7 | MIMIC-IV: LR 4.6, LGBM 5.9, **GRU 9.1** | Not reported | Multi-centre, per-dataset internal | **Yes** — https://github.com/rvandewater/YAIB + https://github.com/rvandewater/YAIB-cohorts |
| 2 | **Moor et al., eClinicalMedicine 2023;62:102124.** doi:10.1016/j.eclinm.2023.102124 · https://pubmed.ncbi.nlm.nih.gov/37588623/ | MIMIC-III, MIMIC-IV, eICU, HiRID, AUMCdb | **136,478 ICU admissions** (published abstract [V]); preprint states 156,309 stays, 26,734 (17.1%) septic [V-sub] | Sepsis-3 **harmonised via `ricu`**: hourly SOFA + SOI windows from cultures & antibiotics; acute SOFA rise ≥2 in SOI window | Hourly sepsis risk; earliness measured | Not stated in abstract; pipeline is prodrome-oriented | Deep (attention/RNN ensemble) | **0.846 internal / 0.761 external** | 0.177–0.252 across sites [S] | **1.4 false alerts per true alert**; 80% of septic patients detected **3.7 h (95% CI 3.0–4.3)** before onset | **True external** across 5 databases, 3 countries | **Yes** — https://github.com/BorgwardtLab/multicenter-sepsis (Apache-2.0) |
| 3 | **Do, Rockenschaub, Boie, et al., J Med Internet Res 2026;28:e72083.** doi:10.2196/72083 · https://www.jmir.org/2026/1/e72083 | **MIMIC-IV v2.0** (train) → BerlinICU (external) | MIMIC-IV **67,056** admissions, 5.6% septic; BerlinICU 40,132, 10.3% | Sepsis-3; SI = antibiotics ≥1/24 h for ≥3 days; SI window −48 h/+24 h | Hourly; horizons swept **1–100 h** | **Yes — this is the paper's subject.** Compares fixed-horizon, peak-score, continuous | TCN | MIMIC-IV internal continuous 6 h **0.84**; BerlinICU continuous **0.67**, fixed-horizon **0.61** | Not extracted | Not reported | Internal + **external** | **Yes** (stated; repo cited as ref 44) |
| 4 | **Stylianides et al., IEEE J Biomed Health Inform 2026.** doi:10.1109/JBHI.2026.3725473 · https://pubmed.ncbi.nlm.nih.gov/42616625/ | MIMIC-IV (version **not stated**); eICU | **45,285 patients** | "Sepsis" — **exact definition not stated in abstract** | **12 h in advance** | **Not stated** | Ensemble: gradient boosting + hybrid LSTM, 21 features | **0.87** | **0.88** | Sensitivity 0.79 at specificity 0.81 | Internal + eICU "comparable" | Not stated |
| 5 | **Dalal, Ardabili, Bonavia. medRxiv 2024.11.21.24317716** · https://pmc.ncbi.nlm.nih.gov/articles/PMC11601686/ | **MIMIC-IV v2.2** (accessed 2024-06-10) + eICU-CRD | MIMIC-IV **104,767 patients** (4,632 septic, 4.4%); eICU 82,486 (4,687 septic) | MIMIC-IV: "Sepsis-3 diagnostic criteria" (unspecified implementation). **eICU: patients with the string "sepsis" in the diagnosis field** | 6 h, 12 h, 24 h before onset | **Not addressed.** Uses labs "within 6 h prior to the time point of interest" | LSTM encoder–decoder + attention + conformal prediction | **0.99 / 0.98 / 0.96** (6/12/24 h), *identical* on eICU | Not reported | 57% reduction in false positives with conformal (6 h) | 80:20 internal + external eICU | "On reasonable request" |
| 6 | **Dickens A. "Falsification Testing of Sepsis Prediction Models." medRxiv 2026.** doi:10.64898/2026.03.17.26348414 | **MIMIC-IV v3.1** (primary); eICU v2.0, MIMIC-III v1.4, CinC 2019 (replication) | **65,241 adult ICU stays** | Sepsis-3 (clinical) vs administrative coding, compared | Not an early-warning model per se — a pre-registered falsification study | n/a | GBM-family | Biological features predict Sepsis-3 at **0.901**; minimal loss when care-intensity features removed | Not reported | n/a | Pre-registered on OSF before data access; 4 datasets | Not stated |
| 7 | **Huang, Yang, Rahmani. "MIMIC-Sepsis", arXiv:2510.24500 (2025)** · https://arxiv.org/abs/2510.24500 | **MIMIC-IV v3.1** | **35,239 ICU patients** | Sepsis-3 (custom SQL; SI = abx within 24 h before positive culture, or culture within 72 h before abx) | **Not onset prediction** — early mortality, LOS, shock onset | n/a | Transformer + baselines | Not extracted | Not extracted | n/a | Internal | **Yes** — https://github.com/yongh7/MIMIC-sepsis (MIT) |
| 8 | **Zhang M, Zhong M, Cheng Y, Zhang T. JMIR Med Inform 2025;13:e74940.** doi:10.2196/74940 | MIMIC-IV (version not stated) | **224 patients in test cohort** | Not stated in abstract | Real-time, 3-h dynamic temporal features | Not stated | Tree ensembles + TreeSHAP | **0.76** (95% CI 0.74–0.77) | Not reported | Web platform; no burden numbers | Internal | Platform described, code not stated |
| 9 | **Yamamoto, Wu, Sprehe, Abeer, Celi, Tohyama. medRxiv 2026.** doi:10.64898/2026.04.05.26350209 | **MIMIC-IV** (n=30,218) → **eICU-CRD** (n=31,403) | 30,218 septic adults, 16.3% mortality | Sepsis-3 | **Mortality**, not onset — included because of its observation-process finding (§5) | n/a | LR + GBT, 7 specifications | Internal 0.819→0.834 with measurement counts; **external drop −0.047 without vs −0.082 with counts** | Not reported | Calibration slope 1.007 → **0.417** | Internal + external | Not stated |

### 1.2 MIMIC-III and PhysioNet/CinC 2019 — context and method precedents

| # | Study | Dataset | Cohort | Sepsis definition | Target / horizon | Post-onset? | Model | AUROC | AUPRC | Lead / burden | Validation | Code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | **Reyna et al., Crit Care Med 2020;48(2):210–217.** doi:10.1097/CCM.0000000000004145 · https://physionet.org/content/challenge-2019/1.0.0/ | **PhysioNet/CinC 2019** — 3 hospital systems | >60,000 ICU patients; 40,336 public (Sets A+B), 22,761 sequestered | **The canonical timing rule.** `t_suspicion` (abx ≥72 consecutive h + culture, 24 h/72 h pairing); `t_SOFA` = 2-pt SOFA deterioration within 24 h; `t_sepsis` = min, valid only if `t_SOFA ∈ [t_suspicion−24 h, t_suspicion+12 h]`; **labels shifted 6 h earlier** | Hourly binary; utility scored | **Records truncated near onset** (see §2.3) | 104 teams, 853 entries | Ranked by **utility score**, not AUROC. Winner (Morrill signature) **0.360** | — | Utility rewards [t−12 h, t+3 h]; −0.05 early, −2.0 late, −0.05 false alarm | Hidden test set from a 3rd hospital | Evaluation code yes (https://github.com/physionetchallenges/evaluation-2019); **label-generation code never released** |
| 11 | **Cohen et al., Sci Rep 2024;14:1920.** doi:10.1038/s41598-024-51989-6 · https://pmc.ncbi.nlm.nih.gov/articles/PMC10803347/ | MIMIC-III | 867–2,178 septic ICU admissions **depending on definition** | **Three onset interpretations**: H1 = `t_SOFA`; H2 = `t_suspicion`; H3 = min | 6 h (T swept) | Varies with definition | LGBM, LSTM, CoxPHM | H1/H2/H3 — LGBM **0.832/0.869/0.829**; LSTM 0.805/0.856/0.793; Cox 0.799/0.844/0.780 | Not extracted | — | Internal | **Yes** — https://zenodo.org/record/5168789 |
| 12 | **Horn, Moor, Bock, Rieck, Borgwardt. "Set Functions for Time Series" (SeFT), ICML 2020.** arXiv:1909.12064 | PhysioNet/CinC 2019 (P-Sepsis) | Positive-class prevalence **1.8%** | CinC-provided labels | **Hourly, sepsis within next 6–12 h** | Per challenge | SeFT-Attn + baselines | SeFT-Attn **76.8±0.9**; GRU-Simple 78.1; GRU-D 67.4; Phased-LSTM 75.4; Latent-ODE 64.6; **IP-Nets 94.1; Transformer 97.3** | SeFT 4.84; GRU-D 5.33; GRU-Simple 6.10; **IP-Nets 29.4; Transformer 53.4** | Utility reported | Internal | **Yes** — https://github.com/BorgwardtLab/Set_Functions_for_Time_Series |
| 13 | **Shao et al., BMC Med Inform Decis Mak 2026.** doi:10.1186/s12911-026-03610-1 | MIMIC-III | **1,634** ICU patients after excluding Sepsis-3 within first 24 h; **349 (21.4%)** developed sepsis later | Sepsis-3 | Onset **after** the first 24 h | Onset-after-window design (a mild form of prodrome targeting) | 9 models; XGBoost-DART best | **0.881** (0.854–0.908); range across 9 models 0.794–0.881 | Not reported | — | Internal | Not stated |
| 14 | **Khorram & Kouchaki, BMJ Health Care Inform 2026.** doi:10.1136/bmjhci-2025-101762 | PhysioNet/CinC 2019 | 40,336 patients; 2,932 septic (7.3%) | CinC labels | **Up to 12 h in advance** | Per challenge | LSTM + graph attention (GAT) | **0.853±0.005** | Not reported; F1 0.627±0.006 | — | Internal | Not stated |
| 15 | **Yang, Yi, Chen. Sci Rep 2026.** doi:10.1038/s41598-026-49672-z | PhysioNet/CinC 2019 | 40,336 | CinC labels | Early onset | Per challenge | Fine-tuned small LLMs (Gemma-2-9B) via a semantic rule engine | **0.9307** | Not reported | — | Internal | Not stated |
| 16 | **Mahmoud et al., Sci Rep 2026.** doi:10.1038/s41598-026-61652-x | ICU time series (dataset not identified in abstract) | Not stated | Not stated | Early detection | Not stated | 12 models; BiLSTM & TCN best | **0.9566 / 0.9595** | Not reported; F1 0.85 | — | Internal | Not stated |
| 17 | **Yang X et al., JMIR Med Inform 2026.** doi:10.2196/82762 | Public ICU databases | **>400 patients** evaluated | Not stated | Online, vital signs only | Not stated | Multi-scale temporal contrastive learning | **0.8834** | Not reported; sens 0.893, spec 0.73 | — | Internal | Not stated |

### 1.3 Deployed / prospectively evaluated systems — see §4 for detail

| # | System | Study | Setting | Cohort | Definition | AUROC | Lead / burden | Design |
|---|---|---|---|---|---|---|---|---|
| 18 | **Epic Sepsis Model** | Wong et al., JAMA Intern Med 2021;181(8):1065–70 | Michigan Medicine | 27,697 patients / **38,455 hospitalizations** | Composite: CDC surveillance **or** ICD-10 + 2 SIRS + 1 organ dysfunction within 6 h | **0.63** (0.62–0.64) | Alerts on **18%** of hospitalizations; sens 33%, PPV 12%; **NNE 109**; missed 67% of septic patients | Retrospective external |
| 19 | **Epic Sepsis Model** | **Kamran et al., NEJM AI 2024;1(3), AIoa2300032** | U. Michigan 2018–20 | **77,582 hospitalizations**, 3,766 sepsis (4.9%) | CDC surveillance + SEP-1 bundle composite | **0.62 → 0.47** when restricted to before treatment | — | Retrospective; the key leakage result |
| 20 | **Epic "Early Detection of Sepsis v2"** | **Dutta et al., JAMA Netw Open 2026;9(4):e265599** | Mass General Brigham, 9 hospitals | **198,494 encounters** | **Three compared**: Sepsis-3 / SEP-1 / CDC ASE | **0.89 / 0.94 / 0.85** | AUPRC 0.24/0.16/0.11; PPV 11.4/6.8/5.9%; **lead 3.4 / 4.5 / 1.4 h**; at Youden threshold 21.2% of encounters alert, **NNE 8.7** | Retrospective, locally trained |
| 21 | **COMPOSER** | Shashikumar et al., npj Digit Med 2021, doi:10.1038/s41746-021-00504-6 | UCSD, 6 cohorts | **515,720 patients**, >6M prediction windows | Sepsis-3 | **ICU 0.925–0.953; ED 0.938–0.945** | Lead **ICU 12.2 h [3.2–22.8]**, ED 2.1 h [0.8–4.5] before first antibiotic order; ~20% of non-septic flagged indeterminate | Retrospective + temporal |
| 22 | **COMPOSER (deployed)** | Boussina et al., npj Digit Med 2024, doi:10.1038/s41746-023-00986-6 | UCSD EDs, Jan 2021–Apr 2023 | **6,217 septic patients** | — | not reported | **Mortality −1.9% absolute (17% relative, 95% CI 0.3–3.5%)**; SEP-1 compliance **+5.0% absolute**; 72-h SOFA change −4% | **Before–after quasi-experimental, Bayesian causal inference** |
| 23 | **TREWS** | Adams et al., Nat Med 2022, doi:10.1038/s41591-022-01894-0 | 5 hospitals | **590,736 monitored**; 6,877 sepsis identified pre-antibiotic | — | not reported | **Mortality 3.3% absolute / 18.7% relative reduction** when alert confirmed within 3 h | **Prospective multi-site** |
| 24 | **TREWS (adoption)** | Henry et al., Nat Med 2022, doi:10.1038/s41591-022-01895-z | same | 9,805 retrospective sepsis cases | — | — | **82% sensitivity; 89% of alerts evaluated by a provider; 38% confirmed; −1.85 h median time to first antibiotic order** | Prospective deployment analysis |
| 25 | **Sepsis Watch** | Valan et al., npj Digit Med 2025;8:350 | Duke → 4 Summa Health EDs | **205,005 encounters**, 101,584 patients | **Sepsis-2** (2 SIRS + blood culture order + organ damage); 3.37% within 36 h | Duke internal **0.882**, temporal 0.943; **external 0.906–0.960** | **AUPRC only 0.177–0.252**; at 20% precision: **1–27 alerts/day**, lead **3.58–5.07 h** | Retrospective multi-site external |
| 26 | **Penn severe-sepsis EWS** | Giannini et al., Crit Care Med 2019, doi:10.1097/CCM.0000000000003891 | Penn Medicine | — | severe sepsis | — | Sens **26%**, spec 98%, **PPV 29%**; increased lactate testing and fluids; **no mortality or ICU-transfer difference** | Prospective silent-then-live |

### 1.4 A note on what dominates the "MIMIC-IV sepsis ML" literature

When I searched Europe PMC for `sepsis AND MIMIC-IV AND (early prediction OR early warning OR
onset prediction)` sorted by citation count, **every one of the top 13 results was about
mortality, complications, or prognosis among patients who already had sepsis**, not about
predicting onset [V]. Examples: Hu et al. *Infect Dis Ther* 2022 (in-hospital mortality,
XGBoost AUC 0.884), Zhou et al. *Eur J Med Res* 2024 (SIC 28-day mortality, XGBoost AUC
0.828–0.923), Zhang et al. *Sci Rep* 2023 (sepsis-associated delirium, AUROC 0.793/0.701).

This matters for the write-up: **the apparently enormous "sepsis ML on MIMIC-IV" literature is
mostly not our task.** The genuinely comparable set — hourly onset prediction on MIMIC-IV with
a stated Sepsis-3 timing rule — is small. YAIB, Moor et al., and Do et al. are the three
serious comparators; everything else in §1.1 has either an unstated definition, an unstated
post-onset policy, or a cohort construction that makes the number non-comparable.

---

## 2. The comparability problem

### 2.1 The published range

| Source | Scope | AUROC range reported |
|---|---|---|
| Fleuren et al., *Intensive Care Med* 2020;46:383–400 [V-sub] | 28 papers, 130 models | **ICU 0.68–0.99**; wards 0.96–0.98; ED 0.87–0.97 |
| Moor et al., *Front Med* 2021;8:607952 [V-sub] | 22 ICU studies | **0.78–0.99** |
| Wang et al., *npj Digit Med* 2025;8:190 [V] | 91 studies | median **0.886** (6 h, partial-window internal), **0.861** (12 h), **0.783** (full-window external) |
| Shanmugam et al., *Indian J Crit Care Med* 2025;29(6) [V] | 13 studies | **0.753–0.99** |
| González Garcés et al., *Front Digit Health* 2026;8:1794922 [V] | 37 studies | ML **0.81–0.99**; DL up to 0.93 |

**A ~0.3 AUROC spread on nominally the same task.** Our 0.751 sits at the bottom of every one
of these ranges. That is the central fact the advisor needs, and §2.2–§2.5 explain why it is
not a statement about our model.

### 2.2 What drives the range — with published magnitudes

Each row below is a *measured* effect from the literature, not speculation.

| Driver | Published magnitude | Source |
|---|---|---|
| **Including post-treatment hours** | AUROC **0.62 → 0.47** when evaluation is restricted to before treatment initiation | Kamran et al., NEJM AI 2024 [V] |
| **Evaluation strategy** (continuous vs fixed-horizon vs peak) | **0.67 vs 0.61** on the same model and data; case–control onset matching moves peak-score AUROC by **+0.08** and fixed-horizon by **−0.14** | Do et al., JMIR 2026 [V] |
| **Partial- vs full-window validation** | median **0.886 → 0.783**; only 54.9% of 91 studies used full-window validation | Wang et al., npj Digit Med 2025 [V] |
| **Onset-time definition within Sepsis-3** | **0–6% AUROC** (H1 0.832 / H2 0.869 / H3 0.829 for LGBM) | Cohen et al., Sci Rep 2024 [V-sub] |
| **Model class** | **1–5% AUROC** — *smaller than the definition effect* | Cohen et al. 2024 [V-sub] |
| **Sepsis definition family** (Sepsis-3 vs SEP-1 vs ASE) | **0.85 – 0.94** for one model on one cohort | Dutta et al., JAMA Netw Open 2026 [V] |
| **Prediction horizon** | 0.886 (6 h) → 0.861 (12 h) | Wang et al. 2025 [V] |
| **Site shift, holding the label harmonised** | **0.846 → 0.761** | Moor et al. 2023 [V] |
| **Measurement-count features** | +0.015 internal, but external drop doubles (−0.047 → −0.082); calibration slope 1.007 → 0.417 | Yamamoto et al. 2026 [V] |

**Cohen et al.'s formulation is the one to quote:** *the best-performing model on the worst
definition had lower AUROC than the worst model on the best definition* [V-sub]. The label
choice dominates the architecture choice. YAIB reaches the same conclusion independently:
*"the choice of dataset, cohort definition, and preprocessing have a major impact on the
prediction performance, often more so than model class"* [V-sub].

### 2.3 How many papers state their post-onset truncation, window, and thresholds?

Honest answer: **the field does not report this systematically, and no review counts it
directly.** What I can establish:

- **Post-onset truncation.** No systematic review I found tabulates it. The closest proxy is
  Wang et al. 2025: **only 54.9% of 91 studies used "full-window validation with both model-
  and outcome-level metrics"; the rest used partial-window approaches that exclude negative
  cases and thereby inflate performance** [V]. In my own reading of the papers in §1, **3 of
  17** state their post-onset policy explicitly and unambiguously: YAIB (features stopped 6 h
  before onset), Do et al. (the paper's entire subject), and CinC 2019 (records truncated
  shortly after onset). Papers 4, 5, 8, 16, 17 do not state it at all.
- **Prediction window.** Better reported — horizons of 3–24 h are near-universally stated
  (González Garcés et al.: horizons "ranged from approximately 3 to 24 h" [V]). But the
  *window semantics* differ: Do et al. note that the horizon "only determines which time
  points before onset are labeled as positive, but does **not** restrict the range of time
  points included in the evaluation" [V] — two papers can both say "6 hours" and mean
  different things.
- **Fixed-threshold / operating-point evaluation.** Rare. Shanmugam et al. found **only 2 of
  13** studies reported anything like alert burden [V]. Wang et al. found **only 2 of 91**
  studies had prospective validation [V]. Among §1.1, only Moor et al. reports an alert-burden
  figure (1.4 false alerts per true alert).
- **Case–control matching.** Moor et al. 2021: **72.7% of studies gave no matching
  methodology at all** [V-sub] — and Do et al. show matching moves AUROC by up to 0.08 [V].

### 2.4 Papers whose headline numbers are likely inflated — with reasons

These are flagged on internal evidence. I am not alleging error; I am saying the number is not
comparable to a per-hour, post-onset-truncated, patient-grouped evaluation.

| Study | Headline | Why it is likely inflated |
|---|---|---|
| **Dalal et al. (medRxiv 2024)** — MIMIC-IV v2.2 | **AUROC 0.99 / 0.98 / 0.96** at 6/12/24 h | Three independent red flags. (a) **The 24-h-ahead number (0.96) is higher than most papers' at-onset numbers** — discrimination should *fall* with horizon (Wang et al.: 0.886→0.861 over 6→12 h). (b) **The external eICU numbers are identical to the internal MIMIC-IV numbers to two decimals at all three horizons.** Genuine external validation essentially never does this (cf. Moor 0.846→0.761). (c) The eICU label is *"patients with the term 'sepsis' in their diagnosis string"* [V] — a string match on a discharge diagnosis field, with no onset timestamp, evaluated against a model that "uses laboratory data within 6 h prior to the time point of interest". This is consistent with a per-patient classification with post-onset labs in the window, not a prospective hourly prediction. |
| **Stylianides et al. (IEEE JBHI 2026)** — MIMIC-IV, 45,285 patients | AUROC 0.87, **AUPRC 0.88** | **AUPRC > AUROC is only possible at high positive prevalence** (AUPRC's baseline is the prevalence itself). At the ~1–3% timestep prevalence of this task, an AUPRC of 0.88 is not attainable. The number implies a **balanced or heavily undersampled evaluation set**, i.e. per-patient classification on a matched cohort, not per-hour risk. Compare YAIB on the same database: **AUPRC 5.9–9.1** at ~1% prevalence [V-sub]. |
| **Mahmoud et al. (Sci Rep 2026)** | AUROC 0.9566 / 0.9595 | Abstract states the data had "large data gaps and class disparities" addressed by "precise categorizations" — i.e. resampling. Neither the dataset nor the sepsis definition is identified in the abstract. Not interpretable as reported. |
| **Yang, Yi & Chen (Sci Rep 2026)** — LLM on CinC 2019 | AUC 0.9307 | The CinC 2019 challenge was won at a **utility score of 0.360**, and the best AUROCs on that dataset in careful hands are ~0.76–0.85 (SeFT 0.768; LSTM-GAT 0.853). 0.93 on the same data, reported without a utility score, warrants checking whether evaluation is per-patient rather than per-hour. |
| **SeFT paper's own baseline table** (Horn et al., ICML 2020) | **IP-Nets 0.941 / AUPRC 29.4; Transformer 0.973 / AUPRC 53.4** on P-Sepsis, vs SeFT-Attn 0.768 / 4.84 | I flag this **against the authors' own proposed method**. An AUPRC of 53.4 at 1.8% prevalence on the CinC task is far outside everything else published on that dataset, including the challenge winners. Something in those two baseline pipelines is almost certainly leaking. This is a useful example for §2 precisely because it appears in a well-cited ICML paper and, to my knowledge, has never been remarked on. **[V]** — I read the table; the interpretation is mine. |
| **Epic Sepsis Model, vendor-reported** | AUC 0.76–0.83 | Externally measured at **0.63** (Wong et al.) and **0.47** when restricted to before treatment (Kamran et al.). The canonical case. |

**And, for symmetry, one likely-deflated case: ours.** Our own experiments show the +3 h vs
+24 h truncation choice alone moves AUROC 0.630 → 0.718. A reviewer comparing our 0.751
against a table of 0.9s will draw the wrong conclusion unless §2 is in the paper.

### 2.5 The AUROC-vs-lead-time question specifically

The advisor asked how much of our anti-correlation finding is already published. Carefully:

**What is published:**
- **Horizon → AUROC decline** is documented: 0.886 (6 h) → 0.861 (12 h) across 91 studies
  [V]; a CNN-LSTM reported 0.879 at 15 min → 0.752 at 24 h [S]; an LSTM 0.96 at 1 h → 0.92 at
  4 h [S]; a JEPA sepsis model reports a 16.8% AUROC drop from H0 to H10 [S].
- **Do et al. 2026 state the mechanism**: continuous-evaluation performance *increases as the
  prediction window narrows*, because "the assigned sepsis labels align more closely with the
  sepsis-related signals in the patient data" [V].
- **Dutta et al. 2026** show a definition that gives higher AUROC (SEP-1, 0.94) giving *lower*
  AUPRC (0.16) and lower PPV — the discrimination/utility divergence [V].

**What is not published, as far as I can determine:** a paper that (a) names the AUROC–lead-time
relationship as a *reportable property of the evaluation*, (b) demonstrates it across multiple
independent manipulations of the same pipeline, or (c) demonstrates it with a synthetic control
where the signal magnitude is held exactly. The delegated search agent independently reached the
same conclusion: *"the trade-off is empirically well-documented but I did not find a paper that
states it as a formal, named principle."* **Two agents and I searched for this; none of us found
it. That is a reasonably strong negative, but it is a negative.** See §6.

---

## 3. Sepsis definition landscape

### 3.1 The taxonomy, and what each gets wrong

| Definition family | What it is | Known critique | Key citation |
|---|---|---|---|
| **Sepsis-3 (clinical)** | SOFA rise ≥2 attributable to infection. **The consensus paper defines no onset timestamp** — this is the root of the whole comparability problem | Requires an operational SOI rule that Singer et al. never specified; every implementation differs | Singer et al., JAMA 2016;315:801 [S] |
| **Sepsis-3 + Seymour SOI timing** | SOI = antibiotics + cultures; abx-first → culture within 24 h; culture-first → abx within 72 h; SOI time = earlier event | The window is asymmetric and pipelines silently re-symmetrise it | Seymour et al., JAMA 2016;315:762 [V-sub] |
| **PhysioNet/CinC 2019** | Seymour SOI **plus** abx must continue ≥72 consecutive h; `t_SOFA` = 2-point *deterioration* within 24 h; validity window **[t_susp−24 h, t_susp+12 h]**; labels shifted 6 h earlier | Label-generation code never released; not reproducible outside the challenge data | Reyna et al., Crit Care Med 2020 [V] |
| **mimic-code `sepsis3`** | SOI as above, **but** organ dysfunction = **absolute `sofa_24hours ≥ 2` with baseline assumed zero**; validity window **[SOI−48 h, SOI+24 h]**; **one row per ICU stay** | Not a rise, not hourly, and a different window from CinC. See §3.3 | https://github.com/MIT-LCP/mimic-code [V] |
| **CDC Adult Sepsis Event (ASE) / eSOFA** | Blood culture + ≥4 qualifying antibiotic days + simplified organ dysfunction | Smaller, sicker cohort; validation sensitivity vs Sepsis-3 ranges **50.8%–91.6%** across studies | Rhee et al., Crit Care Med 2019 [V-sub] |
| **SEP-1 bundle** | CMS quality-measure definition | Highest AUROC but lowest AUPRC and PPV in a head-to-head | Dutta et al., JAMA Netw Open 2026 [V] |
| **Angus criteria (ICD)** | 122 ICD-9 codes (109 infection + 13 organ dysfunction) | **PPV 70.7%, sensitivity 50.4%** vs chart review | Iwashyna et al., Med Care 2014 [V-sub] |
| **Explicit ICD-9/10 sepsis codes** | Direct sepsis codes | **Sensitivity 32.3% vs 69.7% for clinical criteria**; incidence trend **+10.3%/y from coding drift** while clinical incidence was flat | Rhee et al., JAMA 2017;318:1241 [V-sub] |

### 3.2 How many papers use which — the honest count

**No systematic review breaks this down cleanly for MIMIC-IV specifically.** What exists:

- **Moor et al. 2021** (22 ICU studies, the only clean breakdown I found): **Sepsis-2 in 12
  (54.5%), Sepsis-3 in 9 (40.9%), expert clinician labels in 1** [V-sub]. Note this is a
  pre-2021 sample — the balance has since shifted toward Sepsis-3.
- **Wang et al. 2025** (91 studies): **58.2% used PhysioNet/CinC 2019 data** [V] — so the
  single most common "sepsis label" in the literature is the CinC label, inherited rather than
  computed.
- **González Garcés et al. 2026** (37 studies): "considerable heterogeneity"; definitions
  included Sepsis-3, SIRS-based, composite clinical rules, and administrative coding; **the
  review does not provide counts** [V].
- **In our §1 table**: of the 17 onset-prediction studies, **6 use CinC-provided labels**
  (papers 10, 12, 14, 15, and CinC replications in 6), **4 use a stated Sepsis-3
  implementation with SOI timing** (1 via ricu, 2 via ricu, 3 own SQL, 11 three variants),
  **1 uses a diagnosis-string match** (5, for its external set), and **6 do not state the
  definition precisely enough to classify** (4, 8, 16, 17, and parts of 5, 7).

**So: roughly a third of the onset-prediction papers I examined cannot be classified by sepsis
definition from their abstracts.** That is the headline for this section.

### 3.3 The critique that matters most for us: "Sepsis-3" names several different labels

This is the finding I would put in the paper. Verified directly from source files [V]:

- **mimic-code `sepsis3.sql`** joins SOFA to SOI on
  `sofa.endtime BETWEEN suspected_infection_time − 48 h AND suspected_infection_time + 24 h`,
  filters `sofa_24hours >= 2`, and returns **`WHERE rn_sus = 1` — one row per `stay_id`**. The
  code comments state that **baseline SOFA is assumed zero** on ICU admission.
- **PhysioNet/CinC 2019** requires `t_SOFA ∈ [t_suspicion − 24 h, t_suspicion + 12 h]`, defines
  `t_SOFA` as a **2-point deterioration**, and requires **≥72 consecutive hours of
  antibiotics**.
- **ricu `sep3()`** defaults to `si_lwr = 48 h`, `si_upr = 24 h` (matching mimic-code, not
  CinC) but uses a genuine **delta** criterion (`delta_fun = delta_cummin`, i.e. rise vs
  running minimum) — configurable, and the only public tool that offers the choice [V-sub].

**Three tools, all called "Sepsis-3", with different windows (−48/+24 vs −24/+12), different
organ-dysfunction semantics (absolute ≥2 from an assumed-zero baseline vs a 2-point rise), and
different antibiotic-duration requirements (none vs ≥72 h).** Cohen et al. quantified the cost
of just the *onset-time* choice at 0–6% AUROC; nobody has quantified the cost of the window
and delta-semantics choices.

### 3.4 The strongest empirical citations for "the definition determines the cohort"

- **Johnson et al., Crit Care Med 2018;46(4):494–499** [V-sub]: in a *single* database, sepsis
  incidence ranged from **31.9% (CDC method) to 9.0% (explicit ICD coding)** — a 3.5× swing.
- **Bauer et al., Crit Care 2025;29:523** [V-sub]: 25 hospitals. CDC ASE **139,267** vs
  Sepsis-3 **234,601** vs ICD **64,461** patients. **κ = 0.39, Jaccard = 0.31** between ASE and
  Sepsis-3. **65% of sepsis cases were identified by only one criterion.** Mortality: 14.9%
  for cases meeting both, but **2.2% for Sepsis-3-only** cases.
- **Dickens (medRxiv 2026)** [V]: on **MIMIC-IV v3.1**, mean **Jaccard ≈ 0.32** between clinical
  and administrative sepsis definitions at the primary site, **0.20** across multi-centre
  cohorts.
- **Rhee et al., JAMA 2017** [V-sub]: ICD-based sensitivity **32.3%** vs clinical **69.7%**;
  claims-based incidence rising **10.3%/y** while clinical incidence was flat.

**On ICD timing specifically:** ICD codes are assigned at the *hospitalization* level and carry
**no onset timestamp at all**. The critique is therefore structural rather than a matter of
degree. I looked for, and **could not find, a study quantifying the timing offset in hours
between ICD-coded sepsis and Sepsis-3 clinical onset** — because the quantity is largely
undefined. State it that way rather than as "ICD timing is poor by N hours".

---

## 4. Deployment-oriented work

### 4.1 The Epic Sepsis Model arc — the field's cautionary tale, in three acts

**Act 1 — Wong et al., JAMA Intern Med 2021;181(8):1065–1070.** doi:10.1001/jamainternmed.2021.2626 ·
https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307 [V-sub, numbers
consistent across three independent retrievals]

27,697 patients / **38,455 hospitalizations**, Michigan Medicine, Dec 2018–Oct 2019. Sepsis
label was itself a composite (CDC surveillance criteria **or** ICD-10 + 2 SIRS + 1 organ
dysfunction within 6 h) — note that even the paper that indicted Epic used a hybrid definition.

- **AUROC 0.63 (95% CI 0.62–0.64)** vs vendor-claimed 0.76–0.83
- At deployed threshold (score ≥6): **sensitivity 33%, specificity 83%, PPV 12%**
- Alerts on **6,971 of 38,455 hospitalizations (18%)**; **missed 1,709 (67%) of septic patients**
- **Number needed to evaluate: 109** patients per one sepsis case detected earlier

Accompanying editorial: Habib, Lin & Grant, *"The Epic Sepsis Model Falls Short — The Importance
of External Validation"*, JAMA Intern Med 2021, doi:10.1001/jamainternmed.2021.3333 [S].

**Act 2 — Kamran et al., NEJM AI 2024;1(3), AIoa2300032.** doi:10.1056/AIoa2300032 ·
open copy at https://par.nsf.gov/biblio/10522981 [V]

This is, in my view, **the single most important paper for our §2 argument** and the advisor
should read it. 77,582 hospitalizations, U. Michigan 2018–2020, 3,766 sepsis (4.9%). They
re-evaluated the ESM restricting predictions to **before any indication of treatment**
(antibiotics, fluids, blood culture, or lactate measurement).

> **AUROC 0.62 (0.61–0.63) conventionally → 0.47 (0.46–0.48) before treatment.**

Below chance. The authors' framing: *"an alert arriving 3 hours before the patient becomes
overtly septic with organ dysfunction but 2 hours after clinicians have initiated antibiotics
is of little value."*

**Act 3 — Dutta et al., JAMA Netw Open 2026;9(4):e265599.** doi:10.1001/jamanetworkopen.2026.5599 ·
https://pmc.ncbi.nlm.nih.gov/articles/PMC13058769/ [V]

Epic's revised, **locally trained** "Early Detection of Sepsis Model v2", evaluated across nine
Mass General Brigham hospitals, **198,494 encounters**, March–August 2024. The point of the
paper is that performance is a function of the yardstick:

| Definition | Incidence | AUROC | AUPRC | PPV | Median lead time |
|---|---|---|---|---|---|
| Sepsis-3 | 2.9% | 0.89 (0.89–0.90) | 0.24 | 11.4% | **3.4 h** |
| SEP-1 | 1.2% | **0.94** | **0.16** | 6.8% | 4.5 h |
| CDC ASE | 2.0% | 0.85 | 0.11 | 5.9% | **1.4 h** |

At the Youden-optimal threshold for Sepsis-3, **21.2% of all encounters alerted**, false-positive
rate 19.3%, **8.7 encounters needing evaluation per true positive**. Note that the *highest*
AUROC (SEP-1, 0.94) comes with the *lowest* AUPRC and PPV — a clean published instance of
discrimination and utility pulling apart.

**I did not find** a published Epic rebuttal in the peer-reviewed literature, nor an independent
validation of Epic's own retrained model outside the vendor's customers. Epic's public response
to Wong et al. was made through trade press rather than a journal, as far as I can establish [?].
A second external validation exists in two county EDs (*JAMIA Open* 2024;7(4):ooae133, AUC
0.60–0.64) [S].

### 4.2 COMPOSER — the strongest deployment evidence

- **Shashikumar, Wardi, Malhotra, Nemati. npj Digit Med 2021,** doi:10.1038/s41746-021-00504-6 [V]
  515,720 patients, >6M prediction windows, six cohorts. **ICU AUROC 0.925–0.953; ED
  0.938–0.945.** Median lead **12.2 h [IQR 3.2–22.8] in ICU**, 2.1 h [0.8–4.5] in ED, measured
  *before first antibiotic order*. Conformal prediction abstains on ~20% of non-septic and 8%
  of septic windows ("learns to say I don't know").
- **Boussina et al., npj Digit Med 2024,** doi:10.1038/s41746-023-00986-6 [V]
  Before–after quasi-experimental deployment at two UC San Diego EDs, Jan 2021–Apr 2023, 6,217
  septic patients, Bayesian causal inference. **In-hospital mortality −1.9% absolute (17%
  relative, 95% CI 0.3–3.5%); SEP-1 bundle compliance +5.0% absolute (10% relative, 95% CI
  2.4–8.0%); 72-hour SOFA change −4%.**

**Caveat worth stating to the advisor:** COMPOSER's lead time is measured *relative to first
antibiotic order*, not relative to a Sepsis-3 onset timestamp. That is arguably the more
clinically meaningful anchor, but it makes the 12.2 h figure **not directly comparable to our
20.6 h**, which is measured relative to `t_sepsis`.

### 4.3 TREWS — the only prospective multi-site outcome evidence

- **Adams et al., Nat Med 2022,** doi:10.1038/s41591-022-01894-0 [V]
  **590,736 patients monitored** across five hospitals; 6,877 sepsis cases identified before
  antibiotics. **Adjusted absolute mortality reduction 3.3%; 18.7% relative reduction** among
  patients whose alert was confirmed by a provider within 3 hours.
- **Henry et al., Nat Med 2022,** doi:10.1038/s41591-022-01895-z [V]
  **82% sensitivity; 89% of alerts evaluated by a provider; 38% confirmed; median time to first
  antibiotic order reduced by 1.85 h.** The 89% evaluation rate is the number to contrast with
  the 86% *override* rate reported for a conventional NEWS deployment [S].
- **Henry et al., npj Digit Med 2022,** doi:10.1038/s41746-022-00597-7 [S] — qualitative
  human-machine teaming study; the finding is that adoption, not AUROC, was the binding
  constraint.
- Predecessor: **Henry, Hager, Pronovost, Saria, Sci Transl Med 2015** (TREWScore) [S].

### 4.4 InSight / Dascena, Sepsis Watch, and negative results

- **InSight** — Desautels et al., *JMIR Med Inform* 2016;4(3):e28, "Prediction of sepsis in the
  ICU with minimal electronic health record data: a machine learning approach"; MIMIC-III [S].
  Followed by **Shimabukuro et al., BMJ Open Respir Res 2017;4:e000234**, a small **randomized
  controlled trial** at two UCSF wards reporting reduced length of stay and in-hospital
  mortality [S]. **I was unable to verify the RCT's numbers from the primary source in this
  session** — flag before citing. This RCT is frequently cited as *the* RCT evidence for sepsis
  ML and it is small and single-centre.
- **Sepsis Watch (Duke)** — Valan et al., *npj Digit Med* 2025;8:350 [V]. Described as "the first
  deep learning model implemented in routine clinical care in the United States", running at
  Duke since November 2018. External validation on **205,005 encounters** at four Summa Health
  EDs. **AUROC 0.906–0.960 externally — but AUPRC only 0.177–0.252.** At a 20%-precision
  threshold: **1–27 alerts/day** depending on site, lead time **3.58–5.07 h**. Uses a **Sepsis-2**
  label. Code not public ("proprietary reasons"). The paper itself notes that *neither AUROC nor
  AUPRC considers alert timing*.
- **Negative result — Giannini et al., Crit Care Med 2019** [V]: a deployed severe-sepsis ML
  alert achieved **sensitivity 26%, PPV 29%**, modestly increased lactate testing and IV fluids,
  and produced **no significant difference in mortality or ICU transfer**.
- **Negative result — conventional EWS** [S]: a NEWS implementation fired 175,357 alerts across
  85,322 patients in 12 months, some patients receiving >100 alerts/day, an **86% override
  rate**, and no measurable effect on ICU transfers or mortality.

### 4.5 Alert burden as a metric, and bedside-score baselines

**Alert-burden numbers that exist in the literature** (useful for calibrating our ≤1.0 false
alerts per nonseptic patient-day):

| Source | Burden |
|---|---|
| Moor et al. 2023 [V] | **1.4 false alerts per true alert** at 80% recall |
| Sepsis Watch external [V] | **1–27 alerts/day per ED** at 20% precision |
| Epic v2, Dutta 2026 [V] | **21.2% of all encounters** alert at Youden threshold; **NNE 8.7** |
| Epic v1, Wong 2021 [V-sub] | **18% of hospitalizations** alert; **NNE 109** |
| Romero-Brufau et al. [S, unverified] | a clinically acceptable alarm burden is roughly **3–10 alarms per 100 patient-days** |
| ML vs threshold alarms [S] | 0.032 ML alarms/patient-hour vs 0.132 threshold alarms/patient-hour |

**Our ≤1.0 false alerts per nonseptic patient-day is ~10–30× more permissive than the
Romero-Brufau figure**, if that figure is right. I could **not** verify Romero-Brufau et al.,
*"Why the C-statistic is not informative to evaluate early warning scores and what metrics to
use"*, Crit Care 2015;19:285 from the primary source — it is the ideal citation for our §2
argument and the advisor should have someone confirm it before we lean on it.

**Bedside-score baselines on MIMIC.** Our NEWS2 0.626 / qSOFA 0.587 / SIRS 0.551 are consistent
with the literature. Seymour et al. 2016 [V-sub] report, **in ICU encounters**, SOFA 0.74,
qSOFA 0.66, SIRS 0.64 for *mortality* (a different endpoint, and the reason those numbers look
higher). **Our decision to exclude SOFA as a comparator because the label is defined by a SOFA
rise is correct and, as far as I can tell, is not made explicit in any of the papers above** —
several of which do report "our model beats SOFA" on Sepsis-3 labels.

### 4.6 Regulatory

**Prenosis Sepsis ImmunoScore** received **FDA De Novo marketing authorization in April 2024**,
reported as the first FDA-authorized AI diagnostic for sepsis [S]. A development-and-validation
paper exists in *NEJM AI* (doi:10.1056/AIoa2400867, "FDA-Authorized AI/ML Tool for Sepsis
Prediction: Development and Validation") but **the publisher returned 403 to every retrieval
attempt and it is not indexed in Europe PMC; I could not verify a single number from it.** [?]
Flag as a lead.

---

## 5. Methods most relevant to us

### 5.1 Irregular-time-series architectures — and whether any of them were tested on sepsis

| Model | Citation | Sepsis? | Note |
|---|---|---|---|
| **GRU-D** | Che, Purushotham, Cho, Sontag, Liu. *Sci Rep* 2018;8:6085. doi:10.1038/s41598-018-24271-9 | **No** (MIMIC-III & PhysioNet 2012 mortality) | **The origin of the term "informative missingness"** — and it argues *for* exploiting it. This is the paper our synthetic control is in tension with. |
| **mTAND** | Shukla & Marlin, ICLR 2021. arXiv:2101.10318 | No | Continuous-time embeddings + attention over reference times. Code: github.com/reml-lab/mTAN |
| **SeFT** | Horn, Moor, Bock, Rieck, Borgwardt, ICML 2020. arXiv:1909.12064 | **Yes** — CinC 2019 | The only one in this family evaluated on sepsis. **SeFT-Attn AUROC 0.768, AUPRC 4.84** at 1.8% prevalence [V]. On MIMIC-III mortality it is *worse* than GRU-D (AUPRC 46.3 vs 52.0) and sells on ~10× speed. |
| **Raindrop** | Zhang, Zeman, Tsiligkaridis, Zitnik, ICLR 2022. arXiv:2110.05357 | **Partly** — uses the "P19" CinC-2019 data, but as whole-stay binary classification, not per-hour risk | Claims up to +11.4 F1 over SOTA incl. SeFT |
| **Warpformer** | Zhang, Zheng, Cao, Bian, Li, KDD 2023. arXiv:2306.09368 | **No** | Learned warping to unify granularity. Notable for us: one of its five MIMIC-III tasks is **"will this variable be measured next"** — the field already treats the observation process as a learnable target |
| **Latent ODE / ODE-RNN** | Rubanova, Chen, Duvenaud, NeurIPS 2019. arXiv:1907.03907 | No | Can model observation *times* as a Poisson process |
| **CRU** | Schirmer et al., ICML 2022. arXiv:2111.11344 | No | |
| **ContiFormer** | Chen et al., NeurIPS 2023. arXiv:2402.10635 | No | |
| **STraTS** | Tipirneni & Reddy, *ACM TKDD* 2022;16(6):105. arXiv:2107.14293 | **No** | **The closest architectural precedent to ours**: (time, variable, value) triplets, continuous value embedding, no gridding, self-supervised forecasting pretraining |

**The most useful single number in this section** is STraTS's own baseline table on MIMIC-III
mortality [V-sub]: plain **GRU 0.886** → **STraTS 0.891**. The entire span from a vanilla RNN to
the best specialised irregular-TS architecture is **0.005–0.026 AUROC**. Our Transformer-minus-XGBoost
gap of **0.015** sits squarely inside that band — which is a defensible thing to say, not an
embarrassing one.

**Gap:** I found **no** paper in this architecture family that evaluates **per-hour sepsis risk on
MIMIC-IV**. SeFT and Raindrop use CinC-2019; everything else uses MIMIC-III/PhysioNet-2012
mortality. Per-hour sepsis on MIMIC-IV appears only in the *benchmark-suite* literature (YAIB).

### 5.2 Transformers for ICU EHR

BEHRT (*Sci Rep* 2020), Med-BERT (*npj Digit Med* 2021), TransformEHR (*Nat Commun* 2023;14:7857,
doi:10.1038/s41467-023-43715-z), EHRSHOT (NeurIPS 2023 D&B, arXiv:2307.02028), Context Clues
(ICLR 2025, arXiv:2412.16178). **All of these operate on visit-level diagnosis-code sequences,
not hourly physiological signals, and none reports a sepsis-onset task** [V-sub]. The EHR
foundation-model literature and the sepsis-timing literature are essentially disjoint. A
recurring stated limitation in the 2024–2026 ICU foundation models (PULSE-ICU arXiv:2511.22199;
"Foundation Models for Clinical Records at Health System Scale" arXiv:2507.00574 [S]) is that
quadratic attention forces truncation of long ICU event streams — a motivation for a causal,
streaming design like ours.

### 5.3 Gradient boosting vs deep learning

- **Grinsztajn, Oyallon, Varoquaux, NeurIPS 2022 D&B.** arXiv:2207.08815 [V-sub]. 45 datasets.
  Trees remain SOTA at ~10k samples; the gap is **not** explained by categorical features and
  does **not** close with tuning. Three diagnosed NN weaknesses: **non-robustness to
  uninformative features**, rotation invariance, inability to fit irregular targets. Our
  feature matrix — many rarely-measured channels, mask/delta channels, a non-smooth target — is
  exactly this regime.
- **Shwartz-Ziv & Armon, *Information Fusion* 2022;81:84–90** [S]. XGBoost wins; deep tabular
  models are harder to tune; the deeper lesson is about **evaluation-set selection bias**.
- **ICU-specific, and the evidence points both ways:**
  - **HiRID-ICU-Benchmark** (Yèche et al., NeurIPS 2021 D&B, arXiv:2111.08536) concludes
    explicitly that *"boosted ensembles of decision trees outperform current deep learning
    approaches on medical time series problems"* — but **it has no sepsis task** (circulatory
    and respiratory failure) [V-sub].
  - **YAIB** (ICLR 2024), which *does* have an hourly sepsis task on MIMIC-IV, finds the
    opposite: **GRU 83.6 vs LGBM 77.5 AUROC**, GRU ahead on every dataset [V-sub].
  - **Liao et al., arXiv:2211.06034** [V]: on CinC-2019 sepsis, DL beats non-DL **only** (a) on
    some metrics (AUROC, AUPRC, sensitivity, FNR) and not others, and (b) once training size
    reaches thousands.

**Important caution for our write-up.** The delegated agent found **no** paper whose headline
result is "XGBoost ≈ Transformer for per-hour sepsis *onset* prediction on MIMIC-IV", and the
strongest same-task published benchmark (YAIB) points the *other* way by ~6 AUROC points. So
our finding that flat XGBoost matches the Transformer is genuinely interesting — **but we must
present it as our Reconciling-the-two-parts analysis does**: the gap is representation (causal
ffill + mask + Δt), not architecture. Framed as "GBDT beats deep learning" it would collide
with YAIB and lose.

### 5.4 Benchmark suites

| Suite | Sepsis task? | MIMIC-IV? | Reported sepsis AUROC |
|---|---|---|---|
| **YAIB** (ICLR 2024) https://github.com/rvandewater/YAIB | **Yes** — "sepsis within next 6 h", hourly | **Yes** | **LR 77.1 / LGBM 77.5 / GRU 83.6**; AUPRC 4.6 / 5.9 / 9.1 |
| **HiRID-ICU-Benchmark** (NeurIPS 2021) | No | No | — |
| **MIMIC-Extract** (CHIL 2020) | No | **No — MIMIC-III only**, never ported | — |
| **ricu** (*GigaScience* 2023;12:giad041) | Provides a `sep3` *concept*, not a benchmark task | Yes (`miiv`) | — |
| **MIMIC-Sepsis** (arXiv:2510.24500) | No — mortality/LOS/shock onset | **Yes, v3.1** | — |

**YAIB is our baseline anchor: ~83.6 AUROC / 9.1 AUPRC (GRU) and 77.5 / 5.9 (LGBM) for hourly
sepsis on MIMIC-IV at ~1% positive-bin prevalence.** Our 0.751 is below both — and the first
thing a reviewer will ask is why. The answer is our post-onset truncation and pre-onset target
(YAIB keeps a standard 6-h prediction window and truncates features 6 h before onset, which is
a *different and easier* target than positives confined to `[t_sepsis − 12 h, t_sepsis)` with
truncation at onset). **We should reproduce the YAIB target on our cohort and report both
numbers.** Without that, the comparison will be made for us, unfavourably.

### 5.5 Label-definition sensitivity, prodrome windows, and measurement-frequency confounding

**Label-definition sensitivity** — Cohen et al., *Sci Rep* 2024;14:1920 is the key prior work
(§2.2, §3). It varies onset time within Sepsis-3 on MIMIC-III with three model classes and
finds definition variance ≥ model variance.

**Prodrome / pre-onset window as a training target.** The strongest *motivation* in the
literature is:

> **Weissman, Hubbard, Himes, et al., "Sepsis Prediction Models are Trained on Labels that
> Diverge from Clinician-Recommended Treatment Times", AMIA Annu Symp Proc 2024, PMID 40417569**
> [V]. 153 clinicians at three centres reviewed vignettes from eight real sepsis cases and
> recommended starting antibiotics **an average of 7.0 hours (95% CI 5.3–8.8) BEFORE the
> Sepsis-3 onset time.** They name the problem **"label bias"** and conclude that "predicting
> Sepsis-3 onset as a treatment prompt could lead to inappropriate and delayed treatment
> recommendations."

This is close to a direct argument for our pre-onset target and should be cited as such. It
motivates the design; it does not implement or evaluate one.

**Measurement frequency as a confounder** — the closest published work to our synthetic control:

> **Yamamoto, Wu, Sprehe, Abeer, Celi, Tohyama, "Observation-process features are associated
> with larger domain shift in sepsis mortality prediction: a cross-database evaluation using
> MIMIC-IV and eICU-CRD", medRxiv 2026, doi:10.64898/2026.04.05.26350209** [V]. MIMIC-IV
> n=30,218 → eICU n=31,403. Seven model specifications, each fit **with and without measurement
> counts**. Internal AUROC 0.819 → 0.834 with counts; **external drop −0.047 without counts vs
> −0.082 with counts**; **calibration slope 1.007 → 0.417**.

Two limits: the task is **mortality, not onset**, and it is a **feature ablation with
cross-site transfer**, not a synthetic control that holds the physiological signal fixed.

> **Dickens, "Falsification Testing of Sepsis Prediction Models", medRxiv 2026,
> doi:10.64898/2026.03.17.26348414** [V]. A **pre-registered (OSF, before data access)**
> falsification study on **MIMIC-IV v3.1, n=65,241 adult ICU stays**, testing exactly the
> hypothesis that sepsis models learn care-process intensity rather than biology. **The
> hypothesis was not confirmed**: biological features predicted Sepsis-3 at AUROC 0.901 with
> minimal loss when care-intensity features were removed. Care-intensity signal was larger at
> community hospitals than at the academic centre.

**This is the single most important paper for the advisor to see, and it partially cuts against
us.** It is a pre-registered study, on the same database version, with a near-identical cohort
size (65,241 vs our 63,672), asking a version of our question and getting "no". We should
engage with it directly rather than let a reviewer find it. Note the differences: it works at a
much higher AUROC operating point (0.901), suggesting a substantially different label/target
construction, and it tests *feature removal*, whereas our control *injects* a known signal at
controlled amplitude into channels of differing measurement density. Those are different
experiments and can both be right.

Supporting: Groenwold, *Diagn Progn Res* 2020;4, doi:10.1186/s41512-020-00077-0 [V] — informative
missingness in EHR, and the point that **the missingness mechanism changes once a model is
deployed and clinicians respond to it**. Sisk et al., *Stat Methods Med Res* 2023,
doi:10.1177/09622802231165001 [V] — missing indicators help but are **harmful under
outcome-dependent missingness**. Also: the CinC-2019 entry literally titled *"Utilizing
Informative Missingness for Early Prediction of Sepsis"* (CTL-Team, 5th place) [V-sub] — a
demonstration that missingness alone carries sepsis signal.

---

## 6. Where our work is and is not novel

Blunt assessment. I have separated "someone has published this" from "I searched and did not
find it".

### 6.1 Already published — we would be duplicating

| Our claim | Prior art | How much is left for us |
|---|---|---|
| **Retaining post-onset hours inflates AUROC while making the model clinically worse** | **Kamran et al., NEJM AI 2024**: ESM AUROC **0.62 → 0.47** when restricted to before treatment. **Do et al., JMIR 2026**: evaluation strategy moves AUROC 0.61↔0.67 and matching moves it ±0.14. **Wang et al., npj Digit Med 2025**: partial-window 0.886 vs full-window external 0.783 | **The core claim is published, twice, with larger effects than ours (0.15 vs our 0.09).** What is not published: the *dose–response curve* over truncation horizon on a single cohort (our +0h/+3h/+6h/+24h/none table with prevalence 1.5%→25.4%) and the paired demonstration that lead time moves in the opposite direction. Present ours as a **replication with a dose–response extension**, not a discovery. |
| **Label definition materially changes measured performance** | **Cohen et al., Sci Rep 2024** (within Sepsis-3, 0–6% AUROC, code on Zenodo); **Dutta et al., JAMA Netw Open 2026** (across definition families, 0.85–0.94) | Fully covered. Cite, do not claim. |
| **AUROC is a poor metric for rare-event early warning** | Moor et al. 2021; Wang et al. 2025 (Utility 0.381 → −0.164 externally while AUROC held at 0.783); Sepsis Watch (AUROC 0.96 / AUPRC 0.18) | Fully covered. |
| **Sepsis-3 onset is partly a treatment-decision timestamp** | Kamran et al. 2024; Weissman et al. 2024 ("label bias"); Hagmann/Schamoni/Riezler arXiv:2311.03037 (formal circularity argument); Moor et al. 2021 ("circularity") | Fully covered as an argument. |
| **Deep sequence models barely beat well-featurised flat models** | STraTS's own table (GRU 0.886 → best specialised 0.891); Grinsztajn 2022; Shwartz-Ziv & Armon 2022 | The general claim is covered. **Our specific version — that the gap is representation (ffill + mask + Δt) rather than architecture, measured as 0.640 → 0.722 → 0.736 — I did not find published for sepsis.** Modest but real. |
| **Models may learn clinician behaviour rather than physiology** | GRU-D (proposes exploiting it); Yamamoto et al. 2026 (quantifies the transfer cost); **Dickens 2026 (pre-registered, MIMIC-IV v3.1 — tested it and found the hypothesis NOT confirmed)** | **The question is claimed, and the most rigorous attempt reached a different conclusion from ours.** We must engage with Dickens directly. |

### 6.2 Appears genuinely unclaimed — with the caveat attached

Ordered by how confident I am. **In each case, "I found nothing" means three independent search
passes (mine plus two delegated agents) over Google Scholar, arXiv, Europe PMC/PubMed, and
GitHub found nothing — not that nothing exists.**

**(c) The measurement-frequency-vs-effect-size synthetic control — strongest claim.**
I found no paper that injects a synthetic prodrome of controlled amplitude into channels of
differing measurement density and compares detectability. The nearest work is Yamamoto et al.
(feature ablation + cross-site transfer, mortality task) and Dickens (feature removal). Our
result — **a 0.25 SD drift in a 93%-measured channel (AUROC 0.972) beats a 4 SD shift in a
2.6%-measured one (0.813)**, with the control arm reproducing the real baseline at 0.722 — is a
different experimental design and produces a *quantitative design specification* rather than a
diagnosis. **This is the most defensible novel contribution and I would lead with it.** It also
doubles as a pipeline-validity proof, which reviewers will value.
*Confidence: high that the specific experiment is unpublished. Moderate that no reviewer knows
of an analogue in another clinical domain — I did not search outside sepsis/ICU.*

**(a) AUROC anti-correlated with lead time, demonstrated across independent manipulations.**
The horizon→AUROC decline is well documented (§2.5) and Do et al. articulate the mechanism.
What I did not find: (i) any paper naming this as a property that must be reported alongside
AUROC, (ii) any paper showing it across *five different kinds of change to the same pipeline*
(model class, truncation, feature count, synthetic amplitude, architecture), or (iii) the
synthetic demonstration that a **16× stronger prodrome yields higher AUROC and shorter warning**.
Item (iii) is the sharpest, because it isolates the mechanism with the signal held exactly.
*Confidence: moderate-to-high. Two agents and I searched specifically for this framing. But it
is the kind of observation that may exist as a remark in a discussion section somewhere.*

**(d) Equal-alert-burden model comparison instead of fixed recall.**
Alert burden is widely *reported* (Moor: 1.4 false alerts per true alert; Epic: NNE 8.7 and 109;
Sepsis Watch: alerts/day at 20% precision), and workload-constrained threshold selection exists
as a concept [S]. **What I did not find is a paper that uses equal alert burden as the
comparison protocol between models** — i.e. holding burden fixed and reading off recall and lead
time, rather than holding recall fixed. Our argument that burden "is the constraint a unit
actually imposes and, unlike recall, it forces nothing" is, as far as I can tell, unmade.
*Confidence: moderate. This is a methodological convention, and conventions are hard to prove
absent. It is a good framing contribution but a weak novelty claim on its own.*

**(b) Confining positives to a pre-onset window as a training target — weakest claim.**
Be careful here.
- **CinC 2019 already shifts labels 6 h earlier**, and its utility function already rewards
  only [t−12 h, t+3 h] [V].
- **YAIB already truncates features 6 h before onset and uses a 6-h prediction window** [V-sub].
- **Shao et al. 2026** already excludes patients septic within the first 24 h and predicts
  later onset — a mild prodrome-targeting design [V].
- **Do et al. 2026** already sweep the prediction window from 1 to 100 h and report that
  narrowing it raises measured performance [V].
- **Weissman et al. 2024** already argue on clinical grounds that the target should be earlier
  than `t_sepsis` [V].

So the *idea* is thoroughly present. **What I did not find** is a controlled study that treats
the prodrome window length as the object of study — sweeping it, and reporting the resulting
AUROC / lead time / alert-burden surface. Our `prodrome_window_h` is a parameter; if we sweep it
and publish the surface, that is a contribution. **If we only report the 12-hour setting as "our
method", it will read as a hyperparameter choice that others have already made.**
*Confidence: high that the raw idea is claimed. Moderate that the systematic sweep is not.*

### 6.3 Three additional things that are ours and that I did not see elsewhere

1. **A CinC-2019-faithful sepsis label on MIMIC-IV.** My delegated agent searched GitHub
   extensively and found **no repository that re-implements the CinC-2019 labelling rules on
   MIMIC-IV**, and confirmed that **PhysioNet never released the label-generation code** — only
   the prose definition and `evaluate_sepsis_score.py` [V-sub]. mimic-code implements a
   *different* label (§3.3). If our `sepsentinel/data/sepsis3.py` with its enumerated
   `DEVIATIONS` is released, it is, as far as I can determine, the first public one. **This may
   be the most immediately citable artefact we have.**
2. **The demonstration that post-onset truncation at +3 h reconstructs the PhysioNet timestep
   prevalence (2.3% vs PhysioNet's 2.2%)** — i.e. reverse-engineering an undocumented cohort
   filter of the challenge. I found nothing like this.
3. **Explicitly excluding SOFA as a comparator on the grounds that the label is a SOFA rise.**
   Several papers report beating SOFA on Sepsis-3 labels. I did not find one that names the
   tautology.

### 6.4 The honest summary for the advisor

Our *headline discrimination number is not competitive and should not be the headline.* 0.751
against YAIB's 0.836 on the same database, and against a literature median near 0.89, will read
as a weak result unless the paper is framed as a **measurement paper**: what the number means,
what moves it, and why the thing that moves it most is not the model.

The strongest contribution is **§6.2(c), the synthetic control**, followed by the
**CinC-faithful MIMIC-IV labeller** as a released artefact, followed by the
**prodrome-window sweep** if we actually run it. The post-onset-inflation finding should be
positioned as a replication of Kamran et al. with a dose–response curve, not as new.

**And the one thing to do before writing anything:** reproduce the **YAIB sepsis target** on our
cohort and report our model under it alongside our own target. Without that, our 0.751 has no
anchor and the paper is defenceless on its central number.

---

## 7. Reproducibility and code

### 7.1 Sepsis-labelling implementations available for MIMIC-IV

| Implementation | What it gives you | License / status | Suitable for CinC-style hourly labels? |
|---|---|---|---|
| **MIT-LCP/mimic-code** — https://github.com/MIT-LCP/mimic-code | `sepsis3.sql`, `suspicion_of_infection.sql`, `sofa.sql`. **`sofa.sql` IS hourly** (built on `icustay_hourly`, 24-h rolling window `ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING`). **`sepsis3.sql` is NOT** — one row per `stay_id`, absolute `sofa_24hours ≥ 2`, window [SOI−48 h, SOI+24 h] | MIT; Zenodo DOI 10.5281/zenodo.6818823; ~2,374 commits, active | **Partially.** Hourly SOFA yes; the CinC 2-point-rise and t_sepsis logic must be built on top. **Ships an official DuckDB dialect** at `mimic-iv/concepts_duckdb/` (auto-generated) |
| **ricu** — https://github.com/eth-mds/ricu | `sep3` / `sep3_alt` concepts over MIMIC-III, **MIMIC-IV**, eICU, HiRID, AUMCdb. Outputs an hourly `ts_tbl`. `delta_fun` is configurable (`delta_cummin` default = rise vs running minimum) | GPL-3.0; CRAN + GitHub; active. Paper: *GigaScience* 2023;12:giad041 | **Closest available.** Genuine delta criterion and hourly output, but SI window defaults to [−48 h, +24 h], not CinC's [−24 h, +12 h] |
| **YAIB-cohorts** — https://github.com/rvandewater/YAIB-cohorts | ricu-based sepsis task definition; `stop_obs_at(offset = 6h)`, 6-h outcome window, excludes onset <6 h after admission | MIT; active | It is a *different* target, cleanly specified. Best available reference implementation of a reproducible hourly sepsis task |
| **PhysioNet CinC 2019** — https://physionet.org/content/challenge-2019/1.0.0/ | Prose definition + `evaluate_sepsis_score.py` (https://github.com/physionetchallenges/evaluation-2019, BSD-2) | CC BY 4.0 | **The label-generation code was never released.** Only the definition and the scorer |
| **alistairewj/sepsis3-mimic** | MIMIC-III; the codebase for Johnson et al. 2018 | Zenodo 1256723 | MIMIC-III only |
| **yongh7/MIMIC-sepsis** — https://github.com/yongh7/MIMIC-sepsis | MIMIC-IV **v3.1**, 35,239 patients, own SQL | MIT; 19 stars | Cohort only; tasks are mortality/LOS/shock |
| **philipdarke/mimic4** — https://github.com/philipdarke/mimic4 | Pure-Python MIMIC-IV **v3.0** → DuckDB loader with derived concepts incl. SOFA and sepsis | MIT; **12 commits, 2 stars** — low maintenance, use with care | Community DuckDB port |

**Verified negative:** an extensive GitHub search found **no repository that faithfully
implements the CinC-2019 `t_suspicion` / `t_SOFA` / `t_sepsis` rules on MIMIC-IV** [V-sub].

### 7.2 Which papers ship code

| Ships code | Does not / unclear |
|---|---|
| YAIB (MIT) · Moor multicentre (Apache-2.0, https://github.com/BorgwardtLab/multicenter-sepsis) · Moor MGP-TCN (BSD-3) · Cohen et al. (Zenodo 5168789) · SeFT · STraTS · Raindrop · Warpformer · mTAND · MIMIC-Sepsis (MIT) · Do et al. (stated) · all CinC-2019 entrants (PhysioNet hosts every submission as a zip) | Dalal et al. ("on reasonable request") · Stylianides et al. · Sepsis Watch ("proprietary reasons") · Epic (proprietary) · COMPOSER · TREWS · most 2026 Sci Rep / JMIR papers in §1.2 |

**The base rate is bad.** Moor et al. 2021: **only 2 of 22 studies shared both analysis code and
label-generation code** [V-sub]. Wang et al. 2025 across 91 studies does not tabulate code
availability at all [V].

### 7.3 CinC 2019 leaderboard and its artefacts

Every challenge submission's source is hosted at
`physionet.org/static/published-projects/challenge-2019/1.0.0/sources/`, and the official
results are at https://moody-challenge.physionet.org/2019/results/ [V-sub]. Top five by utility:
**1. "Can I get your signature?"** (Morrill et al., signature methods; utility **0.360**;
GitHub rewrite at https://github.com/jambo6/sepsis_competition_physionet_2019, **no license
file**); 2. Sepsyd (gradient-boosted trees); 3. Separatrix (XGBoost ensemble); 4. FlyingBubble
(TASP time-phased model, Ping An); 5. CTL-Team ("Utilizing Informative Missingness").

**Papers with Code is dead** — retired by Meta and sunset **24 July 2025**;
`paperswithcode.com` now redirects to `huggingface.co/papers/trending` [V-sub]. The
PhysioNet-2019 leaderboard pages there are gone. A JSON dump survives at
https://github.com/paperswithcode/paperswithcode-data. **Cite the official CinC results page
instead.**

### 7.4 MIMIC-IV version differences that matter for sepsis

| Version | Released | Patients | Hospital admissions | ICU stays |
|---|---|---|---|---|
| v2.2 | 2023-01-06 | 299,712 | 431,231 | 73,181 |
| v3.0 | 2024-07-23 | 364,627 | 546,028 | 94,458 |
| **v3.1** | **2024-10-11** | **364,627** | **546,028** | **94,458** |

[V-sub, from physionet.org/content/mimiciv/]

Three changes that bear on sepsis work:

1. **v2.2 fixed ~2.5% of medication administration records that lacked `hadm_id`.** Antibiotic →
   `hadm_id` linkage drives suspicion-of-infection, so pre-v2.2 SOI cohorts are suspect.
2. **v3.0 extended admissions from 2019 to 2022**, taking patients from 299,712 to 364,627. That
   brings **COVID-era ICU stays into the cohort**, materially changing case mix and
   ventilation/oxygenation distributions. **Any comparison of our v3.1 numbers against a
   v2.2-derived result is confounded by this**, and it is worth a sentence in the paper.
3. **v3.1 reverted `itemid` changes** in `d_labitems`/`labevents` that v3.0 had introduced, back
   to v2.2 values. **Hard-coded itemids extracted against v3.0 will silently mismatch both v2.2
   and v3.1.** Our `verify_mimic_itemids.py` step is the right defence and should be mentioned
   in the methods.

---

## 8. What I could not verify — read this before citing

1. **Prenosis Sepsis ImmunoScore / NEJM AI 2025 (doi:10.1056/AIoa2400867).** Publisher returned
   403 repeatedly; not in Europe PMC. **No number from it is verified.** The April 2024 FDA De
   Novo authorization is secondary-sourced only.
2. **Romero-Brufau et al., "Why the C-statistic is not informative to evaluate early warning
   scores", Crit Care 2015;19:285.** Not fetched. The "3–10 alarms per 100 patient-days
   acceptable burden" figure attributed to this line of work is **unverified** and is the number
   most likely to be challenged if we use it.
3. **Shimabukuro et al. 2017 InSight RCT (BMJ Open Respir Res 4:e000234).** Not fetched; effect
   sizes unverified. Frequently over-cited as *the* RCT evidence; it is small and single-centre.
4. **Fleuren et al. 2020 pooled AUROC and pooled sensitivity/specificity.** Springer and PubMed
   both blocked retrieval. The **0.68–0.99 ICU range** comes from snippets; check the PDF.
5. **Epic's formal response to Wong et al.** — I found no peer-reviewed rebuttal. Absence here
   may reflect that it was made through trade press, not that none exists.
6. **The SeFT P-Sepsis table.** I read it via ar5iv [V], but two sources disagreed on the SeFT
   row (B-Acc 70.9 / AUPRC 4.84 vs 74.50 / 8.78) — likely different table rows. **Open the PMLR
   PDF before quoting any specific SeFT number.**
7. **Moor et al. 2023 cohort size.** The published abstract says **136,478 ICU admissions**; the
   arXiv preprint (2107.05230) says **156,309 ICU stays, 26,734 (17.1%) septic**. Use the
   published figure and note the discrepancy if it matters.
8. **MIMIC-IV version used by YAIB.** The paper predates v3.0; the cohort is described as ~73k
   MIMIC-IV stays, consistent with **v2.x**. Our v3.1 cohort is ~94k stays. **Our numbers and
   YAIB's are therefore not on identical data**, which is another reason to reproduce their
   target ourselves.
9. **ricu's `susp_inf` window directions.** The rendered documentation describes
   "antibiotic-first → sampling within 72 h; culture-first → antibiotic within 24 h", which is
   the **mirror image** of the Seymour-2016 / CinC-2019 / mimic-code convention. This may be a
   documentation wording artefact rather than a code difference, but **verify in source** before
   asserting cross-tool equivalence — and if it is real, it is itself reportable.
10. **Our own 93,224 qualifying ICU stays** against v3.1's 94,458 total (see §0).

---

## 9. Suggested citation spine

If the advisor wants the shortest defensible argument, this chain is entirely verified:

**Bauer 2025** (three definitions → κ=0.39, 65% single-criterion, mortality 14.9% vs 2.2%)
→ **Johnson 2018** (same database, incidence 31.9% → 9.0% by method)
→ **Cohen 2024** (within Sepsis-3 alone, definition variance ≥ model variance)
→ **Moor 2021** (2 of 22 studies released label code; AUROC 0.78–0.99 not comparable)
→ **Do 2026** (evaluation strategy alone moves AUROC 0.61↔0.67; matching moves it 0.08)
→ **Wang 2025** (Utility 0.381 → −0.164 externally while AUROC holds at 0.783)
→ **Kamran 2024** (0.62 → 0.47 before treatment)
→ **Wong 2021** (deployed reality: AUC 0.63, PPV 12%, NNE 109, 67% missed)
→ **our contribution**.
