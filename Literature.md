# Literature review — machine learning for sepsis prediction / early warning on MIMIC-IV

**Prepared for advisor review.** Compiled 2026-09-07.

**Scope.** ML studies predicting *sepsis onset* (early warning), on MIMIC-IV primarily, with
MIMIC-III and PhysioNet/CinC 2019 where directly relevant. Deliberately excluded from the main
tables: the much larger literature predicting *mortality or complications among patients who are
already septic*. That distinction turns out to matter a great deal — see §1.5.

**Evidence grading.** Every claim is tagged. This is not decoration; several widely-repeated
numbers in this field are misattributed, and §9 lists the ones I could not stand behind.

| Tag | Meaning |
|---|---|
| **[V]** | I fetched the primary source in this session and read the number there |
| **[V-sub]** | A delegated search agent fetched the primary source and reported reading it |
| **[S]** | Secondary — search snippet, citing paper, or listing page; **not** read at source |
| **[?]** | Could not verify; recorded as a lead only |

Where I found nothing, I say so. **"I searched and found nothing" is not the same as "this is
novel"** — §7 keeps those separate.

---

## 0. Our work, for positioning

Numbers taken from `RESULTS.md` as of 2026-09-07. Note these differ slightly from the brief
I was given (which said Transformer 0.751 and median lead 19.3 h); `RESULTS.md` now reports a
seed-averaged 0.756 and 19.0 h. **Fix the discrepancy before anything is written up.**

| | SepSentinel |
|---|---|
| Task | Per-hour sepsis risk from ICU vitals + labs, hourly grid |
| Dataset | **MIMIC-IV v3.1** |
| Labels | Sepsis-3 with **PhysioNet/CinC 2019 timing rules** — `t_suspicion` from antibiotic–culture pairing; `t_SOFA` = ≥2-point SOFA rise vs prior-24 h minimum; `t_sepsis` = min of the two inside a window. DuckDB port of MIT-LCP mimic-code concepts, deviations enumerated in `sepsis3.DEVIATIONS` |
| Cohort | 93,224 qualifying ICU stays → 63,672–64,236 episodes; 48,150 unique patients |
| Prevalence | 11.5–12.5% patient-level; 2.2–2.6% timestep-level |
| Best results (extended features, pre-onset target, ≤1.0 false alerts/nonseptic patient-day) | Transformer **AUROC 0.756 / AUPRC 0.088**, recall 0.50, patient precision 0.272, median lead **19.0 h**; XGBoost **0.736 / 0.072**, recall 0.64, patient precision 0.225, median lead **20.6 h**; logreg 0.705 / 0.057 |
| Comparators | NEWS2 **0.626**, qSOFA 0.587, SIRS 0.551. SOFA deliberately excluded (label is a SOFA rise) |
| Distinctive findings | (a) post-onset hours inflate AUROC ~0.09 while worsening utility; (b) AUROC anti-correlated with lead time across five manipulations; (c) measurement-frequency-vs-effect-size synthetic control; (d) equal-alert-burden model comparison |

**Two sanity checks to settle before submission.**

1. **Cohort denominator.** MIMIC-IV v3.1 contains **94,458 ICU stays in total** [V-sub, PhysioNet
   v3.1 landing page]. Our "93,224 qualifying ICU stays (age ≥18, LOS ≥6 h)" is **98.7%** of
   that. MIMIC-IV has been adult-only since v2.0 so the age filter removes little, but LOS ≥6 h
   should remove more than 1.3%. A reviewer will check this against the published total.
2. **Alert-burden units.** Our "≤1.0 false alerts per nonseptic patient-day" is directly
   comparable to COMPOSER's **0.029–0.047 false alarms per patient-hour in ICU** [V-sub] — i.e.
   **0.70–1.13 per patient-day**. We are operating at essentially the same burden as the best
   deployed system in the field. **This is a favourable comparison and we are currently not
   making it.** Make it.

---

## 1. Summary table — sepsis onset prediction

**"Post-onset?"** records whether the paper states what it does with hours after `t_sepsis`.
This is the single most consequential undocumented choice in the literature (§2).

### 1.1 MIMIC-IV, hourly onset prediction — the direct comparators

| # | Study | MIMIC-IV version | Cohort / prevalence | Sepsis definition | Target / horizon | Post-onset? | Model | AUROC | AUPRC | Lead time / alert burden | Validation | Code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **Backes J, Tsanda A, Knopp T, Renz W, Schöll E. "Combining machine learning and physiological network models for sepsis prediction." *Front Netw Physiol* 2026.** doi:10.3389/fnetp.2026.1852577 · https://pmc.ncbi.nlm.nih.gov/articles/PMC13327882/ | **v2.2** [V-sub] | 73,181 → **63,425** after filtering; **3,320 septic (5.2%)** | Sepsis-3: suspected/confirmed infection + SOFA increase ≥2 in consecutive assessments | Onset **within 6 h**, continuous | Not explicitly addressed | GRU encoder → coupled-oscillator physiological network (physics-informed hybrid) | **0.841 ± 0.008** | **0.099 ± 0.009** | Not reported | 5×5-fold CV (25 splits), 80/10/10 stratified; patient-grouping [?] | Not stated |
| **2** | **van de Water R, et al. "Yet Another ICU Benchmark" (YAIB), ICLR 2024.** arXiv:2306.05109 | **v2.x** (~73k stays) [V-sub] | ~73k stays; **~1% positive hourly bins** | Sepsis-3 via **ricu `sep3_alt`**; excludes onset <6 h post-admission | **Sepsis within next 6 h**, hourly | **Yes — explicit.** `stop_obs_at(offset = 6 h)` | LR / LGBM / GRU / LSTM / TCN / Transformer | LR **77.1±0.4**, LGBM **77.5±0.3**, **GRU 83.6±0.3** | LR 4.6, LGBM 5.9, **GRU 9.1** | Not reported | Multi-centre (also eICU 77.4, HiRID 80.6, AUMCdb 79.7 for GRU) | **Yes** — https://github.com/rvandewater/YAIB · https://github.com/rvandewater/YAIB-cohorts |
| **3** | **Do DK, Rockenschaub P, Boie SD, et al. *J Med Internet Res* 2026;28:e72083.** doi:10.2196/72083 · https://www.jmir.org/2026/1/e72083 | **v2.0** [V-sub] | **67,056** stays, **3,730 septic (5.6%)**; BerlinICU 40,132 / 4,134 (10.3%) external | Sepsis-3 with **adapted antibiotic-only SI** (no microbiology at Berlin): abx ≥24 h apart for ≥3 days; SOFA rise ≥2 in [−48 h, +24 h] | Onset **within next 6 h**, hourly; horizons swept **1–100 h** | **Onset hour excluded**; the paper's whole subject is evaluation strategy | TCN; L2 logreg baseline | MIMIC-IV internal continuous **0.84 (0.83–0.85)**; BerlinICU continuous **0.67**, fixed-horizon **0.61** | Not extracted | Not reported | 10× repeated 64/16/20 split internally; **full external** on BerlinICU | **Yes** (stated) |
| **4** | **Tranchellini F, Farag Y, Jutzeler C, Meegahapola L. "Evaluating deep learning sepsis prediction models in ICUs under distribution shift." *npj Digit Med* 2026.** doi:10.1038/s41746-026-02364-4 · https://pmc.ncbi.nlm.nih.gov/articles/PMC13066616/ | Not stated [?] | **216,536 stays** across MIMIC-IV + eICU + HiRID; **MIMIC-IV 3,321 septic (5.24%)** | "Strict Sepsis-3: suspicion of infection plus organ dysfunction"; windows [?] | Onset **6 h in advance** | Not stated | CNN, InceptionTime, LSTM + 5 transfer strategies | Per-model absolutes [?]; **domain adaptation gives +3–8% AUROC** | Reports **normalized AUPRC**; +0.2–0.6 nAUPRC from DA | Not reported | Systematic cross-database transfer study | **Yes** — https://gitlab.ethz.ch/BMDSlab/publications/sepsis/transfer-tearning-for-sepsis-detection |
| **5** | **Li J, Xi F, Yu W, Sun C, Wang X. "Real-Time Prediction of Sepsis in Critical Trauma Patients." *JMIR Form Res* 2023;7:e42452.** doi:10.2196/42452 · https://pmc.ncbi.nlm.nih.gov/articles/PMC10131736/ | **v1.0** [V-sub] | **4,603** trauma ICU patients; **1,196 septic (26%)** | **True Sepsis-3**: suspected infection AND acute SOFA rise ≥2; onset = earliest coincidence | Hourly; windows **4, 6, 8, 12, 24 h** | **Yes — explicit. Records after onset excluded** | XGBoost (ensemble-averaged); L2 logreg comparator | **0.83–0.88** across windows | **0.23–0.27** | At 6 h: **sensitivity 73% at 10% precision → 9 false alerts per true alert** | 70/15/15, stratified 5-fold CV; patient-grouping [?] | On request — not public |
| **6** | **Dickens A. "Falsification Testing of Sepsis Prediction Models." medRxiv 2026.** doi:10.64898/2026.03.17.26348414 | **v3.1** [V] | **65,241 adult ICU stays** | Sepsis-3 (clinical) vs administrative coding, compared | Pre-registered falsification study, not an EWS | n/a | GBM-family | Biological features predict Sepsis-3 at **0.901**; minimal loss removing care-intensity features | Not reported | n/a | **Pre-registered on OSF before data access**; replications on eICU v2.0, MIMIC-III v1.4, CinC 2019 | Not stated |
| **7** | **Stylianides C, et al. *IEEE J Biomed Health Inform* 2026.** doi:10.1109/JBHI.2026.3725473 | Not stated [?] | **45,285 patients** | **Not stated in abstract [?]** | **12 h in advance** | **Not stated** | GBM + hybrid LSTM ensemble, 21 features | **0.87** (eICU external 0.88) | **0.88** ⚠️ | Sens 0.79 at spec 0.81 | Split scheme [?] | Not stated |
| **8** | **Dalal S, Khaleghi Ardabili A, Bonavia AS.** medRxiv 2024.11.21.24317716; journal version PMID 40450820 · https://pmc.ncbi.nlm.nih.gov/articles/PMC11601686/ | **v2.2** (accessed 2024-06-10) [V] | 104,767 patients; **4,632 septic (~4.4%)**; eICU 82,486 external | MIMIC-IV: "Sepsis-3 criteria" (implementation unspecified). **eICU: EMR diagnosis-string match for "sepsis"** | **6 / 12 / 24 h** before onset | **Not addressed**; labs "within 6 h prior to the time point of interest" | LSTM encoder–decoder + attention + conformal prediction | **0.99 / 0.98 / 0.96** ⚠️ (near-identical on eICU) | Not reported | 57% reduction in false positives via conformal (6 h) | 80:20 internal + eICU external | On request |
| **9** | **Li Q, Li D, Jiao H, Wu Z, Nie W. "CISepsis: a causal inference framework." *Front Cell Infect Microbiol* 2024;14:1488130.** doi:10.3389/fcimb.2024.1488130 | Not stated [?] | From 53,150 patients / 69,211 admissions; analytic N [?] | **"SOFA ≥ 2 after suspicion of infection" — absolute SOFA, not a ≥2-point rise** [V-sub] | 0–6 h before onset | [?] | Deep learning + causal inference (IV + backdoor) | **0.919–0.926** ⚠️ | **Not reported** | Not reported | **Test set enriched to ~4:1 negative:positive**; grouping [?] | None |
| **10** | **Nie W, et al. "Clinically Interpretable Sepsis Early Warning via LLM-Guided Simulation." arXiv:2604.20924 (2026)** | **v2.2** [V-sub] | **7,021 samples, ~43% positive** (balanced case-control); eICU v2.0 4,488, ~34% | **Onset = first SOFA ≥ 2. No suspicion-of-infection timing described** [V-sub] | 4 / 6 / 8 / 12 / 24 h before onset | [?] | LLM-guided spatiotemporal features + agent post-processing | 0.861 (24 h) → **0.903 (4 h)** | Not reported | Not reported | 7:2:1; grouping [?] | Not stated |
| **11** | **Jin H, Lee H. "Leakage-Aware Federated Learning for ICU Sepsis Early Warning: Fixed Alert-Rate Evaluation on PhysioNet/CinC 2019 and MIMIC-IV." *Applied Sciences* 2026;16(6):2735.** doi:10.3390/app16062735 | Not stated [?] | [?] (MDPI 403) | "Sepsis-3-aligned" for MIMIC-IV; CinC labels for the challenge arm | Hourly early warning | [?] | LSTM, Transformer, XGBoost; centralized vs federated | Values behind 403 [?] | [?] | **Evaluates at a FIXED ALERT RATE (α = 5%)** with stay-level detection rates and lead-time distributions | **Enforces patient-group and temporal separation; includes leakage stress tests** | [?] |
| **12** | **Huang Y, Yang Z, Rahmani A. "MIMIC-Sepsis." arXiv:2510.24500 (2025)** | **v3.1** (repo); version absent from paper [V-sub] | **35,239 ICU patients** | SI = **antibiotics OR positive cultures (disjunction — looser than standard)**; onset = earliest SOFA rise ≥2 with infection | **No sepsis-onset task.** Mortality / LOS / vasopressor / **septic shock onset** | Window fixed at [−24 h, +72 h] around infection onset | Linear, LSTM, Transformer | Shock onset **Transformer 0.925**; IHM 0.863; vasopressor 0.927 | Vasopressor AUPRC 0.870 with treatments | n/a | Random 80/20; **does not state whether patient-grouped** | **Yes** — https://github.com/yongh7/MIMIC-sepsis (MIT) |
| **13** | **Yang J, Karstens L, Ross C, Yala A. "AI Gone Astray: Technical Supplement." arXiv:2203.16452 (2022)** | Not stated [?] | [?] | Replicates **commercial Dascena and Epic feature sets** | Sepsis onset | [?] | RNN | **0.729 → 0.525 over a decade of temporal drift** | Not reported | n/a | **Temporal/calendar-year evaluation** | [?] |
| **14** | **Yamamoto R, Wu F, Sprehe LK, Abeer A, Celi LA, Tohyama T.** medRxiv 2026. doi:10.64898/2026.04.05.26350209 | Not stated [?] | **30,218** septic adults (16.3% mortality) → eICU 31,403 | Sepsis-3 | **Mortality, not onset** — included for its observation-process finding (§6.5) | n/a | LR + GBT, 7 specifications | Internal 0.819→0.834 with measurement counts; **external drop −0.047 without vs −0.082 with** | Not reported | **Calibration slope 1.007 → 0.417** | Internal + external | Not stated |
| **15** | **Zhang M, Zhong M, Cheng Y, Zhang T. *JMIR Med Inform* 2025;13:e74940.** doi:10.2196/74940 | Not stated [?] | **224 patients in test cohort** | Not stated [?] | Real-time, 3-h dynamic temporal features, 8 noninvasive vitals | [?] | Tree ensembles + TreeSHAP | **0.76 (0.74–0.77)** | Not reported | Web platform; no burden numbers | Internal | Platform described |

### 1.2 MIMIC-III and CinC 2019 — the anchors and method precedents

| # | Study | Dataset | Cohort | Sepsis definition | Horizon | Post-onset? | Model | AUROC | AUPRC | Code |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | **Reyna MA, et al. "Early Prediction of Sepsis From Clinical Data: The PhysioNet/CinC Challenge 2019." *Crit Care Med* 2020;48(2):210–217.** doi:10.1097/CCM.0000000000004145 | CinC 2019, 3 hospital systems | >60,000 ICU patients; 40,336 public, 22,761 hidden | **The canonical timing rule** — see §4.1 | Hourly, utility-scored | Post-onset hours **retained**; labels shifted 6 h earlier | 104 teams, 853 entries | **Ranked by utility, not AUROC.** Winner 0.360 | — | Evaluation code yes; **label-generation code never released** |
| 17 | **Cohen SN, et al. *Sci Rep* 2024;14:1920.** doi:10.1038/s41598-024-51989-6 | MIMIC-III | **867–2,178 septic admissions depending on definition** | **Three onset interpretations**: H1 = `t_SOFA`, H2 = `t_suspicion`, H3 = min | 6 h (T swept) | Varies | LGBM, LSTM, CoxPHM | LGBM **0.832 / 0.869 / 0.829**; LSTM 0.805/0.856/0.793; Cox 0.799/0.844/0.780 | Not extracted | **Yes** — https://zenodo.org/record/5168789 |
| 18 | **Rosnati M, Fortuin V. "MGP-AttTCN." *PLOS ONE* 2021;16(5):e0251248.** doi:10.1371/journal.pone.0251248 | MIMIC-III | **7,936 cases with their labels vs 1,797 with Moor's — a >4× difference from labelling alone** | Sepsis-3; **key divergence: unmeasured SOFA contributors assumed *healthy* rather than *unknown*** | 1–6 h | Controls length-matched to cases from admission | MGP-AttTCN | **0.660** at 5 h (MGP-TCN 0.635, InSight 0.490) | **0.483** at 5 h | **Yes** — https://github.com/mmr12/MGP-AttTCN + https://github.com/mmr12/MIMIC-III-sepsis-3-labels |
| 19 | **Moor M, Horn M, Rieck B, Roqueiro D, Borgwardt K. "Early Recognition of Sepsis with MGP-TCN and DTW." MLHC 2019, PMLR 106:2–26.** arXiv:1902.01659 | MIMIC-III | [?] | Sepsis-3 at hourly resolution | 7 h before onset | [?] | MGP-TCN (GP adapter + TCN) | — | **AUPRC 0.25 (prior SOTA) → 0.35 (DTW) → 0.40 (MGP-TCN)** | **Yes** — https://github.com/BorgwardtLab/mgp-tcn (the reference PostgreSQL Sepsis-3 extraction) |
| 20 | **Moor M, Bennett N, Plečko D, et al. *eClinicalMedicine* 2023;62:102124.** doi:10.1016/j.eclinm.2023.102124 | MIMIC-III, eICU, HiRID, AUMCdb (+Salzburg in preprint) | Published: **136,478 admissions / 25,694 septic (18.8%)**; preprint: 156,309 / 26,734 (17.1%) | Sepsis-3 harmonised via **`ricu`**, hourly | Continuous hourly | [?] | Deep self-attention | **0.846 internal / 0.761 external** (0.807 fine-tuned) | **0.177–0.252** across sites [S] | **1.4 false alerts per true alert**; 80% of septic detected **3.7 h (3.0–4.3)** before onset; 39% precision at 80% recall | **Yes** — https://github.com/BorgwardtLab/multicenter-sepsis (Apache-2.0) |
| 21 | **Horn M, Moor M, Bock C, Rieck B, Borgwardt K. "Set Functions for Time Series" (SeFT), ICML 2020.** arXiv:1909.12064 | CinC 2019 | Positive prevalence **1.8%** | CinC labels | **Hourly, sepsis within next 6–12 h** | Per challenge | SeFT-Attn + baselines | SeFT **76.8**; GRU-Simple 78.1; GRU-D 67.4; Phased-LSTM 75.4; Latent-ODE 64.6; **IP-Nets 94.1 ⚠️; Transformer 97.3 ⚠️** | SeFT 4.84; GRU-D 5.33; **IP-Nets 29.4 ⚠️; Transformer 53.4 ⚠️** | **Yes** — https://github.com/BorgwardtLab/Set_Functions_for_Time_Series |
| 22 | **bin Mansoor U, Rashid M, Naqvi R. "A Framework for Early Sepsis Prediction via JEPA and Federated Representation Learning." arXiv:2607.16681 (2026)** | MIMIC-III + CinC 2019 | 40,336 stays | **Unusually explicit**: SI = blood culture + antibiotics within ±24 h; SOFA rise ≥2 | H = 0…10 h | **Post-onset hours RETAINED** (label = 1 for all hours after onset) | TCN under JEPA / VICReg / semi-supervised / supervised | At H0: JEPA+XGB **0.838**; VICReg+XGB 0.850; supervised 0.801 | **0.636 / 0.510 / 0.474** | Not stated |
| 23 | **Chen et al. "SEPRES." *BMC Med Inform Decis Mak* 2022;22:343.** doi:10.1186/s12911-022-02090-3 | MIMIC-III v1.4 + Ruijin Hospital + prospective (67 pts) | 6,891 (1,057 septic vs 5,834) — **1:5.5 case-control** | Sepsis-3 | 1–5 h | [?] | LightGBM, MLP, SVM, XGBoost, LSTM | **0.98 internal ⚠️**; external 0.82–0.86 → **0.94 after transfer**; prospective 0.86–0.90 | Not reported | **Yes** — "SEPRES" |
| 24 | **Tang Y, Zhang Y, Li J. *BMC Med Res Methodol* 2024;24:23.** doi:10.1186/s12874-023-02138-6 | eICU primary, MIMIC external | Train 17,932 with **9,092 septic = 50.7% (balanced by construction)** | **Weak — generic Sepsis-3 wording + "valid diagnostic status at discharge"**, i.e. discharge-diagnosis-based | 4 / 8 / 12 h | **Yes — data taken only before diagnosis timestamp** | CNN-Transformer, LSTM-Transformer | **0.99 at 12 h ⚠️**; MIMIC external ~0.9 | Not reported | On request |
| 25 | **Nemati S, Holder A, Razmi F, et al. "AISE." *Crit Care Med* 2018;46(4):547–553.** doi:10.1097/CCM.0000000000002936 | Emory (>31k) + MIMIC-III (>52k) | [S] | [S] | 4 / 6 / 8 / 12 h | [?] | Modified Weibull-Cox, 65 features | **0.83–0.85** [S] | [?] | [?] |
| 26 | **Shashikumar SP, Josef CS, Sharma A, Nemati S. "DeepAISE." arXiv:1908.04759 / *J Biomed Inform* 2021** | Emory + external | [?] | **Notable**: couples onset criterion with a *treatment policy* and ranks candidate criteria by **offline policy evaluation** | Hourly | [?] | Recurrent neural survival model | **0.90 internal / 0.87 external** | — | **False alarm rates 0.20 / 0.26** | [?] |
| 27 | **Shao X, Li R, Lan Y, et al. *BMC Med Inform Decis Mak* 2026.** doi:10.1186/s12911-026-03610-1 | MIMIC-III | **1,634** after excluding Sepsis-3 within first 24 h; **349 (21.4%)** later septic | Sepsis-3 | Onset **after** the first 24 h | Onset-after-window design (mild prodrome targeting) | 9 models; XGBoost-DART best | **0.881 (0.854–0.908)**; range 0.794–0.881 | Not reported | Not stated |
| 28 | **Khorram B, Kouchaki S. *BMJ Health Care Inform* 2026.** doi:10.1136/bmjhci-2025-101762 | CinC 2019 | 40,336; 2,932 septic (7.3%) | CinC labels | Up to 12 h | Per challenge | LSTM + graph attention | **0.853 ± 0.005** | F1 0.627 | Not stated |
| 29 | **Lauritsen SM, et al. *Artif Intell Med* 2020;104:101820** | Danish multi-hospital EHR | [S] | [S] | 3 h / 10 h | [?] | CNN-LSTM on raw event sequences | **0.856 (3 h) → 0.756 (10 h)** [S] — numbers disputed, see §9 | [?] | [?] |

### 1.3 Papers whose headline numbers I flag as inflated

Full reasoning in §2.4. Flagged: **#7 Stylianides** (AUPRC 0.88 > AUROC 0.87), **#8 Dalal**
(0.99 with identical external numbers), **#9 CISepsis** (4:1 enriched test set), **#10 Nie**
(43% positive, absolute-SOFA label), **#21 SeFT's own baselines** (IP-Nets/Transformer),
**#23 SEPRES** (0.98 internal, 1:5.5 case-control), **#24 Tang** (50.7% positive by
construction, discharge-diagnosis labels), and **Chang et al. arXiv:2603.15651 (AUC 0.956)**,
**KA-Transformer (AUROC 0.962/0.944/0.984 — non-monotonic in horizon)**.

### 1.4 Deployed / prospectively evaluated systems — detail in §5

| # | System | Study | Setting | Cohort | Definition | AUROC | Lead / burden | Design |
|---|---|---|---|---|---|---|---|---|
| 30 | **Epic ESM v1** | Wong et al., *JAMA Intern Med* 2021;181(8):1065–70 | Michigan Medicine | 27,697 patients / **38,455 hospitalizations**; **2,552 sepsis (6.6%)** | CDC surveillance **or** ICD-10 + 2 SIRS + 1 organ dysfunction within 6 h | **0.63 (0.62–0.64)** | Alerts on **18%**; sens 33%, PPV 12%; **NNE 8 (hospitalization) / 109 (4-h horizon)**; missed 67%; median lead **2.5 h** | Retrospective external |
| 31 | **Epic ESM v1** | **Kamran et al., *NEJM AI* 2024;1(3), AIoa2300032** | U. Michigan 2018–20 | **77,582 hospitalizations**, 3,766 sepsis (4.9%) | CDC surveillance + SEP-1 composite | **0.62 → 0.47 before treatment** | — | The key leakage result |
| 32 | **Epic ESM v2** | **Wong et al., *JAMA Netw Open* 2026;9(2):e260181** | **Prospective**, 4 health systems | **227,091 encounters**, 7,401 sepsis (3.3%) | **Sepsis-3** (ΔSOFA ≥2 + suspected infection) | **0.82 / 0.85 / 0.90 / 0.92** by site; **0.75–0.85 at 12-h horizon** | PPV 0.13–0.26 at 60% sens; **NNE 21–35** at 12 h; lead **1.9–10.3 h**; 8-h silencing cut volume without accuracy gain | **Prospective multi-site** |
| 33 | **Epic ESM v1 vs v2** | **Currey D, Tarabichi Y. *Am J Emerg Med* 2025;97:147–151** | ED | 35,076 encounters, **648 (1.8%) Sepsis-3** | Sepsis-3 | **v1 0.77, v2 0.90** → **0.70 / 0.85 restricted to before clinical recognition** | *"Both models tended to alert for sepsis after evidence of clinical recognition"* | Retrospective external |
| 34 | **Epic ESM v2** | Ostermayer et al., *JAMIA Open* 2024;7(4):ooae133 | 2 county EDs, safety-net | 145,885 encounters | — | Not reported | 6-h window: **sens 14.7%, PPV 7.6%**; 4.9% alert rate; **median lead 0 min**; **alerts fired after onset in 50% of positives** | Retrospective external |
| 35 | **Epic ESM** | Cull et al., *Crit Care Explor* 2023;5(7):e0941 | Single academic centre | 11,512 encounters, **1,171 (10.2%) sepsis** | — | **0.834** | Sens 86%, PPV 33.8%; mortality among alerted-not-yet-treated **24.3% → 15.9%**, adj. **OR 0.56 (0.39–0.80)** | Before-after; **an outlier — see §5.1** |
| 36 | **Epic ESM** | Wong et al., *JAMA Netw Open* 2021;4(11):e2135286 | 24 hospitals, 4 systems | — | — | — | COVID onset: alerts/day **953→1,363 (+43%)** while census fell 35%; **% patients alerted/day 9%→21%**; **Michigan paused ESM alerts April 2020** | Dataset-shift study |
| 37 | **COMPOSER** | Shashikumar et al., *npj Digit Med* 2021;4:134 | UCSD + Emory, 6 cohorts | **515,720 encounters**, >6M windows | Sepsis-3 | **ICU 0.925–0.953; ED 0.938–0.945** | **False alarms per patient-hour 0.029–0.047 (ICU) = 0.7–1.1/patient-day**; PPV 24–38% ICU; lead **12.2 h (IQR 3.2–22.8)** before first antibiotics; conformal abstains on 20% non-septic / 8% septic → 75–85% false-alarm reduction | Retrospective + temporal |
| 38 | **COMPOSER (deployed)** | Boussina et al., *npj Digit Med* 2024;7:14 | 2 UCSD EDs | **6,217 septic patients** | — | — | Threshold at 80% sens / **PPV 20.1%**; **~235 alerts/month = ~1.65 per nurse per month**; **mortality −1.9% absolute (17% relative, 95% CI 0.3–3.5)**; **SEP-1 +5.0% absolute**; 72-h SOFA −4% | **Before–after quasi-experimental, Bayesian causal inference** |
| 39 | **COMPOSER-LLM** | Shashikumar et al., *npj Digit Med* 2025 | UCSD ED | **754 prospective encounters** (18.4% septic) | — | — | COMPOSER 70.8% sens / 25.1% PPV / **FAPH 0.034** → COMPOSER-LLM 70.8% / **58.2% PPV** / **FAPH 0.0086**; 62% of remaining FPs had bacterial infections | **Prospective** |
| 40 | **TREWScore** | Henry, Hager, Pronovost, Saria, *Sci Transl Med* 2015;7:299ra122 | MIMIC-II | 13,014 dev / 3,011 val | **Septic shock** (not sepsis) | **0.83 (0.81–0.85)**; MEWS 0.73 | Sens 0.85 at spec 0.67; **median lead 28.2 h (IQR 10.6–94.2)**; 68.8% identified before organ dysfunction | Retrospective |
| 41 | **TREWS** | Adams et al., *Nat Med* 2022;28:1455–1460 | 5 hospitals | **590,736 monitored**; 6,877 sepsis pre-antibiotic | — | — | **Mortality 3.3% absolute / 18.7% relative reduction** when confirmed ≤3 h | **Prospective multi-site cohort** (not randomized) |
| 42 | **TREWS (adoption)** | Henry et al., *Nat Med* 2022;28:1447–1454 | same | 9,805 sepsis cases | — | — | **82% sensitivity; 89% of alerts evaluated; 38% confirmed** (a clinician-adjudicated PPV); **−1.85 h (1.66–2.00) median time to first antibiotics** | Prospective deployment analysis |
| 43 | **Sepsis Watch** | Valan et al., *npj Digit Med* 2025;8:350 | Duke → 4 Summa Health EDs | **205,005 encounters**, 101,584 patients | **Sepsis-2** (2 SIRS + culture order + organ damage); 3.37% in 36 h | Duke internal **0.882**, temporal 0.943; **external 0.906–0.960** | **AUPRC only 0.177–0.252**; at 20% precision **1–27 alerts/day**, lead **3.58–5.07 h** | Retrospective multi-site external |
| 44 | **Sepsis Watch (dev)** | Bedoya et al., *JAMIA Open* 2020;3(2):252–260 | Duke | 42,979 train / 39,786 temporal val | Sepsis-2 | **MGP-RNN 0.88**; RF 0.836; Cox 0.849; SIRS 0.756; **NEWS 0.619; qSOFA 0.481** | Median lead **5 h (IQR 2–20)**; **at a fixed budget of 3 alarms/hour: MGP-RNN captured 10.5 of 17.9 daily cases vs SIRS 5.76, NEWS 3.03, qSOFA 2.21** | Internal + temporal |
| 45 | **InSight** | Desautels et al., *JMIR Med Inform* 2016;4(3):e28 | MIMIC-III v1.3 (Metavision only) | **22,853 ICU stays, 2,577 sepsis (11.3%)** | Sepsis-3-style (ΔSOFA ≥2 + suspected infection); onsets <7 h or >500 h excluded | **0.880 (SD 0.006)** at onset; **0.74 at 4 h** | **AUPRC 0.595**; SIRS 0.609/0.160; qSOFA 0.772/0.277; MEWS 0.803/0.327; SOFA 0.725/0.284 | 4-fold CV | Retrospective |
| 46 | **InSight** | Mao et al., *BMJ Open* 2018;8:e017833 | 684,443 encounters, UCSF + MIMIC-III + 4 hospitals | UCSF 90,353; sepsis 1.30% | **ICD-9 codes** — authors self-acknowledge label leakage | Sepsis **0.92**; severe sepsis 0.87; **septic shock 0.9992 ⚠️** | — | — | Retrospective multi-site |
| 47 | **Penn severe-sepsis EWS** | Giannini et al., *Crit Care Med* 2019 | Penn Medicine | — | severe sepsis | — | Sens **26%**, spec 98%, **PPV 29%**; ↑lactate testing and fluids; **no mortality or ICU-transfer difference** | Prospective silent-then-live |

### 1.5 What actually dominates the "MIMIC-IV sepsis ML" literature

When I searched Europe PMC for `sepsis AND MIMIC-IV AND (early prediction OR early warning OR
onset prediction)` sorted by citations, **all 13 top results were about mortality,
complications, or prognosis among already-septic patients** [V]. The delegated agent found the
same pattern and catalogued ~17 such papers (mortality, AKI, delirium, encephalopathy,
coagulopathy) [V-sub].

**Consequence for the write-up:** the apparently vast "sepsis ML on MIMIC-IV" literature is
mostly not our task. The genuinely comparable set is **small**: Backes, YAIB, Do, Tranchellini,
and Li (trauma). Everything else in §1.1 has an unstated definition, an unstated post-onset
policy, or a cohort construction that makes the number incomparable.

**Most important structural finding, from the delegated survey [V-sub]:** *no MIMIC-IV v3.0 or
v3.1 sepsis-**onset** early-warning paper was found.* v3.0/v3.1 appear only in prognosis papers
(Han 2025, Yu 2026, Yang PCM 2025, Tripathi 2026) and in the Dickens falsification study. The
onset-prediction literature is on **v1.0 (Li 2023), v2.0 (Do 2026), v2.2 (Backes 2026, Nie 2026,
Dalal 2024)**. **Our v3.1 onset work appears to be first.** See §7.3 for why that matters and how
much weight it will bear.

---

## 2. The comparability problem

### 2.1 The published range

| Source | Scope | AUROC range |
|---|---|---|
| Fleuren et al., *Intensive Care Med* 2020;46:383–400 [V-sub] | 28 papers, 130 models | **ICU 0.68–0.99**; wards 0.96–0.98; ED 0.87–0.97 |
| Moor et al., *Front Med* 2021;8:607952 [V-sub] | 22 ICU studies | **0.78–0.99** |
| Wang et al., *npj Digit Med* 2025;8:190 [V] | 91 studies | median **0.886** (6 h, partial-window internal), **0.861** (12 h), **0.783** (full-window external) |
| Shanmugam et al., *Indian J Crit Care Med* 2025;29(6) [V] | 13 studies | **0.753–0.99** |
| González Garcés et al., *Front Digit Health* 2026;8:1794922 [V] | 37 studies | ML **0.81–0.99**; DL up to 0.93 |
| **This review's own tally of MIMIC-IV onset papers** | 15 papers in §1.1 | **0.61 – 0.99** |

**A ~0.3–0.4 AUROC spread on nominally the same task.** Our 0.756 sits at the bottom of every
published range. That is the fact the advisor needs, and the rest of §2 explains why it is not a
statement about our model.

**The credible band.** Restricting to studies that are unbalanced, patient-grouped where stated,
and continuously evaluated, the honest range is **AUROC 0.76–0.85 with AUPRC 0.10–0.30**.
Anchors, all verified:

| Study | AUROC | AUPRC | Prevalence |
|---|---|---|---|
| **Backes 2026 (MIMIC-IV v2.2)** | **0.841** | **0.099** | 5.2% patient-level |
| YAIB GRU (MIMIC-IV) | 0.836 | 0.091 | ~1% hourly |
| YAIB LGBM (MIMIC-IV) | 0.775 | 0.059 | ~1% hourly |
| Moor 2023 (5 databases) | 0.846 int / 0.761 ext | 0.177–0.252 | 17–19% |
| Li 2023 (MIMIC-IV trauma) | 0.83–0.88 | 0.23–0.27 | 26% |
| Do 2026 (MIMIC-IV → Berlin) | 0.84 int / 0.67 ext | — | 5.6% |
| Rosnati 2021 (MIMIC-III, 5 h) | 0.660 | 0.483 | case-control |
| **SepSentinel (MIMIC-IV v3.1)** | **0.756** | **0.088** | 11.5% patient / 2.6% timestep |

**Backes et al. is the single most useful comparator we have** — same database family, same
version era, honest unbalanced evaluation, and an AUPRC of 0.099 that is almost exactly ours
(0.088) at roughly double our timestep prevalence. **Put this in the paper.**

### 2.2 What drives the range — with published magnitudes

Each row is a *measured* effect, not speculation.

| Driver | Magnitude | Source |
|---|---|---|
| **Including post-treatment hours** | AUROC **0.62 → 0.47** | Kamran et al., *NEJM AI* 2024 [V] |
| **Including post-clinical-recognition hours** | ESM v1 **0.77 → 0.70**; ESM v2 **0.90 → 0.85** | Currey & Tarabichi, *Am J Emerg Med* 2025 [V-sub] |
| **Evaluation strategy** (continuous / fixed-horizon / peak) | **0.67 vs 0.61** on identical model+data; onset matching moves peak-score **+0.08**, fixed-horizon **−0.14** | Do et al., *JMIR* 2026 [V] |
| **Partial- vs full-window validation** | median **0.886 → 0.783**; only **54.9% of 91 studies** used full-window | Wang et al., *npj Digit Med* 2025 [V] |
| **Onset-time definition within Sepsis-3** | **0–6% AUROC** | Cohen et al., *Sci Rep* 2024 [V-sub] |
| **Model class** | **1–5% AUROC** — *smaller than the definition effect* | Cohen et al. 2024 [V-sub] |
| **Missingness assumption on unmeasured SOFA components** | **cohort size 1,797 → 7,936 (>4×) in the same database** | Rosnati & Fortuin 2021 [V-sub] |
| **Sepsis definition family** (Sepsis-3 / SEP-1 / ASE) | **0.85 – 0.94** for one model on one cohort | Dutta et al., *JAMA Netw Open* 2026 [V] |
| **Prediction horizon** | 0.886 (6 h) → 0.861 (12 h) | Wang et al. 2025 [V] |
| **Site shift, label harmonised** | **0.846 → 0.761** | Moor et al. 2023 [V] |
| **Site shift, MIMIC-IV → BerlinICU** | **0.84 → 0.67** | Do et al. 2026 [V] |
| **Temporal drift over a decade** | **0.729 → 0.525** | Yang et al., "AI Gone Astray" [V-sub] |
| **Measurement-count features** | +0.015 internal; external drop doubles (−0.047 → −0.082); calibration slope 1.007 → 0.417 | Yamamoto et al. 2026 [V] |
| **Class balancing** | Balanced test sets in §1.3 report 0.92–0.99; unbalanced ones 0.76–0.88 | This review's tally |

**Cohen et al.'s formulation is the one to quote:** *the best-performing model on the worst
definition had lower AUROC than the worst model on the best definition* [V-sub]. YAIB reaches
the same conclusion independently: *"the choice of dataset, cohort definition, and preprocessing
have a major impact on the prediction performance, often more so than model class"* [V-sub].

### 2.3 How many papers state their post-onset truncation, window, and thresholds?

**The field does not report this systematically, and no review counts it.** What I can establish:

**Post-onset truncation.** No systematic review tabulates it. Closest proxy: Wang et al. 2025 —
**only 54.9% of 91 studies used full-window validation; the rest used partial-window approaches
that exclude negative cases and inflate performance** [V]. From my own reading of the 29 onset
papers in §1:

- **Explicitly truncate:** Li 2023 (records after onset excluded), Tang 2024 (data only before
  the diagnosis timestamp), YAIB (features stopped 6 h before onset) — **3**.
- **Explicitly retain:** bin Mansoor 2026 (label = 1 for all post-onset hours), CinC 2019
  (post-onset hours retained, records truncated shortly after) — **2**.
- **Partial:** Do 2026 excludes only the onset hour itself — **1**.
- **Not stated at all:** the remaining **~23**.

**Prediction window.** Better reported — horizons 3–24 h are near-universally stated [V]. But
window *semantics* differ. Do et al. note the horizon "only determines which time points before
onset are labeled as positive, but does **not** restrict the range of time points included in the
evaluation" [V]. Two papers can both say "6 hours" and mean different things.

**Fixed-threshold / operating-point evaluation.** Rare. Shanmugam et al.: **only 2 of 13**
studies reported anything like alert burden [V]. Wang et al.: **only 2 of 91** had prospective
validation [V]. Among §1.1–1.2, only Moor 2023, Li 2023, DeepAISE, and Jin & Lee 2026 report an
alert-burden figure.

**Case–control matching.** Moor et al. 2021: **72.7% of studies gave no matching methodology at
all** [V-sub] — and Do et al. show matching moves AUROC by up to 0.08 [V].

**Patient-grouped splitting.** Almost never explicitly reported. **Jin & Lee 2026 is the only
paper I found that makes leakage-aware group and temporal separation its central contribution**;
MIMIC-Sepsis explicitly does *not* state whether its 80/20 split is grouped [V-sub].

### 2.4 Papers whose headline numbers are likely inflated — with reasons

Not allegations of error; statements that the number is not comparable to a per-hour,
post-onset-truncated, patient-grouped, unbalanced evaluation.

| Study | Headline | Why |
|---|---|---|
| **Dalal et al. 2024** (MIMIC-IV v2.2) | **0.99 / 0.98 / 0.96** at 6/12/24 h | Three flags. (a) The **24-h number (0.96) exceeds most papers' at-onset numbers**; discrimination should *fall* with horizon. (b) **External eICU numbers are identical to internal MIMIC-IV numbers to two decimals at all three horizons** — genuine external validation essentially never does this (cf. Moor 0.846→0.761, Do 0.84→0.67). (c) The eICU label is a **diagnosis-string match for "sepsis"** with no onset timestamp, while the model uses "laboratory data within 6 h prior to the time point of interest" [V]. Consistent with per-patient classification containing post-onset labs. |
| **Stylianides et al. 2026** (MIMIC-IV, 45,285 patients) | AUROC 0.87, **AUPRC 0.88** | **AUPRC > AUROC is only possible at high positive prevalence** — AUPRC's no-skill baseline *is* the prevalence. At the ~1–5% prevalence of this task, 0.88 is unattainable. Implies a **balanced or heavily undersampled evaluation set**. Compare Backes on the same database: **AUPRC 0.099** at 5.2%. |
| **CISepsis (Li Q et al. 2024)** | 0.919–0.926 | **Test set enriched to ~4:1 negative:positive** [V-sub]. Also the label is **absolute SOFA ≥ 2**, not a ≥2-point rise — a materially weaker and easier target. AUPRC not reported. |
| **Nie et al. 2026** | 0.861–0.903 | **43% positive** by construction. Onset defined as **first SOFA ≥ 2 with no suspicion-of-infection timing at all** [V-sub]. |
| **Tang et al. 2024** | 0.99 at 12 h | **50.7% positive** by construction; labels effectively **discharge-diagnosis-based** [V-sub]. |
| **SEPRES (Chen et al. 2022)** | 0.98 internal on MIMIC-III | 1:5.5 case-control design. Note the external number drops to 0.82–0.86 — the internal figure is the artefact. |
| **Chang et al. arXiv:2603.15651** | AUC **0.956** | Far above every rigorously-evaluated study. No cohort, definition, horizon, or split scheme stated in the abstract [V-sub]. |
| **KA-Transformer 2025** | 0.962 (1 h) / 0.944 (6 h) / **0.984 (12 h)** | **Non-monotonic in horizon** — 12 h better than 1 h. A strong signature of a case-control/window-selection artefact rather than predictive skill [V-sub]. |
| **Mao et al. 2018** | septic shock **0.9992** | ICD-9 labels; the authors themselves write that *"laboratory tests are contingent on physician suspicion"* so *"timing of these inputs may reflect clinician judgement rather than true onset time"* [V-sub]. Treat as a leakage artefact the authors flagged. |
| **SeFT's own baseline table (ICML 2020)** | **IP-Nets 0.941 / AUPRC 29.4; Transformer 0.973 / AUPRC 53.4** on CinC-2019 at 1.8% prevalence | I flag this **against the authors' own proposed method** (SeFT-Attn is 0.768 / 4.84). An AUPRC of 53.4 at 1.8% prevalence is far outside everything published on that dataset, including the challenge winners. Something in those two baseline pipelines is almost certainly leaking. **[V]** — I read the table; the interpretation is mine, and to my knowledge it has never been remarked on. |
| **Epic ESM, vendor-reported** | 0.76–0.83 | Externally **0.63** (Wong), **0.47** before treatment (Kamran). The canonical case. |

**And, for symmetry, one likely-deflated case: ours.** Our own experiments show the +3 h vs +24 h
truncation choice alone moves AUROC 0.630 → 0.718. A reviewer comparing 0.756 against a table of
0.9s will draw the wrong conclusion unless §2 is in the paper.

### 2.5 The AUROC-vs-lead-time question specifically

**What is published:**
- **Horizon → AUROC decline** is documented: 0.886 (6 h) → 0.861 (12 h) across 91 studies [V];
  Epic ESM v2 0.82–0.92 at onset → **0.75–0.85 at a 12-h horizon** [V-sub]; InSight 0.880 at
  onset → **0.74 at 4 h** [V-sub]; Lauritsen 0.856 (3 h) → 0.756 (10 h) [S]; a JEPA sepsis model
  16.8% AUROC drop from H0 to H10 [S].
- **Do et al. 2026 state the mechanism**: continuous-evaluation performance *increases as the
  prediction window narrows*, because "the assigned sepsis labels align more closely with the
  sepsis-related signals in the patient data" [V].
- **Dutta et al. 2026** show a definition giving higher AUROC (SEP-1, 0.94) giving *lower* AUPRC
  (0.16) and lower PPV — discrimination and utility pulling apart [V].
- **Epic ESM v2** shows AUROC 0.82–0.92 alongside **median lead times of 1.9–10.3 h** across
  sites, i.e. the sites with the best AUROC do not have the best lead time [V-sub].

**What I did not find:** a paper that (a) names the AUROC–lead-time relationship as a property
that must be reported alongside AUROC, (b) demonstrates it across multiple independent
manipulations of one pipeline, or (c) demonstrates it with a synthetic control holding signal
magnitude exactly. Two delegated agents reached the same conclusion independently: *"the
trade-off is empirically well-documented but I did not find a paper that states it as a formal,
named principle."* **Three independent searches; none found it. Reasonably strong, but a
negative.** See §7.

---

## 3. Sepsis definition landscape

### 3.1 The taxonomy and its critiques

| Family | What it is | Critique | Citation |
|---|---|---|---|
| **Sepsis-3 (clinical)** | SOFA rise ≥2 attributable to infection. **Defines no onset timestamp** — the root of the comparability problem | Every implementation must invent an operational SI rule | Singer et al., *JAMA* 2016;315:801 [S] |
| **Sepsis-3 + Seymour SOI timing** | SI = antibiotics + cultures; abx-first → culture within 24 h; culture-first → abx within 72 h; SOI time = earlier event | Asymmetric window; pipelines silently re-symmetrise | Seymour et al., *JAMA* 2016;315:762 [V-sub] |
| **PhysioNet/CinC 2019** | Seymour SOI **plus** abx ≥72 consecutive h; `t_SOFA` = 2-point *deterioration* within 24 h; validity window **[t_susp−24 h, t_susp+12 h]**; labels shifted 6 h earlier | Label-generation code never released | Reyna et al., *Crit Care Med* 2020 [V] |
| **mimic-code `sepsis3`** | SOI as above, **but** organ dysfunction = **absolute `sofa_24hours ≥ 2`, baseline assumed zero**; window **[SOI−48 h, SOI+24 h]**; **one row per ICU stay** | Not a rise, not hourly, different window from CinC. §3.3 | https://github.com/MIT-LCP/mimic-code [V] |
| **CDC Adult Sepsis Event / eSOFA** | Blood culture + ≥4 qualifying antibiotic days + simplified organ dysfunction | Smaller, sicker cohort; validation sensitivity vs Sepsis-3 ranges **50.8%–91.6%** across studies | Rhee et al., *Crit Care Med* 2019 [V-sub] |
| **SEP-1 bundle** | CMS quality measure | Highest AUROC but lowest AUPRC and PPV head-to-head | Dutta et al. 2026 [V] |
| **Angus criteria** | 122 ICD-9 codes (109 infection + 13 organ dysfunction) | **PPV 70.7%, sensitivity 50.4%** vs chart review | Iwashyna et al., *Med Care* 2014 [V-sub] |
| **Martin criteria** | 6 septicaemia/bacteraemia codes | **Sens 27.6%, PPV 78.2%** | Jolley et al., *Crit Care* 2016 [V-sub] |
| **Explicit ICD-9/10 sepsis codes** | Direct codes | **Sensitivity 32.3% vs 69.7% clinical**; incidence trend **+10.3%/y from coding drift** while clinical incidence was flat | Rhee et al., *JAMA* 2017;318:1241 [V-sub] |

### 3.2 How many papers use which — the honest count

**No systematic review breaks this down for MIMIC-IV specifically.** What exists:

- **Moor et al. 2021** (22 ICU studies, the only clean breakdown found): **Sepsis-2 in 12
  (54.5%), Sepsis-3 in 9 (40.9%), expert clinician labels in 1** [V-sub]. Pre-2021 sample; the
  balance has shifted toward Sepsis-3 since.
- **Wang et al. 2025** (91 studies): **58.2% used PhysioNet/CinC 2019 data** [V] — so the single
  most common "sepsis label" in the field is the CinC label, **inherited rather than computed**.
- **González Garcés et al. 2026** (37 studies): "considerable heterogeneity"; **no counts given** [V].

**In this review's own 29 onset papers**, the distinct verified variants are:

| Variant | Papers | Count |
|---|---|---|
| **True Sepsis-3 with explicit SI window (abx + cultures) + SOFA rise ≥2** | Li 2023 (#5), bin Mansoor (#22), Backes (#1) | 3 |
| **Sepsis-3 with an *adapted* antibiotic-only SI rule (no microbiology)** | Do 2026 (#3) | 1 |
| **Sepsis-3 via `ricu`** | Moor 2023 (#20), YAIB (#2) | 2 |
| **SI = antibiotics OR cultures (disjunction — looser)** | MIMIC-Sepsis (#12) | 1 |
| **Absolute SOFA ≥ 2, no rise** | CISepsis (#9), Nie (#10) | 2 |
| **Discharge diagnosis / ICD codes** | Tang (#24), Mao (#46), Yang PCM | 3 |
| **CinC-2019-provided labels** | Reyna (#16), SeFT (#21), Khorram (#28), Jin & Lee CinC arm (#11) | 4 |
| **Sepsis-2** | Sepsis Watch (#43, #44) | 2 |
| **Not stated precisely enough to classify** | Stylianides (#7), Dalal MIMIC arm (#8), Jin & Lee MIMIC arm (#11), Tranchellini (#4), Zhang JMIR (#15), AI Gone Astray (#13), CSRA, SepsisSuite, and others | **~9+** |

**Roughly a third of onset-prediction papers cannot be classified by sepsis definition from
their abstracts.** That is the headline for this section.

### 3.3 The critique that matters most for us: "Sepsis-3" names several different labels

Verified directly from source files [V]:

- **mimic-code `sepsis3.sql`** joins SOFA to SOI on
  `sofa.endtime BETWEEN suspected_infection_time − 48 h AND suspected_infection_time + 24 h`,
  filters **`sofa_24hours >= 2` (absolute)**, and returns **`WHERE rn_sus = 1` — one row per
  `stay_id`**. Code comments state **baseline SOFA is assumed zero** on ICU admission. Its
  `suspicion_of_infection.sql` implements culture ≤72 h before antibiotics OR ≤24 h after, with
  `suspected_infection_time = COALESCE(last72_charttime, antibiotic_time)`.
- **`sofa.sql` IS hourly** — built on `icustay_hourly` with `ROWS BETWEEN 23 PRECEDING AND 0
  FOLLOWING`. So mimic-code gives you an hourly SOFA series but a per-stay sepsis flag.
- **PhysioNet/CinC 2019** requires `t_SOFA ∈ [t_suspicion − 24 h, t_suspicion + 12 h]`, defines
  `t_SOFA` as a **2-point deterioration**, and requires **≥72 consecutive hours of antibiotics**.
- **ricu `sep3()`** defaults to `si_lwr = 48 h`, `si_upr = 24 h` (matching mimic-code, not CinC)
  but uses a genuine **delta** criterion (`delta_fun = delta_cummin`, rise vs running minimum),
  configurable to `delta_start` or `delta_min` — the only public tool offering the choice
  [V-sub].

**Three tools, all called "Sepsis-3", with different windows (−48/+24 vs −24/+12), different
organ-dysfunction semantics (absolute ≥2 from an assumed-zero baseline vs a 2-point rise), and
different antibiotic-duration requirements (none vs ≥72 h).** Cohen et al. quantified the cost of
the *onset-time* choice alone at 0–6% AUROC; nobody has quantified the cost of the window and
delta-semantics choices.

### 3.4 The strongest empirical citations for "the definition determines the cohort"

- **Johnson et al., *Crit Care Med* 2018;46(4):494–499** [V-sub]: in a *single* database, sepsis
  incidence ranged **31.9% (CDC method) → 9.0% (explicit ICD coding)** — a 3.5× swing. This is
  the paper `mimic-code`'s `sepsis3` is based on.
- **Bauer et al., *Crit Care* 2025;29:523** [V-sub]: 25 hospitals. CDC ASE **139,267** vs
  Sepsis-3 **234,601** vs ICD **64,461** patients. **κ = 0.39, Jaccard = 0.31**. **65% of cases
  identified by only one criterion.** Mortality 14.9% for cases meeting both, **2.2% for
  Sepsis-3-only**.
- **Dickens, medRxiv 2026** [V]: on **MIMIC-IV v3.1**, mean **Jaccard ≈ 0.32** between clinical
  and administrative definitions at the primary site, **0.20** across multi-centre cohorts.
- **Rosnati & Fortuin 2021** [V-sub]: **7,936 vs 1,797 cases** from the same MIMIC-III database
  purely from a missingness assumption about unmeasured SOFA components.
- **Rhee et al., *JAMA* 2017** [V-sub]: ICD sensitivity **32.3%** vs clinical **69.7%**;
  claims-based incidence rising **10.3%/y** while clinical incidence was flat.
- **PaO₂/FiO₂ imputation choice alone**: incidence 14.2% → 19.7%, mortality 10.1% → 7.8% [S].

**On ICD timing specifically:** ICD codes are assigned at the *hospitalization* level and carry
**no onset timestamp at all**. The critique is structural, not a matter of degree. I looked for
and **could not find** a study quantifying the timing offset in hours between ICD-coded sepsis
and Sepsis-3 onset — because the quantity is largely undefined. **State it that way rather than
as "ICD timing is poor by N hours".** Relatedly, I found **no paper reporting an agreement
statistic between ICD-coded sepsis and the `sepsis3` derived table within MIMIC-IV** — Dickens's
Jaccard 0.32 is the closest, and it is a preprint. This looks like a genuine, small, easily
fillable gap.

---

## 4. The CinC 2019 label, in full

Because our whole labelling approach descends from it, and because §3.3 shows it is *not* what
mimic-code implements. Verified from the PhysioNet dataset page [V]:

- **`t_suspicion`** = earlier of IV antibiotics and blood culture, where **antibiotics must be
  given for ≥72 consecutive hours**; cultures within **24 h if antibiotics first**, within
  **72 h if cultures first**.
- **`t_SOFA`** = a **≥2-point SOFA deterioration within a 24-hour period**.
- **`t_sepsis`**: if **`t_suspicion − 24 h ≤ t_SOFA ≤ t_suspicion + 12 h`**, then
  **`t_sepsis = min(t_suspicion, t_SOFA)`**; otherwise the patient is **not** septic.
- **`SepsisLabel` = 1 for all t ≥ `t_sepsis − 6 h`** — a built-in 6-hour prodrome shift.
- **Utility score**: max reward **1.0** in **[t_sepsis − 12 h, t_sepsis + 3 h]**; **−0.05** for
  predictions >12 h early; **−2.0** late/missed; **−0.05** per false alarm on non-septic; **0**
  for true negatives. Normalised so perfect = 1.0, no-predictions = 0.0.
- Data: **40,336 public subjects** (Set A 20,336 Beth Israel, Set B 20,000 Emory), hidden Set C
  from a third system; 40 variables, hourly.
- Winner: **"Can I get your signature?"** (Morrill et al., signature methods), **utility 0.360**
  on the full hidden test set. Top five: Sepsyd (GBDT), Separatrix (XGBoost ensemble),
  FlyingBubble (TASP time-phased), CTL-Team (*"Utilizing Informative Missingness"*).
- Reyna et al.'s own conclusion: *"generalizability to different hospital systems remains a
  challenge"* — utility correlated between the two training hospitals but poorly with the third
  [V].
- Post-hoc: **Reyna & Clifford, arXiv:2012.11013** — ensembling 70 challenge algorithms
  outperforms individuals *"especially on a hidden test set on which most algorithms failed to
  generalize"* [V-sub].

**Two things worth noting for our paper.** First, the CinC label already contains a 6-hour
prodrome shift and a utility function that rewards only a bounded pre-onset window — so the
*idea* of a pre-onset target is baked into the most-used benchmark in the field (see §7.2b).
Second, the challenge's own headline metric is **not AUROC**, which is why CinC numbers and
MIMIC AUROCs are frequently and wrongly compared.

---

## 5. Deployment-oriented work

### 5.1 The Epic Sepsis Model arc — the field's cautionary tale, now in five acts

**Act 1 — Wong et al., *JAMA Intern Med* 2021;181(8):1065–1070.** doi:10.1001/jamainternmed.2021.2626 ·
https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2781307 [V-sub, full text]

27,697 patients / **38,455 hospitalizations**, Michigan Medicine, Dec 2018–Oct 2019; **2,552
sepsis (6.6%)**. Sepsis label was itself a composite (CDC surveillance **or** ICD-10 + 2 SIRS +
1 organ dysfunction within 6 h) — even the paper that indicted Epic used a hybrid definition.

- **AUROC 0.63 (0.62–0.64)** vs vendor-claimed 0.76–0.83
- By horizon: 24 h 0.72, 12 h 0.73, 8 h 0.74, 4 h 0.76
- At threshold ≥6: **sensitivity 33%, specificity 83%, PPV 12%, NPV 95%**
- Alerts on **6,971 / 38,455 (18%)**; **missed 1,709 of 2,552 (67%)**; median lead **2.5 h**
- **NNE 8 at hospitalization level; NNE 109 at a 4-hour prospective horizon.** *These two numbers
  are routinely confused in citing literature — the 109 is the operationally meaningful one and
  should be quoted with its framing.*
- Only **183 of 2,552 (7%)** of sepsis cases identified were ones clinicians had not already
  treated in time.

**Editorial** — Habib AR, Lin AL, Grant RW, *JAMA Intern Med* 2021;181(8):1040–1041,
doi:10.1001/jamainternmed.2021.3333, open PDF at https://gwern.net/doc/ai/tabular/2021-habib.pdf
[V-sub]. Source of the widely-cited "Epic claimed 0.76–0.83". Proposes a rule of thumb that
*"models with poor combined specificity and sensitivity, defined as less than 1.5… must be
incorporated into care with caution"* — ESM sums to **1.16**. Calls for *"pragmatic randomized
clinical trials."*

**Act 2 — Kamran et al., *NEJM AI* 2024;1(3), AIoa2300032.** doi:10.1056/AIoa2300032 · open
accepted manuscript at https://par.nsf.gov/biblio/10522981 [V]

**The single most important paper for our §2 argument.** 77,582 hospitalizations, U. Michigan
2018–2020, 3,766 sepsis (4.9%). Re-evaluated the ESM restricting predictions to **before any
indication of treatment** (antibiotics, fluids, blood culture, or lactate).

> **AUROC 0.62 (0.61–0.63) conventionally → 0.47 (0.46–0.48) before treatment.** Below chance.

Their framing: *"an alert arriving 3 hours before the patient becomes overtly septic with organ
dysfunction but 2 hours after clinicians have initiated antibiotics is of little value."*

**Act 3 — dataset shift.** Wong A, Cao J, Lyons PG, Dutta S, Major VJ, Ötleş E, Singh K,
*JAMA Netw Open* 2021;4(11):e2135286 [V-sub]. 24 hospitals, 4 health systems, around each
system's first COVID case: alerts/day **953 → 1,363 (+43%)** while census fell **35%**;
**proportion of patients alerting per day 9% → 21%**. **University of Michigan paused ESM alerts
entirely in April 2020** after nursing reports of overalerting. The best primary source in the
field for "distribution shift blows up alert burden in deployment."

**Act 4 — the revised model.** **Wong A, et al., *JAMA Netw Open* 2026;9(2):e260181,**
doi:10.1001/jamanetworkopen.2026.0181 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12949446/
[V-sub]. **Prospective** validation of ESM v2 at 4 health systems (Michigan, OHSU, Emory,
MetroHealth), Aug 2023–Mar 2025, **227,091 encounters, 7,401 sepsis (3.3%)**, Sepsis-3.

- **AUROC 0.82 / 0.85 / 0.90 / 0.92 by site**; **0.75–0.85 at a 12-h horizon**
- At standardised 60% sensitivity: specificity 0.83–0.96, **PPV 0.13–0.26**
- **NNE 21–35** at the 12-h horizon; median lead **1.9–10.3 h** (enormous inter-site variability)
- Alerts heavily front-loaded post-admission; an **8-hour silencing strategy cut volume without
  improving accuracy**
- Conclusion: *"improved discrimination… but high institutional variability, low positive
  predictive value, and high alert burden."*

**Act 5 — and the leakage persists.** **Currey D, Tarabichi Y, *Am J Emerg Med* 2025;97:147–151,**
doi:10.1016/j.ajem.2025.07.056 [V-sub]. 35,076 ED encounters, 648 (1.8%) Sepsis-3.
**v1 = 0.77, v2 = 0.90** — but restricted to scores **before evidence of clinical recognition**,
**v1 = 0.70, v2 = 0.85**. *"Both models tended to alert for sepsis after evidence of clinical
recognition."* **This is an independent replication of Kamran's finding on the improved model,
and it is the most important methodological caveat for anyone benchmarking against Epic.**

**The dissenting datapoint.** Cull J, et al., *Crit Care Explor* 2023;5(7):e0941 [V-sub] — a rare
*positive* ESM study. Before-after at a single 746-bed academic centre, 11,512 encounters, 10.2%
sepsis. **AUC 0.834**, sens 86%, PPV 33.8%; mortality among alerted patients not yet on
appropriate antibiotics **24.3% → 15.9%**, adjusted **OR 0.56 (0.39–0.80)**. Treat as an outlier
and say why: single-centre, before-after, and a sepsis prevalence (10.2%) two to three times
every other ESM validation.

**Two further external validations.** Ostermayer et al., *JAMIA Open* 2024;7(4):ooae133 [V-sub]:
145,885 ED encounters at two safety-net county EDs, ESPM v1. Within a 6-hour window,
**sensitivity 14.7%, PPV 7.6%**; **median lead time 0 minutes**; **alerts fired *after* sepsis
onset in 50% of positive cases.**

**Regulatory status:** the Epic Sepsis Model has **never been FDA-submitted or reviewed** — it is
deployed at hundreds of hospitals as unregulated clinical decision support [S, converging
sources; the absence of an FDA record is itself the point]. Contrast with §5.6.

**I found no peer-reviewed Epic rebuttal.** Epic's response to Wong et al. was made through a
company blog post and trade press (STAT, 2021-09-27 and 2022-10-24), arguing the authors used a
*"hypothetical configuration"* at a threshold appropriate for a rapid-response team rather than a
tuned deployment [S, press paraphrase only; I could not retrieve Epic's own text]. The same STAT
investigation reported that **ESM used whether a clinician had already ordered antibiotics as an
input** — the key label-leakage critique — which I flag as **[S] and would not cite without
confirmation.**

### 5.2 COMPOSER — the strongest deployment evidence, and our closest burden comparator

**Note the title.** The frequently-cited *"COMPOSER: a deep learning model for sepsis prediction
with conformal prediction"* **does not exist**. The 2021 paper is:

**Shashikumar SP, Wardi G, Malhotra A, Nemati S. "Artificial intelligence sepsis prediction
algorithm learns to say 'I don't know'." *npj Digit Med* 2021;4:134.**
doi:10.1038/s41746-021-00504-6 · https://pmc.ncbi.nlm.nih.gov/articles/PMC8429719/ [V-sub].
COMPOSER = **CO**nformal **M**ultidimensional **P**rediction **O**f **SE**psis **R**isk.

Retrospective, 2 health systems, **six cohorts, 515,720 encounters**, 2016–2020, ICU + ED.

| | ICU | ED |
|---|---|---|
| AUROC | **0.925–0.953** | **0.938–0.945** |
| Sensitivity | 78.9–92.3% | 70.5–96.0% |
| Specificity | 90.7–93.6% | 92.4–94.7% |
| **PPV** | 24.3–38.0% | 13.4–20.8% |
| **False alarms per patient-hour** | **0.029–0.047** | 0.038–0.042 |
| Lead time before first antibiotics | **12.2 h (IQR 3.2–22.8)** | 2.1 h (IQR 0.8–4.5) |

Conformal prediction flags **20% of non-septic and 8% of septic windows as indeterminate**,
yielding a **75–85% reduction in false alarms** at matched sensitivity. Baselines included
Epic's ESPM.

**The false-alarms-per-patient-hour metric is the best-specified alert-burden metric in the
sepsis ML literature, and it converts directly to ours: 0.029–0.047/patient-hour =
0.70–1.13 per patient-day, against our ≤1.0. We are at the same operating burden as COMPOSER.**

**Deployment.** Boussina A, Shashikumar SP, Malhotra A, et al., *npj Digit Med* 2024;7:14,
doi:10.1038/s41746-023-00986-6 · https://pmc.ncbi.nlm.nih.gov/articles/PMC10805720/ [V-sub].
Before–after quasi-experimental with Bayesian counterfactual inference, 2 UC San Diego EDs.
Pre: Jan 2021–Dec 2022 (705 days, 5,065 septic); post: Dec 2022–Apr 2023 (145 days, 1,152
septic); **6,217 total**. Nurse-facing Best Practice Advisory at **80% sensitivity / PPV 20.1%**,
4-hour prediction window.

- **~235 alerts/month; ~1.65 alerts per nurse per month**; 5.9% exited without acknowledgment;
  ~55% triggered "Will Notify MD Immediately"
- **In-hospital sepsis mortality: observed 9.49% vs counterfactual 11.39% (9.79–13.00) →
  absolute −1.9% (0.3–3.5), 17% relative, p=0.014** (≈22 additional survivors in 5 months)
- **SEP-1 compliance: 53.42% observed vs 48.38% expected → +5.0% absolute (2.4–8.0)**
- 72-h SOFA: 3.56 vs 3.71 → **−4% (1.1–7.1)**, p=0.013
- **Not reported:** false alarms per patient-day during the intervention period

**The ~1.65 alerts/nurse/month figure is the strongest published counter-example to the
alarm-fatigue narrative** — contrast with Epic's 18–21% of patients alerted.

**COMPOSER-LLM.** Shashikumar et al., *npj Digit Med* 2025 ·
https://pmc.ncbi.nlm.nih.gov/articles/PMC11952477/ [V-sub]. **Prospective, 754 ED encounters
(18.4% septic), May–Jun 2024.** An LLM layer performing a differential-diagnosis check on
high-uncertainty cases: sensitivity unchanged at 70.8%, **PPV 25.1% → 58.2%**, **FAPH 0.034 →
0.0086** — a >4× reduction in false alarms at unchanged sensitivity. Chart review found **62% of
remaining false positives had bacterial infections**, i.e. arguably reasonable flags.

**Caveat for the advisor:** COMPOSER's lead time is measured **relative to first antibiotic
order**, not relative to a Sepsis-3 onset timestamp. That is arguably the more clinically
meaningful anchor, but **it makes 12.2 h not directly comparable to our 20.6 h**, which is
measured relative to `t_sepsis`. Say so explicitly if we compare.

### 5.3 TREWS — the only prospective multi-site outcome evidence

- **Henry KE, Hager DN, Pronovost PJ, Saria S, *Sci Transl Med* 2015;7(299):299ra122** [V-sub].
  MIMIC-II, **13,014 development / 3,011 validation**. Target is **septic shock**, not sepsis.
  **AUC 0.83 (0.81–0.85)**; MEWS 0.73 (0.71–0.76). At specificity 0.67 → sensitivity 0.85;
  **median lead 28.2 h (IQR 10.6–94.2)**; **68.8% of identified patients had no sepsis-related
  organ dysfunction at identification.** Note: a 28.2 h lead on septic *shock* is not comparable
  to a 20 h lead on sepsis onset — different, later endpoint.
- **Adams R, et al., *Nat Med* 2022;28:1455–1460** [V-sub]. **590,736 patients monitored** across
  five hospitals; analysis cohort **6,877** sepsis patients alerted before antibiotics.
  **Adjusted absolute mortality reduction 3.3% (1.7–5.1); 18.7% relative (9.4–27.0)** when the
  alert was confirmed within 3 hours. **Design caveat to state: this is a prospective cohort,
  not a randomized trial, and the comparison is within-cohort by alert-confirmation timing** —
  clinicians who confirm quickly may differ systematically.
- **Henry KE, et al., *Nat Med* 2022;28:1447–1454** [V-sub]. **82% sensitivity; 89% of alerts
  evaluated by a physician/APP; 38% of evaluated alerts confirmed** — effectively a real-world
  clinician-adjudicated PPV, and remarkably high against Epic's 12%. **Median time to first
  antibiotic order reduced 1.85 h (1.66–2.00).**
- **Henry KE, et al., *npj Digit Med* 2022;5, doi:10.1038/s41746-022-00597-7** [S] — qualitative
  human-machine teaming; the finding is that *adoption*, not AUROC, was the binding constraint.
- **Gap:** TREWS never publishes a raw alert-rate denominator — only the 89%/38% conditional
  rates. Alerts per patient-day are **not obtainable** from the published record [V-sub].
- **Regulatory:** Bayesian Health reportedly received FDA Breakthrough Device Designation (2023)
  and a 510(k) clearance (April 2026) [S, press only — **no 510(k) number confirmed**].

### 5.4 InSight / Dascena — and the RCT that everyone over-cites

- **Desautels T, et al., *JMIR Med Inform* 2016;4(3):e28, doi:10.2196/medinform.5909** [V-sub].
  **MIMIC-III v1.3, Metavision subsystem only** (CareVue excluded for underreported negative
  cultures), **22,853 ICU stays, 2,577 sepsis (11.3%)**, Sepsis-3-style. **This is the standard
  source for bedside-score baselines on MIMIC** — see §5.7.
- **Mao Q, et al., *BMJ Open* 2018;8:e017833** [V-sub]. 684,443 encounters. Sepsis AUROC 0.92,
  severe sepsis 0.87, **septic shock 0.9992**. **Labels from ICD-9 codes**, and the authors
  themselves write that *"timing of these inputs may reflect clinician judgement rather than true
  onset time."* Treat the shock number as a leakage artefact the authors flagged.
- **Shimabukuro DW, Barton CW, Feldman MD, Mataraso SJ, Das R, *BMJ Open Respir Res*
  2017;4(1):e000234** [V-sub]. **The only individually-randomized trial of an ML sepsis alert.**
  Open-label RCT, single centre, two UCSF medical-surgical ICUs, Dec 2016–Feb 2017.
  **N = 142 (75 control / 67 intervention).** LOS **13.0 → 10.3 days (−20.6%), p=0.042**;
  in-hospital mortality **21.3% → 8.96%, p=0.018**.
  **Cite this as a pilot, never as evidence of mortality benefit.** An independent commentary
  ("Artificial Intelligence for Early Sepsis Detection: A Word of Caution", PMC10111986) notes
  the population *"was largely diluted with patients without sepsis and included only **25 actual
  cases**"* [V-sub]. A 12-point mortality difference on 25 sepsis cases is not a credible effect
  estimate. Two authors were Dascena employees.

### 5.5 Negative and null results — the pattern to state explicitly

- **Bedoya AD, Clement ME, Phelan M, Steorts RC, O'Brien C, Goldstein BA, *Crit Care Med*
  2019;47(1):49–55** [V-sub]. Before/after NEWS + best-practice-alert deployment, **N = 85,322**.
  **Primary outcome (ICU transfer or death): no significant change. 175,357 alerts triggered.
  Alert PPV 2.2% (academic) vs 7.4% (community).** Conclusion verbatim: NEWS *"had poor
  performance characteristics and was generally ignored by frontline nursing staff."*
  **Arguably the most quotable alarm-fatigue datapoint in the field.**
- **Downing NL, Rolnick J, Poole SF, et al., *BMJ Qual Saf* 2019;28(9):762–768** [V-sub].
  Single-blind **patient-level RCT**, N = 1,123, non-ICU wards. **Primary outcome (new antibiotic
  order within 3 h): 35% vs 37%, p=0.53 — null.** No difference in 30-day mortality, LOS, ICU
  transfer, or fluids.
- **Giannini HM, Ginestra JC, Chivers C, et al., *Crit Care Med* 2019** [V]. Sens **26%**, spec
  98%, **PPV 29%**; modestly increased lactate testing and IV fluids; **no mortality or
  ICU-transfer difference.**
- **Ginestra JC, et al., *Crit Care Med* 2019;47(11):1477–1484** [S]. 362 alerts; **only 24% of
  nurses and 13% of providers identified new clinical findings after the alert**; agreement that
  sepsis was present at alert time: nurses 13%, providers 40%.
- **Boussina et al., *JAMA Netw Open* 2026;9(6):e2611885** [V-sub]. Cluster RCT of LLM-based
  SEP-1 abstraction and feedback: **compliance 70.1% → 82.9%, +13.0% absolute (2.5–23.4),
  OR 2.10 (1.15–3.81)** — but **no significant difference in 30-day mortality or ICU admission.**
- **SCREEN trial, *JAMA* 2025;333(9):763–773** [S, cross-confirmed across two secondary sources].
  Stepped-wedge cluster RCT, 5 hospitals, **N = 60,055**. **90-day in-hospital mortality adjusted
  RR 0.85 (0.77–0.93), p<0.001.** Alert fired in ~15% of screened patients. Also **↑code blue
  activations, ↑renal replacement therapy initiation, ↑C. difficile infections.**
  **Crucially, this is a rule-based qSOFA alert, not ML** — the strongest randomized mortality
  evidence in sepsis alerting comes from a simple score.

**The pattern, and it belongs in our discussion:** *process measures move; mortality usually does
not.* The three studies reporting a mortality benefit — TREWS (prospective cohort, stratified by
confirmation timing), COMPOSER (before-after quasi-experiment), SCREEN (cluster-randomized,
rule-based) — **none is an individually randomized blinded trial of an ML alert.** Shimabukuro is
the only individually randomized ML trial and rests on 25 sepsis cases.

### 5.6 Regulatory — Prenosis Sepsis ImmunoScore, FDA De Novo DEN230036

Verified from the FDA primary documents [V-sub]:
https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN230036.pdf (47-page decision summary) ·
https://www.accessdata.fda.gov/cdrh_docs/pdf23/DEN230036.pdf (decision letter)

- Prenosis, Inc.; submitted **2023-05-05**, **granted 2024-04-02**
- **Class II, 21 CFR 880.6316, product code SAK** — a **new generic type**: *"software device to
  aid in the prediction or diagnosis of sepsis"*. **The first device of this type; it created the
  classification.** Follow-on devices must submit a 510(k).
- Indication: risk of sepsis **within 24 hours**, for ED/inpatient patients **for whom sepsis is
  suspected and a blood culture was ordered**; *"should not be used as the sole basis to determine
  the presence of sepsis."* Up to 22 EHR inputs, four discrete risk bands.
- Pivotal: retrospective analysis of prospectively collected data, **3 validation sites
  independent of training sites**, **N = 746** (735 analyzable).
- **AUROC 0.81 (0.76–0.86)** forced-majority adjudication; 0.84 (0.78–0.90) forced-unanimous.
  Pre-specified goal was 0.75.
- PPV by band: Low **3.02%**, Medium **12.74%**, High **36.59%**, Very High **69.70%**.
- **The caveat that matters most:** of 151 sepsis-within-24 h cases, **99 were diagnostic**
  (score after onset) and **only 52 predictive**. Predictive-only PPVs: 2.17 / 5.52 / 14.22 /
  50.0%. **This is largely a diagnostic aid, not an early-warning system** — and the FDA summary
  says so.
- **Special controls** (effectively FDA's minimum evidentiary bar for a sepsis model): clinical
  and non-clinical performance testing, missing-input/imputation study, **monotonicity of risk
  outputs**, human factors, **subgroup analyses by demographics, site, and acquisition system**,
  **disjoint train/test data**, and a mandatory **post-market performance management plan** with
  real-world generalizability monitoring.

**This special-controls list is the most useful regulatory artefact for a deployment-oriented
paper**: Epic's ESM satisfies almost none of it while being deployed at hundreds of hospitals.

A larger published validation exists (7 US sites, **N = 6,027**, AUROC sepsis **0.82
(0.81–0.83)**; NEWS 0.69, qSOFA 0.67, SIRS 0.59, SOFA 0.72, PCT 0.70, CRP 0.61) at
PMC13360300 [V-sub], noting a turnaround delay of *"potentially several hours"* because PCT/CRP
assays are required. The *NEJM AI* development paper (doi:10.1056/AIoa2400867) returned 403 to
every retrieval attempt and is **not verified** [?].

### 5.7 Alert burden as a metric, and bedside-score baselines

**Alert-burden numbers that exist**, for calibrating our ≤1.0 false alerts/nonseptic patient-day:

| Source | Burden |
|---|---|
| **COMPOSER (Shashikumar 2021)** [V-sub] | **0.029–0.047 false alarms per patient-hour (ICU) = 0.70–1.13/patient-day** |
| **COMPOSER-LLM (2025)** [V-sub] | 0.034 → **0.0086** FAPH prospectively |
| **COMPOSER deployed (Boussina 2024)** [V-sub] | ~235 alerts/month; **~1.65 alerts per nurse per month**; PPV 20.1% |
| **DeepAISE** [V-sub] | False alarm rates **0.20 / 0.26** |
| Moor et al. 2023 [V] | **1.4 false alerts per true alert** at 80% recall |
| Li et al. 2023 (MIMIC-IV trauma) [V-sub] | **9 false alerts per true alert** at 73% sensitivity, 6-h window |
| Bedoya 2020 (Duke) [V-sub] | At a **fixed budget of 3 alarms/hour**: MGP-RNN 10.5/17.9 daily cases vs SIRS 5.76, NEWS 3.03, qSOFA 2.21 |
| Sendak 2020 (Duke) [V-sub] | Design constraint: **≤4 high-risk alerts/hour per rapid-response nurse** |
| Sepsis Watch external [V] | **1–27 alerts/day per ED** at 20% precision |
| Epic v2 (Wong 2026) [V-sub] | **NNE 21–35** at 12 h; PPV 0.13–0.26 at 60% sensitivity |
| Epic v1 (Wong 2021) [V-sub] | **18% of hospitalizations**; **NNE 8 / 109** |
| Epic under COVID shift [V-sub] | 9% → **21% of patients alerting per day** |
| Bedoya 2019 (NEWS) [V-sub] | 175,357 alerts; **PPV 2.2% / 7.4%**; "generally ignored" |
| Jin & Lee 2026 [V-sub] | Evaluates at **fixed alert rate α = 5%** with stay-level detection and lead-time distributions |
| Romero-Brufau et al. [S, **unverified**] | Clinically acceptable burden **3–10 alarms per 100 patient-days** |

**Systematic-review verdict on this literature.** *"Sepsis Alerts in Emergency Departments: A
Systematic Review of Accuracy and Quality Measure Impact"* (PMC7514413) [V-sub] reports
sensitivity 10–100%, specificity 78–99%, **PPV 5.8–54%**, and explicitly states it **found no
quantitative data on alerts-per-patient-day, override/dismissal rates, or alarm-fatigue survey
evidence, and that "none of the studies addressed potential harms."** That is a citable,
primary-source-verified statement of the evidence gap and is **the strongest available
justification for a paper that does report alert burden.**

**Bedside-score baselines on MIMIC.** Our NEWS2 0.626 / qSOFA 0.587 / SIRS 0.551 are consistent
with the literature.

| Source | Task | SIRS | qSOFA | MEWS | NEWS | SOFA |
|---|---|---|---|---|---|---|
| **Desautels 2016** [V-sub] | MIMIC-III sepsis onset, AUROC | **0.609** | **0.772** | **0.803** | — | 0.725 |
| **Desautels 2016** [V-sub] | same, **AUPRC** | 0.160 | 0.277 | 0.327 | — | 0.284 |
| **Bedoya 2020** [V-sub] | Duke sepsis onset, AUROC | 0.756 | **0.481** | — | **0.619** | — |
| Churpek 2017 [V-sub] | ward mortality, AUROC | 0.65 | 0.69 | 0.73 | **0.77** | — |
| Seymour 2016 [V-sub] | **ICU** mortality, AUROC | 0.64 | 0.66 | — | — | **0.74** |
| Prenosis 2025 [V-sub] | sepsis, 7 sites | 0.59 | 0.67 | — | 0.69 | 0.72 |

**Our NEWS2 0.626 is essentially identical to Bedoya's 0.619 on a comparable task — a useful
external corroboration of our baseline pipeline.** Note also that **no published study reports
NEWS/NEWS2, qSOFA, SIRS and MEWS AUROCs on a MIMIC sepsis-onset task in one place** [V-sub];
Desautels 2016 (no NEWS) is the closest. **We are in a position to publish that table**, which is
a small but genuinely useful contribution.

**On SOFA.** Our decision to exclude SOFA as a comparator because the label is defined as a SOFA
rise is correct and, as far as I can determine, **is not made explicit in any of the papers
above** — several of which report "our model beats SOFA" on Sepsis-3 labels. Say it plainly.

---

## 6. Methods most relevant to us

### 6.1 Irregular-time-series architectures — and whether any were tested on sepsis

| Model | Citation | Sepsis? | Note |
|---|---|---|---|
| **GRU-D** | Che, Purushotham, Cho, Sontag, Liu, *Sci Rep* 2018;8:6085 | **No** (MIMIC-III + PhysioNet 2012 mortality) | **The origin of the term "informative missingness"** — and it argues *for* exploiting it. The paper our synthetic control is in tension with |
| **mTAND** | Shukla & Marlin, ICLR 2021, arXiv:2101.10318 | No | Continuous-time embeddings + attention over reference times |
| **SeFT** | Horn, Moor, Bock, Rieck, Borgwardt, ICML 2020 | **Yes** — CinC 2019 | The only one in this family evaluated on sepsis. **0.768 / AUPRC 4.84** at 1.8% prevalence. On MIMIC-III mortality it is *worse* than GRU-D (AUPRC 46.3 vs 52.0) and sells on ~10× speed |
| **Raindrop** | Zhang, Zeman, Tsiligkaridis, Zitnik, ICLR 2022 | **Partly** — uses the CinC-2019 "P19" data, but as whole-stay binary classification | Claims up to +11.4 F1 over SOTA incl. SeFT |
| **Warpformer** | Zhang, Zheng, Cao, Bian, Li, KDD 2023 | **No** | One of its five MIMIC-III tasks is **"will this variable be measured next"** — the field already treats the observation process as a learnable target |
| **Latent ODE / ODE-RNN** | Rubanova, Chen, Duvenaud, NeurIPS 2019 | No | Can model observation *times* as a Poisson process |
| **CRU** | Schirmer et al., ICML 2022 | No | |
| **ContiFormer** | Chen et al., NeurIPS 2023 | No | |
| **STraTS** | Tipirneni & Reddy, *ACM TKDD* 2022;16(6):105 | **No** | **The closest architectural precedent to ours**: (time, variable, value) triplets, continuous value embedding, no gridding, self-supervised forecasting pretraining |
| **MGP-TCN / MGP-AttTCN** | Moor MLHC 2019; Rosnati *PLOS ONE* 2021 | **Yes** — MIMIC-III | The sepsis-specific branch of this literature. AUPRC 0.40 at 7 h (MGP-TCN); AUROC 0.660 at 5 h (MGP-AttTCN) |

**The most useful single number here** is STraTS's own baseline table on MIMIC-III mortality
[V-sub]: plain **GRU 0.886** → **STraTS 0.891**. The entire span from a vanilla RNN to the best
specialised irregular-TS architecture is **0.005–0.026 AUROC**. **Our Transformer-minus-XGBoost
gap of 0.020 (0.756 vs 0.736) sits squarely inside that band** — which is a defensible thing to
say, not an embarrassing one.

**Gap:** I found **no** paper in this architecture family that evaluates **per-hour sepsis risk on
MIMIC-IV**. SeFT and Raindrop use CinC-2019; MGP-TCN/MGP-AttTCN use MIMIC-III; everything else
uses MIMIC-III/PhysioNet-2012 mortality. Per-hour sepsis on MIMIC-IV appears only in the
benchmark-suite literature (YAIB) and the applied papers in §1.1.

### 6.2 Transformers for ICU EHR

BEHRT (*Sci Rep* 2020), Med-BERT (*npj Digit Med* 2021), TransformEHR (*Nat Commun* 2023;14:7857),
EHRSHOT (NeurIPS 2023 D&B), Context Clues (ICLR 2025, best result Mamba at 16k tokens, mean AUROC
0.807). **All operate on visit-level diagnosis-code sequences, not hourly physiological signals,
and none reports a sepsis-onset task** [V-sub]. The EHR foundation-model literature and the
sepsis-timing literature are essentially disjoint. A recurring stated limitation in the 2024–2026
ICU foundation models (PULSE-ICU arXiv:2511.22199; arXiv:2507.00574 [S]) is that quadratic
attention forces truncation of long ICU event streams — a motivation for a causal streaming
design like ours.

### 6.3 Gradient boosting vs deep learning

- **Grinsztajn, Oyallon, Varoquaux, NeurIPS 2022 D&B, arXiv:2207.08815** [V-sub]. 45 datasets.
  Trees remain SOTA at ~10k samples; the gap is **not** explained by categorical features and does
  **not** close with tuning. Three diagnosed NN weaknesses: **non-robustness to uninformative
  features**, rotation invariance, inability to fit irregular targets. Our feature matrix — many
  rarely-measured channels, mask/delta channels, a non-smooth target — is exactly this regime.
- **Shwartz-Ziv & Armon, *Information Fusion* 2022;81:84–90** [S]. XGBoost wins; deep tabular
  models are harder to tune; the deeper lesson is **evaluation-set selection bias**.
- **ICU-specific, and the evidence points both ways:**
  - **HiRID-ICU-Benchmark** (Yèche et al., NeurIPS 2021 D&B, arXiv:2111.08536) concludes
    explicitly that *"boosted ensembles of decision trees outperform current deep learning
    approaches on medical time series problems"* — but **it has no sepsis task** [V-sub].
  - **YAIB** (ICLR 2024), which *does* have an hourly sepsis task on MIMIC-IV, finds the
    opposite: **GRU 83.6 vs LGBM 77.5**, GRU ahead on every dataset [V-sub].
  - **Liao et al., arXiv:2211.06034** [V]: on CinC-2019 sepsis, DL beats non-DL **only** (a) on
    some metrics (AUROC, AUPRC, sensitivity, FNR) and not others, and (b) once training size
    reaches thousands.
  - **Jin & Lee 2026** [V-sub] report that **centralized XGBoost had the strongest detection on
    MIMIC-IV** under a fixed-alert-rate evaluation — the closest published result to ours.

**Caution for our write-up.** No paper's headline result is "XGBoost ≈ Transformer for per-hour
sepsis *onset* prediction on MIMIC-IV", and the strongest same-task published benchmark (YAIB)
points the *other* way by ~6 AUROC points. So our finding is genuinely interesting — **but present
it as our "Reconciling the two parts" analysis does**: the gap is *representation* (causal ffill +
observation mask + time-since-last-measurement), not architecture. Framed as "GBDT beats deep
learning" it collides with YAIB and loses.

### 6.4 Benchmark suites

| Suite | Sepsis task? | MIMIC-IV? | Sepsis AUROC |
|---|---|---|---|
| **YAIB** (ICLR 2024) | **Yes** — "sepsis within next 6 h", hourly | **Yes (v2.x)** | **LR 77.1 / LGBM 77.5 / GRU 83.6**; AUPRC 4.6 / 5.9 / 9.1 |
| **HiRID-ICU-Benchmark** (NeurIPS 2021) | No (circulatory/respiratory failure) | No | — |
| **MIMIC-Extract** (CHIL 2020) | No | **No — MIMIC-III only**, never ported | — |
| **ricu** (*GigaScience* 2023;12:giad041) | Provides a `sep3` *concept*, not a benchmark task | Yes (`miiv`) | — |
| **MIMIC-Sepsis** (arXiv:2510.24500) | No — mortality / LOS / vasopressor / **shock onset** | **Yes, v3.1** | Shock onset Transformer 0.925 |

**YAIB is our baseline anchor: ~83.6 AUROC / 9.1 AUPRC (GRU), 77.5 / 5.9 (LGBM) at ~1%
positive-bin prevalence.** Our 0.756 is below both, and a reviewer will ask why. The answer is our
post-onset truncation and pre-onset target — YAIB keeps a standard 6-h prediction window and
truncates *features* 6 h before onset, which is a **different and easier target** than positives
confined to `[t_sepsis − 12 h, t_sepsis)` with truncation at onset.

> **Strongest single recommendation in this document: reproduce the YAIB sepsis target on our
> cohort and report our model under both targets.** Without that anchor our 0.756 is
> indefensible. Note that YAIB is built on MIMIC-IV v2.x (~73k stays) and we are on v3.1 (~94k),
> so we should say so and, ideally, run our pipeline on both.

### 6.5 Label-definition sensitivity, prodrome windows, and measurement-frequency confounding

**Label-definition sensitivity** — Cohen et al., *Sci Rep* 2024;14:1920 is the key prior work
(§2.2, §3), varying onset time within Sepsis-3 on MIMIC-III across three model classes and finding
definition variance ≥ model variance. Rosnati & Fortuin 2021 is the complementary demonstration
that a single missingness assumption changes the case count >4×.

**Label circularity, formalised** — **Hagmann M, Schamoni S, Riezler S. "Validity problems in
clinical machine learning by indirect data labeling using consensus definitions." ML4H 2023,
arXiv:2311.03037** [V-sub]. When the target label is defined by measurements that are also model
inputs, the model learns to *reconstruct the label definition* rather than acquire diagnostic
ability; such models "show perfect performance on similarly constructed test data but will fail
catastrophically on real-world examples where the defining fundamental measurements are not or
only incompletely available." Early sepsis prediction is the case study, and the paper proposes a
detection procedure. Companion: Schamoni/Lindner/Riezler arXiv:1909.09557, proposing
physician-questionnaire ground truth as a non-circular alternative. **This is arguably the single
most important methodological citation for a rigorous review of this literature.**

**Prodrome / pre-onset window as a training target.** The strongest *motivation*:

> **Weissman GE, Hubbard RA, Himes BE, et al., "Sepsis Prediction Models are Trained on Labels
> that Diverge from Clinician-Recommended Treatment Times", *AMIA Annu Symp Proc* 2024,
> PMID 40417569** [V]. 153 clinicians at three geographically diverse centres reviewed vignettes
> from eight real sepsis cases and recommended starting antibiotics **an average of 7.0 hours
> (95% CI 5.3–8.8) BEFORE the Sepsis-3 onset time.** They name it **"label bias"** and conclude
> that "predicting Sepsis-3 onset as a treatment prompt could lead to inappropriate and delayed
> treatment recommendations."

Cite this as the clinical argument for our pre-onset target. It motivates the design; it does not
implement or evaluate one.

**Measurement frequency as a confounder** — the two closest papers to our synthetic control:

> **Yamamoto R, Wu F, Sprehe LK, Abeer A, Celi LA, Tohyama T, "Observation-process features are
> associated with larger domain shift in sepsis mortality prediction", medRxiv 2026,
> doi:10.64898/2026.04.05.26350209** [V]. MIMIC-IV n=30,218 → eICU n=31,403. Seven specifications
> fit **with and without measurement counts**. Internal AUROC 0.819 → 0.834 with counts; **external
> drop −0.047 without vs −0.082 with**; **calibration slope 1.007 → 0.417**. Limits: the task is
> **mortality, not onset**, and it is a **feature ablation with cross-site transfer**, not a
> synthetic control holding the physiological signal fixed.

> **Dickens A, "Falsification Testing of Sepsis Prediction Models", medRxiv 2026,
> doi:10.64898/2026.03.17.26348414** [V]. A **pre-registered (OSF, before data access)**
> falsification study on **MIMIC-IV v3.1, n=65,241 adult ICU stays**, testing exactly the
> hypothesis that sepsis models learn care-process intensity rather than biology. **The hypothesis
> was NOT confirmed**: biological features predicted Sepsis-3 at AUROC 0.901 with minimal loss
> when care-intensity features were removed. Care-intensity signal was larger at community
> hospitals than at the academic centre.

**This is the single most important paper for the advisor to see, and it partially cuts against
us.** Pre-registered, same database version, near-identical cohort size (65,241 vs our
63,672–64,236), asking a version of our question and getting "no". **We must engage with it
directly rather than let a reviewer find it.** The differences worth arguing: it works at a much
higher AUROC operating point (0.901), implying a substantially different label/target
construction; and it tests *feature removal*, whereas our control *injects* a known signal at
controlled amplitude into channels of differing measurement density. Those are different
experiments and can both be right. But we should say that, in the paper, ourselves.

**Supporting:** Groenwold, *Diagn Progn Res* 2020;4, doi:10.1186/s41512-020-00077-0 [V] —
informative missingness in EHR, and the point that **the missingness mechanism changes once a
model is deployed and clinicians respond to it**. Sisk et al., *Stat Methods Med Res* 2023,
doi:10.1177/09622802231165001 [V] — missing indicators help but are **harmful under
outcome-dependent missingness**. And the CinC-2019 fifth-place entry is literally titled
*"Utilizing Informative Missingness for Early Prediction of Sepsis"* [V-sub] — missingness alone
carries sepsis signal, and a top-5 team exploited it deliberately.

**Metamorphic testing** — Wu JJW, E F, Chen B, arXiv:2607.22984 (2026) [V-sub]: 12 metamorphic
relations for ICU tasks including **sepsis onset** on MIMIC-III/IV; models with AUROC 0.849–0.900
violate clinically-grounded relations **27–87%** of the time, and **an injected sign-negation
fault in a blood-pressure feature is invisible to AUROC but raises violation rates by 31–67
points.** A strong independent argument that AUROC alone is insufficient, and a natural companion
to our synthetic control.

---

## 7. Where our work is and is not novel

Blunt. "Someone has published this" is kept separate from "I searched and did not find it".

### 7.1 Already published — we would be duplicating

| Our claim | Prior art | What is left |
|---|---|---|
| **Retaining post-onset hours inflates AUROC while making the model clinically worse** | **Kamran et al., *NEJM AI* 2024**: 0.62 → **0.47** before treatment. **Currey & Tarabichi 2025**: 0.77→0.70 (v1), 0.90→0.85 (v2) before clinical recognition. **Do et al. 2026**: evaluation strategy moves AUROC 0.61↔0.67, matching ±0.14. **Wang et al. 2025**: partial-window 0.886 vs full-window external 0.783 | **The core claim is published at least three times, with larger effects than ours (0.15 vs our 0.09).** Unpublished: the **dose–response curve over truncation horizon** on a single cohort (+0h/+3h/+6h/+24h/none, prevalence 1.5%→25.4%) with the paired demonstration that lead time moves *opposite*. **Present as replication with a dose–response extension, not discovery.** |
| **Label definition materially changes measured performance** | Cohen 2024 (0–6% AUROC, code on Zenodo); Dutta 2026 (0.85–0.94 across definition families); Rosnati 2021 (>4× cohort size from a missingness assumption); Bauer 2025 (κ=0.39) | Fully covered. Cite, do not claim. |
| **AUROC is a poor metric for rare-event early warning** | Moor 2021; Wang 2025 (Utility 0.381 → **−0.164** externally while AUROC held at 0.783); Sepsis Watch (AUROC 0.96 / AUPRC 0.18); the ED systematic review's explicit statement of the gap; Wu 2026 metamorphic testing | Fully covered. |
| **Sepsis-3 onset is partly a treatment-decision timestamp** | Kamran 2024; Weissman 2024 ("label bias", 7.0 h); **Hagmann/Schamoni/Riezler 2023** (formal circularity); Moor 2021 ("circularity"); Mao 2018 (authors' own admission) | Fully covered as an argument. |
| **Deep sequence models barely beat well-featurised flat models** | STraTS's own table (GRU 0.886 → best specialised 0.891); Grinsztajn 2022; Shwartz-Ziv & Armon 2022; Jin & Lee 2026 (XGBoost strongest on MIMIC-IV) | The general claim is covered. **Our specific version — the gap is representation not architecture, measured as 0.640 → 0.722 → 0.736 — I did not find published for sepsis.** Modest but real. |
| **Models may learn clinician behaviour rather than physiology** | GRU-D (proposes exploiting it); a CinC top-5 entry built on it; Yamamoto 2026 (quantifies the transfer cost); **Dickens 2026 (pre-registered, MIMIC-IV v3.1 — tested it and found the hypothesis NOT confirmed)** | **The question is claimed, and the most rigorous attempt reached a different conclusion.** Engage directly. |

### 7.2 Appears unclaimed — with the caveat attached

Ordered by confidence. In each case, "found nothing" means **three independent search passes**
(mine plus two delegated agents) over Google Scholar, arXiv, Europe PMC/PubMed, and GitHub.

**(c) The measurement-frequency-vs-effect-size synthetic control — strongest claim.**
I found no paper that injects a synthetic prodrome of controlled amplitude into channels of
differing measurement density and compares detectability. Nearest work: Yamamoto (feature ablation
+ cross-site transfer, mortality), Dickens (feature removal), Wu 2026 (metamorphic perturbation,
but for fault detection not measurement density). Our result — **a 0.25 SD drift in a 93%-measured
channel (AUROC 0.972) beats a 4 SD shift in a 2.6%-measured one (0.813)**, with the control arm
reproducing the real baseline at 0.722 — is a different design and yields a *quantitative design
specification* rather than a diagnosis. It also doubles as a pipeline-validity proof, which
reviewers value. **This is the most defensible novel contribution; lead with it.**
*Confidence: high the specific experiment is unpublished; moderate that no analogue exists in
another clinical domain — I did not search outside sepsis/ICU.*

**(a) AUROC anti-correlated with lead time, across independent manipulations.**
The horizon→AUROC decline is well documented (§2.5) and Do et al. articulate the mechanism. Not
found: (i) any paper naming this as a property that must be reported alongside AUROC, (ii) any
paper showing it across *five different kinds of change to one pipeline* (model class, truncation,
feature count, synthetic amplitude, architecture), or (iii) the synthetic demonstration that a
**16× stronger prodrome yields higher AUROC and shorter warning**. (iii) is the sharpest, because
it isolates the mechanism with signal held exactly.
*Confidence: moderate-to-high. Three searches specifically for this framing. But it is the kind of
observation that may sit in someone's discussion section.*

**(d) Equal-alert-burden model comparison instead of fixed recall — DOWNGRADE THIS.**
My initial assessment was too generous. Two papers do essentially this:
- **Jin & Lee, *Applied Sciences* 2026;16(6):2735** [V-sub] — "**Fixed Alert-Rate Evaluation**" is
  in the title. Evaluates at **α = 5% alert rate** with stay-level detection rates and lead-time
  distributions, on **CinC 2019 and MIMIC-IV**, explicitly arguing this is more clinically
  interpretable than AUROC/AUPRC alone, with leakage stress tests and decision-curve analysis.
  **This is very close to our (d) and to parts of our framing generally.**
- **Bedoya et al., *JAMIA Open* 2020** [V-sub] — compares MGP-RNN, SIRS, NEWS and qSOFA at a
  **fixed budget of 3 alarms/hour**, reporting cases captured per day.
Also: Sendak 2020 states a **≤4 alerts/hour/nurse** design constraint; COMPOSER reports
false-alarms-per-patient-hour; Epic v2 reports NNE at a standardised sensitivity.
**Conclusion: fixed-burden comparison is claimed. Our contribution is at most the specific
argument that burden "unlike recall, forces nothing", and the pairing of burden with lead time and
capture-≥6h. Do not present (d) as novel — present it as adopting the right convention, and cite
Jin & Lee and Bedoya.** I could not retrieve Jin & Lee's numbers (MDPI 403); **get the open-access
PDF and read it before writing anything about (d).**

**(b) Confining positives to a pre-onset window as a training target — weakest claim.**
The idea is thoroughly present:
- **CinC 2019 already shifts labels 6 h earlier** and its utility function rewards only
  [t−12 h, t+3 h] [V].
- **YAIB already truncates features 6 h before onset** with a 6-h prediction window [V-sub].
- **Li et al. 2023 already excludes all post-onset records** [V-sub].
- **Shao et al. 2026** excludes patients septic in the first 24 h and predicts later onset [V].
- **Do et al. 2026** sweep the prediction window 1–100 h and report that narrowing raises measured
  performance [V].
- **Weissman et al. 2024** argue on clinical grounds the target should be ~7 h earlier than
  `t_sepsis` [V].

**Not found:** a controlled study treating the **prodrome window length as the object of study** —
sweeping it and reporting the resulting AUROC / lead-time / alert-burden surface. Our
`prodrome_window_h` is already a parameter. **If we sweep it and publish the surface, that is a
contribution. If we only report the 12-hour setting as "our method", it reads as a hyperparameter
choice others have already made.**
*Confidence: high the raw idea is claimed; moderate the systematic sweep is not.*

### 7.3 Additional things that appear to be ours

1. **A CinC-2019-faithful sepsis label on MIMIC-IV.** Extensive GitHub searching found **no
   repository that re-implements the CinC-2019 `t_suspicion` / `t_SOFA` / `t_sepsis` rules on
   MIMIC-IV**, and confirmed **PhysioNet never released the label-generation code** — only the
   prose definition and `evaluate_sepsis_score.py` [V-sub]. mimic-code implements a *different*
   label (§3.3); ricu is configurable but defaults elsewhere. If `sepsentinel/data/sepsis3.py`
   with its enumerated `DEVIATIONS` is released, it is, as far as I can determine, **the first
   public one. This may be our most immediately citable artefact.**
2. **First sepsis-onset early-warning work on MIMIC-IV v3.1.** No v3.0/v3.1 onset paper was found
   [V-sub]. **Weight this carefully** — "first on a version" is a weak claim on its own, and v3.1
   brings COVID-era stays (§8.4) which arguably makes our numbers *less* comparable to the
   v2.x literature, not more. Frame it as a reproducibility contribution, not a novelty claim.
3. **The demonstration that post-onset truncation at +3 h reconstructs the PhysioNet timestep
   prevalence (2.3% vs PhysioNet's 2.2%)** — reverse-engineering an undocumented cohort filter of
   the challenge. Found nothing like it.
4. **A single-source MIMIC table of NEWS2 + qSOFA + SIRS on a sepsis-onset task.** No published
   study reports these together on MIMIC [V-sub]; Desautels 2016 (no NEWS) is the closest.
5. **Explicitly excluding SOFA as a comparator because the label is a SOFA rise.** Several papers
   report beating SOFA on Sepsis-3 labels; I found none that names the tautology.

### 7.4 The honest summary for the advisor

**Our headline discrimination number is not competitive and should not be the headline.** 0.756
against YAIB's 0.836 and Backes's 0.841 on the same database will read as weak unless the paper is
framed as a **measurement paper**: what the number means, what moves it, and why the thing that
moves it most is not the model. The one number that *is* directly competitive is **AUPRC 0.088
against Backes's 0.099 at roughly half his prevalence** — and our **alert burden, which matches
COMPOSER's.** Lead with those.

Ranked contributions:
1. **§7.2(c) the synthetic control** — the strongest genuinely novel result.
2. **The CinC-faithful MIMIC-IV labeller** as a released artefact.
3. **The prodrome-window sweep**, if we actually run it.
4. **The post-onset dose–response curve**, positioned as a replication of Kamran/Currey with an
   extension.
5. The bedside-score comparator table.

**Three things to do before writing anything:**
- **Reproduce the YAIB sepsis target** on our cohort and report both.
- **Read Dickens 2026 in full** and write our response into the paper.
- **Get the Jin & Lee 2026 PDF** and rewrite our (d) framing around it.

---

## 8. Reproducibility and code

### 8.1 Sepsis-labelling implementations for MIMIC-IV

| Implementation | What it gives | License / status | CinC-style hourly labels? |
|---|---|---|---|
| **MIT-LCP/mimic-code** — https://github.com/MIT-LCP/mimic-code | `sepsis3.sql`, `suspicion_of_infection.sql`, `sofa.sql`. **`sofa.sql` IS hourly** (`icustay_hourly`, `ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING`). **`sepsis3.sql` is NOT** — one row per `stay_id`, absolute `sofa_24hours ≥ 2`, window [SOI−48 h, SOI+24 h], baseline SOFA assumed 0 | MIT; Zenodo DOI 10.5281/zenodo.6818823; ~2,374 commits, active. **Ships an official DuckDB dialect** at `mimic-iv/concepts_duckdb/` (auto-generated) | **Partially.** Hourly SOFA yes; the 2-point-rise and `t_sepsis` logic must be built on top |
| **ricu** — https://github.com/eth-mds/ricu | `sep3` / `sep3_alt` over MIMIC-III, **MIMIC-IV**, eICU, HiRID, AUMCdb; outputs an hourly `ts_tbl`. `delta_fun` configurable (`delta_cummin` default = rise vs running minimum) | GPL-3.0; CRAN + GitHub; active. Paper: *GigaScience* 2023;12:giad041; ~100 concepts across 395,941 ICU admissions | **Closest available.** Genuine delta criterion + hourly output, but SI window defaults to [−48 h, +24 h], not CinC's [−24 h, +12 h] |
| **YAIB-cohorts** — https://github.com/rvandewater/YAIB-cohorts | ricu-based sepsis task; `stop_obs_at(offset = 6h)`, 6-h outcome window, excludes onset <6 h post-admission | MIT; active | A *different* target, cleanly specified. **The best available reference implementation of a reproducible hourly sepsis task** |
| **PhysioNet CinC 2019** — https://physionet.org/content/challenge-2019/1.0.0/ | Prose definition + `evaluate_sepsis_score.py` (https://github.com/physionetchallenges/evaluation-2019, BSD-2) | CC BY 4.0 | **Label-generation code never released** |
| **BorgwardtLab/mgp-tcn** | The canonical PostgreSQL Sepsis-3 case/control extraction (MIMIC-III) | BSD-3; 70 stars | MIMIC-III only |
| **mmr12/MIMIC-III-sepsis-3-labels** | Alternative Sepsis-3 labels with different missingness assumptions (Rosnati) | — | MIMIC-III only |
| **alistairewj/sepsis3-mimic** | Codebase for Johnson et al. 2018 | Zenodo 1256723 | MIMIC-III only |
| **yongh7/MIMIC-sepsis** — https://github.com/yongh7/MIMIC-sepsis | MIMIC-IV **v3.1**, 35,239 patients, own SQL; SI = abx OR culture | MIT; 19 stars | Cohort only; no onset task |
| **on1262/sepsisdataprocessing** | MIMIC-IV **v2.2** processing pipeline; expects an external `sepsis3.csv` | No license file; 117 commits | Processing only |
| **philipdarke/mimic4** — https://github.com/philipdarke/mimic4 | Pure-Python MIMIC-IV **v3.0** → DuckDB loader with derived concepts incl. SOFA and sepsis | MIT; **12 commits, 2 stars** — low maintenance | Community DuckDB port; use with care |
| **JP5635/Early_Sepsis_Detection_in_ICU_MIMIC-IV** | MIMIC-IV **v3.1**, hourly SOFA across six systems; **cohort by ICD string matching + SOFA** | MIT; 5 commits, 0 stars | Not CinC labels |

**Verified negative [V-sub]:** no repository faithfully implements the CinC-2019
`t_suspicion`/`t_SOFA`/`t_sepsis` rules on MIMIC-IV.

**Flag for our own methods [V-sub]:** ricu's rendered documentation for `susp_inf` describes
"antibiotic-first → sampling within **72 h**; culture-first → antibiotic within **24 h**", which is
the **mirror image** of the Seymour-2016 / CinC-2019 / mimic-code convention. This may be a
documentation wording artefact rather than a code difference. **Verify in source before asserting
cross-tool equivalence — and if it is real, it is itself reportable.**

### 8.2 Which papers ship code

| Ships code | Does not / unclear |
|---|---|
| YAIB (MIT) · YAIB-cohorts · Moor multicentre (Apache-2.0) · Moor MGP-TCN (BSD-3) · Rosnati MGP-AttTCN + labels · Cohen et al. (Zenodo 5168789) · SeFT · STraTS · Raindrop · Warpformer · mTAND · MIMIC-Sepsis (MIT) · Do et al. (stated) · Tranchellini (ETH GitLab) · SEPRES · SepsisSuite · all CinC-2019 entrants (PhysioNet hosts every submission as a zip) | Backes 2026 · Li 2023 (on request) · Stylianides 2026 · Dalal 2024 (on request) · CISepsis · Nie 2026 · Tang 2024 (on request) · Sepsis Watch ("proprietary reasons") · Epic (proprietary) · COMPOSER · TREWS · most 2026 Sci Rep / JMIR papers |

**The base rate is bad.** Moor et al. 2021: **only 2 of 22 studies shared both analysis code and
label-generation code** [V-sub]. Wang et al. 2025 across 91 studies does not tabulate code
availability at all [V]. **Note the pattern: the best-performing MIMIC-IV onset papers (Li,
Stylianides, Dalal) all withhold code, and the best-documented ones (YAIB, Moor, Do) report the
lowest numbers.**

### 8.3 CinC 2019 leaderboard and Papers with Code

Every challenge submission's source is hosted at
`physionet.org/static/published-projects/challenge-2019/1.0.0/sources/`; official results at
https://moody-challenge.physionet.org/2019/results/ [V-sub]. Top five by utility:
**1. "Can I get your signature?"** (Morrill et al., signature methods, **0.360**; GitHub rewrite
at https://github.com/jambo6/sepsis_competition_physionet_2019 — **no license file**, and the
README notes it is a rewrite, not the submitted artefact); 2. Sepsyd; 3. Separatrix;
4. FlyingBubble (TASP); 5. CTL-Team.

**Papers with Code is dead** — retired by Meta, sunset **24 July 2025**; `paperswithcode.com` now
redirects to `huggingface.co/papers/trending` [V-sub]. The PhysioNet-2019 leaderboards there are
gone; a JSON dump survives at https://github.com/paperswithcode/paperswithcode-data. **Cite the
official CinC results page instead.**

**No published study attempted to re-run a CinC-2019 winning entry end-to-end and report a
reproduction failure** [V-sub]. The nearest is Reyna & Clifford arXiv:2012.11013, which shows most
of 70 algorithms failed to generalize to the hidden test set.

### 8.4 MIMIC-IV version differences that matter for sepsis

| Version | Released | Patients | Hospital admissions | ICU stays |
|---|---|---|---|---|
| v2.2 | 2023-01-06 | 299,712 | 431,231 | 73,181 |
| v3.0 | 2024-07-23 | 364,627 | 546,028 | 94,458 |
| **v3.1** | **2024-10-11** | **364,627** | **546,028** | **94,458** |

[V-sub, physionet.org/content/mimiciv/]

1. **v2.2 fixed ~2.5% of medication administration records lacking `hadm_id`.** Antibiotic →
   `hadm_id` linkage drives suspicion-of-infection, so pre-v2.2 SOI cohorts are suspect.
2. **v3.0 extended admissions from 2019 to 2022** (299,712 → 364,627 patients), bringing
   **COVID-era ICU stays into the cohort** and materially changing case mix and
   ventilation/oxygenation distributions. **Every comparison of our v3.1 numbers against a
   v2.x-derived result — including YAIB, Backes, Do, and Dalal — is confounded by this.** It needs
   a sentence in the paper, and it partly undercuts the "first on v3.1" framing in §7.3(2).
3. **v3.1 reverted `itemid` changes** in `d_labitems`/`labevents` that v3.0 introduced, back to
   v2.2 values. **Hard-coded itemids extracted against v3.0 silently mismatch both v2.2 and
   v3.1.** Our `verify_mimic_itemids.py` step is the right defence and should be in the methods.

---

## 9. What I could not verify — read before citing

1. **Prenosis / *NEJM AI* 2025 development paper (doi:10.1056/AIoa2400867).** Publisher returned
   403 repeatedly; not in Europe PMC. **No number from it is verified.** The FDA De Novo record
   (§5.6) *is* verified and should be cited instead.
2. **Romero-Brufau et al., "Why the C-statistic is not informative to evaluate early warning
   scores", *Crit Care* 2015;19:285.** Not fetched. The **"3–10 alarms per 100 patient-days"**
   figure attributed to this line of work is **unverified** and is the number most likely to be
   challenged if we use it. Also unverified: **Saito & Rehmsmeier 2015 PLOS ONE** (the canonical
   "precision-recall is more informative than ROC for imbalanced data"), which did not surface but
   is the standard methodological citation and should be added.
3. **Shimabukuro et al. 2017 InSight RCT.** Abstract verified [V-sub]; the "25 actual cases"
   critique is verified from a citing commentary, not from the RCT itself. Frequently over-cited
   as *the* RCT evidence.
4. **Fleuren et al. 2020 pooled AUROC and pooled sensitivity/specificity.** Springer and PubMed
   both blocked retrieval. The **0.68–0.99 ICU range** comes from snippets; check the PDF.
5. **Epic's formal response to Wong et al.** No peer-reviewed rebuttal found. The claim that ESM
   used antibiotic orders as an input is **[S] from a STAT investigation** and should not be cited
   without confirmation.
6. **Bayesian Health / TREWS FDA status.** Breakthrough Designation (2023) and 510(k) (April 2026)
   are **press-only; no 510(k) number confirmed.**
7. **TREWS alert-rate denominators.** Not obtainable — Nature Medicine paywalled, medRxiv preprint
   403. Only the 89%-evaluated / 38%-confirmed conditional rates exist publicly.
8. **Jin & Lee 2026 numeric results.** MDPI returned 403; metadata verified via Crossref. **The
   open-access PDF is retrievable directly from MDPI — get it, because it bears on §7.2(d).**
9. **The SeFT P-Sepsis table.** I read it via ar5iv [V], but two sources disagreed on the SeFT row
   (B-Acc 70.9 / AUPRC 4.84 vs 74.50 / 8.78) — likely different table rows. **Open the PMLR PDF
   before quoting a specific SeFT number.**
10. **Moor et al. 2023 cohort size.** Published abstract says **136,478 admissions / 25,694 septic
    (18.8%)**; the arXiv preprint says **156,309 stays / 26,734 (17.1%)**. The cohort was refined
    between preprint and publication [V-sub]. Use the published figures.
11. **Lauritsen et al. 2020 numbers.** Two sources gave "0.856 at 3 h / 0.756 at 10 h" and "0.84 /
    0.79"; could not reconcile. **Do not cite either without the primary source.**
12. **Nemati 2018 (AISE) and Calvert 2016** — paywalled at every route; numbers are **[S]**.
13. **YAIB's MIMIC-IV version.** The paper predates v3.0; ~73k stays is consistent with **v2.x**.
    Our v3.1 cohort is ~94k. **Our numbers and YAIB's are not on identical data.**
14. **Our own 93,224 qualifying ICU stays** against v3.1's 94,458 total (§0).
15. **Our own Transformer AUROC** — the brief said 0.751/19.3 h, `RESULTS.md` now says
    0.756/19.0 h. Reconcile.
16. **Ewig 2023, Sepsyn-OLCP, CSRA, and both Düsing federated papers** — abstract-only; cohort and
    label details unknown.
17. **Papers named in my search brief that I could not locate at all:** Guidi, Joshi, Rolnick, and
    Kang sepsis-alert papers; an RCT under the name "ePASS" (Ginestra 2019 is observational). A
    widely-circulating "87.9% of sepsis BPAs cancelled/timed out" statistic traces to a conference
    abstract with no locatable citation — **do not cite.**

---

## 10. Suggested citation spine

The shortest defensible argument, entirely verified:

**Bauer 2025** (three definitions → κ=0.39, 65% single-criterion, mortality 14.9% vs 2.2%)
→ **Johnson 2018** (same database, incidence 31.9% → 9.0% by method)
→ **Rosnati 2021** (same database, 1,797 vs 7,936 cases from one missingness assumption)
→ **Cohen 2024** (within Sepsis-3 alone, definition variance ≥ model variance)
→ **Moor 2021** (2 of 22 studies released label code; AUROC 0.78–0.99 not comparable)
→ **Do 2026** (evaluation strategy alone moves AUROC 0.61↔0.67; matching moves it 0.08)
→ **Wang 2025** (Utility 0.381 → −0.164 externally while AUROC holds at 0.783)
→ **Kamran 2024** and **Currey & Tarabichi 2025** (0.62→0.47 and 0.90→0.85 before treatment/recognition)
→ **Wong 2021 / Wong 2026** (deployed reality: 0.63 then 0.82–0.92, but NNE 109 then 21–35)
→ **Backes 2026** and **YAIB** (what an honest MIMIC-IV number looks like: 0.84/0.099 and 0.836/0.091)
→ **our contribution.**
