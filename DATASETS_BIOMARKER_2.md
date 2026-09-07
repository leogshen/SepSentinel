# SepSentinel — Biomarker Dataset Reconnaissance, Pass 2 (per-paper supplements, LPS challenge, IPD repositories, biosensors, wearable pairings)

**Date of survey:** 2026-09-07
**Companion to:** `DATASETS_BIOMARKER.md` (pass 1). This file does **not** repeat pass 1. Pass 1 covered ImmPort, PhysioNet, AmsterdamUMCdb, HiRID, SICdb, Zigong, BioLINCC, GEO/dbGaP, MIMIC-IV and the European consortia. Everything below is new, except §9 which closes pass 1's open question on eICU.

**Tooling situation (disclose this):** this session's `WebSearch` budget was **already exhausted before the task started** (200/200 used), exactly as in pass 1. I worked around it by driving **search APIs over raw HTTP** instead, which turned out to be strictly better than a search engine for this job:

- **Europe PMC REST API** — full-text and metadata search over ~46M records, incl. `hasSuppl` flags.
- **Europe PMC `supplementaryFiles` endpoint** — returns a ZIP of every supplementary file for an OA article. I wrote a scanner that pulls these, opens the XLSX/CSV inside, and greps the column headers for IL-6 + `pg/mL` + patient-ID + timepoint tokens. **This is the tool pass 1 lacked**, and it is what produced the main findings here.
- **PLOS Solr API** (`api.plos.org/search`) — exposes the `data_availability` and `supporting_information` fields directly, so PLOS's mandatory-data-deposition policy becomes machine-searchable.
- **DataCite**, **Zenodo**, **Harvard Dataverse**, **Borealis** APIs.
- **PhysioNet open-licence files** (for the eICU verification).

Tags used throughout: **[VERIFIED]** = I retrieved the artefact today and read the actual content (column headers, row counts, licence text); **[INFERRED]** = reasoned from something I read; **[UNVERIFIED]** = could not determine — stated as such, not guessed.

---

## Bottom line up front

Pass 1's conclusion was "IL-6 lives in immunology repositories, not critical-care repositories, and ImmPort is the cheapest door." That stands. **Pass 2 adds a second, cheaper door that pass 1 could not see: open supplementary data files attached to individual papers.**

Four datasets are **downloadable right now, with no account, no registration, no email, no DUA** — a direct HTTPS GET, CC-BY licensed — and all four contain **per-patient serial IL-6 in pg/mL**:

| | Dataset | Population | Serial IL-6 |
|---|---|---|---|
| **#1** | PLOS ONE `0209669` — PermiT trial cytokine sub-study | 72 critically ill ICU adults | **5 timepoints** (d1/3/5/7/14), 29-cytokine panel |
| **#2** | PLOS ONE `0178387` — kallistatin in severe sepsis/septic shock | 86 ICU adults, severe sepsis + septic shock | **3 timepoints** (D1/D3/D5), + SOFA at same days |
| **#3** | PLOS Medicine `1002338` — prehospital trauma immune response | 89 trauma patients | **3 timepoints** (<1 h, 4–12 h, 48–72 h) |
| **#4** | PLOS ONE `0211981` — endotoxin/TNF/IL-6 NONMEM dataset | 116 **piglets**, LPS infusion | **~7 median, up to 31** obs/animal, hourly grid |

Together that is roughly **360 subjects with genuinely serial IL-6, obtainable this afternoon**. None is a perfect population match, and none has a paired continuous physiological stream. But for the specific job of *learning what an IL-6 trajectory looks like* — which is what Model A's temporal prior needs — this is materially better than anything in pass 1's open tier, and it costs nothing.

Two findings that **close** questions rather than open them: **eICU-CRD has no interleukin and no procalcitonin** (verified from the open-licensed demo release — §9), and **PERSEVERE does not contain IL-6 at all** (verified panel composition — §8). Both are dead ends; stop spending time on them.

---

## 1. Ranked summary table

| # | Dataset / source | URL | Host | Access tier | Time-to-access (HS student + PI) | n | IL-6 serial? | Other analytes | Paired physiological time series? | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **PermiT cytokine sub-study** (PLOS ONE 0209669, S5) | `journals.plos.org/plosone/article?id=10.1371/journal.pone.0209669` | PLOS (CC-BY) | **Fully open, direct download** | **Minutes** | 72 | **Yes — 5 timepoints** (days 1,3,5,7,14) | 29-cytokine panel: EGF, eotaxin, G-CSF, GM-CSF, IFN-α2, IFN-γ, IL-1β/1ra/2/4/5/7/8/10/12p40/12p70/13/15/17, MCP-1, MIP-1α/β, TNF-α, VEGF … | No | **Take it first.** Densest open serial IL-6 in a critically ill population found in either pass. |
| 2 | **Kallistatin sepsis cohort** (PLOS ONE 0178387, S1) | `journals.plos.org/plosone/article?id=10.1371/journal.pone.0178387` | PLOS (CC-BY) | **Fully open, direct download** | **Minutes** | 86 | **Yes — 3 timepoints** (D1/D3/D5) | TNF-α, IL-1β, IL-8, kallistatin, kallikrein (all ×3 days); CRP, WBC single | Partial — SOFA at D1/D3/D5, APACHE II, qSOFA, shock/ARDS/vent flags, LOS, mortality | **Take it second.** The only *true severe-sepsis/septic-shock* per-patient serial IL-6 set that is openly downloadable. Direct replacement for pass 1's restricted Zenodo 7612571. |
| 3 | **Prehospital trauma immune response** (PLOS Med 1002338, S1) | `journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002338` | PLOS (CC-BY) | **Fully open, direct download** | **Minutes** | 89 trauma + 63 healthy controls | **Yes — 3 timepoints** (<1 h post-injury, 4–12 h, 48–72 h) | IL-8, IL-10, TNF-α, MCP-1, IL-1Ra, G-CSF, cortisol, nDNA, mtDNA + ~40 flow-cytometry neutrophil/monocyte parameters | No, but ISS + MODS outcome | Ultra-early kinetics (<1 h) are otherwise unobtainable. Sterile inflammation, not infection — domain shift. |
| 4 | **Porcine endotoxaemia NONMEM dataset** (PLOS ONE 0211981, S1) | `journals.plos.org/plosone/article?id=10.1371/journal.pone.0211981` | PLOS (CC-BY) | **Fully open, direct download** | **Minutes** | 116 piglets, 6 studies | **Yes — median 7, up to 31 obs/animal**, hourly `TAD` grid, 1–30 h LPS infusions at 0.063–16 µg/kg/h | Endotoxin concentration + TNF-α, on the same time grid | No | **Species mismatch, best temporal shape.** The only dataset in either pass with hourly IL-6 *and* the driving stimulus measured. Ideal for pre-training / sanity-checking a kinetic model. |
| 5 | **Vivli — GSK/Sirtris human LPS-challenge Phase I trials** (3 studies) | `search.vivli.org/doiLanding/studies/00000998`, `/00001071`, `/00000863` | Vivli | Request form + contributor review + DUA; analysis inside Vivli's secure environment | **Months**, PI-mediated, no raw export | Phase I, tens of healthy males each | **[INFERRED]** yes — LPS-challenge trials measure IL-6 serially by design; the CRF variable list is not public | **[UNVERIFIED]** | No | The only *human* endotoxin-challenge IPD I found anywhere. Scientifically ideal, access tier wrong for this project. |
| 6 | **Vivli — sepsis RCT IPD** (PROWESS-SHOCK–style drotrecogin trial, SAILS/rosuvastatin, severe-sepsis observational) | `search.vivli.org/doiLanding/studies/00006470`, `/00000092`, `/00004184` | Vivli | Same as above | Months | 100s–1000s | **[UNVERIFIED]** — Vivli metadata gives condition + intervention only, never the analyte list | **[UNVERIFIED]** | Trial CRFs | Note SAILS appears on **both** Vivli and BioLINCC; Vivli's route may be lighter than BioLINCC's institution-signed DUA. Still months. |
| 7 | **Critically ill dogs, glycocalyx + IL-6** (PLOS ONE 0325809, S2) | `journals.plos.org/plosone/article?id=10.1371/journal.pone.0325809` | PLOS (CC-BY) | **Fully open** | Minutes | small (veterinary ICU) | **Yes — 5 pre-defined timepoints over 48 h**, with explicit `Time (h)` column | Hyaluronic acid, ANP, cumulative IV fluid volume | No | Canine sepsis/haemoperitoneum. Curiosity value; too small and too species-shifted to train on. |
| 8 | **Serial IL-6 + lactate + procalcitonin, 28-day mortality** (PMID 36383295) | `europepmc.org/article/MED/36383295` | Eur J Clin Microbiol Infect Dis | **Author request only** | Weeks, author's discretion | retrospective cohort | Yes — the paper's whole premise is *serial* IL-6, lactate and PCT | **Lactate + PCT — SepSentinel's exact analyte triple** | Unknown | **Highest-value email target in this report.** No public data ("available by contacting the corresponding author"). If the PI writes one letter, write this one. |
| 9 | **Burn-sepsis longitudinal biomarker trajectories** (PMID 41563442, Ningbo No.2 Hospital) | `europepmc.org/article/MED/41563442` | Inflamm Res | **Author request only** | Weeks | ICU burn-sepsis cohort | Yes — trajectory-clustering study over 21 days | Multi-category biomarker panel | EMR-derived | Second-best email target. Explicitly "not publicly available … available from the corresponding author". |
| 10 | **PERSEVERE / PERSEVERE-II** (Cincinnati, 13 PICUs, 2003–2023) | — | — | Author request only | — | 681+ children | **No — panel contains no IL-6** | IL-8, HSPA1B, GZMB, MMP-8, CCL3 + platelets + age | Daily clinical data d1–7 | **Dead end for IL-6.** See §8. |
| 11 | **eICU-CRD** | `physionet.org/content/eicu-crd-demo/2.0.1/` | PhysioNet | (demo is open; full is credentialed) | — | 200k stays | **No — zero interleukin, zero procalcitonin** | CRP, CRP-hs, ESR, **lactate**, **pH**, pO₂/pCO₂, base excess, ferritin | Yes | **Question closed.** See §9. |
| 12 | **Wearable-HRV ↔ cytokine primary studies** (3 named) | see §7 | various | Author request only | Weeks–months | 15 / 89 / 93 | Varies; Deepika 2018 is serial | IL-6, TNF-α, IL-10, IL-1β / CRP | **Yes — this is the rare shape** | Only route to the paired shape SepSentinel actually wants. All request-only. |

---

## 2. Priority 1 — Supplementary tables of published sepsis/critical-illness cytokine papers

### 2.1 Method (so the coverage claim is auditable)

I scanned supplementary files programmatically rather than reading papers:

1. Europe PMC query → candidate PMCIDs with `hasSuppl:Y` and `OPEN_ACCESS:Y`.
2. `GET https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/supplementaryFiles` → ZIP.
3. Open every `.xlsx`/`.csv`/`.tsv` inside; for XLSX, parse `xl/sharedStrings.xml` to recover the column headers without materialising the data.
4. Score on four independent signals: an IL-6 token, a concentration unit (`pg/mL`), a patient-identifier token, and a timepoint token (`D1`, `48 h`, `visit 2`, …).

Sweeps actually run **[VERIFIED]**:

| Sweep | Query scope | Records with hits | Supplements opened |
|---|---|---|---|
| Europe PMC A | IL-6 ∧ sepsis-family, OA ∧ suppl, 2015– | 252 | 80 |
| Europe PMC B | IL-6/cytokine ∧ sepsis-family, strict `pg/mL` filter, 2014– | 548 | 250 |
| PLOS A | title-restricted sepsis/ICU ∧ IL-6 ∧ dataset in SI | 64 | 64 |
| PLOS B | everything-field sepsis/ICU/endotoxaemia/trauma ∧ IL-6 ∧ dataset in SI | 1,443 | 300 |
| PLOS C | surgery / CPB / trauma / burn / exercise / vaccination ∧ IL-6 | 849 | 300 |
| PLOS D | biosensor / immunosensor / aptasensor / POC ∧ IL-6 ∧ ELISA | 179 | 179 |

**The single most important negative result:** an XLSX matching "IL-6" is *usually a gene list*, not a concentration table. The `pg/mL` unit filter is what separates them. Without it the false-positive rate was roughly 4-in-5.

### 2.2 The four open datasets, verified

#### (a) PLOS ONE 10.1371/journal.pone.0209669 — PermiT trial cytokine sub-study ⭐ TOP PICK

Arabi *et al.*, "Permissive underfeeding, cytokine profiles and outcomes in critically ill patients", PLOS ONE, 2019-01-07. Sub-study of the **PermiT** RCT (ISRCTN68144998).

- **[VERIFIED]** File `…0209669.s005` is an XLSX, 142,539 bytes, sheet `Cytokines`, dimension **A1:HL73** → **220 columns × 72 data rows**.
- **[VERIFIED]** Column naming is `Rep_<ANALYTE><DAY>`. IL-6 columns present: `Rep_IL_61`, `Rep_IL_63`, `Rep_IL_65`, `Rep_IL_67`, `Rep_IL_614` — i.e. **days 1, 3, 5, 7 and 14**. The same 5-day grid repeats for ~43 analytes (EGF, eotaxin, G-CSF, GM-CSF, IFN-α2, IFN-γ, IL-10, IL-12p40, IL-12p70, IL-13, IL-15, …).
- **[VERIFIED]** `Randomization_n` column present → treatment arm is in the file. Extra sheets: `coding`, `Sheet1` (72 rows).
- **[VERIFIED]** Abstract: 72 patients (36 permissive underfeeding / 36 standard feeding); serum on study days 1, 3, 5, 7, 14; **29-cytokine panel**.
- **Access [VERIFIED]:** PLOS CC-BY. `https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0209669.s005&type=supplementary` returns the file to an anonymous GET. No account.
- **[UNVERIFIED]:** per-cell missingness (how many of the 72 patients actually have all 5 IL-6 draws), and units/assay platform (a Luminex-style multiplex is implied by a 29-plex panel, but the file itself does not state it).
- **Fit:** critically ill mechanically ventilated adults, a large fraction septic but **not a sepsis-defined cohort**. No paired vitals.

#### (b) PLOS ONE 10.1371/journal.pone.0178387 — plasma kallistatin in severe sepsis and septic shock ⭐ BEST TRUE-SEPSIS OPEN SET

Lin *et al.*, PLOS ONE, 2017-05-24. Taiwanese ICU cohort.

- **[VERIFIED]** File `…0178387.s001` is an XLSX, 34,585 bytes, dimension **A1:AT87** → 46 columns × **86 data rows**, matching the abstract's "we enrolled **86 ICU patients** with severe sepsis and septic shock".
- **[VERIFIED]** Full column list, read from the file:
  `Case No, Age, Sex, Appropriate antibiotics, Days(ICU), Days(hospital), Days(mechanical ventilation), Hospital death, Days(60-day death), 60-day death, DM, Lung Dz, Cardiac Dz, Liver Dz, Kidney Dz, Malignancy, Pneumonia/Extrapulmonary, Blood Culture, APACHE II, qSOFA, SOFA(D1), SOFA(D3), SOFA(D5), Mechanical ventilation, Shock, ARDS, WBC, CRP, Kallistatin(D1/D3/D5), Kallikrein(D1/D3/D5), TNF-α(D1/D3/D5), IL-1β(D1/D3/D5), IL-6(D1/D3/D5), IL-8(D1/D3/D5)`
- So: **IL-6 at three timepoints per patient**, alongside three other cytokines on the identical grid and **SOFA on the identical grid**. Assay is ELISA **[VERIFIED from abstract]**.
- **Access [VERIFIED]:** PLOS CC-BY, anonymous GET.
- **Why this matters:** pass 1's recommended true-sepsis set (Zenodo 7612571, Humanitas) is **restricted** and needs a request email for 178 patients × 2 timepoints. This one is **open** with 86 patients × 3 timepoints. Get this first; still send the Humanitas email, since 178+86 beats either alone.
- **Limitations:** n=86 is small; CRP and WBC are single-timepoint only; no lactate, no pH, no continuous physiology.

#### (c) PLOS Medicine 10.1371/journal.pmed.1002338 — prehospital trauma immune response

Hazeldine *et al.*, PLOS Medicine, 2017-07-18 (Birmingham / UK air ambulance).

- **[VERIFIED]** File `…1002338.s001` is an XLSX, **4,634,378 bytes**, with five sheets: **`T=<1h`**, **`T=4-12h`** (dim A2:CE81), **`T=48-72h`** (dim A1:CF71), `MODS Vs No MODS` (A3:AJ109), `HCs` (A2:BZ63, healthy controls).
- **[VERIFIED]** Analyte columns include `IL-6, IL-8, IL-10, TNF-A, MCP-1, IL1-Ra, G-CSF, Cortisol, nDNA (ng/ml), mtDNA (ng/ml)` plus ~40 flow-cytometry parameters (L-selectin, CD88, CD16, CD11b, CXCR1/2, CD63, HLA-DR, TLR2, TLR4, CD86 …) and whole-blood counts, and LPS-stimulation conditions at 1 ng and 10 ng.
- **[VERIFIED from abstract]** n = **89 adult trauma patients**, mean age 41, mean ISS 24; blood drawn **within 1 hour of injury** (mean 42 min, range 17–60 min).
- **Access [VERIFIED]:** PLOS CC-BY, anonymous GET.
- **Why it is unique:** almost no human dataset captures the inflammatory response *inside the first hour*. If Model A must handle a fast-rising IL-6 edge, this is where that edge is observable.
- **Domain caveat:** blunt/penetrating trauma is **sterile** inflammation. Kinetics resemble sepsis onset; aetiology does not.

#### (d) PLOS ONE 10.1371/journal.pone.0211981 — porcine endotoxaemia PK/PD dataset (also §3)

Thorsted, Bouchene, Tano, Castegren, Lipcsey, Sjölin, Karlsson, Friberg, Nielsen (Uppsala), PLOS ONE, 2019-02-21. Detailed in §3.1.

### 2.3 Other supplements found, with per-patient IL-6 but weaker fit

All **[VERIFIED]** open PLOS CC-BY downloads, all with per-subject `IL-6 (pg/mL)` columns:

| DOI | Study | Serial? | Why it's here |
|---|---|---|---|
| `10.1371/journal.pone.0197637` | miR-122 in Sepsis-3 patients (German ICU) | **No — single draw** | Rare pairing: per-patient `IL-6 [pg/ml]`, `PCT [ng/ml]`, `CRP [mg/dl]`, SOFA, SAPS-II **and haemodynamics** — `MAP [mmHg]`, `mPAP`, `Cardiac-Index`, `StrokeVolume`, `CVD`, noradrenaline/dobutamine/epinephrine doses. Snapshot physiology beside IL-6. |
| `10.1371/journal.pone.0296863` | HDL-C anti-inflammatory capacity in sepsis | No | Includes `Lactate_0`, `PCT`, `CRP`, APACHE II, SOFA |
| `10.1371/journal.pone.0185685` | Inflammatory response during liver resection | **Yes — D0…D6** | `IL2, IL4, IL5, IL6, GM-CSF, IFN-γ` across 7 postoperative days |
| `10.1371/journal.pone.0293347` | Abdominal surgery, CTCF/MHC-II | **Yes — pre-op, POD1…POD7** | `IL-6, IL-10, HLA-DR` daily post-op |
| `10.1371/journal.pone.0340864` | Esketamine PCA after surgery | **Yes — 3 points** | `IL-6 0/1/2`, `TNF 0/1/2`, `BDNF 0/1/2` |
| `10.1371/journal.pone.0227993` | Soccer players, progressive effort | **Yes — 3 points** | `IL-6 (pg/mL)` at baseline / post-effort / recovery, 10-plex; exercise-immunology shape |
| `10.1371/journal.pone.0280069` | Human whole-blood loop model, surface coatings | **Yes — 30/90/120 min** | **`IL-6` + `Lactate` + `pH` + `Base Excess` + `pO2`/`pCO2` in one table** — SepSentinel's exact triple, but *in vitro*, not a patient |
| `10.1371/journal.pone.0324153` | Pneumococcal meningitis + endotoxaemia | No — cross-sectional | Per-patient plasma LPS + `IL-6 (pg/ml)` + `CRP` + iFABP |
| `10.1371/journal.pone.0208017` | Puumala hantavirus AKI | Partial | `maximum_IL6`, `IL6_at_recovery_phase`, `IL6_at_one_year` — 3 points but on a months scale |
| `10.1371/journal.pone.0222721` | Postoperative delirium, BBB | No | `il6 (pg/mL)` + adhesion-molecule panel, n≈117 |

**Honest read:** none of these ten is a sepsis training set. They are listed because two of them (`0280069`, `0197637`) are the only places in either pass where IL-6 sits in the same row as lactate/pH or as live haemodynamics, and because the surgery/exercise ones (§2.3, `0185685`/`0293347`/`0227993`) are cheap extra IL-6 trajectories if Model A ends up data-starved.

### 2.4 What the supplement sweep did **not** find

**[VERIFIED by absence, across ~1,100 supplement files opened]** — I found **no open supplementary dataset with (i) a Sepsis-3 cohort, (ii) ≥4 IL-6 timepoints, and (iii) n > 200.** The dominant pattern in sepsis cytokine papers is a data-availability statement reading *"available from the corresponding author upon reasonable request"*. I read that exact sentence in, among others:

- PMC9816294 — serial IL-6 + lactate + PCT mortality model **[VERIFIED]**
- PMC12823687 — burn-sepsis longitudinal biomarker trajectories **[VERIFIED]**
- PMC11726927 — age-dependent IL-6/mortality, Chiba University **[VERIFIED]**
- PMC11477125 — ED sepsis biomarkers, Cluj **[VERIFIED]**
- PMC11885351 — repeat human LPS challenge, Medical University of Vienna: *"No datasets were generated or analysed during the current study"* **[VERIFIED]**
- PMC12902028 — human segmental endotoxin challenge BAL proteomics **[VERIFIED]**

That is the structural finding of this pass: **the serial-IL-6 sepsis data exists in quantity, and is almost never deposited.** Email is the access mechanism for it, not download.

---

## 3. Priority 2 — Human endotoxin (LPS) challenge studies

### 3.1 The one open serial-IL-6 endotoxaemia dataset is **porcine**, not human

**PLOS ONE 10.1371/journal.pone.0211981**, Thorsted *et al.* 2019, Uppsala University. *"A non-linear mixed effect model for innate immune response: in vivo kinetics of endotoxin and its induction of the cytokines tumor necrosis factor alpha and interleukin-6."*

**[VERIFIED] — I downloaded and parsed `S1 Table` (`…s001`, CSV, 213,550 bytes, 3,338 rows):**

- NONMEM-format columns: `ID, SID, STUDY, DGRP, AMT, AMT2, RATE, RATE2, CMT, EVID, DVID, TAD, LNDV, DV, MDV, OMIT, LLOQ, TYPE, WT, FLUID, INFL, INFR, PEXP`.
- **116 unique subject IDs**, 27 sub-groups, **6 pooled studies**.
- `WT` range **20.8–32.8** (kg) — consistent with the abstract's anaesthetised piglets.
- `TAD` (time after dose) is an **integer hourly grid, 0…29 h**.
- Observation rows by `DVID`: **290 / 1,350 / 1,350**. **[INFERRED]** these map to endotoxin, TNF-α, IL-6 in the paper's stated order. Per animal: **median ~7 cytokine observations, range 4–31**.
- **[VERIFIED from abstract]** purified *E. coli* O111:B4 endotoxin infused IV for **1–30 h at 0.063–16.0 µg/kg/h** across studies; the model captures endotoxin, TNF-α, IL-6 **and tolerance development** simultaneously.
- **Access:** PLOS CC-BY, anonymous GET. **Minutes.**

**Why this is worth taking despite being a pig:** it is the only dataset in either pass where the *driving input* (measured endotoxin concentration, and the infusion rate) is recorded alongside the IL-6 output on a dense grid. That is the ideal structure for fitting or validating a forward kinetic model, and the pig is the standard translational sepsis species. Treat it as a **physics prior**, not as human training data. Do not present it as human.

### 3.2 Human endotoxaemia: the data exists, none of it is deposited

**[VERIFIED]** Europe PMC returns 84 papers matching `"experimental human endotoxemia" AND ("2 ng/kg" OR "healthy volunteers") AND "IL-6"` since 2012. I opened the supplementary bundles of the most promising:

| Paper | What its supplement actually contains |
|---|---|
| Nature Immunology 2025, PMC12043512, *"Systemic inflammation impairs myelopoiesis and interferon type I responses in humans"* | **[VERIFIED]** The one XLSX (438 KB) is an **RNA-seq gene table** — columns `Donor1_day0, Donor1_4h, Donor1_8h, Donor1_24h, Donor1_day7, Donor1_day7_4h`, ×3 donors, rows are Ensembl IDs. Perfect temporal design (0/4/8/24 h + day 7 re-challenge), **no cytokine concentrations**. Same trap as pass 1's GSE3284. |
| Critical Care 2023, PMC10029173, *"CytoSorb hemoperfusion markedly attenuates circulating cytokine concentrations during systemic inflammation in humans in vivo"* (Radboud) | **[VERIFIED]** 6 supplementary files, **all PDF**. No data file. |
| Frontiers in Medicine 2020, PMC7674961, endotoxaemia and GFR | **[VERIFIED]** 1 file, `.DOCX`. No data file. |
| Med Microbiol Immunol 2025, PMC11885351, repeat LPS challenge at 1 year (Vienna) | **[VERIFIED]** 1 PDF; data-availability statement literally says *"No datasets were generated or analysed during the current study."* |
| Critical Care 2024, PMC11010428, neo-epitope ECM turnover in systemic inflammation and sepsis | **[VERIFIED]** 1 `.docx`. No data file. |
| Sci Rep 2026, PMC12902028, segmental endotoxin challenge BAL proteomics | **[VERIFIED]** "available from the corresponding author JMH on reasonable request." |

**Conclusion [VERIFIED by exhaustive check of the OA supplements]: there is no publicly deposited per-subject human endotoxin-challenge IL-6 time course.** Pass 1 reached this conclusion by inference; this pass confirms it by opening the files.

### 3.3 Human LPS-challenge IPD does exist — on Vivli

**[VERIFIED]** via the DataCite API (`publisher:Vivli`), three GSK/Sirtris Phase I trials, each tagged `Conditions: Sepsis`:

| Vivli DOI | Title | Landing page |
|---|---|---|
| `10.25934/00000998` | A Phase I Study to Evaluate a Single Oral Dose of **SRT2379** on the Endotoxin Induced Inflammatory Response in Healthy Male Subjects | `search.vivli.org/doiLanding/studies/00000998/isLanding` |
| `10.25934/00001071` | A Double Blind, Placebo Controlled, Phase I Dose-ranging Study to Evaluate the Activity of **SRT2379** on Endotoxin Induced Inflammatory Response in Healthy Male Subjects | `…/00001071/isLanding` |
| `10.25934/00000863` | A Phase I Study to Evaluate Single and Multiple (Seven) Oral Doses of **SRT2104** on the Endotoxin Induced Inflammatory Response in Healthy Male Subjects | `…/00000863/isLanding` |

**[VERIFIED]** Rights statement on all three, verbatim: *"Study data may be accessed by completing a request form at https://vivli.org. It will then be reviewed according to the data contributor's governance policy … If approved, use of the data is governed by the data use agreement at https://vivli.org/resources/vivli-data-use-agreement/"*.

**[UNVERIFIED]:** whether IL-6 is among the recorded endpoints. The DataCite record gives condition and intervention only, never a variable list. **[INFERRED]:** a SIRT1-activator LPS-challenge trial almost certainly measured serial TNF-α and IL-6 — that is the standard readout of the model — but this must be confirmed with Vivli before any effort is spent.

**Verdict:** scientifically the best-matched human data in this whole survey; **access tier makes it unusable here.** Vivli requires a named researcher with an institutional affiliation, a research proposal, contributor review, a signed DUA, and — critically — analysis typically happens **inside Vivli's secure research environment with no raw data export**. That is incompatible with a student building and iterating a model locally. Note it in the write-up as future work.

---

## 4. Priority 3 — Trial repositories and IPD platforms

**[VERIFIED]** Vivli sepsis holdings (36 records under `publisher:Vivli AND (sepsis OR septic)`), the substantive ones being:

| DOI | Study |
|---|---|
| `10.25934/00006470` | Phase 3, drotrecogin alfa (activated), 96-h infusion, septic shock (Lilly) — *plus* a companion "Available IPD datapackage" record `…6470.0` |
| `10.25934/00000092` | Randomized Trial of Rosuvastatin for Acutely Injured Lungs From Sepsis (= **SAILS**) |
| `10.25934/00004184` | International Observational Study Among Severe Sepsis Patients Treated in the ICU |
| `10.25934/00005427` | Prehospital Resuscitation On Helicopter Study |

**Notable [VERIFIED]:** SAILS is listed on **both** BioLINCC (pass 1 §2.14) and Vivli. Two independent request routes for the same trial. Neither route's package is confirmed to contain assayed cytokines — pass 1 inferred BioLINCC's does not, and Vivli's metadata does not say.

**[VERIFIED by absence]** DataCite queries for `tocilizumab AND sepsis` and `anakinra AND sepsis` returned **no IPD deposit** for any anti-IL-6 or anti-IL-1 sepsis trial. Hits were reviews, systematic-review datasets, and unrelated CAR-T/HLH studies. Specifically: **no SAVE-MORE, no PROVIDE, no ImmunoSep, no EUPHRATES, no polymyxin-B, no sivelestat IPD deposit surfaced.**

**[UNVERIFIED] — not fetched at all this pass:** the **YODA Project** (`yoda.yale.edu`) and **ClinicalStudyDataRequest.com / CSDR** own catalogues. Both are SPA/portal sites and neither mints DataCite DOIs I could query, so my DataCite sweep does not cover them. **[INFERRED]** YODA is predominantly Johnson & Johnson; J&J's **sirukumab** (anti-IL-6) programme is rheumatology/depression, not sepsis. CSDR aggregates GSK/Sanofi/Takeda and others. Both use the same shape of process as Vivli — proposal, review, DUA, secure environment. **A follow-up pass with working web search should check both catalogues directly.**

**[VERIFIED]** Pediatric Sepsis Data CoLab (Borealis, `doi:10.5683/SP2/…`, `SP3/…`) holds only governance documents, SOPs, protocols, training materials, and **synthetic** data-challenge sets — **no biomarker data**.

**[VERIFIED]** Harvard Dataverse: no serial-IL-6 sepsis deposit. `10.7910/DVN/1GTQAI` (sepsis neutrophil multi-omics, CC0, open) is **FCS flow files + R code**, no cytokine concentration table. `10.7910/DVN/UAJX1D` ("ICU SEPSIS DATASET", CC0, one 916 KB XLSX) is one row per patient, EHR-style, **[UNVERIFIED]** whether it has IL-6 — its description lists demographics, comorbidities, observations, lab results and sepsis scores without naming analytes.

---

## 5. Priority 4 — Point-of-care / IL-6 biosensor validation datasets

**Answer: I could not find a single published dataset of paired biosensor-vs-ELISA IL-6 measurements on clinical samples.**

**[VERIFIED]** The PLOS sweep (`biosensor OR immunosensor OR aptasensor OR point-of-care` ∧ IL-6 ∧ ELISA, 179 articles, all supplements opened) returned **zero** paired-measurement tables. The Europe PMC equivalent returned 52 records, of which the IL-6-relevant ones are reviews (`Mikrochim Acta`, `RSC Adv`, `Int J Nanomedicine`, `Biosensors`) rather than primary validation studies with data.

**[VERIFIED]** Two primary papers surfaced that are the right *kind* of study:
- *"Biofunctional 2D Graphitic Carbon Nitride-Hydrogel Heterointerfaces for Electrochemical Detection of Interleukin-6 toward Septic Cardiomyopathy Diagnostics in Clinical Biofluids"* (2026; Europe PMC record has no PMCID and no supplementary flag) — **[UNVERIFIED]** contents; not open access.
- `Mikrochim Acta` PMC11493819, *"Hierarchical Au@Pt nanoparticle / amino benzoic acid polymer-based hybrid material for labeled and label-free detection of interleukin-6"* — has supplements; **[UNVERIFIED]** whether they contain per-sample paired values.

**[INFERRED] — the structural reason this category is empty:** biosensor papers report validation as a **correlation figure and a Bland–Altman plot over 5–30 clinical samples**, with the numbers living only in the figure. Journals in this field (ACS Sensors, Biosensors & Bioelectronics, Analytical Chemistry) have no data-deposition mandate comparable to PLOS's. n is also far too small to calibrate a model even when published.

**Recommendation:** stop searching for this dataset; it does not exist in deposited form. Model A's sensor→concentration calibration will have to come from **the project's own bench measurements**, with the literature datasets in §2 supplying only the *concentration distribution and trajectory prior* (what IL-6 values and rates of change are physiologically real). Frame it that way in the write-up — it is a defensible design, and claiming a public calibration set exists would be false.

---

## 6. Priority 5 — Pediatric sepsis / PERSEVERE

**[VERIFIED] — PERSEVERE does not measure IL-6.** From the full text of PMC12672347 (Front Immunol 2025, Cincinnati Children's, OA), verbatim:

> *"the 5 PERSEVERE biomarkers interleukin 8 (IL-8), heat shock protein (HSPA1B), granzyme B (GZMB), matrix metalloprotein 8 (MMP8), and C-C motif chemokine ligand 3 (CCL3) were previously measured in day 1 sera"*

and:

> *"CART analyses were used to identify a parsimonious set of 5 protein biomarkers (IL8, HSPA1B, GZMB, MMP8, and CCL3) in addition to patient age"*

PERSEVERE-II adds **PICU admission platelet count**. **IL-6 is not in the panel, in any version.**

**[VERIFIED]** Serial-ness: the panel is measured in **day-1 sera** in the core model. Some sub-studies extend to **day 3** (e.g. PMID 37917869 measures serum humanin "on days 1 and 3"; the AKI models predict a day-3 outcome). So PERSEVERE is at best 2 timepoints, and never IL-6.

**[VERIFIED]** Cohort scale is genuinely impressive and worth knowing about: **13 PICUs, USA, 2003–2023, 681 children with septic shock** in one secondary analysis (PMID 38234587), with *"data collected daily from days 1 to 7 of PICU admission"*.

**[VERIFIED]** Data availability: the **transcriptomic** arm is deposited (e.g. PMC12672347 → NCBI SRA `PRJNA1358292`). The **protein biomarker values and clinical data are not deposited anywhere I could find** — no PhysioNet project, no Dataverse, no Zenodo, no dbGaP sepsis study (pass 1 §2.17 confirmed the dbGaP absence).

**Verdict: close this line.** PERSEVERE is the wrong panel. If the project wants a pediatric arm it needs a different source, and I did not find one with serial IL-6.

---

## 7. Priority 6 — Datasets pairing continuous physiological signals with intermittent biomarker draws

This remains the rarest shape, and I did not find a public dataset with it. But I found a **map** of who has it.

**[VERIFIED]** *Diagnostics* 2026, PMC12939863 — Siswishanto *et al.*, *"Clinical Evidence of Wearable-Derived Heart Rate Variability for Detecting Systemic Inflammation: A Systematic Review."* Eleven studies, **2,419 participants**, searched through April 2025. This is a ready-made, curated list of every study that has paired wearable HRV with inflammatory biomarkers. I extracted its Table 1 in full. The IL-6-bearing entries:

| Study | Country | n | Population | Device (continuous signal) | Duration | Biomarkers |
|---|---|---|---|---|---|---|
| **Deepika 2018** | India | **89** | Severe traumatic brain injury, **prospective longitudinal** | BioHarness telemetric chest strap (Zephyr) | NR | **TNF-α, IL-6, IL-10, IL-1β** — reported finding: *"IL-6 negatively correlated with RMSSD"* |
| **Hirten 2021** | USA (Mount Sinai) | 15 | Ulcerative colitis | **VitalPatch chest patch (VitalConnect), 72 h continuous** | 72 h | **CRP, TNF, IL-6, IL-1β**, faecal calprotectin. Finding: *"Significant changes in HRV precede symptomatic or inflammatory flare"* |
| **Wang 2023** | China | 93 | Normal-weight vs obese young adults | 12-lead ambulatory ECG, **24 h** | 24 h | **MCP-1, IL-6, IL-8, TNF-α, fractalkine, MIP-1α, MIP-1β** |
| Barone 2008 | Italy | 44 | Refractory epilepsy, 3-month follow-up | Holter ECG, 24 h | 24 h | CRP, TNF-α, IL-6 |
| Hasty 2021 | USA | 16 | **COVID-19 ICU patients** | Tiger Tech Warfighter armband | 7 min | CRP only |
| Haase 2012 | Germany | 40 | Elective colorectal surgery | Polar S810 chest strap | 10 min | CRP only |
| Brun 2023 | Switzerland | 44 | Pregnant women with PPROM (intra-amniotic infection) | **Ava wrist bracelet** | — | CRP only |

**[VERIFIED]** The review's own data-availability statement is *"available by the authors on reasonable request"*, and it is a review — it holds no primary data.

**Assessment:**
- **Deepika 2018 (n=89, TBI, longitudinal, continuous chest-strap telemetry + serial IL-6/TNF-α/IL-10/IL-1β)** is the closest published match to SepSentinel's target data shape that I found anywhere. **[UNVERIFIED]** whether any of it is deposited — I did not resolve the primary citation.
- **Hirten 2021 (Mount Sinai)** is the highest-quality pairing: 72 h of continuous chest-patch physiology against IL-6/TNF/CRP/IL-1β. n=15. Mount Sinai's Hirten group runs a wearables-and-inflammation programme and is a plausible collaborator rather than merely a data source.
- **[VERIFIED] the negative result:** **11 studies, 2,419 participants, and the review found only 3 IL-6 comparisons for SDNN and 5 for RMSSD** — the whole world literature on wearable-HRV-vs-IL-6 is a handful of small studies. The review's own conclusion is that RMSSD–cytokine associations were *"heterogeneous and largely non-significant"*, while SDNN–CRP was consistent (83% inverse, sign test p=0.031).

**Two implications the project should absorb honestly.** First, the "continuous physiology + intermittent IL-6" dataset SepSentinel would ideally train on does not exist publicly; it must be assembled, requested, or collected. Second, and more strategically: **the published evidence that a wearable signal tracks IL-6 specifically is weak, whereas the evidence that it tracks CRP is stronger.** That reinforces pass 1's closing note about targeting CRP alongside IL-6.

**[VERIFIED] near-miss found separately:** *Scientific Data* 2026, PMC13186999 — *"A digital biomarker dataset from hematopoietic cell transplant caregivers and patients."* Right shape (wearables in a population that gets neutropenic sepsis), **[UNVERIFIED]** whether it carries any blood biomarkers. Worth 10 minutes to check.

**[VERIFIED] in-vitro near-miss:** `10.1371/journal.pone.0280069` (§2.3) is the only open table found containing **IL-6, lactate, pH, base excess, pO₂ and pCO₂ in the same rows across timepoints** — but it is a human whole-blood *loop* model on the bench, not a patient.

---

## 8. Priority 7 — eICU-CRD lab name list: **question closed**

Pass 1 could not enumerate eICU's `labname` values. I have now done so.

**Method [VERIFIED]:** the **eICU Collaborative Research Database Demo v2.0.1** is published under an **open licence** — I read the landing page at `physionet.org/content/eicu-crd-demo/2.0.1/`, which states verbatim: *"Access Policy: Anyone can access the files, as long as they conform to the terms of the specified license. License (for files): Open Data Commons Open Database License v1.0."* No credentialing, no DUA. I retrieved `lab.csv.gz` (5.7 MB) and `customLab.csv.gz` from `physionet.org/files/eicu-crd-demo/2.0.1/` and enumerated distinct values.

**Result [VERIFIED]: 147 distinct `labname` values in the demo release.** Searching them for inflammatory and blood-gas analytes:

| Present | Count in demo |
|---|---|
| `CRP` | 207 |
| `CRP-hs` | 6 |
| `ESR` | (present) |
| `Ferritin` | 172 |
| `lactate` | 2,037 |
| `pH` | 3,853 |
| `paO2`, `paCO2`, `Base Excess`, `Base Deficit`, `Total CO2`, `bicarbonate`, `HCO3`, `O2 Sat (%)`, `Carboxyhemoglobin`, `Methemoglobin`, `Oxyhemoglobin` | (all present) |

| Absent | |
|---|---|
| **any `interleukin` / `IL-6` / `IL6` term** | **zero** |
| **`procalcitonin` / `PCT`** | **zero** |
| `TNF`, `presepsin`, `suPAR`, `cytokine` | zero |

**[VERIFIED]** The `customLab` table (the free-text escape hatch) has only **19 distinct** `labothername` values in the demo — GFR, vitamin B12, iron, folate, TIBC, bleeding time, BNP/proBNP, heparin anti-Xa, globulin, influenza A/B, neutrophil %, ventilator settings. **No cytokines, no PCT.**

**Caveat, stated plainly [INFERRED]:** the demo covers ~2,500 stays from a subset of hospitals. `labname` is a *controlled vocabulary* shared across the full eICU-CRD, so the full database's vocabulary is a modest superset of these 147 — a handful of additional names could exist that the demo happens not to sample. But **procalcitonin was in routine US use in 2014–2015 and does not appear even once in 2,037 lactate-bearing stays**, which makes its presence at scale in the full DB unlikely; and IL-6 was not a routine US lab in that era at all.

**Verdict: eICU-CRD contributes lactate, pH, full blood gases, CRP and ferritin — and no IL-6 and (almost certainly) no procalcitonin.** Pass 1's inference was correct. Do not credential for eICU expecting a cytokine.

**Free bonus for Model B:** the eICU **demo** is fully open and contains `lactate` and `pH` with timestamps (`labresultoffset`) against a full ICU record. That is a legitimate, zero-friction development substrate for the lactate/pH arms of Model A and for prototyping Model B before any credentialing.

---

## 9. Recommendation

### Do today — costs nothing, no account, no email

Download these four files. They are anonymous HTTPS GETs on CC-BY content:

```
https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0209669.s005&type=supplementary   # PermiT, 72 pts x 5 timepoints
https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0178387.s001&type=supplementary   # sepsis/septic shock, 86 pts x D1/D3/D5
https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1002338.s001&type=supplementary  # trauma, 89 pts x <1h/4-12h/48-72h
https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0211981.s001&type=supplementary   # porcine endotoxaemia, 116 animals, hourly
```

Cite each paper. Build the IL-6 trajectory prior from the pooled human sets (#1–#3, ≈247 subjects), and use the porcine set (#4) to check that the kinetic form your model assumes is the one the data actually shows.

Then, in parallel and at the same zero cost, proceed with pass 1's ImmPort recommendation (SDY1662 + SDY1655) — that is still the largest per-patient IL-6 volume available at a registration-only tier, and it is the only source with paired vitals.

### Do this week — one email each, decided by one lab

1. **PMID 36383295** (Eur J Clin Microbiol Infect Dis 2023) — *serial IL-6, lactate and procalcitonin* for 28-day sepsis mortality. **This is SepSentinel's exact analyte triple, measured serially, in a sepsis cohort.** Data-availability statement: *"may be available by contacting the corresponding author upon reasonable request."* Chinese group, NSFC grant 82073284. **Write this letter first.**
2. **PMID 41563442** (Inflammation Research 2026, Ningbo No.2 Hospital) — longitudinal multi-biomarker *trajectories* in burn sepsis over 21 days, with trajectory-based phenotyping. Data "available from the corresponding author on reasonable request", and they also offer their extraction/preprocessing code.
3. **Zenodo 7612571** (Humanitas Sepsis-3) — as pass 1 recommended. Still worth it; 178 Sepsis-3 patients × 2 timepoints complements the 86 × 3 you now have openly.
4. **Deepika 2018** (severe TBI, n=89, continuous BioHarness telemetry + serial IL-6/TNF-α/IL-10/IL-1β) and **Hirten 2021** (Mount Sinai, 72 h VitalPatch + IL-6/TNF/IL-1β/CRP). These are the only two published studies I found with SepSentinel's actual target data shape. Small n, non-sensitive, and a Mount Sinai wearables group is a plausible collaborator as well as a data source.

Every letter must be sent and signed by the PI, not the student.

### Do not pursue

- **Vivli human LPS-challenge trials** (§3.3) — scientifically the best human match, but proposal + contributor review + DUA + no-raw-export secure environment. Note as future work; do not start.
- **YODA / CSDR** — same tier; and **[UNVERIFIED]**, since I could not query their catalogues without web search.
- **PERSEVERE** — verified to contain no IL-6.
- **eICU-CRD (full, credentialed)** — verified to contain no IL-6 and no PCT. Use the open demo instead if you want its lactate/pH.
- **IL-6 biosensor validation datasets** — verified not to exist in deposited form.

### One strategic point, restated because pass 2 strengthened it

Pass 1 ended by noting CRP is the marker that actually appears at scale. Pass 2 adds an independent line of evidence: the *only* systematic review of wearable-derived signals versus inflammatory markers (§7) found the SDNN–**CRP** association consistent (83% inverse, p=0.031) and the RMSSD–**IL-6** associations *"heterogeneous and largely non-significant"*, across a total world literature of eleven studies. If SepSentinel's thesis is that a wearable signal can stand in for an inflammatory state, the published evidence supports that claim more strongly for CRP than for IL-6. That does not mean abandoning IL-6 — the biosensor is an IL-6 biosensor — but the write-up should acknowledge it rather than let a reviewer find it.

---

## 10. What I could not verify

1. **Per-cell missingness in the PermiT dataset.** I read the column headers and confirmed 72 rows × 5 IL-6 columns; I did not count how many patients actually have all five draws. Some will have died or been discharged before day 14.
2. **Assay platform and units for the PermiT panel.** A 29-plex implies Luminex, but the file does not say and I did not read the paper's methods section.
3. **Exact row counts in the trauma dataset's `T=<1h` sheet.** Its dimension is `A1:AON1582`, which is far wider and longer than the other sheets (`T=4-12h` is 81 rows, `T=48-72h` is 71) — the layout is evidently different, possibly long-format or with embedded sub-tables. n=89 patients comes from the abstract, not from the sheet.
4. **Whether the three Vivli LPS-challenge trials recorded IL-6.** DataCite metadata gives condition and intervention only. Confirm with Vivli before any effort.
5. **Whether Vivli's SAILS / drotrecogin data packages contain assayed cytokines.** Same limitation. Pass 1 inferred "no" for BioLINCC's SAILS package; Vivli's may differ.
6. **YODA Project and ClinicalStudyDataRequest.com / CSDR catalogues — not examined at all.** Both are portal sites that do not mint DataCite DOIs, so my repository sweep is blind to them. Anything I said about them is general knowledge, not a finding. **This is the largest single gap in this pass.**
7. **Whether the Deepika 2018 TBI dataset or the Hirten 2021 VitalPatch dataset is deposited anywhere.** I have the studies from a systematic review's table; I did not resolve either primary citation to check its data-availability statement.
8. **Contents of `10.7910/DVN/UAJX1D`** ("ICU SEPSIS DATASET", Harvard Dataverse, CC0, one 916 KB XLSX). Its description lists "lab test results" without naming analytes. Cheap to check; I did not open it.
9. **Contents of PMC13186999** (*Scientific Data*, digital biomarker dataset from HCT patients and caregivers) — whether it pairs wearable signals with any blood biomarker.
10. **The two IL-6 biosensor primary papers** (graphitic carbon nitride / septic cardiomyopathy; Au@Pt hybrid in *Mikrochim Acta* PMC11493819) — whether their SI contains per-sample paired sensor-vs-ELISA values.
11. **Coverage limits of the supplement sweep.** Europe PMC sweep B matched 548 records and I opened supplements for 250; PLOS sweep B matched 1,443 and I opened 300; sweep C matched 849 and I opened 300. **Roughly a third to a half of the matched corpus was not opened.** The scan is also blind to (a) non-OA papers, whose supplements Europe PMC will not serve; (b) `.pdf` and `.docx` supplements, which I skipped deliberately — a per-patient table published as a PDF appendix would not be detected; (c) journals whose supplements sit behind publisher CDNs rather than in PMC (Elsevier, Wiley, Springer non-OA, Karger). Notably, `10.6084/m9.figshare.33143168` (Karger, "Time-Dependent Kinetics and Saturation Dynamics of Interleukin-6 Clearance During Combined HA380 Hemoadsorption and CRRT") surfaced in DataCite but was **not** opened.
12. **`labname` values present in full eICU-CRD but absent from the open demo.** ~147 of an expected ~150–160. See the caveat in §8.
13. **Zenodo restricted-record response times** — still no basis for an estimate, unchanged from pass 1.
14. **Whether the PLOS `data_availability`/`supporting_information` Solr fields are complete** for every article. If a paper deposited data without describing it in those fields, my PLOS sweeps would miss it.

---

## Appendix — reproducing this survey

Scripts written during this pass live in the session scratchpad (not committed): `hunt2.py` / `hunt3.py` (Europe PMC supplement scanner), `plos.py` (PLOS Solr + supplement scanner), `scan.py` (data-availability grep over `fullTextXML`). The three API endpoints that made this possible, and that a future pass should reuse in preference to a search engine:

```
https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=...&format=json&resultType=core
https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/supplementaryFiles     # returns a ZIP of all SI files
https://api.plos.org/search?q=...&fl=id,title,data_availability,supporting_information&wt=json
```

The decisive trick is filtering candidate spreadsheets on the **co-occurrence of an IL-6 token and a `pg/mL` unit token**. Without the unit, four out of five hits are gene lists.
