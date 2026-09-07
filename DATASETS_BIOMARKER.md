# SepSentinel — Biomarker Dataset Reconnaissance (IL-6 and inflammatory markers)

**Date of survey:** 2026-09-07
**Scope:** find obtainable datasets with IL-6 and related inflammatory biomarkers measured **longitudinally** in sepsis / critical illness, with paired physiological time series where possible.
**Method:** direct HTTP retrieval of repository APIs and dataset landing pages (no downloads, no accounts created). Every claim below is tagged **[VERIFIED]** (I read it in a page or API response I fetched today), **[INFERRED]** (reasoned from something I read, not stated outright), or **[UNVERIFIED]** (could not check).

**Bottom line up front:** the user's finding is confirmed — there is no open ICU EHR database with IL-6. The realistic IL-6 sources are *immunology* repositories, not *critical-care* repositories. The best access-tier match by a wide margin is **ImmPort**, which requires only free self-service registration (no DUA, no PI, no IRB, no institutional signing official). Two ImmPort studies (SDY1662, SDY1655) contain per-patient IL-6 with repeat sampling. The population is COVID-19, not bacterial sepsis — this is the trade the project has to accept.

---

## 1. Ranked summary table

| # | Dataset | IL-6? | Serial? | n (IL-6) | Access tier | Realistic time-to-access | Paired physio/lab time series | Verdict |
|---|---------|-------|---------|----------|-------------|--------------------------|-------------------------------|---------|
| 1 | **ImmPort SDY1662** (Mount Sinai COVID cytokine signature) | Yes — IL-6, IL-8, TNF-α, IL-1β | Repeat, ~2 draws/patient | 3,075 IL-6 results across ~1,484 patients | **Open with free registration**, click-through agreement | **Same day** | Yes — 130,230 blood-chemistry results, 27,328 CBCs, 21,273 vital-sign records | **Pursue first.** Largest per-patient IL-6 set obtainable by a high-schooler unaided. |
| 2 | **ImmPort SDY1655** (Yale IMPACT, Lucas *et al.* Nature 2020) | Yes — 72-analyte panel incl. IL-6, TNF-α, IL-10, IL-1β, IFN-γ | **Yes, genuinely longitudinal** — 8 planned visits | 428 IL-6 results, cohort n=248 (patients + HCW controls) | **Open with free registration** | **Same day** | Partial — clinical severity/outcome metadata; not a dense vitals stream | **Pursue in parallel.** The only truly serial trajectory-shaped IL-6 in the open tier. Best fit for Model A's temporal shape. |
| 3 | **Zenodo 7612571** — Davoudian *et al.*, Humanitas sepsis cohort | Yes — IL-6, IL-8, IL-10, IL-1β, IL-18, IL-1ra, TNF-α, PTX3, sIL-1R2 | 2 timepoints (ED admission + day 5) | 178 **Sepsis-3** patients | **Restricted** — one-click "request access" to depositor | Days–weeks (author's discretion) | SOFA score, clinical params; no high-res vitals | **Best true-sepsis population.** Small, but the only Sepsis-3 IL-6 dataset with a lightweight request path. |
| 4 | **AmsterdamUMCdb** | **No IL-6** | Procalcitonin + CRP serial | ~1,923 PCT results, ~82,746 CRP results | Registration + DUA (institutional) | Weeks–months | Yes — full ICU EHR, high granularity | Good Model B substrate; **PCT/CRP only**, no IL-6. |
| 5 | **HiRID** (Bern) | **No IL-6** | CRP + procalcitonin as lab variables | ~34k ICU admissions | PhysioNet credentialed + CITI + DUA | 1–3 months | Yes — 2-minute-resolution monitoring | Best physiological resolution; **PCT/CRP only**. |
| 6 | **ImmPort SDY1760 / SDY2112 (IMPACC)** | Likely (Olink proteomics) [INFERRED] | **Yes — 18 planned visits**, n=1,185 | 1,185 hospitalized COVID | **Controlled** — NIAID AccessClinicalData, IAL2 identity proofing + federal DAR review + DUA | Many months; **not realistic** for a minor | Yes — dense longitudinal clinical | Scientifically ideal, access-wise out of reach. |
| 7 | **SICdb (Salzburg)** | Unknown | — | 27,000+ admissions | PhysioNet credentialed + DUA **+ per-project contributor review** | 2–6 months | Yes | Worst tier; lab dictionary only visible after download. Deprioritise. |
| 8 | **Zigong Fourth People's Hospital ICU infection DB** (PhysioNet) | Unknown [UNVERIFIED] — plausible PCT/IL-6 (Chinese ICU practice) | — | ICU infection/sepsis cohort, 2019–2020 | PhysioNet credentialed + CITI + DUA | 1–3 months | Yes — labs, nursing chart, drugs, ICD | Worth a look *if* credentialing happens anyway; IL-6 presence is a gamble. |
| 9 | **BioLINCC ARDSNet-SAILS** | Assay values **not confirmed**; plasma vials exist at Day 0/3/6/12 | — | 745 sepsis-ARDS patients, 2010–2013 | Formal application + institution-signed DUA | 2–4 months | Yes — trial CRFs | You'd likely get specimens, not IL-6 numbers. Poor fit. |
| 10 | **eICU-CRD** | Could not verify lab name list | — | 200k+ stays | PhysioNet credentialed + DUA | 1–3 months | Yes | See "could not verify". |
| 11 | **GEO GSE65682 (MARS consortium)** | **No protein cytokines** — transcriptomic only | Serial RNA sampling | 802 samples | **Fully open** | Same day | No | Not usable for Model A. |
| 12 | **GEO GSE3284 (human endotoxin challenge)** | **No protein cytokines in GEO** — transcriptomic only | 0/2/4/6/9/24 h | 110 arrays, ~8 subjects | **Fully open** | Same day | No | Right temporal shape, wrong modality. |
| 13 | **PhysioNet as a whole** | **Zero** cytokine/interleukin datasets found | — | — | — | — | — | Confirms the project's prior finding. |

---

## 2. Dataset-by-dataset detail

### 2.1 ImmPort — access model (this is the key finding)

URLs fetched:
- `https://docs.immport.org/help/user-registration/`
- `https://docs.immport.org/home/agreement/`
- `https://docs.immport.org/download/controlled/`
- `https://docs.immport.org/download/guide/`
- Search API: `https://www.immport.org/shared/data/query/api/search/study?term=<query>`

**[VERIFIED]** Registration is self-service and instant: "Good news getting an account is simple and quick!" — click Register, fill in user details, receive an email. No DUA, no institutional signing official, no IRB, no CITI training, no PI attestation. The only obligation is a click-through User Agreement whose substantive terms are (1) do not attempt re-identification, (2) do not share your password.

**[VERIFIED]** "The majority of ImmPort studies are openly available to any registered ImmPort user. Only studies explicitly designated as controlled-access require the process described on this page." The controlled-access list is explicitly the IMPACC studies **SDY1760 and SDY2112**, migrating to NIAID AccessClinicalData (ACD) effective 2026-05-20, which imposes IAL2 identity verification, federal DAR review, and DUA execution.

**[INFERRED]** SDY1662 and SDY1655 are therefore in the open (registration-only) tier — they are not on the named controlled list. This should be confirmed on the study download page once an account exists.

Practical notes **[VERIFIED]**: downloads are via the web Data Browser, a Download Cart (ZIP + emailed CloudFront link), an `ALL_DATA` complete-study ZIP per release, a GA4GH DRS API, or the ImmPort Download Client with an API key. The Aspera legacy browser retires July 2026.

**Caveat on account creation:** this recon did not create an account, per instructions. Registration is free and instant, but ImmPort's terms are between the account holder and NIAID — the sponsoring PI should be the account holder, or at minimum should approve a minor registering.

---

### 2.2 ImmPort **SDY1662** — "An Inflammatory Cytokine Signature Predicts Covid-19 Severity And Survival" ⭐ TOP PICK

- **URL:** `https://www.immport.org/shared/study/SDY1662` (SPA; machine-readable record retrieved from the ImmPort search API)
- **DOI:** 10.21430/M3ODKGM8O5
- **PI:** Sacha Gnjatic, Icahn School of Medicine at Mount Sinai. **PubMed 32839624** (Del Valle *et al.*, *Nature Medicine* 2020).
- **Program:** NIH / HIPC (RFA-AI-15-041); grants AI118610, CA224319.
- **First release** 2020-09-30; **latest release DR58**, 2025-10-30. Size 15.66 MB.

**[VERIFIED] from the study record:**
- `actual_enrollment` = **2,340**; arms "Original Cohort" and "Validation Cohort".
- `analyte_preferred_count` = **IL1B (3075), IL6 (3075), IL8 (3075), TNFA (3075)** — i.e. 3,075 IL-6 measurements.
- `assay_method_count` = **ELISA (3075)**. Biosample type: **Serum**.
- `lab_test_panel_count` = **Blood Chemistry (130,230)**, **Blood Cell Count (27,328)**, SARS-CoV-2 PCR (1,484).
- `assessment_panel_count` = **Vital Signs (21,273)**, Comorbidities (21,545), Disease Outcome (17,374).
- `planned_visit_total_count` = **2**.
- Condition: COVID-19. Age 0–89. Mixed race/ethnicity.

**[INFERRED]** 3,075 IL-6 results against the 1,484 PCR-tested patients is ≈ **2.07 IL-6 draws per patient** — consistent with `planned_visit_total_count = 2`. So this is **repeat-measures, not densely serial**: admission plus roughly one follow-up. That is still strictly better than a single admission draw, and it gives you a Δ (rate of change) per patient.

**Why it matters for SepSentinel:** it is the only registration-only dataset found that pairs per-patient IL-6 with a real EHR-style lab and vital-sign stream. 130k chemistry results + 21k vital-sign records against ~1,484 patients means **Model B has features and Model A has targets in the same subjects** — the linkage SepSentinel actually needs.

**Limitations:** COVID-19, not bacterial sepsis. IL-6 by ELISA in serum (fine for Model A calibration targets). Lactate and pH are not listed as analytes — **[UNVERIFIED]** whether the Blood Chemistry panel includes lactate/blood gas; that must be checked in the downloaded `lab_test` tables.

---

### 2.3 ImmPort **SDY1655** — Yale IMPACT, "Longitudinal Analyses Reveal Immunological Misfiring in Severe COVID-19" ⭐ SECOND PICK

- **DOI:** 10.21430/M3BYPHCD6F. **PubMed 32717743** (Lucas *et al.*, *Nature* 2020). PI Carolina Lucas / Akiko Iwasaki, Yale.
- Released 2020-09-30. Size **18.66 GB** (dominated by flow cytometry FCS files).
- Companion study **SDY1648** (Takahashi *et al.*, sex differences, PMID 32846427) — record explicitly says *"THE DATA FOR THIS STUDY IS CONTAINED IN SDY1655"*.

**[VERIFIED]:**
- `actual_enrollment` = **248**; `planned_visit_total_count` = **8**.
- **72 analytes.** IL-relevant: **IL6 (428)**, **TNFA (435)**, **IFNG (411)**, **IL10 (215)**, **IL1B (215)**.
- Assays: ELISA (339), Flow Cytometry (659), **Protein microarray (215)**, RT-PCR (164).
- Biosamples: Plasma, PBMC, nasopharyngeal swab, saliva.
- Description: *"Longitudinal analyses of COVID-19 patients revealed increasingly dysregulated and sustained inflammatory immune responses… persistent type 1, type 2 and type 3 cytokines…"*

**[INFERRED]** 8 planned visits with 428 IL-6 measurements over a cohort of 248 (which includes healthcare-worker controls; the Nature paper's patient cohort is ~98) implies roughly **4+ IL-6 timepoints per patient**, sampled every 3–7 days through admission. **This is the genuinely longitudinal IL-6 trajectory** the brief asks for.

**Trade-off vs SDY1662:** SDY1655 has the temporal density Model A wants; SDY1662 has the paired vitals/labs Model B wants. **Take both.** They are the same access tier and cost nothing but disk.

---

### 2.4 Zenodo 7612571 — Humanitas Sepsis-3 cytokine cohort (best *true sepsis* population)

- **URL:** `https://zenodo.org/records/7612571` (API: `https://zenodo.org/api/records/7612571`)
- Supplement to **PMID 36189302 / doi:10.3389/fimmu.2022.979232** — Davoudian *et al.*, "A cytokine/PTX3 prognostic index as a predictor of mortality in sepsis", *Front Immunol* 2022. Creators include S. Davoudian, A. Desai, S. N. Mapelli, R. Leone, M. Sironi (Humanitas Research Hospital / Alberto Mantovani's group, Milan).
- Deposited 2023-02-06.

**[VERIFIED] from the record's own abstract:** plasma from **178 patients diagnosed by Sepsis-3 criteria**, sampled **at ED admission and after 5 days of hospitalisation**. Measured by ELISA: **PTX3, sIL-1R2, IL-1β, IL-6, IL-8, IL-10, IL-18, IL-1ra, TNF-α**. Correlated with SOFA score and 90-day mortality. Cytokines fell significantly between admission and day 5 — i.e. the two timepoints carry real trajectory information.

**[VERIFIED]** Access is **`restricted`** — the API returns an empty `files` list. Zenodo restricted records expose a "Request access" button that emails the depositor; this is the lightest-weight by-request path of anything in this survey (no institutional DUA machinery, decided by one lab).

**Also seen (same group, same pattern):** Zenodo **20122208**, "Monocyte-macrophage membrane expression of IL-1R2 is a severity biomarker in sepsis" (Supino & Garlanda, *Cell Death Dis* 2025, PMID 40204720) — also restricted, flow-cytometry-centred, SOFA-stratified septic patients.

**Verdict:** 178 patients × 2 timepoints is small for Model A training, but it is the **only genuine bacterial-sepsis IL-6 dataset** identified with a tractable access route. It is the right thing to use as an external validation / domain-shift check on a model trained on COVID cytokines.

---

### 2.5 AmsterdamUMCdb — procalcitonin and CRP verified, no IL-6

- **URL fetched:** `https://raw.githubusercontent.com/AmsterdamUMC/AmsterdamUMCdb/master/amsterdamumcdb/dictionary/dictionary.csv` (13,154 rows — the official concept dictionary, OMOP-mapped).

**[VERIFIED] by grep of the full dictionary:**
- `Procalcitonine (bloed)` — itemid 15775 → LOINC 33959-8 — **1,921 measurements**; plus legacy `PROCALCITON (bloed)` itemid 15565 (2 measurements).
- `CRP (bloed)` — itemid 10079 → LOINC 1988-5 — **82,347 measurements**; `CRP` itemid 6825 (399); `CRP (overig)` itemid 18854 (16).
- **No match for `interleuk`, `IL-6`, `IL6`, `TNF`, `presepsin`, or `suPAR` anywhere in the dictionary.**

So AmsterdamUMCdb gives serial **PCT + CRP** against a very rich ICU time series, and **no IL-6 at all**.

**[UNVERIFIED]** Access process. AmsterdamUMCdb is distributed via amsterdammedicaldatascience.nl under an access request + data use agreement rather than PhysioNet credentialing; I did not fetch and confirm the current form. Treat as "registration + institutional DUA, weeks to months".

---

### 2.6 HiRID (Bern University Hospital) — procalcitonin and CRP verified, no IL-6

- **URLs fetched:** `https://hirid.intensivecare.ai/data-details`; variable reference `https://raw.githubusercontent.com/ratschlab/HIRID-ICU-Benchmark/master/preprocessing/resources/varref.tsv` (710 rows).

**[VERIFIED] in varref.tsv, category `infection/inflamation`:**
- `C-reactive protein`, variableid **20002200**, LOINC "C reactive protein [Mass/volume] in Serum or Plasma", range 0–600.
- `procalcitonin`, variableid **24000570**, LOINC "Procalcitonin [Mass/volume] in Serum or Plasma", range 0–200.
- **No interleukin variable of any kind.**

**[VERIFIED]** HiRID's selling point is time resolution: most bedside monitoring parameters recorded **every 2 minutes**, plus labs, ventilator settings, drugs, fluids. Dates shifted to 2100–2200; age/height/weight binned; k-anonymised.

**Access:** PhysioNet credentialed + CITI + DUA. **[INFERRED]** — I read HiRID's own site, not its PhysioNet access block, but HiRID is distributed on PhysioNet under the standard credentialed licence.

**Verdict:** if the project ever needs the highest-resolution physiological substrate for Model B, HiRID is it — but it will never supply an IL-6 label.

---

### 2.7 SICdb (Salzburg) — worst access tier, contents unknown

- **URL fetched:** `https://physionet.org/content/sicdb/`

**[VERIFIED]:** 27,000+ ICU admissions, four ICUs at University Hospital Salzburg, **2013–2021**, 41 beds. Both once-per-hour aggregated and **once-per-minute granular** data. Case info, vital signs, laboratory results, medication, preceding surgery.

**[VERIFIED] Access Policy:** *"Only credentialed users who sign the DUA can access the files. **In addition, users must have individual studies reviewed by the contributor.**"* Licence: PhysioNet Contributor Review Health Data License 1.5.0. Required training: CITI Data or Specimens Only Research.

This is the tier the brief flags as adding months. Confirmed: credentialing + CITI + DUA + **per-project contributor review**. The lab dictionary is only inside the restricted download, so **[UNVERIFIED]** whether SICdb carries IL-6 or PCT. Given a European surgical/mixed ICU, PCT is likely and IL-6 is possible (German-speaking ICUs do order IL-6 more than US ones) — but that is a guess, not a finding, and it is not worth a multi-month access process on a guess.

---

### 2.8 Zigong Fourth People's Hospital ICU infection database (PhysioNet)

- **URL fetched:** `https://physionet.org/content/icu-infection-zigong-fourth/1.1/` — DOI 10.13026/xpt9-z726 (v1.1, June 2022; v1.0 Sept 2021).

**[VERIFIED]:** ICU patients with infection, Zigong Fourth People's Hospital, Sichuan, China, **Jan 2019 – Dec 2020**; adults >18; *"A large proportion of these patients have sepsis and/or septic shock."* Six CSV tables — `dtBaseline`, `dtDrugs`, `dtNursingChart` (incl. temperature), plus lab findings, ICD codes and follow-up, keyed by `PATIENT_ID`/`INP_NO`. Times are offsets in hours from hospital admission. No waveforms; ventilator params low granularity.

**[VERIFIED] Access:** *"Only credentialed users who sign the DUA can access the files"*, CITI training required. **Not** contributor-review — one tier easier than SICdb.

**[UNVERIFIED]** Whether the lab table contains IL-6 or procalcitonin. The landing page does not enumerate lab analytes. **[INFERRED]** Chinese tertiary ICUs routinely order PCT, and IL-6 is a common routine assay in Chinese hospital labs — so this is the single most plausible *routine-care* source of serial IL-6 paired with an ICU time series. But it is a credentialed dataset whose contents I cannot confirm without downloading, which is exactly the wrong risk profile for a months-long access process. **Recommended action: ask the corresponding author directly whether the lab table contains IL-6, before starting credentialing.**

---

### 2.9 ImmPort IMPACC (SDY1760, SDY2112) — scientifically ideal, access-blocked

**[VERIFIED]:** SDY1760, "Immunophenotyping Assessment in a COVID-19 Cohort (IMPACC): A Prospective Cohort Study to Assess Longitudinal Immune Responses in Hospitalized Patients with COVID-19" — `actual_enrollment` **1,185**, `planned_visit_total_count` **18**, released 2022-07-26. SDY2112 = "HTP proteomics, DIA", same 18-visit structure, released 2024-02-29.

**[VERIFIED]:** these are the two studies ImmPort names as **controlled-access**, moving to NIAID AccessClinicalData effective 2026-05-20, requiring IAL2 identity verification, prospective federal DAR review, and DUA execution.

**[INFERRED]:** IMPACC's proteomic panels (Olink / DIA-MS) include IL-6, and the cohort has 18 scheduled longitudinal visits with paired daily clinical data — this would be the single best dataset in the survey on scientific merit. IAL2 identity proofing plus federal review is not realistic for a high-school student, and probably not fast for the PI either. **Do not pursue now; revisit if the PI has an existing NIAID data pathway.**

---

### 2.10 PhysioNet as a whole — no cytokine data

**URLs fetched:** `https://physionet.org/content/?topic=sepsis`, `https://physionet.org/about/database/`.

**[VERIFIED]** Grepping the full PhysioNet database index for `cytokin|interleukin|IL-6|inflammat|biomarker|procalcitonin|sepsis` returns **no project with cytokine or interleukin measurements**. Sepsis-topic projects are: PhysioNet/CinC Challenge 2019, "Chest CT for patients with sepsis in the ED", "Clinical Time Series Datasets for Trajectory Flow Matching (ICU Sepsis / Cardiac Arrest / GIB)", the Health Gym synthetic MIMIC-III sepsis dataset, TherLid, and the Zigong infection database. The only "biomarker" hits are a voice-biomarker project and an image-derived cardiomegaly biomarker set. **This independently confirms the project's prior finding.**

---

### 2.11 MIMIC-IV and derivatives

The project has already verified directly: **zero interleukin items and zero procalcitonin items** in MIMIC-IV 3.1 `d_labitems` and `d_items`.

**[UNVERIFIED by me, but expected]** MIMIC-IV does carry **C-Reactive Protein** in `d_labitems` — worth confirming locally, since CRP is the one inflammatory marker that would let a MIMIC-based Model B carry an inflammation feature.

**[VERIFIED]** No cytokine sub-study or linked biobank for MIMIC surfaced anywhere in this survey — not on PhysioNet, not in ImmPort, not in GEO. **[INFERRED]** MIMIC is de-identified retrospective EHR with no consented biospecimen arm, so a linked cytokine biobank is structurally unlikely to exist.

---

### 2.12 eICU-CRD

- **URL fetched:** `https://eicu-crd.mit.edu/eicutables/lab/` — returned the documentation site chrome and table index but **no enumeration of `labname` values**. The `MIT-LCP/eicu-code` GitHub tree API returned no lab-concept dictionary file either.

**[UNVERIFIED]** Whether eICU's `lab` / `customLab` tables contain CRP or procalcitonin. **[INFERRED]** eICU is a US multi-centre telehealth extract from 2014–2015; IL-6 was not routine US practice then, so IL-6 is very unlikely. Access is PhysioNet credentialed + DUA. **Not worth pursuing for biomarkers.**

---

### 2.13 European sepsis cohorts and trials

| Cohort | What I found | Access |
|---|---|---|
| **MARS** (Molecular Diagnosis and Risk Stratification of Sepsis, Amsterdam) | **[VERIFIED]** Public arm is **GEO GSE65682**, "Genome-wide blood transcriptional profiling in critically ill patients — MARS consortium", **802 samples**, PAXgene blood RNA "at ICU admission and throughout ICU length-of-stay", CAP/HAP/non-infectious. **Transcriptomic only — no measured protein cytokines in the GEO record.** | GEO is fully open. The MARS plasma-biomarker data (the group published IL-6/IL-8 panels separately) is **not** in GEO — by request from the Amsterdam UMC investigators. **[UNVERIFIED]** |
| **GAinS** (Genomic Advances in Sepsis, Oxford) | **[VERIFIED]** Related open deposit: **Zenodo 7924238**, Kwok *et al.* 2023, "Neutrophil and emergency granulopoietic drivers of sepsis immune suppression…" — single-cell RNA/ATAC multi-omics, ~7 GB of matrices, n=39/53/27 across sub-cohorts. Abstract states *"We observed elevated plasma G-CSF and IL-6 in SRS1"* — so **plasma IL-6 was measured**, but the deposited files are single-cell count matrices; **the plasma IL-6 values are not in the deposit** [VERIFIED from the file list]. | Zenodo files open; the IL-6 protein values would be by request from the Knight lab, Oxford. **[UNVERIFIED]** |
| **ProCESS** | **[VERIFIED]** Not in BioLINCC (searched `process`, `protocolized`, `ARISE`, `albios`, `vasopressin` — none returned it). **[INFERRED]** ProCESS measured serial IL-6/IL-10 at 0/6/24/72 h in a biomarker sub-study; individual data would be by request from the University of Pittsburgh investigators. **[UNVERIFIED]** |
| **ARISE / ProMISe** | Not in BioLINCC (ANZICS-CTG and ICNARC respectively). **[UNVERIFIED]** — both operate formal IPD-request committees; months, PI-mediated. |
| **ALBIOS** | No Zenodo/BioLINCC deposit found. **[UNVERIFIED]** — Italian, Mario Negri Institute; by-request only. Note that the Humanitas dataset (§2.4) is from the same Milan research ecosystem and *is* deposited, which suggests a direct email to Italian sepsis-cytokine groups is a live route. |
| **VANISH** | No public deposit found. **[UNVERIFIED]** |

**Net:** every European sepsis consortium biobank is a by-request, PI-mediated, multi-month path. None is realistic as a first move.

---

### 2.14 BioLINCC / NHLBI trials (ARDSNet, PETAL)

- **URLs fetched:** `https://biolincc.nhlbi.nih.gov/studies/?q=sepsis`, `https://biolincc.nhlbi.nih.gov/studies/sails/`.

**[VERIFIED]** Sepsis-relevant holdings: **ARDSNet-SAILS**, ARDSNet-Omega, PETAL-CLOVERS, PETAL-ASTER, PETAL-VIOLET, plus ARDSNet/FACTT/ALVEOLI/EDEN/ALTA.

**SAILS detail [VERIFIED]:** 745 patients with sepsis-associated ARDS (379 rosuvastatin / 366 placebo), 2010–2013, NCT00979121, primary pub PMID 24835849, datasets last updated 2024-10-02. **Available biospecimens: plasma, DNA, urine — plasma vials at Day 0 (1,969), Day 3 (1,860), Day 6 (2,041), Day 12 (96).** Documents: Data Dictionary (PDF), Forms, Protocol. Consent restricts non-genetic biospecimen use to lung-injury/critical-care research; no commercial use of specimens.

**Critical caveat [VERIFIED by absence]:** grepping the SAILS, ARDSNet, FACTT and ALVEOLI pages for `biomarker|interleukin|IL-6|IL-8|cytokine` returned essentially nothing (one incidental "cytokine" on the ARDSNet page). **The BioLINCC data packages appear to contain trial CRF data, not assayed cytokine values.** The serial plasma exists as *frozen vials you would have to assay yourself* — which is a wet-lab project, not a data project.

**Access:** formal request with institution-signed DUA and NHLBI/Biorepository review; account registration required. **[VERIFIED]** the site currently carries the banner *"This repository is under review for potential modification in compliance with Administration directives"* — treat BioLINCC availability as an additional schedule risk.

**Verdict: poor fit.** Months of paperwork for data that probably has no IL-6 in it.

---

### 2.15 Human endotoxin challenge studies

- **[VERIFIED]** ImmPort contains **zero** studies matching `endotoxemia`, `healthy volunteers LPS`, or `LPS challenge`, and the 10 hits for `endotoxin` are all vaccine/adjuvant or allergy studies with no IL-6 analyte.
- **[VERIFIED]** **GEO GSE3284** — Calvano *et al.*, "A network-based analysis of systemic inflammation in humans" (the Glue Grant / Inflammation & Host Response to Injury endotoxin experiment). Design: *"Healthy male and female subjects were intravenously administered either endotoxin or only vehicle. Arterial blood samples were collected before infusion (0 hours) and at post infusion times of 2, 4, 6, 9, and 24 hours."* 110 arrays. **Fully open.** But GEO holds only leukocyte microarray data — **no plasma cytokine concentrations**. The serial IL-6 curves from that programme live in the paper's figures, not in a per-subject data file.
- **[VERIFIED]** "Glue Grant" as an ImmPort search term returns 0 studies.

**Verdict:** the endotoxin-challenge literature has exactly the dense IL-6 temporal shape Model A wants (peak ~2 h, resolution by 6–8 h), but **no per-subject IL-6 concentration dataset is publicly deposited** that I could find. The realistic route is an email to an experimental-endotoxemia group (Radboud UMC Nijmegen runs the largest ongoing human endotoxemia programme) asking for de-identified serial IL-6 per subject. n would be tens, not thousands — but these datasets are small, non-sensitive, and PIs often share them.

---

### 2.16 COVID-19 cohorts with serial cytokines (beyond ImmPort)

- **ImmPort is the winner in this category** — SDY1662 (§2.2) and SDY1655 (§2.3) are both COVID cohorts with per-patient IL-6, at the lowest access tier of anything surveyed.
- Other ImmPort COVID studies with IL-6 **[VERIFIED from the search index]**:
  - **SDY1669** "Mild and severe COVID-19" (Rouphael, Emory; 76 patients + 69 controls, Hong Kong + Atlanta; **9 planned visits**; IL6/IL1B/TNFA/IFNG at 93 each; also LPS quantification (69); CyTOF, flow, RNA-seq; 4.30 GB; PMID 32788292). ~1 IL-6 result per sample-set; smaller but multi-visit.
  - **SDY1641** (Shenzhen Third People's Hospital; 42 COVID+hypertension patients; **IL6 (126)** = 3 IL-6 draws per patient; plus Blood Chemistry (924) and blood cell counts (504) and MAP/hospitalisation assessments; PMID 32228222). Tiny but genuinely serial and lab-paired.
- **ISARIC 4C / N3C:** not fetched in this survey. **[INFERRED]** both require formal application with institutional affiliation and DUA; N3C additionally requires an approved Data Use Request tied to an institution's Data Use Agreement. **Neither is realistic for a minor without heavy PI involvement, and neither is known to carry per-patient serial IL-6** — N3C is EHR-derived (so it inherits the same "IL-6 is not routinely ordered" problem) and ISARIC 4C is primarily clinical CRF data.

---

### 2.17 GEO / ArrayExpress / dbGaP transcriptomic cohorts with measured cytokine protein

**[VERIFIED]** A GEO (`gds`) search for sepsis + cytokine/IL-6 in humans returned 30 series; inspecting the top 15 (GSE314568, GSE345117, GSE341494, GSE339365, GSE335824, GSE303333, GSE302306, GSE301821, GSE297509, GSE294890, …) shows **all are small transcriptomic/epigenomic studies (n = 3–120 samples) with no deposited protein cytokine measurements**. Representative: GSE335824 "Assessment of differences in immune responses to albumin compared with crystalloids for resuscitation in sepsis" (n=69); GSE303333 "Aberrant STAT signaling and T cell dysregulation define a targetable pediatric sepsis endotype" (n=24).

**[VERIFIED]** A dbGaP (`gap`) search for `sepsis` returned no sepsis-specific study — the top hits are WHI, GTEx, Framingham, LOGIC, UDN, i.e. keyword collisions. **No dbGaP sepsis cytokine cohort surfaced.** dbGaP in any case requires an eRA Commons account, a signed institutional DUA and a Data Access Committee review — the wrong tier entirely.

**Verdict:** GEO/ArrayExpress/dbGaP are open but supply **transcriptomes, not IL-6 concentrations**. They cannot serve as Model A regression targets. If the project ever wants a *transcriptomic* sepsis endotype label, GSE65682 (MARS, 802 samples, open) is the standard reference — but that is a different project.

---

## 3. Recommendation

### Pursue first (do this week, costs nothing but a free account)

**1. ImmPort SDY1662 + SDY1655, together.** The sponsoring PI registers an ImmPort account (free, instant, click-through agreement only — no DUA, no IRB, no institutional signatory), then downloads both `ALL_DATA` study packages.
- SDY1662 gives ~3,075 IL-6 values across ~1,484 patients **with 130k paired chemistry results and 21k paired vital-sign records** → this is the Model A ↔ Model B linkage.
- SDY1655 gives ~4+ IL-6 timepoints per patient across 8 visits → this is the **trajectory shape** Model A needs to learn a time-varying sensor-to-concentration mapping.
- Add SDY1669 (9 visits) and SDY1641 (3 IL-6 draws/patient with paired chemistry) as free bonus cohorts from the same account.

**Honesty requirement for the write-up:** these are **COVID-19 hyperinflammation cohorts, not bacterial sepsis**. The IL-6 dynamic range and kinetics overlap sepsis substantially, but the project must state this as a domain-shift limitation rather than paper over it.

### Pursue second (a single email, decided by one lab)

**2. Zenodo 7612571 — Humanitas Sepsis-3 cohort (Davoudian / Garlanda / Mantovani).** Click "Request access" on the Zenodo record. This is the only **true Sepsis-3** IL-6 dataset with a lightweight path, and it would let the project validate a COVID-trained Model A on real sepsis.

Draft request text (Zenodo access request / email to the corresponding author):

> Subject: Request for access to Zenodo dataset 7612571 (cytokine/PTX3 prognostic index in sepsis)
>
> Dear Dr Davoudian and colleagues,
>
> I am writing on behalf of a student research project I am supervising at [INSTITUTION]. We are developing a wearable electrochemical biosensor platform for early sepsis warning. One model component maps raw sensor signals to circulating biomarker concentrations (IL-6, lactate, pH), and a second predicts sepsis risk from the resulting time series.
>
> We would like to request access to the dataset deposited at https://zenodo.org/records/7612571, supporting Davoudian et al., Front Immunol 2022 (doi:10.3389/fimmu.2022.979232). Specifically we are interested in the per-patient plasma cytokine concentrations (IL-6, IL-8, IL-10, IL-1β, TNF-α, IL-18, IL-1ra, PTX3, sIL-1R2) at both the emergency-department admission and day-5 timepoints, together with the accompanying SOFA scores and any routine clinical/laboratory variables included in the deposit.
>
> Our intended use is strictly methodological: to characterise the distribution and rate of change of IL-6 in a Sepsis-3 population, and to use it as an external-validation cohort for a model calibrated on COVID-19 cytokine data. We would not attempt any re-identification, would not redistribute the data, and would cite the original publication and the Zenodo DOI in any output. We are happy to sign whatever data use terms you require, and to share our derived code and results with you.
>
> If the deposit does not include per-patient longitudinal values, we would be grateful to know whether a de-identified two-timepoint IL-6 table could be shared separately.
>
> With thanks for your consideration,
> [PI NAME, TITLE, INSTITUTION, EMAIL]

Note: the request must come from and be signed by the PI, not the student.

### Cheap parallel probe (one email, potentially high payoff)

**3. Email the Zigong database corresponding author** (contact visible on the PhysioNet page after login) and ask a single yes/no question before committing to credentialing: *"Does the laboratory table in the Zigong ICU infection database contain interleukin-6 and/or procalcitonin results?"* If yes, this becomes the only dataset in the survey with **routine-care serial IL-6 paired with a full ICU time series in a sepsis population** — and PhysioNet credentialed access (no contributor review) is achievable in 1–3 months with the PI as the credentialed user.

**4. Email a human-endotoxemia group** (Radboud UMC Nijmegen is the highest-volume programme) asking for a de-identified per-subject serial IL-6 table (0/1/2/4/6/8/24 h). Small n, but it is the cleanest possible IL-6 kinetic curve and exactly the temporal shape Model A must learn. These datasets are non-sensitive and often shared on request.

### Do not pursue now

SICdb (contributor review, contents unknown), IMPACC (IAL2 + federal DAR), BioLINCC (institution-signed DUA for data that probably has no IL-6), dbGaP, N3C, ISARIC, and the European trial consortia (MARS/GAinS/ProCESS/ARISE/ProMISe/ALBIOS/VANISH) — all are multi-month, PI-mediated, and none is confirmed to contain deposited per-patient serial IL-6.

### Strategic note on the IL-6 premise

Nothing in this survey changes the underlying fact that **IL-6 is not a routine ICU lab in the datasets the ML community uses**. If SepSentinel needs a biomarker that appears serially in every open ICU database, that biomarker is **CRP** (82k results in AmsterdamUMCdb, a variable in HiRID, present in MIMIC-IV) and secondarily **procalcitonin** (~1.9k in AmsterdamUMCdb, a variable in HiRID). Consider whether Model A should target **IL-6 + CRP + lactate + pH**, using ImmPort for the IL-6 arm and an ICU database for the CRP/lactate arm — that keeps the wearable's IL-6 claim while giving Model B a marker it can actually be trained and validated against at scale.

---

## 4. What I could not verify

1. **Whether SDY1662 and SDY1655 are definitively in ImmPort's open (registration-only) tier.** ImmPort states the majority of studies are open and names only SDY1760/SDY2112 as controlled; I inferred the rest. Confirm on the study download page after registering.
2. **The exact per-patient IL-6 timepoint distribution** in SDY1662 and SDY1655. I have total measurement counts and planned-visit counts from the study index, and divided. The actual per-subject sampling schedule is in the downloaded `experiment`/`biosample` tables.
3. **Whether SDY1662's "Blood Chemistry" panel includes lactate, pH/blood gas, CRP, ferritin or D-dimer.** The index reports only the panel name and a count of 130,230 results.
4. **eICU-CRD's `labname` value list** — the MIT documentation page did not enumerate them and no concept dictionary was found in `MIT-LCP/eicu-code`. CRP/procalcitonin presence in eICU is unknown.
5. **Whether Zigong's lab table contains IL-6 or procalcitonin.** The landing page does not list analytes and the files are credentialed. This is the single highest-value open question in the report.
6. **Whether SICdb contains IL-6 or procalcitonin.** Lab dictionary is inside the restricted download, as the brief anticipated. Confirmed the access tier (credentialed + DUA + contributor review); could not confirm contents.
7. **Whether BioLINCC's SAILS/ARDSNet data packages contain any assayed cytokine values.** I inferred "no" from the absence of biomarker keywords on the study pages; the 6.6 MB Data Dictionary PDF would settle it definitively and I did not parse it.
8. **AmsterdamUMCdb's current access procedure** — I verified its *contents* from the official dictionary but not its *access form*. Time-to-access is an estimate.
9. **MIMIC-IV's CRP item.** Expected present but not checked by me; the project can confirm locally in seconds.
10. **ISARIC 4C and N3C** — not fetched at all. Statements about them are inference from general knowledge, not from pages retrieved today.
11. **ProCESS, ARISE, ProMISe, ALBIOS, VANISH individual-patient-data availability.** Confirmed absent from BioLINCC; their own IPD-request mechanisms were not fetched.
12. **Whether MARS or GAinS plasma cytokine (protein) values exist as a deposited dataset anywhere.** Both groups clearly measured plasma IL-6 (GAinS's own Zenodo abstract says so), but the public deposits are transcriptomic/single-cell only.
13. **Zenodo restricted-access response times.** No basis for an estimate beyond "author's discretion".
14. **The nature and completeness of ImmPort's search index.** The `_search` API I used returns only studies matching a free-text term against indexed fields; a study could carry IL-6 data without matching my search terms. Searches run: sepsis (3 hits), septic shock (1), interleukin (241), cytokine (356), IL6 (100), procalcitonin (1), endotoxin (10), endotoxemia (0), LPS challenge (0), critical illness (2), intensive care (11), trauma (15), burn (16), bacteremia (1), systemic inflammation (5), Glue Grant (0). This is a broad but not exhaustive sweep.

**Tooling limitation to disclose:** this session's web-search budget was exhausted before this task began, so all findings come from **direct retrieval of specific URLs and repository APIs**, not from search-engine discovery. Datasets that would only have surfaced through a general web search — small institutional cohorts, supplementary tables attached to individual papers, non-indexed national biobanks — are systematically under-represented here. A follow-up pass with working web search would be worthwhile, particularly targeting per-paper supplementary data files, which are the most common home for exactly this kind of small serial-cytokine table.
