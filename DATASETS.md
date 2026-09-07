# Dataset reference for SepSentinel

Consolidated 2026-09-07. Every dataset considered across the project, what
each is good for, and what it costs to get. Detailed recon lives in
`DATASETS_BIOMARKER.md` (pass 1) and `DATASETS_BIOMARKER_2.md` (pass 2);
this file is the index to reference when writing.

**The central constraint:** no open ICU EHR database contains IL-6. Verified
by direct query, not inference — MIMIC-IV 3.1 has zero interleukin and zero
procalcitonin items in both `d_labitems` and `d_items`; PhysioNet hosts no
cytokine dataset; AmsterdamUMCdb and HiRID carry CRP and procalcitonin but no
interleukin; eICU-CRD has 147 lab names and none is a cytokine. So Model B's
substrate and Model A's biomarker data must come from different sources, and
the bridge between them is the project's core methodological problem.

---

## Tier 1 — in hand, working

| Dataset | Access | Size | Strength | Limitation |
|---|---|---|---|---|
| **MIMIC-IV 3.1** | Credentialed + DUA (held) | 94,458 ICU stays; our cohort 63,672 episodes / 48,150 patients / 3.19M patient-hours | The Model B workhorse. Dense vitals (HR/RR/SpO2 ~94%, MAP 86%, urine 44% of hours), full labs, medications, cultures — everything needed for Challenge-rule Sepsis-3 labels | **No IL-6, no procalcitonin.** Single-centre (BIDMC). Shifted dates make temporal validation impossible |
| **PhysioNet/CinC 2019** | Free (Kaggle mirror) | 14,057 patients, 8.8% septic | The continuity baseline; labels pre-defined by the Challenge; two hospital systems allow cross-site checks | No IL-6. Only 4 dense vitals + sparse labs. Records end ~4h after onset (which turned out to be a *feature* — see RESULTS.md section 5) |

## Tier 2 — downloadable today, no account, no DUA (pass 2)

**This is the discovery that unblocks Model A.** All four are CC-BY
supplementary files with per-patient serial IL-6 in pg/mL.

| Dataset | n | IL-6 sampling | Why it matters | Weakness |
|---|---|---|---|---|
| **PLOS ONE 0209669** (PermiT trial sub-study) | 72 critically ill adults | **Days 1/3/5/7/14** | Longest serial IL-6 in the open tier; 29-cytokine panel; treatment arm included | Nutrition trial, not a sepsis-onset cohort |
| **PLOS ONE 0178387** | 86 **severe sepsis / septic shock** | **D1/D3/D5** | True sepsis population, with SOFA on the same grid. Open replacement for the restricted Zenodo cohort | Only 3 timepoints; no high-res vitals |
| **PLOS Medicine 1002338** | 89 trauma | **<1 h / 4-12 h / 48-72 h** | Ultra-early kinetics — the only dataset covering the first hour, which is where a wearable would live | Trauma, not infection |
| **PLOS ONE 0211981** | 116 piglets | Median 7, up to 31 obs/animal, hourly | **Best temporal shape found anywhere**, and the driving endotoxin concentration is measured alongside — a true dose-response | Porcine. Species mismatch must be stated |

## Tier 3 — free registration, same-day (ImmPort)

| Dataset | n | IL-6 sampling | Strength | Weakness |
|---|---|---|---|---|
| **SDY1662** (Mount Sinai, Del Valle *Nat Med* 2020) | **2,340 subjects** | **1.31 IL-6 draws/subject — VERIFIED from the downloaded package, not inferred.** 82% (1,924) have exactly ONE draw; 416 have >=2; a tail of 16 subjects has >=13, max 30 | **A bridge, not a trajectory dataset.** 3,075 IL-6 values (+IL-8/TNF-a/IL-1b, 12,300 ELISA rows) keyed to the same biosamples as 159,042 clinical lab results, so every cytokine draw has a paired clinical panel | COVID hyperinflammation, not bacterial sepsis. Values are **log2 pg/mL**. 828/3,075 rows carry `STUDY_TIME_COLLECTED = 999`, a sentinel for unknown — not day 999 |
| **SDY1655** (Yale IMPACT, Lucas *Nature* 2020) | 248 | **8 planned visits** | The only genuinely longitudinal human IL-6 trajectory in the open tier; 72-analyte panel | COVID; no dense vitals stream |

## Tier 4 — by request (drafts in `outreach/EMAIL_DRAFTS.md`)

| Target | What it would give | Access cost |
|---|---|---|
| **Zenodo 7612571** (Humanitas) | 178 **Sepsis-3** patients, IL-6/IL-8/IL-10/TNF-a/PTX3 at admission + day 5 | One-click request. Now partly superseded by PLOS ONE 0178387, which is open |
| **PMID 36383295** | Serial **IL-6 + lactate + procalcitonin** — SepSentinel's exact analyte triple | "Available on reasonable request"; needs a PI-signed ask |
| **PMID 41563442** | Burn-sepsis biomarker trajectories over 21 days | Same |
| **Zigong Fourth People's Hospital** (PhysioNet) | Possibly the only routine-care serial IL-6 paired with a full ICU time series | Credentialed + CITI + DUA, 1-3 months, and IL-6 presence is unverified — ask first |

## Tier 5 — evaluated and rejected, with reasons

| Dataset | Why not |
|---|---|
| **SICdb** (Salzburg) | **No microbiology or culture table at all** — Challenge-rule suspicion-of-infection is unreconstructable. Access is credentialed + CITI + DUA **+ per-project contributor review**, with no demo tier. See `SICDB_RECON.md` |
| **eICU-CRD** | Verified from the open demo: 147 lab names, **zero interleukin, zero procalcitonin**. Has lactate, pH, CRP, ferritin, full blood gases. Fine for Model B external validation; useless for IL-6 |
| **AmsterdamUMCdb / HiRID** | CRP and procalcitonin serial, **no interleukin anywhere** in either dictionary. HiRID's 2-minute monitoring resolution is the best available if high-res vitals ever matter |
| **ImmPort IMPACC (SDY1760/SDY2112)** | 18 planned visits, n=1,185 — scientifically ideal, but controlled access via NIAID with IAL2 identity proofing and federal DAR review. Not realistic for a minor |
| **PERSEVERE** (pediatric sepsis) | Panel verified: IL-8, HSPA1B, GZMB, MMP-8, CCL3. **No IL-6 in any version.** Dead end |
| **BioLINCC ARDSNet-SAILS** | Plasma vials exist at D0/3/6/12 but assayed cytokine *values* are not confirmed to be in the data package. You would likely receive specimens, not numbers |
| **GEO / dbGaP sepsis cohorts** | Transcriptomic only — no measured IL-6 protein |

---

### SDY1662, as actually downloaded (2026-09-07)

Verified by opening `SDY1662_DR58_ALL_DATA.zip` (16.4 MB compressed, 118 MB
across 723 files — the apparent small size is because most ImmPort tables are
controlled-vocabulary lookups).

| Table | Rows | Note |
|---|---|---|
| `subject` | 2,340 | |
| `elisa_result` | 12,300 | 3,075 samples x 4 analytes (IL-6, IL-8, TNF-a, IL-1b) |
| `lab_test` | 159,042 | platelets, albumin, glucose, bilirubin, BUN, sodium, ESR, CBC differential |
| `assessment_component` | 60,192 | |
| `intervention` | 67,650 | |

**The `planned_visit = 2` field is a schedule definition, not a draw count.**
Pass-2 recon read it as "~2 draws/patient"; the true figure is 1.31, which
matches the earlier August sweep. Recorded here because two recon passes
disagreed and this settles it.

## Two findings that should shape the write-up

**1. CRP is the better-evidenced wearable-trackable marker, not IL-6.** A 2026
systematic review of wearable HRV against inflammatory markers found
SDNN-CRP associations consistent (83% inverse, p=0.031) but RMSSD-IL-6
associations "heterogeneous and largely non-significant". Independently, CRP
and procalcitonin — not IL-6 — are the inflammatory markers that actually
appear at scale in every open ICU database. If Model A needs a marker with
both a supporting evidence base and available data, that marker is CRP.
IL-6 remains defensible on biology (it rises earlier), but the project should
stop assuming data will appear for it.

**2. No paired biosensor-vs-ELISA IL-6 dataset exists in deposited form** —
checked across 179 PLOS and 52 Europe PMC biosensor papers. Model A's
calibration data will have to come from the project's own bench work. This is
worth knowing before promising a sensor-to-concentration model built on
public data.

## Datasets that would be ideal and do not exist publicly

- Continuous wearable signal paired with intermittent biomarker draws. A 2026
  systematic review names all 11 studies that pair wearable HRV with
  inflammatory markers; the useful ones are **Deepika 2018** (n=89 TBI,
  continuous telemetry + serial IL-6/TNF-a/IL-10/IL-1b) and **Hirten 2021**
  (Mount Sinai, 72h VitalPatch + IL-6/TNF/CRP). Neither has deposited data —
  both are email targets.
- A publicly deposited *human* endotoxin-challenge IL-6 time course. The
  porcine dataset above is the closest substitute.

## Gaps in the search itself

- **YODA and CSDR trial catalogues were never examined** — they are portal
  sites without DataCite DOIs, so the repository sweep was blind to them.
- Roughly a third to half of the matched paper corpus went unopened; the
  supplement scanner skips PDF/DOCX supplements and non-PMC publisher CDNs.
- Both recon passes hit an exhausted web-search budget and worked via direct
  API retrieval, so search-discoverable-only sources are under-covered.
