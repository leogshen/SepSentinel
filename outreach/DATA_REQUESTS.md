# Data Access Outreach — Serial Cytokine Datasets

Status 2026-08-30. Goal: serial IL-6 trajectories for the Phase-3
cytokine-lab bridge (see DATA_ACCESS_SPEC.md and project memory).

---

## 1. Del Valle / Mount Sinai (Nature Medicine 2020) — NO EMAIL NEEDED

**Discovery**: the paper's data availability statement says the dataset is
publicly deposited on ImmPort, study accession **SDY1662**, de-identified
under HIPAA. Cold email unnecessary — just register and download.

Cohort (CORRECTED 2026-08-30 after ImmPort sweep): 1,484 COVID-19 patients
with ELLA cytokines (IL-6, IL-8, TNF-a, IL-1b) — but **mostly single
admission draws**; only a subset of n=244 has repeat measurements (~1,953
specimens total, ~1.3 draws/patient). Its unique strength is clinical
depth: vitals (HR, RR, temp, SpO2, BP) + labs (CRP, D-dimer, ferritin,
platelets, creatinine clearance, ALT) alongside IL-6 — i.e., the best
available cytokine-lab BRIDGE dataset, more than a kinetics dataset.

**Action (do today, ~15 min):**
1. Register (free) at https://www.immport.org — use leogshen@gmail.com.
2. Accept the ImmPort user agreement (https://www.immport.org/agreement).
3. Download study SDY1662; also grab the analysis code at
   https://github.com/delvad03/COVID19ELLA for column semantics.
4. Verify the deposit actually contains the SERIAL timepoints, not only
   baseline draws (deposits sometimes ship reduced tables).

**Contingency — only if serial timepoints are missing from SDY1662:**
email the corresponding author (verified from the paper):
Sacha Gnjatic <sacha.gnjatic@mssm.edu>

Subject: Serial timepoint data for ImmPort SDY1662 (Del Valle et al., Nat Med 2020)

> Dear Dr. Gnjatic,
>
> I am developing a wearable sepsis early-warning platform (SepSentinel)
> that combines an electrochemical IL-6 sensor with a physiological
> time-series model. Your Nature Medicine 2020 cohort is, to our knowledge,
> the largest serial cytokine resource available, and the longitudinal
> IL-6/IL-8/TNF-a/IL-1b trajectories are directly relevant to two questions
> we cannot answer elsewhere: the within-patient kinetics an on-body sensor
> must resolve, and the joint distribution of cytokines with routine labs.
>
> I have downloaded SDY1662 from ImmPort, but the deposited tables appear
> to contain [describe what is missing — e.g., only enrollment draws]. Could
> you point me to the serial measurement tables, or advise whether an
> additional data-use agreement would allow access to them?
>
> Happy to share our analysis plan; results would of course cite the cohort
> and any collaboration terms you prefer.
>
> Best regards,
> Leo Shen
> [affiliation, one line]

---

## 2. Korean CODA COVID-19 multi-omics — FORMAL APPLICATION + INQUIRY

Cohort: ~1,159 patients, 195-plex Luminex cytokine time series with full
EHR labs. Access is a formal application to the National Biobank of Korea /
KDCA, not a cold email — but a short inquiry to the listed data manager
speeds up scoping.

**Action:**
1. Locate the current application portal + contact on the KDCA / National
   Biobank of Korea website (CODA: Clinical & Omics Data Archive,
   https://coda.nih.go.kr — verify URL resolves; contact address is listed
   on the portal, do not guess it).
2. Send the inquiry below to the listed contact, then follow the formal
   application they return.

Subject: Foreign-researcher access inquiry — CODA COVID-19 cytokine time-series

> Dear CODA data management team,
>
> I am a researcher developing machine-learning models for sepsis early
> warning from longitudinal biomarker trajectories (project SepSentinel).
> I am writing to ask about the application procedure for the COVID-19
> multi-omics cohort with serial cytokine measurements described in
> [cite the specific CODA study/publication after identifying it on the
> portal].
>
> Specifically: (1) is access open to researchers outside Korea, and under
> what agreement; (2) does the release include measurement timestamps
> suitable for trajectory modeling; (3) typical review timeline.
>
> I can provide a research plan, institutional information, and ethics
> documentation as required.
>
> Best regards,
> Leo Shen
> [affiliation, one line]

---

## 3. ImmPort sweep results (2026-08-30, via search API)

Headline: **ImmPort holds no serial-IL-6 bacterial-sepsis dataset** — only 3
"sepsis" hits total, none cytokine time-series. Every serial-IL-6 cohort
there is COVID. Ranked candidates:

| Rank | Accession | Cohort | n | Serial? | Access |
|---|---|---|---|---|---|
| 1 | SDY1760 (IMPACC) | Hospitalized COVID, ICU-heavy | 1,185 | up to 6 blood visits / 28d + DAILY vitals & labs | RESTRICTED — AccessClinicalData@NIAID / dbGaP phs002686 |
| 2 | SDY1655 (Yale IMPACT) | Hospitalized COVID + controls | 248 | 8 planned visits, 73-analyte panel incl. IL-6 | Open — best open serial option; but vitals/labs NOT in deposit (paper supplement only) |
| 3 | SDY1662 (Del Valle) | COVID, Mount Sinai | 1,484 (244 serial) | mostly single-draw | Open — best cytokine-lab bridge |
| 4 | SDY1645 | COVID vs ARDS vs sepsis (n=16 sepsis) | 43 | single | UNRESOLVED — accession returns zero API hits despite paper claim; email ImmPort helpdesk |

**Action added**: SDY1760/IMPACC application via AccessClinicalData@NIAID —
formal, weeks-scale, start alongside CODA. Structurally the closest thing
to PhysioNet-plus-IL-6 that exists (serial Olink IL-6 + daily vitals/labs).

**Beyond ImmPort** (leads for serial IL-6 in actual sepsis; unverified):
- VASST trial ancillary: 39 cytokines, 363 septic-shock patients, baseline
  + 24h (2 timepoints only).
- medRxiv 10.1101/2025.05.17.25327533: longitudinal sepsis immune
  profiling, 98 adults, plasma IL-6 day 0 -> 28+; data-availability
  statement unread (403) — check manually, likely the single best lead for
  bacterial-sepsis IL-6 kinetics.

## 4. Parking lot (lower priority)

- **MIMIC-IV / SICdb** — already in progress via PhysioNet credentialing;
  IL-6 census queries specified in DATA_ACCESS_SPEC.md §5.
- **Ozger / Dryad** — already in hand (git history); reference-only.

---

## Workflow note

Gmail connector: once leogshen@gmail.com is authorized (run /mcp in the
session), Claude can place these as ready-to-send drafts in the Drafts
folder — sending always stays manual.
