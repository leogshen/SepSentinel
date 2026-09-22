# SICdb 1.0.8 (Salzburg Intensive Care database) loader and rung-1 sepsis
# labelling.
#
# Companion to sepsis3.py (MIMIC-IV). The two produce the SAME episode schema
# but DIFFERENT labels, and the labels must never be pooled -- see LABEL_RUNG
# below and DATA_ACCESS_SPEC.md section 12.
#
# SICdb contains NO microbiology, culture, specimen or organism data. The
# Challenge-rule t_suspicion we implement for MIMIC (antibiotics PAIRED with
# cultures) therefore cannot be reproduced. This module implements rung 1 of
# the section-12 fallback ladder: suspicion = first systemic antibacterial
# administration, with no culture pairing. That is a materially weaker and
# differently-biased definition; it is recorded in every manifest this module
# writes.
#
# Time base: SICdb `Offset` is SECONDS since PDMS admission, which is the
# pre-surgical hospital admission, NOT ICU admission. Hours from ICU intime
# are (Offset - cases.ICUOffset) / 3600. Median ICUOffset is 1.8 h, so
# ignoring it would shift every stay.

import numpy as np

# --------------------------------------------------------------------------
# Label provenance
# --------------------------------------------------------------------------

LABEL_RUNG = 1
LABEL_NAME = "sicdb_abx_only_sofa_rise"
LABEL_WARNING = (
    "SICdb labels use antibiotic-initiation WITHOUT culture pairing (rung 1 "
    "of DATA_ACCESS_SPEC section 12). They are not comparable to MIMIC-IV "
    "Challenge-rule Sepsis-3 labels and must never be pooled with them. Use "
    "SICdb for self-supervised pretraining, transfer discrimination under the "
    "relabelled definition, or control-only alarm-burden evaluation."
)

DEVIATIONS = [
    "t_suspicion is the start of an antibacterial COURSE, not the first dose. "
    "There is no culture pairing because SICdb has no microbiology at all. "
    "Measured on the real table: 21,260 of 26,763 eligible cases (79%) "
    "receive some systemic antibacterial, so first-dose suspicion would make "
    "most of a perioperative cohort 'suspected' and the label meaningless. "
    "Requiring a course (>=COURSE_MIN_DOSES doses spanning "
    ">=COURSE_MIN_SPAN_H) cuts that to 11,064 cases (41%).",
    "Two prophylaxis filters were designed and ABANDONED after checking the "
    "data. (a) IsSingleDose is useless: 232,266 of 232,271 administrations "
    "carry IsSingleDose=1, because SICdb records each administration event as "
    "a dose. (b) Excluding antibiotics started shortly after surgery end is "
    "not possible in general: HeartSurgeryEndOffset is populated for only "
    "2,343 of 27,350 cases, i.e. cardiac surgery alone. Elective-surgery "
    "prophylaxis therefore remains a contaminant of this label, which is a "
    "large part of why it is rung 1 and not comparable to MIMIC.",
    "SOFA omits the CNS component. GCS was added to SICdb in v1.0.7 but the "
    "maintainer states it is 'not mandatory on our ICU and therefore not "
    "valid enough'; it is absent from the SignalFloat dictionary entirely. "
    "SOFA here therefore ranges 0-20, not 0-24, and a SICdb SOFA is NOT "
    "numerically comparable to a MIMIC SOFA.",
    "Respiratory rate coalesces 2274 (RespRateEKG, 1.71M rows) before 719 "
    "(RespRate, 0.52M) -- the EKG-derived channel is the dominant source "
    "here, the reverse of what the item naming suggests.",
    "FiO2 is DataID 2283 (RespiratorSetting), not 727 (SignalFloat FIO2), "
    "which has zero rows in data_float_h. 2283 is on a PERCENT scale "
    "(median 40.5), matching the MIMIC extract.",
    "BUN is derived as Harnstoff / 2.14. SICdb's 'Harnstoff' (355) is UREA "
    "in mg/dL; MIMIC's BUN is urea NITROGEN. Feeding urea as BUN would "
    "inflate the channel by 2.14x. This is a definition mismatch, not a unit "
    "mismatch, and no amount of normalisation would have caught it.",
    "WBC (301) and platelets (314) are reported in G/L, which is numerically "
    "identical to the K/uL used by the MIMIC extract. No conversion applied.",
    "Urine output (725) is per-hour, not cumulative: cnt averages 1.0 per "
    "hourly bin and the median is 50 mL with p99 500 mL. Verified against the "
    "real table rather than assumed from the '(c)' in the item name.",
    "GCS is EXCLUDED from the shared feature set entirely -- see "
    "POOLED_FEATURES. Carrying it as an all-missing channel would hand any "
    "pooled encoder a perfect hospital identifier.",
]

# --------------------------------------------------------------------------
# Signal mapping. Lists are coalesce priority: first non-null wins.
# --------------------------------------------------------------------------

VITAL_IDS = {
    "heart_rate":       [707, 724, 708],   # ECG, arterial-derived, SpO2-derived
    "spo2":             [710],
    "respiratory_rate": [2274, 719, 2280],
    "temperature":      [709],             # Celsius
    "map":              [703, 706],        # arterial, then non-invasive
    "sbp":              [701, 704],
    "fio2":             [2283],            # percent
    "urine_output":     [725],             # mL per hour
}

# LaboratoryID lists, serum/blood only -- urine, CSF, punctate and dialysate
# variants of the same analyte are deliberately excluded.
LAB_IDS = {
    "lactate":    [454, 465, 657],
    "ph":         [688, 538, 663],
    "creatinine": [367, 368],
    "wbc":        [301],
    "platelets":  [314],
    "bilirubin":  [333],                   # total, not direct (332)
    "pao2":       [689, 444, 664],
    "bun":        [355],                   # urea -> BUN, see UNIT_SCALE
    "glucose":    [331, 348, 656],
}

# Multiplicative conversions applied AFTER coalescing, to reach MIMIC units.
UNIT_SCALE = {
    "bun": 1.0 / 2.14,     # Harnstoff (urea, mg/dL) -> BUN (urea nitrogen)
}

# The MIMIC extended set is 18 features including GCS. GCS does not exist
# usably in SICdb, and a channel that is 100% missing in one source and
# largely present in the other is a PERFECT hospital label -- the encoder
# could read the site off the mask channel alone. Any pooled-pretraining arm
# must therefore drop it from both sources, not carry it as missing.
POOLED_FEATURES = (
    ["heart_rate", "spo2", "respiratory_rate", "temperature", "map", "sbp",
     "fio2", "urine_output"] +
    ["lactate", "ph", "creatinine", "wbc", "platelets", "bilirubin", "pao2",
     "bun", "glucose"]
)
POOLED_VITALS = POOLED_FEATURES[:8]
EXCLUDED_FROM_POOL = ["gcs"]

# Vasopressors for the cardiovascular SOFA component. Verified against
# d_references; matches the ids reported in SICDB_RECON.md section 3.
VASOPRESSOR_IDS = {
    "norepinephrine": 1562,
    "epinephrine": 1502,
    "dopamine": 1618,
    "dobutamine": 1559,
}
# Vasopressin (1550) is extracted but NOT scored, matching sepsis3.py.

# Systemic antibacterials, INN-unified. Selected from d_references Drug
# entries; deliberately EXCLUDES paromomycin (oral, non-absorbed, used for gut
# decontamination rather than systemic infection) and topical preparations
# such as Lidocain/Epinephrin.
ANTIBACTERIAL_IDS = [
    1401,  # Metronidazol
    1406,  # Amoxicillin/Clavulansaeure
    1408,  # Moxifloxacin
    1410,  # Aztreonam
    1418,  # Cefazolin
    1421,  # Cefepim
    1422,  # Ciprofloxacin
    1423,  # Cefotaxim
    1428,  # Clindamycin
    1431,  # Erythromycin
    1433,  # Trimethoprim/Sulfametoxazol
    1436,  # Flucloxacillin
    1439,  # Ceftazidim
    1446,  # Imipenem/Cilastin
    1449,  # Clarithromycin
    1451,  # Meropenem
    1454,  # Penicillin G
    1455,  # Rifampicin
    1456,  # Ceftriaxon
    1457,  # Ampicillin
    1458,  # Levofloxacin
    1459,  # Piperacillin/Tazobactam
    1460,  # Ampicillin/Sulbactam
    1461,  # Vancomycin
    1462,  # Linezolid
    1543,  # Gentamycinsulfat
    1577,  # Cefuroxim
    1605,  # Gentamycin
    1628,  # Fosfomycin
]

# Cohort and timing rules, mirroring sepsis3.py / extract_mimic.py.
MIN_AGE = 18
MIN_LOS_HOURS = 6
SUSPICION_WINDOW_BEFORE_H = 24   # t_SOFA may precede t_suspicion by this much
SUSPICION_WINDOW_AFTER_H = 12
SOFA_LOOKBACK_H = 24             # rise is vs the min of the preceding 24h
SOFA_RISE = 2
# A treatment course, as opposed to perioperative prophylaxis. Prophylaxis is
# typically a single perioperative dose or a <=24h run, so requiring both a
# dose count and a span separates most of it. Measured effect on the real
# cohort: 21,260 cases have any antibacterial, 14,436 have >=3 doses, 11,157
# span >=24h, and 11,064 satisfy both.
COURSE_MIN_DOSES = 3
COURSE_MIN_SPAN_H = 24


def register_sources(con, data_root):
    """Create typed DuckDB views over the SICdb CSVs.

    Types are pinned rather than sniffed. `read_csv_auto` infers per file, and
    a column that is empty in one table comes back VARCHAR, after which every
    numeric comparison fails at bind time -- the trap documented in HANDOFF.md
    for the MIMIC loader.

    NOTE for every query against these views: `Offset` is a RESERVED WORD in
    DuckDB and must be double-quoted ("Offset") or aliased. An unquoted
    reference fails at parse time with a misleading "syntax error at or near"
    pointing one token further on.
    """
    def view(name, cols):
        spec = ", ".join("'%s': '%s'" % (k, v) for k, v in cols.items())
        con.execute(
            "CREATE OR REPLACE VIEW %s AS SELECT * FROM read_csv('%s/%s.csv.gz',"
            " columns={%s}, header=true)" % (name, data_root, name, spec))

    view("cases", {
        "CaseID": "BIGINT", "PatientID": "BIGINT", "AdmissionYear": "BIGINT",
        "TimeOfStay": "BIGINT", "ICUOffset": "BIGINT", "saps3": "DOUBLE",
        "HospitalDischargeType": "BIGINT", "HospitalDischargeDay": "DOUBLE",
        "HospitalStayDays": "DOUBLE", "DischargeState": "BIGINT",
        "DischargeUnit": "BIGINT", "OffsetOfDeath": "DOUBLE",
        "EstimatedSurvivalObservationTime": "BIGINT", "Sex": "BIGINT",
        "WeightOnAdmission": "DOUBLE", "HeightOnAdmission": "DOUBLE",
        "AgeOnAdmission": "DOUBLE", "HospitalUnit": "BIGINT",
        "ReferringUnit": "BIGINT", "ICD10Main": "VARCHAR",
        "ICD10MainText": "VARCHAR", "DiagnosisT2": "VARCHAR",
        "SurgicalSite": "BIGINT", "HoursOfCRRT": "DOUBLE",
        "AdmissionUrgency": "BIGINT", "AdmissionFormHasSepsis": "BIGINT",
        "SurgicalAdmissionType": "BIGINT", "OrbisDataAvailable": "BIGINT",
        "HeartSurgeryAdditionalData": "BIGINT",
        "HeartSurgeryCPBTime": "DOUBLE", "HeartSurgeryCrossClampTime": "DOUBLE",
        "HeartSurgeryBeginOffset": "DOUBLE", "HeartSurgeryEndOffset": "DOUBLE",
        "OffsetAfterFirstAdmission": "DOUBLE"})

    view("data_float_h", {
        "id": "BIGINT", "CaseID": "BIGINT", "DataID": "BIGINT",
        "Offset": "BIGINT", "Val": "DOUBLE", "cnt": "BIGINT",
        "rawdata": "VARCHAR"})

    view("laboratory", {
        "id": "BIGINT", "CaseID": "BIGINT", "LaboratoryID": "BIGINT",
        "Offset": "BIGINT", "LaboratoryValue": "DOUBLE",
        "LaboratoryType": "BIGINT"})

    view("medication", {
        "id": "BIGINT", "CaseID": "BIGINT", "PatientID": "BIGINT",
        "DrugID": "BIGINT", "Offset": "BIGINT", "OffsetDrugEnd": "BIGINT",
        "IsSingleDose": "BIGINT", "Amount": "DOUBLE",
        "AmountPerMinute": "DOUBLE", "GivenState": "BIGINT"})
    return con


def hours_from_icu(offset_seconds, icu_offset_seconds):
    """SICdb offsets are seconds from PDMS (pre-surgical) admission."""
    return (np.asarray(offset_seconds, dtype=float)
            - float(icu_offset_seconds)) / 3600.0


def sofa_components(resp, coag, liver, cardio, renal):
    """Total SOFA from the five scorable components (CNS omitted).

    Returns a value in 0-20. See DEVIATIONS: a SICdb SOFA is not numerically
    comparable to a MIMIC SOFA because the CNS component is missing from one
    and present in the other.
    """
    return (np.nan_to_num(resp) + np.nan_to_num(coag) + np.nan_to_num(liver)
            + np.nan_to_num(cardio) + np.nan_to_num(renal))


def first_sofa_rise(sofa_hourly, lookback_h=SOFA_LOOKBACK_H, rise=SOFA_RISE):
    """First hour whose SOFA is >= `rise` above the min of the preceding window.

    Same Challenge rule as sepsis3.py. Hour 0 cannot qualify: there is no
    preceding window, and treating an empty window as SOFA 0 is exactly the
    bug that made ICU admission itself read as a 2-point rise for 58% of
    stays in the MIMIC demo.
    """
    s = np.asarray(sofa_hourly, dtype=float)
    for t in range(1, len(s)):
        lo = max(0, t - lookback_h)
        window = s[lo:t]
        if window.size == 0:
            continue
        if s[t] - np.nanmin(window) >= rise:
            return t
    return None


def t_sepsis_from(t_suspicion_h, t_sofa_h,
                  before=SUSPICION_WINDOW_BEFORE_H,
                  after=SUSPICION_WINDOW_AFTER_H):
    """min(t_suspicion, t_SOFA) when t_SOFA falls in the suspicion window."""
    if t_suspicion_h is None or t_sofa_h is None:
        return None
    if not (t_suspicion_h - before <= t_sofa_h <= t_suspicion_h + after):
        return None
    return float(min(t_suspicion_h, t_sofa_h))
