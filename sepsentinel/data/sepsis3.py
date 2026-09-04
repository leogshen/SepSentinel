# Sepsis-3 (PhysioNet/CinC 2019 Challenge timing rules) labelling for MIMIC-IV.
#
# Ported to DuckDB from the MIT-LCP `mimic-code` concepts
# (suspicion_of_infection, sofa, sepsis3) per DATA_ACCESS_SPEC.md section 3.
# Every deviation from mimic-code is listed in DEVIATIONS below and is
# reproduced in the report written by scripts/build_sepsis3_labels.py.
#
# Definition implemented (Reyna et al. 2020, the Challenge rules):
#   t_suspicion : earlier of (antibiotic start, culture draw) for a pair where
#                 the culture is <=24h AFTER the antibiotic, or the antibiotic
#                 is <=72h AFTER the culture.
#   t_SOFA      : first hour whose total SOFA is >=2 points above the MINIMUM
#                 of the preceding 24h.
#   t_sepsis    : min(t_suspicion, t_SOFA), valid only when t_SOFA falls in
#                 [t_suspicion - 24h, t_suspicion + 12h].
#
# All times are hours from ICU intime (suspicion may be negative: it can
# precede ICU admission). t_sepsis_hour is UNSHIFTED -- the training label
# shift lives in the loader (gridding.make_labels), per spec section 3.
#
# LEAKAGE: nothing computed here (SOFA, antibiotics, cultures) may become a
# model feature. This module only produces t_sepsis_hour.

import numpy as np

# --- mimic-code deviations, documented (spec section 13 step 4) ------------
DEVIATIONS = [
    "t_SOFA uses the Challenge rule (>=2-point RISE vs the minimum of the "
    "preceding 24h). mimic-code's sepsis3.sql instead requires an absolute "
    "SOFA >=2 in the suspicion window (baseline-0 assumption). Both are "
    "computed; the absolute variant is reported as t_sepsis_hour_mimiccode.",
    "Sepsis window is [t_susp-24h, t_susp+12h] (Challenge). mimic-code uses "
    "[t_susp-48h, t_susp+24h]; that variant is also reported.",
    "Events are binned with floor(hours_from_intime) so hour bins are "
    "[t, t+1), matching gridding.py. mimic-code's sofa.sql anchors its hourly "
    "windows on the bin end time; the difference is sub-hour.",
    "PaO2/FiO2: PaO2 from chartevents 220224 (arterial) and labevents 50821; "
    "FiO2 from chartevents 223835 carried forward at most 6h (ASOF join). "
    "mimic-code pairs them inside its `bg` concept using specimen linkage.",
    "GCS requires all three components (220739/223900/223901) at the same "
    "charttime; mimic-code carries components forward up to 6h. In the demo "
    "3244/3279 GCS charttimes are complete triples.",
    "The urine-output renal criterion is applied only from hour 24 onward, "
    "where the 24h window is full; before that renal scores on creatinine "
    "alone. mimic-code scores a partial-window uo_24hr, which inflates the "
    "renal component early in a stay.",
    "Hourly SOFA is scored only from ICU hour 0 onward; labs drawn up to 24h "
    "before intime are clamped into hour 0 rather than scored in pre-ICU "
    "hours. Scoring empty pre-ICU hours gives SOFA 0 there (missing data read "
    "as normal organ function), which made admission itself look like a "
    ">=2-point rise for nearly every stay. mimic-code's sofa.sql does emit "
    "hr < 0 rows, but its sepsis3.sql uses an absolute threshold, so it never "
    "differences against them.",
    "Vasopressin (222315) is extracted but not scored: classical SOFA and "
    "mimic-code's sofa.sql both score only dopamine/dobutamine/epinephrine/"
    "norepinephrine.",
    "Antibiotic times come from hosp.prescriptions.starttime UNION the "
    "administration times in hosp.emar (event_txt like 'Administered'). "
    "mimic-code's antibiotic.sql uses prescriptions only.",
]

# Antibiotic name patterns (mimic-code antibiotic.sql list, generic + common
# US brand names). Matched case-insensitively as substrings.
ANTIBIOTIC_PATTERNS = [
    "adoxa", "ala-tet", "alodox", "amikacin", "amikin", "amoxicillin",
    "augmentin", "ampicillin", "unasyn", "avelox", "avidoxy", "azactam",
    "azithromycin", "aztreonam", "bactocill", "bactrim", "bethkis", "biaxin",
    "cayston", "cefazolin", "cedax", "cefoxitin", "ceftazidime", "cefaclor",
    "cefadroxil", "cefdinir", "cefditoren", "cefepime", "cefotan", "cefotetan",
    "cefotaxime", "ceftaroline", "cefpodoxime", "cefpirome", "cefprozil",
    "ceftibuten", "ceftin", "ceftriaxone", "cefuroxime", "cephalexin",
    "chloramphenicol", "cipro", "ciprofloxacin", "claforan", "clarithromycin",
    "cleocin", "clindamycin", "colistin", "cubicin", "daptomycin",
    "dicloxacillin", "doryx", "doxycycline", "doxy", "duricef", "dynacin",
    "ery-tab", "eryped", "eryc", "erythrocin", "erythromycin", "ertapenem",
    "factive", "flagyl", "fortaz", "furadantin", "garamycin", "gentamicin",
    "imipenem", "cilastatin", "kanamycin", "keflex", "kefzol", "ketek",
    "levaquin", "levofloxacin", "lincocin", "linezolid", "macrobid",
    "macrodantin", "maxipime", "mefoxin", "meropenem", "meronem",
    "methicillin", "metronidazole", "minocin", "minocycline", "monodox",
    "monurol", "morgidox", "moxatag", "moxifloxacin", "mupirocin", "myrac",
    "nafcillin", "neomycin", "nitrofurantoin", "norfloxacin", "noroxin",
    "ocudox", "ofloxacin", "omnicef", "oracea", "oraxyl", "oxacillin",
    "pediazole", "penicillin", "periostat", "pfizerpen", "piperacillin",
    "tazobactam", "polymyxin", "primsol", "proquin", "raniclor", "rifadin",
    "rifampin", "rocephin", "septra", "seromycin", "smz-tmp", "streptomycin",
    "sulfadiazine", "sulfamethoxazole", "sulfatrim", "sulfisoxazole",
    "suprax", "synercid", "tazicef", "tetracycline", "tigecycline", "timentin",
    "tinidazole", "tobramycin", "trimethoprim", "vancocin", "vancomycin",
    "vantin", "vibativ", "vibra-tabs", "vibramycin", "zinacef", "zithromax",
    "zosyn", "zyvox",
]
# Topical/ophthalmic/otic routes cannot treat a systemic infection.
EXCLUDED_ROUTES = [
    "OU", "OS", "OD", "AU", "AS", "AD", "TP", "IRR", "BOTH EYES", "LEFT EYE",
    "RIGHT EYE", "BOTH EARS", "LEFT EAR", "RIGHT EAR", "DWELL", "NU", "OP",
]
EMAR_ADMINISTERED = [
    "Administered", "Administered in Other Location", "Confirmed",
    "Delayed Administered", "Started", "Applied", "Restarted",
]

# SOFA source itemids (verified against MIMIC-IV 3.1 and demo 2.2 dictionaries).
MAP_ITEMS = (220052, 220181, 225312)
FIO2_ITEM = 223835
PAO2_CHART_ITEM = 220224
GCS_ITEMS = {"eye": 220739, "verbal": 223900, "motor": 223901}
VENT_ITEMS = (225792, 225794)              # invasive / non-invasive
URINE_ITEMS = (226559, 226560, 226561, 226584, 226563, 226564, 226565,
               226567, 226557, 226558, 227489)
VASO_ITEMS = {221906: "norepinephrine", 221289: "epinephrine",
              221662: "dopamine", 221653: "dobutamine", 222315: "vasopressin"}
LAB_PLATELET, LAB_BILIRUBIN, LAB_CREATININE, LAB_PAO2 = 51265, 50885, 50912, 50821

MIN_AGE = 18
MIN_LOS_HOURS = 6
SUSPICION_ABX_TO_CULTURE_H = 24   # culture drawn <=24h AFTER antibiotics
SUSPICION_CULTURE_TO_ABX_H = 72   # antibiotics <=72h AFTER culture
SEPSIS_WINDOW_BEFORE_H = 24       # t_SOFA >= t_susp - 24h
SEPSIS_WINDOW_AFTER_H = 12        # t_SOFA <= t_susp + 12h
MIMICCODE_WINDOW_BEFORE_H = 48
MIMICCODE_WINDOW_AFTER_H = 24
SOFA_LOOKBACK_H = 24


def _quoted(values):
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def _like_clause(column, patterns):
    return " OR ".join("LOWER(%s) LIKE '%%%s%%'" % (column, p) for p in patterns)


def register_sources(con, root, limit_stays=None, stay_ids=None):
    """Register the raw MIMIC-IV CSVs and build the stay cohort.

    Creates the temp table `s3_stays`; every later step joins to it, so a
    limited cohort keeps the whole labelling job small.
    """
    root = str(root).rstrip("/\\").replace("\\", "/")
    # Column types are pinned rather than sniffed: a table that happens to
    # contain no usable rows (or a lab value like '___') otherwise comes back
    # as VARCHAR and every downstream comparison fails at bind time.
    ID, TS, DBL = "BIGINT", "TIMESTAMP", "DOUBLE"
    src = {
        "icustays": ("icu/icustays.csv.gz",
                     {"subject_id": ID, "hadm_id": ID, "stay_id": ID,
                      "intime": TS, "outtime": TS}),
        "patients": ("hosp/patients.csv.gz",
                     {"subject_id": ID, "anchor_age": ID}),
        "chartevents": ("icu/chartevents.csv.gz",
                        {"subject_id": ID, "stay_id": ID, "itemid": ID,
                         "charttime": TS, "valuenum": DBL}),
        "labevents": ("hosp/labevents.csv.gz",
                      {"subject_id": ID, "itemid": ID, "charttime": TS,
                       "valuenum": DBL}),
        "prescriptions": ("hosp/prescriptions.csv.gz",
                          {"subject_id": ID, "starttime": TS,
                           "drug": "VARCHAR", "route": "VARCHAR"}),
        "emar": ("hosp/emar.csv.gz",
                 {"subject_id": ID, "charttime": TS, "medication": "VARCHAR",
                  "event_txt": "VARCHAR"}),
        "microbiologyevents": ("hosp/microbiologyevents.csv.gz",
                               {"subject_id": ID, "charttime": TS,
                                "chartdate": TS}),
        "inputevents": ("icu/inputevents.csv.gz",
                        {"stay_id": ID, "itemid": ID, "starttime": TS,
                         "endtime": TS, "rate": DBL, "patientweight": DBL,
                         "rateuom": "VARCHAR", "statusdescription": "VARCHAR"}),
        "outputevents": ("icu/outputevents.csv.gz",
                         {"stay_id": ID, "itemid": ID, "charttime": TS,
                          "value": DBL}),
        "procedureevents": ("icu/procedureevents.csv.gz",
                            {"stay_id": ID, "itemid": ID, "starttime": TS,
                             "endtime": TS}),
    }
    for name, (path, types) in src.items():
        type_arg = ", ".join("'%s': '%s'" % (c, t) for c, t in types.items())
        con.execute(
            "CREATE OR REPLACE VIEW mv_%s AS SELECT * FROM "
            "read_csv_auto('%s/%s', types = {%s})" % (name, root, path, type_arg)
        )

    where = ["p.anchor_age >= %d" % MIN_AGE,
             "DATEDIFF('second', i.intime, i.outtime) / 3600.0 >= %d" % MIN_LOS_HOURS]
    if stay_ids is not None:
        where.append("i.stay_id IN (%s)" % ",".join(str(int(s)) for s in stay_ids))
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_stays AS
        SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime,
               DATEDIFF('second', i.intime, i.outtime) / 3600.0 AS los_hours,
               p.anchor_age
        FROM mv_icustays i JOIN mv_patients p USING (subject_id)
        WHERE %s
        ORDER BY i.stay_id
        %s
    """ % (" AND ".join(where),
           ("LIMIT %d" % int(limit_stays)) if limit_stays else ""))
    return con.execute("SELECT COUNT(*) FROM s3_stays").fetchone()[0]


def build_suspicion(con):
    """t_suspicion candidates per stay (temp table `s3_suspicion`)."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_abx AS
        SELECT DISTINCT st.stay_id, st.subject_id, t.abx_time
        FROM (
            SELECT subject_id, CAST(starttime AS TIMESTAMP) AS abx_time
            FROM mv_prescriptions
            WHERE starttime IS NOT NULL AND (%s)
              AND (route IS NULL OR UPPER(route) NOT IN (%s))
              AND LOWER(drug) NOT LIKE '%%desensitization%%'
            UNION ALL
            SELECT subject_id, CAST(charttime AS TIMESTAMP) AS abx_time
            FROM mv_emar
            WHERE charttime IS NOT NULL AND (%s)
              AND event_txt IN (%s)
        ) t
        JOIN s3_stays st ON st.subject_id = t.subject_id
         AND t.abx_time >= st.intime - INTERVAL 24 HOUR
         AND t.abx_time <= st.outtime
    """ % (_like_clause("drug", ANTIBIOTIC_PATTERNS), _quoted(EXCLUDED_ROUTES),
           _like_clause("medication", ANTIBIOTIC_PATTERNS),
           _quoted(EMAR_ADMINISTERED)))

    # One row per specimen: microbiologyevents repeats a specimen per organism
    # and per tested antibiotic.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_culture AS
        SELECT DISTINCT subject_id,
               CAST(COALESCE(charttime, chartdate) AS TIMESTAMP) AS cult_time
        FROM mv_microbiologyevents
        WHERE COALESCE(charttime, chartdate) IS NOT NULL
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_suspicion AS
        SELECT a.stay_id,
               DATEDIFF('second', st.intime, LEAST(a.abx_time, c.cult_time))
                   / 3600.0 AS suspicion_hour,
               DATEDIFF('second', st.intime, a.abx_time) / 3600.0 AS abx_hour,
               DATEDIFF('second', st.intime, c.cult_time) / 3600.0 AS culture_hour
        FROM s3_abx a
        JOIN s3_stays st USING (stay_id)
        JOIN s3_culture c ON c.subject_id = a.subject_id
        WHERE (c.cult_time > a.abx_time
               AND c.cult_time <= a.abx_time + INTERVAL %d HOUR)
           OR (c.cult_time <= a.abx_time
               AND a.abx_time <= c.cult_time + INTERVAL %d HOUR)
    """ % (SUSPICION_ABX_TO_CULTURE_H, SUSPICION_CULTURE_TO_ABX_H))


def build_hourly_sofa(con):
    """Hourly SOFA per stay (temp table `s3_sofa`), components + total."""
    chart_items = MAP_ITEMS + (FIO2_ITEM, PAO2_CHART_ITEM) + tuple(GCS_ITEMS.values())
    lab_items = (LAB_PLATELET, LAB_BILIRUBIN, LAB_CREATININE, LAB_PAO2)
    ids = lambda xs: ",".join(str(i) for i in xs)

    # One pass over each event table, restricted to the cohort.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_ce AS
        SELECT c.stay_id, c.itemid, CAST(c.charttime AS TIMESTAMP) AS charttime,
               CAST(c.valuenum AS DOUBLE) AS valuenum,
               CAST(FLOOR(DATEDIFF('second', st.intime,
                    CAST(c.charttime AS TIMESTAMP)) / 3600.0) AS INTEGER) AS hr
        FROM mv_chartevents c JOIN s3_stays st USING (stay_id)
        WHERE c.itemid IN (%s) AND c.valuenum IS NOT NULL
    """ % ids(chart_items))

    # Pre-ICU labs (up to 24h before intime) are clamped into hour 0: they are
    # known at admission and must inform hour 0's 24h window, but they must NOT
    # create pre-ICU grid hours whose SOFA is scored on labs alone -- that
    # partial score is what made every admission look like a >=2-point rise.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_le AS
        SELECT st.stay_id, l.itemid, CAST(l.charttime AS TIMESTAMP) AS charttime,
               CAST(l.valuenum AS DOUBLE) AS valuenum,
               GREATEST(CAST(FLOOR(DATEDIFF('second', st.intime,
                    CAST(l.charttime AS TIMESTAMP)) / 3600.0) AS INTEGER), 0) AS hr
        FROM mv_labevents l
        JOIN s3_stays st ON l.subject_id = st.subject_id
         AND CAST(l.charttime AS TIMESTAMP) >= st.intime - INTERVAL %d HOUR
         AND CAST(l.charttime AS TIMESTAMP) <= st.outtime
        WHERE l.itemid IN (%s) AND l.valuenum IS NOT NULL
    """ % (SOFA_LOOKBACK_H, ids(lab_items)))

    # --- ventilation windows, FiO2, PaO2 -> P/F ratio ----------------------
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_vent AS
        SELECT stay_id, CAST(starttime AS TIMESTAMP) AS starttime,
               CAST(endtime AS TIMESTAMP) AS endtime
        FROM mv_procedureevents
        WHERE itemid IN (%s) AND stay_id IN (SELECT stay_id FROM s3_stays)
    """ % ids(VENT_ITEMS))

    # FiO2 is charted either as a fraction (0.21-1.0) or a percent (21-100).
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_fio2 AS
        SELECT stay_id, charttime,
               CASE WHEN valuenum > 1.0 THEN valuenum ELSE valuenum * 100.0 END AS fio2
        FROM s3_ce
        WHERE itemid = %d
          AND ((valuenum BETWEEN 0.21 AND 1.0) OR (valuenum BETWEEN 21 AND 100))
        ORDER BY stay_id, charttime
    """ % FIO2_ITEM)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_pao2 AS
        SELECT * FROM (
            SELECT stay_id, charttime, hr, valuenum AS pao2 FROM s3_ce
            WHERE itemid = %d AND valuenum BETWEEN 30 AND 700
            UNION ALL
            SELECT stay_id, charttime, hr, valuenum FROM s3_le
            WHERE itemid = %d AND valuenum BETWEEN 30 AND 700
        ) ORDER BY stay_id, charttime
    """ % (PAO2_CHART_ITEM, LAB_PAO2))

    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_pf AS
        WITH paired AS (
            SELECT p.stay_id, p.hr, p.charttime, p.pao2, f.fio2,
                   f.charttime AS fio2_time
            FROM s3_pao2 p
            ASOF LEFT JOIN s3_fio2 f
              ON p.stay_id = f.stay_id AND p.charttime >= f.charttime
        )
        SELECT stay_id, hr,
               pao2 / (fio2 / 100.0) AS ratio,
               EXISTS (SELECT 1 FROM s3_vent v
                       WHERE v.stay_id = paired.stay_id
                         AND paired.charttime >= v.starttime
                         AND paired.charttime <= v.endtime) AS on_vent
        FROM paired
        WHERE fio2 IS NOT NULL AND fio2_time >= charttime - INTERVAL 6 HOUR
    """)

    # --- GCS: all three components at one charttime -------------------------
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_gcs AS
        SELECT stay_id, hr, MIN(gcs) AS gcs
        FROM (
            SELECT stay_id, hr, charttime,
                   MAX(CASE WHEN itemid = %d THEN valuenum END)
                 + MAX(CASE WHEN itemid = %d THEN valuenum END)
                 + MAX(CASE WHEN itemid = %d THEN valuenum END) AS gcs
            FROM s3_ce
            WHERE itemid IN (%s)
            GROUP BY stay_id, hr, charttime
        )
        WHERE gcs IS NOT NULL
        GROUP BY stay_id, hr
    """ % (GCS_ITEMS["eye"], GCS_ITEMS["verbal"], GCS_ITEMS["motor"],
           ids(GCS_ITEMS.values())))

    # --- vasopressors, normalised to mcg/kg/min, expanded over their hours --
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_vaso AS
        WITH inf AS (
            SELECT i.stay_id, i.itemid,
                   CAST(FLOOR(DATEDIFF('second', st.intime,
                        CAST(i.starttime AS TIMESTAMP)) / 3600.0) AS INTEGER) AS hr_start,
                   CAST(FLOOR(DATEDIFF('second', st.intime,
                        CAST(i.endtime AS TIMESTAMP)) / 3600.0) AS INTEGER) AS hr_end,
                   CASE LOWER(i.rateuom)
                       WHEN 'mcg/kg/min'  THEN CAST(i.rate AS DOUBLE)
                       WHEN 'mg/kg/min'   THEN CAST(i.rate AS DOUBLE) * 1000.0
                       WHEN 'mcg/min'     THEN CAST(i.rate AS DOUBLE) / NULLIF(CAST(i.patientweight AS DOUBLE), 0)
                       WHEN 'mg/min'      THEN CAST(i.rate AS DOUBLE) * 1000.0 / NULLIF(CAST(i.patientweight AS DOUBLE), 0)
                       WHEN 'mcg/kg/hour' THEN CAST(i.rate AS DOUBLE) / 60.0
                       WHEN 'mg/kg/hour'  THEN CAST(i.rate AS DOUBLE) * 1000.0 / 60.0
                   END AS rate_mcgkgmin
            FROM mv_inputevents i JOIN s3_stays st USING (stay_id)
            WHERE i.itemid IN (%s) AND i.rate IS NOT NULL
              AND COALESCE(i.statusdescription, '') <> 'Rewritten'
        )
        SELECT stay_id, UNNEST(generate_series(hr_start, hr_end, 1)) AS hr,
               itemid, rate_mcgkgmin
        FROM inf WHERE rate_mcgkgmin IS NOT NULL AND hr_end >= hr_start
    """ % ids(VASO_ITEMS))

    # --- hourly measurement table ------------------------------------------
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_meas AS
        SELECT stay_id, hr,
               MIN(map_v) AS map_min, MIN(platelet) AS platelet_min,
               MAX(bilirubin) AS bilirubin_max, MAX(creatinine) AS creatinine_max,
               MIN(gcs) AS gcs_min, MIN(pf_vent) AS pf_vent,
               MIN(pf_novent) AS pf_novent, SUM(uo) AS uo,
               MAX(rate_norepinephrine) AS rate_norepinephrine,
               MAX(rate_epinephrine) AS rate_epinephrine,
               MAX(rate_dopamine) AS rate_dopamine,
               MAX(rate_dobutamine) AS rate_dobutamine
        FROM (
            SELECT stay_id, hr, valuenum AS map_v,
                   NULL::DOUBLE AS platelet, NULL::DOUBLE AS bilirubin,
                   NULL::DOUBLE AS creatinine, NULL::DOUBLE AS gcs,
                   NULL::DOUBLE AS pf_vent, NULL::DOUBLE AS pf_novent,
                   NULL::DOUBLE AS uo, NULL::DOUBLE AS rate_norepinephrine,
                   NULL::DOUBLE AS rate_epinephrine, NULL::DOUBLE AS rate_dopamine,
                   NULL::DOUBLE AS rate_dobutamine
            FROM s3_ce WHERE itemid IN (%s) AND valuenum BETWEEN 20 AND 200
            UNION ALL
            SELECT stay_id, hr, NULL::DOUBLE,
                   CASE WHEN itemid = %d THEN valuenum END,
                   CASE WHEN itemid = %d THEN valuenum END,
                   CASE WHEN itemid = %d THEN valuenum END,
                   NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE,
                   NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE
            FROM s3_le WHERE itemid IN (%d, %d, %d)
            UNION ALL
            SELECT stay_id, hr, NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE,
                   NULL::DOUBLE, gcs, NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE,
                   NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE
            FROM s3_gcs
            UNION ALL
            SELECT stay_id, hr, NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE,
                   NULL::DOUBLE, NULL::DOUBLE,
                   CASE WHEN on_vent THEN ratio END,
                   CASE WHEN NOT on_vent THEN ratio END,
                   NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE,
                   NULL::DOUBLE
            FROM s3_pf
            UNION ALL
            SELECT o.stay_id,
                   CAST(FLOOR(DATEDIFF('second', st.intime,
                        CAST(o.charttime AS TIMESTAMP)) / 3600.0) AS INTEGER),
                   NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE,
                   NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE,
                   CAST(o.value AS DOUBLE), NULL::DOUBLE, NULL::DOUBLE,
                   NULL::DOUBLE, NULL::DOUBLE
            FROM mv_outputevents o JOIN s3_stays st USING (stay_id)
            WHERE o.itemid IN (%s) AND o.value IS NOT NULL AND o.value >= 0
            UNION ALL
            SELECT stay_id, hr, NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE,
                   NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE, NULL::DOUBLE,
                   NULL::DOUBLE,
                   CASE WHEN itemid = 221906 THEN rate_mcgkgmin END,
                   CASE WHEN itemid = 221289 THEN rate_mcgkgmin END,
                   CASE WHEN itemid = 221662 THEN rate_mcgkgmin END,
                   CASE WHEN itemid = 221653 THEN rate_mcgkgmin END
            FROM s3_vaso
        )
        GROUP BY stay_id, hr
    """ % (ids(MAP_ITEMS), LAB_PLATELET, LAB_BILIRUBIN, LAB_CREATININE,
           LAB_PLATELET, LAB_BILIRUBIN, LAB_CREATININE, ids(URINE_ITEMS)))

    # --- dense hourly grid, 24h rolling windows, component scores ----------
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_sofa AS
        WITH grid AS (
            SELECT stay_id,
                   UNNEST(generate_series(0, CAST(CEIL(los_hours) AS INTEGER), 1))
                       AS hr
            FROM s3_stays
        ),
        joined AS (
            SELECT g.stay_id, g.hr, m.map_min, m.platelet_min, m.bilirubin_max,
                   m.creatinine_max, m.gcs_min, m.pf_vent, m.pf_novent, m.uo,
                   m.rate_norepinephrine, m.rate_epinephrine,
                   m.rate_dopamine, m.rate_dobutamine
            FROM grid g LEFT JOIN s3_meas m USING (stay_id, hr)
        ),
        win AS (
            SELECT stay_id, hr,
                   MIN(map_min)        OVER w AS map_24,
                   MIN(platelet_min)   OVER w AS platelet_24,
                   MAX(bilirubin_max)  OVER w AS bilirubin_24,
                   MAX(creatinine_max) OVER w AS creatinine_24,
                   MIN(gcs_min)        OVER w AS gcs_24,
                   MIN(pf_vent)        OVER w AS pf_vent_24,
                   MIN(pf_novent)      OVER w AS pf_novent_24,
                   SUM(uo)             OVER w AS uo_24,
                   rate_norepinephrine, rate_epinephrine,
                   rate_dopamine, rate_dobutamine
            FROM joined
            WINDOW w AS (PARTITION BY stay_id ORDER BY hr
                         ROWS BETWEEN %d PRECEDING AND CURRENT ROW)
        ),
        scored AS (
            SELECT stay_id, hr,
                CASE WHEN pf_vent_24 < 100 THEN 4
                     WHEN pf_vent_24 < 200 THEN 3
                     WHEN COALESCE(pf_vent_24, pf_novent_24) < 300 THEN 2
                     WHEN COALESCE(pf_vent_24, pf_novent_24) < 400 THEN 1
                     ELSE 0 END AS respiration,
                CASE WHEN platelet_24 < 20 THEN 4 WHEN platelet_24 < 50 THEN 3
                     WHEN platelet_24 < 100 THEN 2 WHEN platelet_24 < 150 THEN 1
                     ELSE 0 END AS coagulation,
                CASE WHEN bilirubin_24 >= 12.0 THEN 4 WHEN bilirubin_24 >= 6.0 THEN 3
                     WHEN bilirubin_24 >= 2.0 THEN 2 WHEN bilirubin_24 >= 1.2 THEN 1
                     ELSE 0 END AS liver,
                CASE WHEN rate_dopamine > 15 OR rate_epinephrine > 0.1
                          OR rate_norepinephrine > 0.1 THEN 4
                     WHEN rate_dopamine > 5 OR rate_epinephrine > 0
                          OR rate_norepinephrine > 0 THEN 3
                     WHEN rate_dopamine > 0 OR rate_dobutamine > 0 THEN 2
                     WHEN map_24 < 70 THEN 1
                     ELSE 0 END AS cardiovascular,
                CASE WHEN gcs_24 < 6 THEN 4 WHEN gcs_24 < 10 THEN 3
                     WHEN gcs_24 < 13 THEN 2 WHEN gcs_24 < 15 THEN 1
                     ELSE 0 END AS cns,
                CASE WHEN creatinine_24 >= 5.0 THEN 4
                     WHEN hr >= %d AND uo_24 < 200 THEN 4
                     WHEN creatinine_24 >= 3.5 THEN 3
                     WHEN hr >= %d AND uo_24 < 500 THEN 3
                     WHEN creatinine_24 >= 2.0 THEN 2
                     WHEN creatinine_24 >= 1.2 THEN 1
                     ELSE 0 END AS renal
            FROM win
        )
        SELECT stay_id, hr, respiration, coagulation, liver, cardiovascular,
               cns, renal,
               respiration + coagulation + liver + cardiovascular + cns + renal
                   AS sofa
        FROM scored
    """ % (SOFA_LOOKBACK_H - 1, SOFA_LOOKBACK_H, SOFA_LOOKBACK_H))


def build_labels(con):
    """Join suspicion + SOFA into t_sepsis (temp table `s3_labels`).

    Returns the labels as a pandas DataFrame, one row per cohort stay.
    """
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_sofa_rise AS
        SELECT stay_id, hr, sofa,
               sofa - MIN(sofa) OVER (PARTITION BY stay_id ORDER BY hr
                    ROWS BETWEEN %d PRECEDING AND CURRENT ROW) AS sofa_delta
        FROM s3_sofa
    """ % SOFA_LOOKBACK_H)

    # Challenge rule: >=2-point rise, inside [t_susp-24h, t_susp+12h].
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_pairs AS
        SELECT su.stay_id,
               LEAST(su.suspicion_hour, CAST(sf.hr AS DOUBLE)) AS t_sepsis_hour,
               su.suspicion_hour AS t_suspicion_hour,
               CAST(sf.hr AS DOUBLE) AS t_sofa_hour,
               sf.sofa, sf.sofa_delta
        FROM s3_suspicion su
        JOIN s3_sofa_rise sf ON sf.stay_id = su.stay_id
        WHERE sf.hr >= 0 AND sf.sofa_delta >= 2
          AND sf.hr >= su.suspicion_hour - %d
          AND sf.hr <= su.suspicion_hour + %d
        QUALIFY ROW_NUMBER() OVER (PARTITION BY su.stay_id
                ORDER BY LEAST(su.suspicion_hour, CAST(sf.hr AS DOUBLE)),
                         sf.hr) = 1
    """ % (SEPSIS_WINDOW_BEFORE_H, SEPSIS_WINDOW_AFTER_H))

    # mimic-code variant: absolute SOFA >= 2, window [-48h, +24h].
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_pairs_mc AS
        SELECT su.stay_id,
               LEAST(su.suspicion_hour, CAST(sf.hr AS DOUBLE))
                   AS t_sepsis_hour_mimiccode
        FROM s3_suspicion su
        JOIN s3_sofa_rise sf ON sf.stay_id = su.stay_id
        WHERE sf.hr >= 0 AND sf.sofa >= 2
          AND sf.hr >= su.suspicion_hour - %d
          AND sf.hr <= su.suspicion_hour + %d
        QUALIFY ROW_NUMBER() OVER (PARTITION BY su.stay_id
                ORDER BY LEAST(su.suspicion_hour, CAST(sf.hr AS DOUBLE)), sf.hr) = 1
    """ % (MIMICCODE_WINDOW_BEFORE_H, MIMICCODE_WINDOW_AFTER_H))

    con.execute("""
        CREATE OR REPLACE TEMP TABLE s3_labels AS
        SELECT st.stay_id, st.subject_id, st.hadm_id, st.los_hours,
               p.t_sepsis_hour, p.t_suspicion_hour, p.t_sofa_hour,
               p.sofa AS sofa_at_onset, p.sofa_delta AS sofa_delta_at_onset,
               CASE WHEN p.t_sepsis_hour IS NULL THEN 0 ELSE 1 END AS sepsis3,
               mc.t_sepsis_hour_mimiccode,
               (SELECT MIN(suspicion_hour) FROM s3_suspicion su
                 WHERE su.stay_id = st.stay_id) AS first_suspicion_hour
        FROM s3_stays st
        LEFT JOIN s3_pairs p USING (stay_id)
        LEFT JOIN s3_pairs_mc mc USING (stay_id)
        ORDER BY st.stay_id
    """)
    return con.execute("SELECT * FROM s3_labels").df()


def run(con, root, limit_stays=None, stay_ids=None, verbose=True):
    """Full pipeline: sources -> suspicion -> hourly SOFA -> labels."""
    def log(msg):
        if verbose:
            print(msg, flush=True)

    n = register_sources(con, root, limit_stays=limit_stays, stay_ids=stay_ids)
    log("  cohort: %d ICU stays (age >= %d, LOS >= %dh)"
        % (n, MIN_AGE, MIN_LOS_HOURS))
    build_suspicion(con)
    n_abx = con.execute("SELECT COUNT(*) FROM s3_abx").fetchone()[0]
    n_sus = con.execute("SELECT COUNT(DISTINCT stay_id) FROM s3_suspicion").fetchone()[0]
    log("  antibiotic administrations in-window: %d" % n_abx)
    log("  stays with a suspicion-of-infection pair: %d" % n_sus)
    build_hourly_sofa(con)
    log("  hourly SOFA built")
    df = build_labels(con)
    log("  septic stays (Challenge rule): %d / %d"
        % (int(df["sepsis3"].sum()), len(df)))
    return df


def load_label_map(path):
    """Read a labels CSV written by scripts/build_sepsis3_labels.py.

    Returns {stay_id (int): t_sepsis_hour (float)} for septic stays only;
    stays absent from the map are controls.
    """
    import csv
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            val = row.get("t_sepsis_hour", "")
            if val not in ("", "NA", "None", "nan"):
                v = float(val)
                if np.isfinite(v):
                    out[int(float(row["stay_id"]))] = v
    return out
