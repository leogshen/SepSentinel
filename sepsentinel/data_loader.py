# Loads real-world datasets with auto-column detection (7-marker panel).

import os
import pandas as pd

COLUMN_PATTERNS = {
    "lactate": ["lactate", "lact", "lactate_level", "lactate_mean", "lactate_max",
                 "lactate_last", "arterial_lactate", "lactate_mmol"],
    "il6": ["il6", "il-6", "interleukin_6", "interleukin6", "il6_level"],
    "ph": ["ph", "arterial_ph", "ph_level", "blood_ph", "ph_mean", "ph_last",
           "ph_arterial"],
    "presepsin": ["presepsin", "scd14", "scd14_st", "presepsin_level"],
    "strem1": ["strem1", "strem-1", "trem1", "trem-1", "strem1_level"],
    "il10": ["il10", "il-10", "interleukin_10", "interleukin10", "il10_level"],
    "cxcl10": ["cxcl10", "ip10", "ip-10", "cxcl10_level"],
    "label": ["label", "sepsis", "sepsislabel", "sepsis_label", "outcome",
              "target", "is_sepsis", "sepsis_flag", "class"],
}


def _find_column(df_columns, patterns):
    """Find a column matching one of the patterns (case-insensitive)."""
    def normalize(name):
        return name.lower().strip().replace(" ", "").replace("_", "")

    normalized_patterns = [normalize(p) for p in patterns]

    for col in df_columns:
        if normalize(col) in normalized_patterns:
            return col

    for col in df_columns:
        col_norm = normalize(col)
        for pattern in normalized_patterns:
            if pattern in col_norm or col_norm in pattern:
                return col

    return None


def detect_columns(df):
    """Auto-detect which columns map to our biomarkers."""
    mapping = {}
    for our_name, patterns in COLUMN_PATTERNS.items():
        mapping[our_name] = _find_column(df.columns.tolist(), patterns)
    return mapping


def load_real_dataset(filepath, interactive=True):
    """Load a CSV/Excel dataset, auto-map columns, clean, and return."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    elif ext == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use CSV or Excel.")

    original_rows = len(df)
    print(f"\n  Loaded {original_rows} rows from {os.path.basename(filepath)}")
    print(f"  Columns found: {list(df.columns)}\n")

    mapping = detect_columns(df)

    print("  Column mapping (auto-detected):")
    for our_name, col_name in mapping.items():
        print(f"    {our_name:>10} -> {col_name or 'NOT FOUND'}")

    if interactive:
        for our_name, col_name in mapping.items():
            if col_name is None:
                print(f"\n  Could not auto-detect '{our_name}'. Available: {list(df.columns)}")
                user_input = input(f"  Enter column name for {our_name} (or 'skip'): ").strip()
                if user_input.lower() != "skip" and user_input in df.columns:
                    mapping[our_name] = user_input
                elif user_input.lower() != "skip":
                    print(f"  '{user_input}' not found. Skipping {our_name}.")

    rename_map = {}
    available_features = []
    for our_name, col_name in mapping.items():
        if col_name is not None:
            rename_map[col_name] = our_name
            available_features.append(our_name)

    if "label" not in available_features:
        raise ValueError("No label/sepsis column found.")

    all_biomarkers = ["lactate", "il6", "ph", "presepsin", "strem1", "il10", "cxcl10"]
    biomarkers_found = [f for f in available_features if f != "label"]
    if not biomarkers_found:
        raise ValueError("No biomarker columns found.")

    selected_cols = [col for col in mapping.values() if col is not None]
    df_clean = df[selected_cols].rename(columns=rename_map)
    df_clean["label"] = df_clean["label"].astype(int)

    rows_before = len(df_clean)
    df_clean = df_clean.dropna()
    rows_dropped = rows_before - len(df_clean)

    report = {
        "original_rows": original_rows,
        "clean_rows": len(df_clean),
        "rows_dropped": rows_dropped,
        "columns_mapped": mapping,
        "biomarkers_available": biomarkers_found,
        "missing_biomarkers": [b for b in all_biomarkers if b not in biomarkers_found],
    }

    print(f"\n  Cleaning complete:")
    print(f"    Rows kept:    {report['clean_rows']} / {original_rows}")
    print(f"    Rows dropped: {rows_dropped} (missing values)")
    print(f"    Biomarkers:   {biomarkers_found}")
    if report["missing_biomarkers"]:
        print(f"    Missing:      {report['missing_biomarkers']}")

    label_counts = df_clean["label"].value_counts()
    print(f"    Healthy (0):  {label_counts.get(0, 0)}")
    print(f"    Septic (1):   {label_counts.get(1, 0)}")

    return df_clean, report
