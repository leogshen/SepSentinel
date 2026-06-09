# data_loader.py
# ----------------
# This file handles loading real-world datasets for training the ML model.
#
# It auto-detects column names and maps them to our 3 biomarkers:
#   - Lactate (mmol/L)
#   - IL-6 (pg/mL)
#   - pH (pH units)
#   - label (0=healthy, 1=septic)
#
# It handles:
#   - CSV and Excel files
#   - Various column naming conventions (e.g., "Lactate", "lactate_level", "LACTATE")
#   - Missing values (drops rows with NaN in required columns)
#   - Column selection when auto-detection fails (interactive prompt)
#
# Supported public datasets:
#   - MIMIC-IV Style ICU Dataset (Kaggle: sinanshereef)
#   - Sepsis Prediction Dataset (Kaggle: tea340yashjoshi)
#   - Any CSV with lactate, pH, and a sepsis/label column

import os
import pandas as pd


# Known column name patterns for each biomarker.
# The loader tries to match dataset columns against these patterns.
COLUMN_PATTERNS = {
    "lactate": ["lactate", "lact", "lactate_level", "lactate_mean", "lactate_max",
                 "lactate_last", "arterial_lactate", "lactate_mmol"],
    "il6": ["il6", "il-6", "interleukin_6", "interleukin6", "il6_level"],
    "ph": ["ph", "arterial_ph", "ph_level", "blood_ph", "ph_mean", "ph_last",
           "ph_arterial"],
    "label": ["label", "sepsis", "sepsislabel", "sepsis_label", "outcome",
              "target", "is_sepsis", "sepsis_flag", "class", "sepsislabel"],
}


def _find_column(df_columns, patterns):
    """
    Try to find a column in the DataFrame that matches one of the patterns.

    Matching is case-insensitive and ignores underscores/spaces.

    Args:
        df_columns: List of column names from the DataFrame.
        patterns: List of pattern strings to match against.

    Returns:
        The matching column name, or None if no match found.
    """
    # Normalize: lowercase, strip spaces, remove underscores for comparison
    def normalize(name):
        return name.lower().strip().replace(" ", "").replace("_", "")

    normalized_patterns = [normalize(p) for p in patterns]

    for col in df_columns:
        if normalize(col) in normalized_patterns:
            return col

    # Partial match: check if any pattern is contained in the column name
    for col in df_columns:
        col_norm = normalize(col)
        for pattern in normalized_patterns:
            if pattern in col_norm or col_norm in pattern:
                return col

    return None


def detect_columns(df):
    """
    Auto-detect which columns in the DataFrame correspond to our biomarkers.

    Returns:
        A dictionary mapping our standard names to the actual column names found.
        Missing mappings will have None as the value.
    """
    mapping = {}
    for our_name, patterns in COLUMN_PATTERNS.items():
        found = _find_column(df.columns.tolist(), patterns)
        mapping[our_name] = found
    return mapping


def load_real_dataset(filepath, interactive=True):
    """
    Load a real dataset from a CSV or Excel file.

    This function:
        1. Reads the file
        2. Auto-detects column mappings
        3. Prompts the user for any columns it can't find (if interactive)
        4. Renames columns to our standard names
        5. Drops rows with missing values in required columns
        6. Returns a clean DataFrame ready for training

    Args:
        filepath: Path to the CSV or Excel file.
        interactive: If True, prompt user for missing column mappings.

    Returns:
        A tuple of (DataFrame, report_dict) where report_dict contains
        loading statistics and column mapping info.
    """
    # --- Step 1: Read the file ---
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

    # --- Step 2: Auto-detect columns ---
    mapping = detect_columns(df)

    print("  Column mapping (auto-detected):")
    for our_name, col_name in mapping.items():
        status = col_name if col_name else "NOT FOUND"
        print(f"    {our_name:>10} -> {status}")

    # --- Step 3: Prompt for missing columns ---
    if interactive:
        for our_name, col_name in mapping.items():
            if col_name is None:
                print(f"\n  Could not auto-detect column for '{our_name}'.")
                print(f"  Available columns: {list(df.columns)}")
                user_input = input(f"  Enter column name for {our_name} (or 'skip' to exclude): ").strip()
                if user_input.lower() != "skip" and user_input in df.columns:
                    mapping[our_name] = user_input
                elif user_input.lower() != "skip":
                    print(f"  Warning: '{user_input}' not found in columns. Skipping {our_name}.")

    # --- Step 4: Rename and select columns ---
    # Build the rename dict (only for columns that were found)
    rename_map = {}
    available_features = []
    for our_name, col_name in mapping.items():
        if col_name is not None:
            rename_map[col_name] = our_name
            available_features.append(our_name)

    # Check that we have at least label and one biomarker
    if "label" not in available_features:
        raise ValueError(
            "Could not find a 'label'/'sepsis' column in the dataset. "
            "The model needs labeled data (0=healthy, 1=septic) to train."
        )

    biomarkers_found = [f for f in available_features if f != "label"]
    if len(biomarkers_found) == 0:
        raise ValueError("No biomarker columns found. Need at least one of: lactate, il6, ph")

    # Select and rename
    selected_cols = [col for col in mapping.values() if col is not None]
    df_clean = df[selected_cols].rename(columns=rename_map)

    # --- Step 5: Clean the data ---
    # Ensure label is integer (0 or 1)
    df_clean["label"] = df_clean["label"].astype(int)

    # Drop rows with missing values in our columns
    rows_before = len(df_clean)
    df_clean = df_clean.dropna()
    rows_dropped = rows_before - len(df_clean)

    # --- Step 6: Build report ---
    report = {
        "original_rows": original_rows,
        "clean_rows": len(df_clean),
        "rows_dropped": rows_dropped,
        "columns_mapped": mapping,
        "biomarkers_available": biomarkers_found,
        "missing_biomarkers": [b for b in ["lactate", "il6", "ph"] if b not in biomarkers_found],
    }

    print(f"\n  Cleaning complete:")
    print(f"    Rows kept:    {report['clean_rows']} / {report['original_rows']}")
    print(f"    Rows dropped: {report['rows_dropped']} (missing values)")
    print(f"    Biomarkers:   {report['biomarkers_available']}")

    if report["missing_biomarkers"]:
        print(f"    Missing:      {report['missing_biomarkers']} (will use synthetic data for these)")

    # Class balance
    label_counts = df_clean["label"].value_counts()
    print(f"    Healthy (0):  {label_counts.get(0, 0)}")
    print(f"    Septic (1):   {label_counts.get(1, 0)}")

    return df_clean, report
