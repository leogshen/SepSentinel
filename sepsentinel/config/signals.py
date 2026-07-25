# Signal definitions for the 7 input features.
# Two sensing modalities:
#   - Electrochemical biomarkers: measured from interstitial fluid (no extraction).
#     Final hardware implementation varies per analyte. The software pipeline
#     treats all three as processed concentration/value inputs.
#   - Physiological: on-skin optical/impedance sensors.
#
# IL-6 specifically uses electrochemical aptamer-based (E-AB) sensing with a
# three-electrode system in direct ISF contact. Lactate and pH hardware
# implementations remain modular — likely electrochemical but final designs
# may differ.

ELECTROCHEMICAL_SIGNALS = {
    "ph": {
        "name": "pH",
        "unit": "pH units",
        "normal_range": (7.35, 7.45),
        "sensing_mode": "potentiometric",
        "description": (
            "Acid-base balance. Likely potentiometric measurement. "
            "Final hardware implementation TBD."
        ),
    },
    "lactate": {
        "name": "Lactate",
        "unit": "mmol/L",
        "normal_range": (0.5, 2.0),
        "sensing_mode": "amperometric",
        "description": (
            "Tissue perfusion marker. Likely amperometric (enzyme-based). "
            "Final hardware implementation TBD."
        ),
    },
    "il6": {
        "name": "IL-6",
        "unit": "pg/mL",
        "normal_range": (0, 7),
        "sensing_mode": "E-AB (SWV)",
        "description": (
            "Inflammatory cytokine. Electrochemical aptamer-based (E-AB) sensor "
            "using square-wave voltammetry via a three-electrode system in direct "
            "contact with ISF."
        ),
    },
}

PHYSIOLOGICAL_SIGNALS = {
    "heart_rate": {
        "name": "Heart Rate",
        "unit": "bpm",
        "normal_range": (60, 100),
        "description": "Cardiac rhythm from optical PPG sensor.",
    },
    "respiratory_rate": {
        "name": "Respiratory Rate",
        "unit": "breaths/min",
        "normal_range": (12, 20),
        "description": "Breathing rate from impedance or accelerometer.",
    },
    "temperature": {
        "name": "Temperature",
        "unit": "\u00b0C",
        "normal_range": (36.1, 37.2),
        "description": "Skin/core temperature from thermistor.",
    },
    "spo2": {
        "name": "SpO2",
        "unit": "%",
        "normal_range": (95, 100),
        "description": "Blood oxygen saturation from pulse oximetry.",
    },
}

ALL_SIGNALS = {**PHYSIOLOGICAL_SIGNALS, **ELECTROCHEMICAL_SIGNALS}

# Canonical feature order for model input tensors.
# Physiological signals first, then electrochemical biomarkers.
# This ordering supports future dual-branch splitting:
#   PHYSIOLOGICAL_FEATURES = FEATURE_ORDER[:4]
#   BIOMARKER_FEATURES     = FEATURE_ORDER[4:]
FEATURE_ORDER = [
    "heart_rate", "respiratory_rate", "temperature", "spo2",
    "ph", "lactate", "il6",
]
PHYSIOLOGICAL_FEATURES = FEATURE_ORDER[:4]
BIOMARKER_FEATURES = FEATURE_ORDER[4:]
NUM_FEATURES = len(FEATURE_ORDER)

# Staged feature sets for Model B development.
# Stage 1: Physiological signals only (available in PhysioNet Challenge data)
# Stage 2: + Lactate, pH (sparse in Challenge data, full in MIMIC-IV)
# Stage 3: + IL-6 (requires Model A or future dataset)
STAGES = {
    1: ["heart_rate", "spo2", "temperature", "respiratory_rate"],
    2: ["heart_rate", "spo2", "temperature", "respiratory_rate", "lactate", "ph"],
    3: FEATURE_ORDER,  # all 7
}

# Column mapping: PhysioNet/CinC 2019 Sepsis Challenge → our signal names
PHYSIONET_COLUMN_MAP = {
    "HR": "heart_rate",
    "O2Sat": "spo2",
    "Temp": "temperature",
    "Resp": "respiratory_rate",
    "pH": "ph",
    "Lactate": "lactate",
    "SepsisLabel": "label",
    "Patient_ID": "patient_id",
    "ICULOS": "hour",
}

# Default pipeline settings
DEFAULT_SAMPLING_INTERVAL_MIN = 5
DEFAULT_HISTORY_WINDOW_MIN = 60


def print_signal_info():
    """Print a summary of all monitored signals."""
    print("=" * 60)
    print("  SepSentinel - Monitored Signals")
    print("=" * 60)

    print("\n  Physiological Sensors:")
    for key in PHYSIOLOGICAL_FEATURES:
        sig = ALL_SIGNALS[key]
        lo, hi = sig["normal_range"]
        print(f"    {sig['name']:20s}  {lo}-{hi} {sig['unit']}")

    print("\n  Electrochemical Biomarkers (in-situ ISF sensing):")
    for key in BIOMARKER_FEATURES:
        sig = ALL_SIGNALS[key]
        lo, hi = sig["normal_range"]
        mode = sig.get("sensing_mode", "")
        print(f"    {sig['name']:20s}  {lo}-{hi} {sig['unit']:12s}  [{mode}]")

    print("\n" + "=" * 60)
