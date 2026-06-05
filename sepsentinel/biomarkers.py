# biomarkers.py
# ---------------
# This file defines the three biomarkers used in SepSentinel Prototype 2.
# Each biomarker has a name, unit, normal range, and a short clinical explanation.
#
# These definitions act as the "knowledge base" for the system.
# In future versions, this could be loaded from a database or config file.

# Dictionary of biomarker definitions.
# Each key is the biomarker's short name, and the value holds all its metadata.
BIOMARKERS = {
    "lactate": {
        "name": "Lactate",
        "unit": "mmol/L",
        "normal_range": (0.5, 2.0),
        "description": (
            "Lactate is a metabolic marker produced when tissues don't get enough oxygen. "
            "Elevated lactate levels are associated with tissue hypoperfusion and are a key "
            "indicator of sepsis severity. Normal range is 0.5-2.0 mmol/L."
        ),
    },
    "il6": {
        "name": "IL-6 (Interleukin-6)",
        "unit": "pg/mL",
        "normal_range": (0, 7),
        "description": (
            "IL-6 is an inflammatory cytokine released by the immune system in response to "
            "infection. Rapidly rising IL-6 levels indicate a strong immune response and are "
            "associated with sepsis progression. Normal range is 0-7 pg/mL."
        ),
    },
    "ph": {
        "name": "pH",
        "unit": "pH units",
        "normal_range": (7.35, 7.45),
        "description": (
            "pH measures the acid-base balance of the blood. In sepsis, metabolic acidosis "
            "can cause pH to drop below normal, reflecting metabolic dysfunction and poor "
            "tissue perfusion. Normal range is 7.35-7.45."
        ),
    },
}


def print_biomarker_info():
    """Print a summary of all biomarkers to the console."""
    print("=" * 60)
    print("  SepSentinel - Biomarker Definitions")
    print("=" * 60)
    for key, bio in BIOMARKERS.items():
        print(f"\n  {bio['name']}")
        print(f"  Unit: {bio['unit']}")
        print(f"  Normal Range: {bio['normal_range'][0]} - {bio['normal_range'][1]} {bio['unit']}")
        print(f"  Info: {bio['description']}")
    print("\n" + "=" * 60)
