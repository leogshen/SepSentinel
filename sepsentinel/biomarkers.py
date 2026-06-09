# Biomarker definitions for Lactate, IL-6, and pH.

BIOMARKERS = {
    "lactate": {
        "name": "Lactate",
        "unit": "mmol/L",
        "normal_range": (0.5, 2.0),
        "description": (
            "Metabolic marker for tissue hypoperfusion and sepsis severity. "
            "Normal: 0.5-2.0 mmol/L."
        ),
    },
    "il6": {
        "name": "IL-6 (Interleukin-6)",
        "unit": "pg/mL",
        "normal_range": (0, 7),
        "description": (
            "Inflammatory cytokine indicating immune response and sepsis progression. "
            "Normal: 0-7 pg/mL."
        ),
    },
    "ph": {
        "name": "pH",
        "unit": "pH units",
        "normal_range": (7.35, 7.45),
        "description": (
            "Blood acid-base balance. Low pH indicates metabolic acidosis "
            "from poor tissue perfusion. Normal: 7.35-7.45."
        ),
    },
}


def print_biomarker_info():
    """Print a summary of all biomarkers."""
    print("=" * 60)
    print("  SepSentinel - Biomarker Definitions")
    print("=" * 60)
    for key, bio in BIOMARKERS.items():
        print(f"\n  {bio['name']}")
        print(f"  Unit: {bio['unit']}")
        print(f"  Normal Range: {bio['normal_range'][0]} - {bio['normal_range'][1]} {bio['unit']}")
        print(f"  Info: {bio['description']}")
    print("\n" + "=" * 60)
