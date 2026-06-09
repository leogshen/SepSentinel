# Biomarker definitions for the expanded 7-marker sepsis panel.

BIOMARKERS = {
    "lactate": {
        "name": "Lactate",
        "unit": "mmol/L",
        "normal_range": (0.5, 2.0),
        "description": "Metabolic marker for tissue hypoperfusion and sepsis severity.",
    },
    "il6": {
        "name": "IL-6 (Interleukin-6)",
        "unit": "pg/mL",
        "normal_range": (0, 7),
        "description": "Pro-inflammatory cytokine indicating immune activation.",
    },
    "ph": {
        "name": "pH",
        "unit": "pH units",
        "normal_range": (7.35, 7.45),
        "description": "Blood acid-base balance. Low pH = metabolic acidosis.",
    },
    "presepsin": {
        "name": "Presepsin (sCD14-ST)",
        "unit": "pg/mL",
        "normal_range": (60, 365),
        "description": "Early sepsis marker released by monocytes/macrophages during bacterial phagocytosis.",
    },
    "strem1": {
        "name": "sTREM-1",
        "unit": "pg/mL",
        "normal_range": (0, 150),
        "description": "Soluble triggering receptor on myeloid cells. Elevated in bacterial infections.",
    },
    "il10": {
        "name": "IL-10 (Interleukin-10)",
        "unit": "pg/mL",
        "normal_range": (0, 10),
        "description": "Anti-inflammatory cytokine. High levels indicate immune dysregulation in sepsis.",
    },
    "cxcl10": {
        "name": "CXCL10 (IP-10)",
        "unit": "pg/mL",
        "normal_range": (0, 300),
        "description": "Chemokine elevated in infection. Helps distinguish viral vs bacterial sepsis.",
    },
}


def print_biomarker_info():
    """Print a summary of all biomarkers."""
    print("=" * 60)
    print("  SepSentinel - Expanded Biomarker Panel")
    print("=" * 60)
    for key, bio in BIOMARKERS.items():
        print(f"\n  {bio['name']}")
        print(f"  Unit: {bio['unit']}")
        print(f"  Normal Range: {bio['normal_range'][0]} - {bio['normal_range'][1]} {bio['unit']}")
        print(f"  Info: {bio['description']}")
    print("\n" + "=" * 60)
