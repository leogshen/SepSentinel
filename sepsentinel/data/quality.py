"""Predeclared, label-independent recording QC for the prospective pipeline.

Broad screening bounds are not normal ranges or absolute physiological limits.
Ambiguous spikes/cross-field disagreement are flags, not deletion criteria.
"""
RANGES = {
    'heart_rate': (10, 300), 'respiratory_rate': (0, 100), 'spo2': (0, 100),
    'temperature': (25, 45), 'map': (10, 250), 'sbp': (20, 350),
    'gcs': (3, 15), 'fio2': (21, 100), 'urine_output': (0, 5000),
    'lactate': (0, 40), 'ph': (6.3, 8.2), 'creatinine': (0, 40),
    'wbc': (0, 250), 'platelets': (0, 2500), 'bilirubin': (0, 100),
    'pao2': (10, 800), 'bun': (0, 300), 'glucose': (5, 2000),
}


def sql_range(feature, column='valuenum'):
    lo, hi = RANGES[feature]
    return f'({column} BETWEEN {lo} AND {hi})'
