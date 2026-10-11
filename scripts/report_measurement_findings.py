"""Summarize released-record and finite-alarm metrics without training or inference."""
import json
import platform
from pathlib import Path
import importlib.metadata
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sepsis_missions_adapt_20261010'

def main():
    rows = json.loads((OUT / 'evaluation.json').read_text())
    report = {'definition': 'Three-seed means; next-12h window; calibration recall target 0.6',
              'environment': {'python': platform.python_version()}, 'arms': []}
    for package in ['torch', 'numpy', 'pandas', 'scikit-learn', 'duckdb']:
        report['environment'][package] = importlib.metadata.version(package)
    for initialization in ['scratch', 'source', 'adapt']:
        for features in [10, 13, 34]:
            for gate in ['all', 'bp_routine_perfusion']:
                selected = [r for r in rows if r['name'].startswith(f'{initialization}_{features}f_')
                            and r['gate'] == gate and r['window'] == [12, 0]
                            and r['target_recall'] == .6]
                assert len(selected) == 3
                metrics = ['patient_precision', 'patient_recall', 'precision', 'recall',
                           'subgroup_recall', 'event_coverage', 'decision_coverage']
                report['arms'].append(dict(initialization=initialization, features=features, gate=gate,
                    means={k: float(np.mean([r['test'][k] for r in selected])) for k in metrics},
                    patient_precision_range=[min(r['test']['patient_precision'] for r in selected),
                                             max(r['test']['patient_precision'] for r in selected)]))
    example = next(r for r in rows if r['name'] == 'source_34f_12h_seed42'
                   and r['gate'] == 'all' and r['window'] == [12, 0] and r['target_recall'] == .6)
    report['example'] = {k: example[k] for k in ['name', 'policy', 'test', 'ci95']}
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
