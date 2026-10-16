"""Separate withholding alerts from recalibration, using the original saved scores."""
import json
import pickle
from pathlib import Path
import numpy as np
import train_rich_transfer as base
from sepsis_mission_metrics import record_statistics, metrics, FIELDS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/sepsis_missions_adapt_20261010'


def main():
    records = pickle.load(open(SOURCE/'local/test.pkl', 'rb'))
    eps = base.view(records, 12)
    evaluation = json.loads((SOURCE/'evaluation.json').read_text())
    output = []
    hit = FIELDS.index('matched')
    eligible = FIELDS.index('eligible_event')
    event = FIELDS.index('event')
    for seed in [42, 123, 456]:
        name = f'scratch_34f_12h_seed{seed}'
        entries = {r['gate']: r for r in evaluation if r['name'] == name
                   and r['window'] == [12, 0] and r['target_recall'] == .6}
        z = np.load(SOURCE/'local'/f'{name}_test_scores.npz')
        full_threshold = entries['all']['policy']['threshold']
        gate_threshold = entries['bp_routine_perfusion']['policy']['threshold']
        full = record_statistics(eps, z['rows'], z['p'], full_threshold, 'all')
        restricted = record_statistics(eps, z['rows'], z['p'], full_threshold, 'bp_routine_perfusion')
        recalibrated = record_statistics(eps, z['rows'], z['p'], gate_threshold, 'bp_routine_perfusion')
        covered = restricted[:, eligible] > 0
        output.append(dict(seed=seed, full_threshold=full_threshold, gate_threshold=gate_threshold,
            all_decisions=metrics(full), restricted_same_threshold=metrics(restricted),
            restricted_recalibrated=metrics(recalibrated),
            events_without_qualifying_warning_time=int(((full[:, event]>0)&~covered).sum()),
            original_hits_without_qualifying_warning_time=int(((full[:, hit]>0)&~covered).sum()),
            original_hits_with_qualifying_warning_time=int(((full[:, hit]>0)&covered).sum()),
            hits_lost_by_gating_at_same_threshold=int(((full[:, hit]>0)&(restricted[:, hit]==0)).sum()),
            hits_gained_by_gating_at_same_threshold=int(((full[:, hit]==0)&(restricted[:, hit]>0)).sum())))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
