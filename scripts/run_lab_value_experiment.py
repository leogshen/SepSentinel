"""Matched value-versus-availability comparison; reuse completed full-input fits."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np

import run_sepsis_controls as control
from lab_value_ablation import blind_collate, BLIND_COLUMNS
from sepsis_mission_metrics import bootstrap, FIELDS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/sepsis_controls_20261011'
CONFIG = ROOT/'configs/sepsis_lab_values_20261011.json'
control.CONFIG = CONFIG
control.C = json.loads(CONFIG.read_text())
control.OUT = ROOT/'results'/control.C['run_name']
control.LOCAL = control.OUT/'local'
ORIGINAL = {k: getattr(control, k) for k in ['arms', 'setup', 'table', 'tree_fit',
            'neural_fit', 'predict', 'summarize']}
ORIGINAL_COLLATE = control.collate
WITHHOLD_VALUES = False


def arms():
    return [dict(a, base_name=a['name'], name=f'{condition}__{a["name"]}', condition=condition)
            for a in ORIGINAL['arms']() for condition in ['values', 'availability']]


def setup():
    norm = ORIGINAL['setup']()
    source_lock = json.loads((SOURCE/'implementation_lock.json').read_text())
    ours = json.loads((control.OUT/'implementation_lock.json').read_text())
    assert source_lock['data'] == ours['data'], 'Comparator data changed'
    assert all(source_lock['code'][k] == v for k, v in ours['code'].items()
               if k in source_lock['code']), 'Comparator implementation changed'
    extra = dict(source_fit_lock=control.sha(SOURCE/'fit_lock.json'),
                 code={str(p.relative_to(ROOT)): control.sha(p) for p in
                       [Path(__file__), ROOT/'scripts/lab_value_ablation.py']},
                 lab_indices=list(__import__('lab_value_ablation').LAB_INDICES))
    path = control.OUT/'ablation_lock.json'
    if path.exists():
        assert json.loads(path.read_text()) == extra
    else:
        control.dump(path, extra)
    return norm


def collate(*args, **kwargs):
    return (blind_collate if WITHHOLD_VALUES else ORIGINAL_COLLATE)(*args, **kwargs)


def table(split, eps, norm):
    rows = control.base.rows_for(eps)
    saved_rows = np.load(SOURCE/'local'/f'{split}_rows.npy')
    assert np.array_equal(rows, saved_rows), 'Decision alignment changed'
    x = np.load(SOURCE/'local'/f'{split}_features.npy', mmap_mode='r')
    if not WITHHOLD_VALUES:
        return x, rows
    path = control.LOCAL/f'{split}_availability_features.npy'
    marker = path.with_suffix('.json')
    if not marker.exists():
        y = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=x.shape)
        for start in range(0, len(rows), 4096):
            block = np.array(x[start:start+4096], copy=True)
            block[:, BLIND_COLUMNS] = 0
            y[start:start+len(block)] = block
        y.flush()
        control.dump(marker, dict(shape=list(x.shape), numeric_lab_columns_zeroed=len(BLIND_COLUMNS)))
    return np.load(path, mmap_mode='r'), rows


def reuse(a):
    report_path = SOURCE/f'{a["base_name"]}.json'
    source_fit_lock = json.loads((SOURCE/'fit_lock.json').read_text())
    assert control.sha(report_path) == source_fit_lock[report_path.name]
    extension = '.pt' if a['kind'] == 'neural' else '.pkl'
    model_path = SOURCE/'local'/f'{a["base_name"]}{extension}'
    assert control.sha(model_path) == source_fit_lock[str(model_path.relative_to(SOURCE))]
    shutil.copy2(model_path, control.LOCAL/f'{a["name"]}{extension}')
    for split in ['cal', 'test']:
        source = SOURCE/'local'/f'{a["base_name"]}_{split}_scores.npz'
        shutil.copy2(source, control.LOCAL/f'{a["name"]}_{split}_scores.npz')
    report = json.loads(report_path.read_text())
    return dict(report, **a, reused_from=str(report_path.relative_to(ROOT)),
                original_elapsed_minutes=report['elapsed_minutes'])


def fit(a, norm):
    global WITHHOLD_VALUES
    if a['condition'] == 'values':
        return reuse(a)
    WITHHOLD_VALUES = True
    return ORIGINAL['neural_fit' if a['kind'] == 'neural' else 'tree_fit'](a, norm)


def predict(a, split, eps, norm):
    global WITHHOLD_VALUES
    WITHHOLD_VALUES = a['condition'] == 'availability'
    return ORIGINAL['predict'](a, split, eps, norm)


def patient_bootstrap(a, b):
    cols = [FIELDS.index(k) for k in ['flagged_event_records', 'flagged_records', 'event']]
    a, b = a[:, cols], b[:, cols]
    rng = np.random.default_rng(control.C['bootstrap_seed'])
    differences = []
    for _ in range(control.C['bootstrap_replicates']):
        ix = rng.integers(0, len(a), len(a))
        x, y = a[ix].sum(0), b[ix].sum(0)
        if x[1] and y[1] and x[2] and y[2]:
            differences.append([x[0]/x[1]-y[0]/y[1], x[0]/x[2]-y[0]/y[2]])
    if not differences:
        return dict(patient_precision=None, patient_recall=None)
    ci = np.quantile(differences, [.025, .975], axis=0)
    return dict(patient_precision=ci[:, 0].tolist(), patient_recall=ci[:, 1].tolist())


def summarize():
    ORIGINAL['summarize']()
    results = json.loads((control.OUT/'evaluation.json').read_text())
    lookup = {(r['name'], r['gate'], r['target_recall']): r for r in results}
    comparisons = []
    for r in results:
        if r['arm']['condition'] != 'values' or r['test'] is None:
            continue
        name = r['name'].replace('values__', 'availability__', 1)
        other = lookup[(name, r['gate'], r['target_recall'])]
        if other['test'] is None:
            continue
        a = np.load(control.LOCAL/r['statistics_file'])['stats']
        b = np.load(control.LOCAL/other['statistics_file'])['stats']
        assert a.shape == b.shape
        np.testing.assert_array_equal(a[:, [0, 1, 4, 5]], b[:, [0, 1, 4, 5]])
        ci = bootstrap(a, control.C['bootstrap_replicates'], control.C['bootstrap_seed'], b)
        ci.update(patient_bootstrap(a, b))
        comparisons.append(dict(model=r['name'], reference=name, gate=r['gate'],
            target_recall=r['target_recall'], difference='values minus availability-only',
            difference_ci95=ci,
            point_difference={k:r['test'][k]-other['test'][k] for k in ci
                              if r['test'][k] is not None and other['test'][k] is not None}))
    control.dump(control.OUT/'paired_lab_value_comparisons.json', comparisons)


def child(stage, arm=None):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), stage]
    if arm:
        command += ['--arm', arm]
    with (control.OUT/f'{stage}_{arm or "all"}.log').open('a') as f:
        subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


control.arms = arms
control.setup = setup
control.table = table
control.collate = collate
control.base.collate = collate
control.tree_fit = fit
control.neural_fit = fit
control.predict = predict
control.summarize = summarize
control.child = child

if __name__ == '__main__':
    control.main()
