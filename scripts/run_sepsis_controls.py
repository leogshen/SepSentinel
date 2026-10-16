"""Sequential tree baselines and matched pretraining controls; separate locked run."""
import argparse
import gc
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import train_rich_transfer as base
import sepsis_missions as missions
from transfer_sequence import collate
from causal_control_features import prefix_features, columns, WINDOWS, STATS
from sepsis_mission_metrics import record_statistics, metrics, bootstrap

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/sepsis_controls_20261011.json'
C = json.loads(CONFIG.read_text())
OUT = ROOT/'results'/C['run_name']
LOCAL = OUT/'local'
DATA = ROOT/'results/sepsis_missions_adapt_20261010/local'


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    base.dump(p, obj)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def status(stage, **kw):
    v = dict(stage=stage, timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    dump(OUT/'status.json', v)
    print(json.dumps(v), flush=True)


def arms():
    out = []
    # CPU trees first, followed by matched neural pairs on the only GPU.
    for seed in C['seeds']:
        for n in C['feature_counts_trees']:
            for kind in ['xgb', 'rf']:
                out.append(dict(name=f'{kind}_{n}f_seed{seed}', kind=kind, features=n, seed=seed))
    for seed in C['seeds']:
        for lr in C['learning_rates']:
            for n in C['feature_counts_neural']:
                for init in C['initializations']:
                    out.append(dict(name=f'{init}_{n}f_lr{lr:g}_seed{seed}', kind='neural',
                                    features=n, seed=seed, lr=lr, initialization=init))
    return out


def setup():
    LOCAL.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(C['threads'])
    base.OUT = OUT
    base.LOCAL = LOCAL
    base.DEADLINE = float('inf')
    paths = [Path(__file__), CONFIG, ROOT/'scripts/causal_control_features.py',
             ROOT/'scripts/sepsis_missions.py', ROOT/'scripts/train_rich_transfer.py',
             ROOT/'scripts/transfer_sequence.py', ROOT/'scripts/sepsis_mission_metrics.py']
    lock = dict(config=C, code={str(p.relative_to(ROOT)): sha(p) for p in paths},
                data={p.name: sha(p) for p in [*(DATA/f'{s}.pkl' for s in ['fit', 'tune', 'cal', 'test']),
                                              DATA/'scaler.npz']},
                source_checkpoints={str(s): sha(base.OLD/f'mae_seed{s}.pt') for s in C['seeds']})
    path = OUT/'implementation_lock.json'
    if path.exists() and json.loads(path.read_text()) != lock:
        raise RuntimeError('Code/config/data changed: use a new run name')
    if not path.exists():
        dump(path, lock)
    return np.load(DATA/'scaler.npz')


def load(split):
    with (DATA/f'{split}.pkl').open('rb') as f:
        return base.view(pickle.load(f), 12)


def weights(eps, rows):
    count = np.bincount(rows[:, 0], minlength=len(eps))
    w = 1/count[rows[:, 0]]
    return w/w.mean()


def label(eps, rows):
    return np.asarray([eps[i]['y'][j] for i, j in rows], np.int64)


def table(split, eps, norm):
    path = LOCAL/f'{split}_features.npy'
    rows = base.rows_for(eps)
    marker = LOCAL/f'{split}_features_complete.json'
    if marker.exists():
        x = np.load(path, mmap_mode='r')
        saved = np.load(LOCAL/f'{split}_rows.npy')
        assert np.array_equal(saved, rows)
        return x, rows
    x = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                  shape=(len(rows), 34*len(WINDOWS)*len(STATS)+5))
    pos = 0
    for i, e in enumerate(eps):
        js = np.flatnonzero(e['eligible'])
        if len(js):
            block = prefix_features(e, norm['median'], norm['std'])
            x[pos:pos+len(js)] = block[js]
            pos += len(js)
        if i % 2000 == 0:
            status('features', split=split, records=i, total=len(eps))
    assert pos == len(rows)
    x.flush()
    np.save(LOCAL/f'{split}_rows.npy', rows)
    dump(marker, dict(rows=len(rows), features=x.shape[1]))
    return x, rows


def tree_fit(a, norm):
    fit, tune = load('fit'), load('tune')
    x, rows = table('fit', fit, norm)
    tx, tr = table('tune', tune, norm)
    cols = columns(a['features'])
    x, tx = np.asarray(x[:, cols], order='C'), np.asarray(tx[:, cols], order='C')
    y, ty = label(fit, rows), label(tune, tr)
    w, tw = weights(fit, rows), weights(tune, tr)
    # Select on the exact tuning decisions and record weights used by neural fits.
    selected_rows = base.rows_for(tune, 'sample')
    selected = np.asarray([tuple(r) for r in tr.tolist()], dtype=np.int64)
    lookup = {tuple(r): k for k, r in enumerate(selected)}
    ix = np.asarray([lookup[tuple(r)] for r in selected_rows])
    select_w = weights(tune, selected_rows)
    prevalence = float(np.average(y, weights=w))
    posweight = min(20., (1-prevalence)/max(prevalence, 1e-6))
    history = []
    best = -np.inf
    for candidate in range(2):
        status('tree_fit', arm=a['name'], candidate=candidate)
        if a['kind'] == 'xgb':
            model = XGBClassifier(n_estimators=500, max_depth=[3, 6][candidate],
                learning_rate=.05, min_child_weight=5, subsample=.8, colsample_bytree=.8,
                reg_lambda=5, tree_method='hist', n_jobs=C['threads'], random_state=a['seed'],
                scale_pos_weight=posweight, eval_metric='aucpr', early_stopping_rounds=40)
            model.fit(x, y, sample_weight=w, eval_set=[(tx[ix], ty[ix])],
                      sample_weight_eval_set=[select_w], verbose=False)
        else:
            model = RandomForestClassifier(n_estimators=300, max_depth=[14, 22][candidate],
                min_samples_leaf=[20, 8][candidate], max_features='sqrt', max_samples=.7,
                n_jobs=C['threads'], random_state=a['seed'], class_weight={0:1., 1:posweight})
            model.fit(x, y, sample_weight=w)
        p = model.predict_proba(tx[ix])[:, 1]
        score = float(average_precision_score(ty[ix], p, sample_weight=select_w))
        history.append(dict(candidate=candidate, tune_ap=score,
                            parameters=model.get_params(), best_iteration=getattr(model, 'best_iteration', None)))
        if score > best:
            best = score
            with (LOCAL/f'{a["name"]}.pkl').open('wb') as f:
                pickle.dump(dict(model=model, columns=cols), f, protocol=5)
        del model
        gc.collect()
    return dict(**a, complete=True, history=history, best_tune_ap=best,
                fit_decisions=len(rows), fit_weighting='equal total weight per released record',
                tuning_decisions=len(ix), test_opened=False)


def init_model(a, fit, norm):
    torch.manual_seed(a['seed'])
    model = base.RichModel(a['features']).cuda()
    init = a['initialization']
    if init in ['source', 'sourcetargetmae']:
        model.source_init(a['seed'])
    adaptation = None
    if init in ['targetmae', 'sourcetargetmae']:
        path = LOCAL/f'{init}_{a["features"]}f_seed{a["seed"]}_encoder.pt'
        metadata = path.with_suffix('.json')
        if path.exists() and metadata.exists():
            model.encoder.load_state_dict(torch.load(path, weights_only=True))
            adaptation = json.loads(metadata.read_text())
        else:
            # Same decoder initialization, masking sampler and dropout RNG across MAE arms.
            torch.manual_seed(a['seed']+5000)
            adaptation = missions.target_adapt(model, fit, norm['median'], norm['std'], a['seed'])
            assert adaptation['steps'] == C['adaptation_steps'] == 1000
            torch.save(model.encoder.state_dict(), path)
            dump(metadata, adaptation)
    return model, adaptation


def neural_fit(a, norm):
    fit, tune = load('fit'), load('tune')
    model, adaptation = init_model(a, fit, norm)
    # MAE/source loading must not change supervised dropout randomness or sampled decisions.
    torch.manual_seed(a['seed']+10000)
    rng = np.random.default_rng(a['seed'])
    opt = torch.optim.AdamW(model.parameters(), lr=a['lr'], weight_decay=1e-4)
    prevalence = np.mean([e['y'][e['eligible']].mean() for e in fit if e['eligible'].any()])
    weight = min(20, (1-prevalence)/max(prevalence, 1e-6))
    tr = base.rows_for(tune, 'sample')
    ty, tw = label(tune, tr), weights(tune, tr)
    best, stale, history = -np.inf, 0, []
    for epoch in range(C['epochs']):
        model.train()
        rows = base.rows_for(fit, 'one', rng)
        rng.shuffle(rows)
        losses = []
        for pos in range(0, len(rows), 32):
            items, y, static = base.batch(fit, rows[pos:pos+32], norm['median'], norm['std'])
            pred = model(collate(items, norm['median'], norm['std'], active=model.active), static)
            loss = nn.functional.binary_cross_entropy_with_logits(pred, y,
                        pos_weight=torch.tensor(weight, device='cuda'))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite supervised loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        p = base.predict(model, tune, tr, norm['median'], norm['std'])
        score = float(average_precision_score(ty, p, sample_weight=tw))
        history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), tune_ap=score))
        status('neural_fit', arm=a['name'], **history[-1])
        if score > best+1e-5:
            best, stale = score, 0
            torch.save(model.state_dict(), LOCAL/f'{a["name"]}.pt')
        else:
            stale += 1
        if stale >= C['patience']:
            break
    return dict(**a, complete=True, history=history, adaptation=adaptation,
                best_tune_ap=best, test_opened=False)


def predict(a, split, eps, norm):
    rows = base.rows_for(eps)
    path = LOCAL/f'{a["name"]}_{split}_scores.npz'
    if path.exists():
        z = np.load(path)
        assert np.array_equal(z['rows'], rows)
        return rows, z['p']
    if a['kind'] == 'neural':
        model = base.RichModel(a['features']).cuda()
        model.load_state_dict(torch.load(LOCAL/f'{a["name"]}.pt', weights_only=True))
        p = base.predict(model, eps, rows, norm['median'], norm['std'])
        del model
        torch.cuda.empty_cache()
    else:
        x, r = table(split, eps, norm)
        assert np.array_equal(r, rows)
        with (LOCAL/f'{a["name"]}.pkl').open('rb') as f:
            obj = pickle.load(f)
        p = obj['model'].predict_proba(np.asarray(x[:, obj['columns']], order='C'))[:, 1]
    assert np.isfinite(p).all()
    np.savez_compressed(path, rows=rows, p=p)
    return rows, p


def freeze_fits():
    frozen = {}
    for a in arms():
        report = OUT/f'{a["name"]}.json'
        assert json.loads(report.read_text())['complete']
        model = LOCAL/f'{a["name"]}{".pt" if a["kind"]=="neural" else ".pkl"}'
        for p in [report, model]:
            frozen[str(p.relative_to(OUT))] = sha(p)
    path = OUT/'fit_lock.json'
    if path.exists():
        assert json.loads(path.read_text()) == frozen
    else:
        dump(path, frozen)


def calibrate(norm):
    freeze_fits()
    eps = load('cal')
    policies = []
    for a in arms():
        status('calibrating', arm=a['name'])
        rows, p = predict(a, 'cal', eps, norm)
        grid = np.unique(np.r_[1.000001, np.quantile(p, np.linspace(0, 1, C['calibration_quantiles'])), 0])[::-1]
        for gate in C['gates']:
            curve = base.evaluate_scores(eps, rows, p, grid, gate)
            dump(OUT/f'{a["name"]}_{gate}_calibration_curve.json', curve)
            for target in C['recall_targets']:
                key = 'recall' if gate == 'all' else 'subgroup_recall'
                feasible = [v for v in curve if v[key] is not None and v[key] >= target]
                policies.append(dict(name=a['name'], gate=gate, target_recall=target,
                                     policy=feasible[0] if feasible else None))
    path = OUT/'policies.json'
    if path.exists():
        assert json.loads(path.read_text()) == policies
    else:
        dump(path, policies)
    dump(OUT/'policy_lock.json', dict(sha256=sha(path)))


def evaluate(norm):
    freeze_fits()
    assert sha(OUT/'policies.json') == json.loads((OUT/'policy_lock.json').read_text())['sha256']
    policies = json.loads((OUT/'policies.json').read_text())
    assert len(policies) == len(arms())*len(C['gates'])*len(C['recall_targets'])
    eps = load('test')
    results = []
    for a in arms():
        status('evaluating_exploratory_test', arm=a['name'])
        rows, p = predict(a, 'test', eps, norm)
        y = label(eps, rows)
        discrimination = dict(auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)))
        for entry in [v for v in policies if v['name'] == a['name']]:
            r = dict(**entry, arm=a, discrimination=discrimination, exploratory=True)
            if entry['policy'] is None:
                r['test'] = None
            else:
                stat = record_statistics(eps, rows, p, entry['policy']['threshold'], entry['gate'])
                token = f'{a["name"]}_{entry["gate"]}_{entry["target_recall"]}'
                np.savez_compressed(LOCAL/f'{token}_stats.npz', stats=stat)
                r.update(test=metrics(stat), ci95=bootstrap(stat, C['bootstrap_replicates'], C['bootstrap_seed']),
                         statistics_file=f'{token}_stats.npz')
            results.append(r)
        dump(OUT/'evaluation_partial.json', results)
    dump(OUT/'evaluation.json', results)


def summarize():
    results = json.loads((OUT/'evaluation.json').read_text())
    groups = {}
    lookup = {(r['name'], r['gate'], r['target_recall']): r for r in results}
    paired = []
    for r in results:
        if r['test'] is None:
            continue
        a = r['arm']
        key = (a['name'].rsplit('_seed', 1)[0], r['gate'], r['target_recall'])
        groups.setdefault(key, []).append(r)
        if a['kind'] == 'neural' and a['initialization'] != 'scratch':
            reference = a['name'].replace(a['initialization']+'_', 'scratch_', 1)
            ref = lookup[(reference, r['gate'], r['target_recall'])]
            if ref['test'] is not None:
                ci = bootstrap(np.load(LOCAL/r['statistics_file'])['stats'], C['bootstrap_replicates'],
                    C['bootstrap_seed'], np.load(LOCAL/ref['statistics_file'])['stats'])
                paired.append(dict(model=a['name'], reference=reference, gate=r['gate'],
                                   target_recall=r['target_recall'], difference_ci95=ci))
    summary = []
    for (model, gate, target), rr in groups.items():
        vals = {}
        for k in ['patient_precision', 'patient_recall', 'precision', 'recall', 'subgroup_recall',
                  'event_coverage', 'alarms_per_patient_day']:
            z = [r['test'][k] for r in rr if r['test'][k] is not None]
            vals[k] = dict(mean=float(np.mean(z)), min=min(z), max=max(z)) if z else None
        summary.append(dict(model=model, gate=gate, target_recall=target, n_seeds=len(rr), metrics=vals))
    dump(OUT/'seed_summary.json', summary)
    dump(OUT/'paired_comparisons.json', paired)


def child(stage, arm=None):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), stage]
    if arm:
        command += ['--arm', arm]
    with (OUT/f'{stage}_{arm or "all"}.log').open('a') as f:
        subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'train', 'calibrate', 'evaluate', 'summarize', 'queue'])
    parser.add_argument('--arm')
    args = parser.parse_args()
    if args.stage == 'plan':
        print(json.dumps(dict(config=C, arms=arms()), indent=2))
        return
    norm = setup()
    if args.stage == 'queue':
        try:
            for a in arms():
                path = OUT/f'{a["name"]}.json'
                if path.exists() and json.loads(path.read_text()).get('complete'):
                    continue
                status('starting_fit', arm=a['name'])
                child('train', a['name'])
            if not (OUT/'policy_lock.json').exists():
                child('calibrate')
            if not (OUT/'evaluation.json').exists():
                child('evaluate')
            child('summarize')
            status('complete', fits=len(arms()), confirmation='not yet available; see CONFIRMATION_AUDIT.md')
        except Exception as exc:
            status('failed', error=repr(exc))
            raise
    elif args.stage == 'train':
        if (OUT/'fit_lock.json').exists():
            raise RuntimeError('Fits are frozen')
        a = next(a for a in arms() if a['name'] == args.arm)
        start = time.time()
        report = neural_fit(a, norm) if a['kind'] == 'neural' else tree_fit(a, norm)
        report['elapsed_minutes'] = (time.time()-start)/60
        dump(OUT/f'{a["name"]}.json', report)
    elif args.stage == 'calibrate':
        calibrate(norm)
    elif args.stage == 'evaluate':
        evaluate(norm)
    else:
        summarize()


if __name__ == '__main__':
    main()
