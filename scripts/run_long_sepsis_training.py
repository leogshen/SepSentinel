"""Longer, matched supervised training with epoch-level recovery and frozen evaluation."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score

import run_sepsis_controls as control
from lab_value_ablation import blind_collate
from sepsis_mission_metrics import bootstrap

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/sepsis_controls_20261011'
LAB_SOURCE = ROOT/'results/sepsis_lab_values_20261011'
CONFIG = ROOT/'configs/sepsis_long_training_20261012.json'
control.CONFIG = CONFIG
control.C = json.loads(CONFIG.read_text())
control.OUT = ROOT/'results'/control.C['run_name']
control.LOCAL = control.OUT/'local'
ORIGINAL = {k:getattr(control, k) for k in ['arms', 'setup', 'predict', 'summarize']}
ORIGINAL_COLLATE = control.collate
WITHHOLD_VALUES = False


def arms():
    planned = ORIGINAL['arms']()
    result = []
    for a in planned:
        result.append(dict(a, condition='values'))
        if a['initialization'] == 'scratch' and a['lr'] == .0003:
            result.append(dict(a, name='availability__'+a['name'], condition='availability'))
    return result


def setup():
    norm = ORIGINAL['setup']()
    source_lock = json.loads((SOURCE/'implementation_lock.json').read_text())
    ours = json.loads((control.OUT/'implementation_lock.json').read_text())
    assert source_lock['data'] == ours['data']
    assert all(source_lock['code'][k] == v for k,v in ours['code'].items() if k in source_lock['code'])
    extra = dict(code={str(p.relative_to(ROOT)):control.sha(p) for p in
                      [Path(__file__), ROOT/'scripts/lab_value_ablation.py']},
                 source_fit_lock=control.sha(SOURCE/'fit_lock.json'),
                 lab_fit_lock=control.sha(LAB_SOURCE/'fit_lock.json'),
                 adaptation_checkpoints={})
    # Reuse the exact MAE initializations; only the supervised budget changes.
    for seed in control.C['seeds']:
        for init in ['targetmae', 'sourcetargetmae']:
            for extension in ['pt', 'json']:
                name = f'{init}_34f_seed{seed}_encoder.{extension}'
                src, dst = SOURCE/'local'/name, control.LOCAL/name
                digest = control.sha(src)
                extra['adaptation_checkpoints'][name] = digest
                if dst.exists():
                    assert control.sha(dst) == digest
                else:
                    shutil.copy2(src, dst)
    path = control.OUT/'long_training_lock.json'
    if path.exists():
        assert json.loads(path.read_text()) == extra
    else:
        control.dump(path, extra)
    return norm


def collate(*args, **kwargs):
    return (blind_collate if WITHHOLD_VALUES else ORIGINAL_COLLATE)(*args, **kwargs)


def stop_training(epoch, stale, c):
    return epoch >= c['epochs'] or (epoch >= c['min_epochs'] and stale >= c['patience'])


def save_checkpoint(path, obj):
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def neural_fit(a, norm):
    global WITHHOLD_VALUES
    WITHHOLD_VALUES = a['condition'] == 'availability'
    fit, tune = control.load('fit'), control.load('tune')
    model, adaptation = control.init_model(a, fit, norm)
    torch.manual_seed(a['seed']+10000)
    rng = np.random.default_rng(a['seed'])
    opt = torch.optim.AdamW(model.parameters(), lr=a['lr'], weight_decay=1e-4)
    prevalence = np.mean([e['y'][e['eligible']].mean() for e in fit if e['eligible'].any()])
    weight = min(20., (1-prevalence)/max(prevalence, 1e-6))
    tr = control.base.rows_for(tune, 'sample')
    ty, tw = control.label(tune, tr), control.weights(tune, tr)
    best, stale, selected_epoch, history = -np.inf, 0, None, []
    step_count, accumulated_seconds = 0, 0.
    recovery_path = control.LOCAL/f'{a["name"]}_recovery.pt'
    best_path = control.LOCAL/f'{a["name"]}.pt'
    if recovery_path.exists():
        saved = torch.load(recovery_path, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        rng.bit_generator.state = saved['sample_rng']
        best, stale, selected_epoch = saved['best'], saved['stale'], saved['selected_epoch']
        history, step_count = saved['history'], saved['step_count']
        accumulated_seconds = saved['elapsed_seconds']
    start = time.time()
    eligible_records = sum(bool(e['eligible'].any()) for e in fit)
    control.status('neural_fit_started', arm=a['name'], gpu=torch.cuda.get_device_name(),
                   device=str(next(model.parameters()).device), starting_epoch=len(history)+1,
                   max_epochs=control.C['epochs'], min_epochs=control.C['min_epochs'],
                   patience=control.C['patience'], examples_per_epoch=eligible_records)
    # A fully saved recovery checkpoint may already satisfy the stop criterion.
    finished = bool(history) and stop_training(len(history), stale, control.C)
    for epoch in range(len(history), control.C['epochs']):
        if finished:
            break
        model.train()
        rows = control.base.rows_for(fit, 'one', rng)
        rng.shuffle(rows)
        losses = []
        for pos in range(0, len(rows), 32):
            items, y, static = control.base.batch(fit, rows[pos:pos+32], norm['median'], norm['std'])
            pred = model(collate(items, norm['median'], norm['std'], active=model.active), static)
            loss = nn.functional.binary_cross_entropy_with_logits(pred, y,
                    pos_weight=torch.tensor(weight, device='cuda'))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            step_count += 1
            losses.append(float(loss.detach()))
        p = control.base.predict(model, tune, tr, norm['median'], norm['std'])
        score = float(average_precision_score(ty, p, sample_weight=tw))
        history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), tune_ap=score,
                            optimizer_steps=step_count))
        if score > best+1e-5:
            best, stale, selected_epoch = score, 0, epoch+1
            save_checkpoint(best_path, model.state_dict())
        else:
            stale += 1
        elapsed = accumulated_seconds+time.time()-start
        save_checkpoint(recovery_path, dict(model=model.state_dict(), optimizer=opt.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            sample_rng=rng.bit_generator.state, history=history, best=best, stale=stale,
            selected_epoch=selected_epoch, step_count=step_count, elapsed_seconds=elapsed))
        control.status('neural_fit', arm=a['name'], **history[-1], best_epoch=selected_epoch,
                       stale_epochs=stale, elapsed_minutes=round(elapsed/60, 2))
        finished = stop_training(epoch+1, stale, control.C)
    return dict(**a, complete=True, history=history, adaptation=adaptation, best_tune_ap=best,
                selected_epoch=selected_epoch, optimizer_steps=step_count,
                examples_per_epoch=eligible_records, hit_epoch_cap=len(history)==control.C['epochs'],
                total_training_minutes=(accumulated_seconds+time.time()-start)/60,
                first20_best_tune_ap=max(h['tune_ap'] for h in history[:20]), test_opened=False)


def predict(a, split, eps, norm):
    global WITHHOLD_VALUES
    WITHHOLD_VALUES = a['condition'] == 'availability'
    return ORIGINAL['predict'](a, split, eps, norm)


def paired(left, right, left_dir, right_dir):
    a = np.load(left_dir/left['statistics_file'])['stats']
    b = np.load(right_dir/right['statistics_file'])['stats']
    assert a.shape == b.shape
    np.testing.assert_array_equal(a[:, [0,1,4,5]], b[:, [0,1,4,5]])
    return bootstrap(a, control.C['bootstrap_replicates'], control.C['bootstrap_seed'], b)


def summarize():
    ORIGINAL['summarize']()
    current = json.loads((control.OUT/'evaluation.json').read_text())
    earlier = json.loads((SOURCE/'evaluation.json').read_text())
    lab = json.loads((LAB_SOURCE/'evaluation.json').read_text())
    old = {(r['name'],r['gate'],r['target_recall']):r for r in earlier+lab}
    lookup = {(r['name'],r['gate'],r['target_recall']):r for r in current}
    comparisons = []
    for r in current:
        if r['test'] is None:
            continue
        ref = old.get((r['name'],r['gate'],r['target_recall']))
        if ref is not None and ref['test'] is not None:
            folder = LAB_SOURCE if r['arm']['condition']=='availability' else SOURCE
            comparisons.append(dict(model=r['name'], reference='short_budget:'+ref['name'],
                gate=r['gate'], target_recall=r['target_recall'],
                difference_ci95=paired(r,ref,control.LOCAL,folder/'local')))
        if r['arm']['condition'] == 'availability':
            full = lookup[(r['name'].removeprefix('availability__'),r['gate'],r['target_recall'])]
            if full['test'] is not None:
                comparisons.append(dict(model=full['name'], reference=r['name'],
                    gate=r['gate'], target_recall=r['target_recall'],
                    difference_ci95=paired(full,r,control.LOCAL,control.LOCAL)))
    control.dump(control.OUT/'budget_and_lab_comparisons.json', comparisons)
    convergence = []
    for a in arms():
        r = json.loads((control.OUT/f'{a["name"]}.json').read_text())
        convergence.append(dict(name=a['name'], epochs=len(r['history']), selected_epoch=r['selected_epoch'],
            hit_epoch_cap=r['hit_epoch_cap'], optimizer_steps=r['optimizer_steps'],
            first20_best_tune_ap=r['first20_best_tune_ap'], final_best_tune_ap=r['best_tune_ap']))
    control.dump(control.OUT/'convergence.json', convergence)


def child(stage, arm=None):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), stage]
    if arm:
        command += ['--arm', arm]
    with (control.OUT/f'{stage}_{arm or "all"}.log').open('a') as f:
        subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


control.arms = arms
control.setup = setup
control.collate = collate
control.base.collate = collate
control.neural_fit = neural_fit
control.predict = predict
control.summarize = summarize
control.child = child

if __name__ == '__main__':
    control.main()
