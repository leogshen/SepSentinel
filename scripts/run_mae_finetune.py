"""Matched one-seed fine-tuning of validation-selected longer target MAE."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
import torch
import run_long_sepsis_training as long

control=long.control
ROOT=long.ROOT
PRE=ROOT/'results/sepsis_long_mae_20261012'
REFERENCE=ROOT/'results/sepsis_long_training_20261012'
control.CONFIG=ROOT/'configs/sepsis_mae_finetune_20261013.json'
control.C=json.loads(control.CONFIG.read_text())
control.OUT=ROOT/'results'/control.C['run_name']
control.LOCAL=control.OUT/'local'

def arms():
    return [dict(a,condition='values') for a in long.ORIGINAL['arms']()]

def copy_verified(src,dst):
    if dst.exists(): assert control.sha(src)==control.sha(dst),str(dst)
    else: shutil.copy2(src,dst)

def setup():
    norm=long.ORIGINAL['setup']()
    previous=json.loads((REFERENCE/'implementation_lock.json').read_text())
    ours=json.loads((control.OUT/'implementation_lock.json').read_text())
    assert previous['data']==ours['data']
    assert all(previous['code'][k]==v for k,v in ours['code'].items() if k in previous['code'])
    checkpoints={}
    for init in ['targetmae','sourcetargetmae']:
        name=f'{init}_34f_seed42'
        report=json.loads((PRE/f'{name}.json').read_text())
        assert report['complete'] and report['best_step']==20000
        path=PRE/'local'/f'{name}_encoder.pt'
        checkpoints[name]=dict(sha256=control.sha(path),report_sha256=control.sha(PRE/f'{name}.json'))
    lock=dict(checkpoints=checkpoints,code={p.name:control.sha(p) for p in [Path(__file__),Path(long.__file__)]},
              reference_fit_lock=control.sha(REFERENCE/'fit_lock.json'),mae_lock=control.sha(PRE/'implementation_lock.json'))
    path=control.OUT/'finetune_lock.json'
    if path.exists(): assert json.loads(path.read_text())==lock
    else: control.dump(path,lock)
    # Exact existing scratch controls, avoiding unnecessary retraining or new RNG draws.
    for a in arms():
        if a['initialization']=='scratch':
            copy_verified(REFERENCE/f'{a["name"]}.json',control.OUT/f'{a["name"]}.json')
            copy_verified(REFERENCE/'local'/f'{a["name"]}.pt',control.LOCAL/f'{a["name"]}.pt')
    return norm

def init_model(a,fit,norm):
    torch.manual_seed(a['seed'])
    model=control.base.RichModel(a['features']).cuda()
    if a['initialization']=='scratch': return model,None
    path=PRE/'local'/f'{a["initialization"]}_34f_seed42_encoder.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    model.encoder.load_state_dict(state,strict=True)
    for key,value in model.encoder.state_dict().items():
        assert torch.equal(value.cpu(),state[key]),key
    return model,dict(steps=20000,encoder_sha256=control.sha(path),selected_on='fixed-mask tuning reconstruction')

def summarize():
    long.ORIGINAL['summarize']()
    previous=json.loads((REFERENCE/'evaluation.json').read_text())
    lookup={(r['name'],r['gate'],r['target_recall']):r for r in previous}
    comparisons=[]
    for r in json.loads((control.OUT/'evaluation.json').read_text()):
        if r['arm']['initialization']=='scratch' or r['test'] is None: continue
        ref=lookup[(r['name'],r['gate'],r['target_recall'])]
        if ref['test'] is not None:
            comparisons.append(dict(model=r['name'],reference='1000_update_MAE:'+ref['name'],
                gate=r['gate'],target_recall=r['target_recall'],
                difference_ci95=long.paired(r,ref,control.LOCAL,REFERENCE/'local')))
    control.dump(control.OUT/'mae_budget_comparisons.json',comparisons)

def child(stage,arm=None):
    command=[sys.executable,'-u',str(Path(__file__).resolve()),stage]
    if arm: command+=['--arm',arm]
    with (control.OUT/f'{stage}_{arm or "all"}.log').open('a') as f:
        subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

control.arms=arms
control.setup=setup
control.init_model=init_model
control.summarize=summarize
control.child=child

if __name__=='__main__': control.main()
