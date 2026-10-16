"""Concurrent, validation-selected 34-channel MAE budget pilot (no test access)."""
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import run_sepsis_controls as control
from transfer_sequence import masked_batch, reconstruction_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/sepsis_long_mae_20261012'
LOCAL = OUT/'local'
C = dict(seed=42, max_steps=20000, min_steps=5000, validation_interval=500,
         patience=8, validation_examples=512, batch_size=32, lr=1e-4,
         memory_fraction=.30, threads=2, initializations=['targetmae','sourcetargetmae'])

def save(path, obj):
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)

def status(**kw):
    obj = dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), **kw)
    control.dump(OUT/'status.json',obj)
    print(json.dumps(obj),flush=True)

def sample(eps, rng, n):
    items=[]
    for _ in range(n):
        e=eps[int(rng.integers(len(eps)))]; end=int(rng.integers(1,len(e['x'])+1))
        first=max(0,end-72)
        items.append(dict(x=e['x'][first:end],t=e['t'][first:end],kind=e['kind']))
    return items

def validate(model, decoder, eps, norm):
    # Reset both prefix and masking RNG on every call, identically across arms.
    rng=np.random.default_rng(20261012)
    model.eval(); decoder.eval(); losses=[]
    with torch.no_grad():
        for _ in range(C['validation_examples']//C['batch_size']):
            b,y,m=masked_batch(sample(eps,rng,C['batch_size']),norm['median'],norm['std'],rng)
            losses.append(float(reconstruction_loss(decoder(model.encoder(*b,active=model.active)),y,m)))
    return float(np.mean(losses))

def fit(init, eps, tune, norm):
    name=f'{init}_34f_seed42'; report=OUT/f'{name}.json'
    if report.exists() and json.loads(report.read_text()).get('complete'): return
    torch.manual_seed(C['seed'])
    model=control.base.RichModel(34).cuda()
    if init=='sourcetargetmae': model.source_init(C['seed'])
    torch.manual_seed(C['seed']+5000)
    decoder=nn.Sequential(nn.Linear(96,96),nn.GELU(),nn.Linear(96,34)).cuda()
    parameters=list(model.encoder.parameters())+list(decoder.parameters())
    opt=torch.optim.AdamW(parameters,lr=C['lr'])
    rng=np.random.default_rng(C['seed']); history=[]; best=float('inf'); stale=0; start=0; best_step=0
    recovery=LOCAL/f'{name}_recovery.pt'
    if recovery.exists():
        state=torch.load(recovery,map_location='cpu',weights_only=False)
        model.encoder.load_state_dict(state['encoder']); decoder.load_state_dict(state['decoder'])
        opt.load_state_dict(state['optimizer']); rng.bit_generator.state=state['numpy_rng']
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        history=state['history']; best=state['best']; stale=state['stale']; start=state['step']; best_step=state['best_step']
    else:
        best=validate(model,decoder,tune,norm)
        history.append(dict(step=0,validation_huber=best))
        save(LOCAL/f'{name}_encoder.pt',model.encoder.state_dict())
    begin=time.time(); losses=[]
    for step in range(start+1,C['max_steps']+1):
        model.train(); decoder.train()
        b,y,m=masked_batch(sample(eps,rng,C['batch_size']),norm['median'],norm['std'],rng)
        loss=reconstruction_loss(decoder(model.encoder(*b,active=model.active)),y,m)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite MAE loss')
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(parameters,1.); opt.step()
        losses.append(float(loss.detach()))
        if step%100==0: status(stage='training',arm=name,step=step,minutes_since_resume=round((time.time()-begin)/60,2))
        if step%C['validation_interval']: continue
        score=validate(model,decoder,tune,norm)
        history.append(dict(step=step,train_huber=float(np.mean(losses)),validation_huber=score)); losses=[]
        if score<best-1e-5:
            best=score; best_step=step; stale=0
            save(LOCAL/f'{name}_encoder.pt',model.encoder.state_dict())
        else: stale+=1
        save(recovery,dict(encoder=model.encoder.state_dict(),decoder=decoder.state_dict(),optimizer=opt.state_dict(),
             numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
             history=history,best=best,best_step=best_step,stale=stale,step=step))
        control.dump(report,dict(complete=False,history=history,best_step=best_step,best_validation_huber=best))
        status(stage='validation',arm=name,step=step,validation_huber=score,best_step=best_step,stale=stale)
        if step>=C['min_steps'] and stale>=C['patience']: break
    control.dump(report,dict(complete=True,history=history,best_step=best_step,best_validation_huber=best,
                            steps=step,hit_cap=step==C['max_steps'],test_opened=False))
    del model,decoder,opt,parameters
    torch.cuda.empty_cache()

def main():
    LOCAL.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(C['threads']); torch.cuda.set_per_process_memory_fraction(C['memory_fraction'])
    lock=dict(config=C,code={p.name:control.sha(p) for p in [Path(__file__),ROOT/'scripts/transfer_sequence.py',ROOT/'scripts/train_rich_transfer.py']},
              data={p.name:control.sha(p) for p in [control.DATA/'fit.pkl',control.DATA/'tune.pkl',control.DATA/'scaler.npz']},
              source_checkpoint=control.sha(control.base.OLD/'mae_seed42.pt'))
    path=OUT/'implementation_lock.json'
    if path.exists(): assert json.loads(path.read_text())==lock,'Frozen implementation changed'
    else: control.dump(path,lock)
    norm=np.load(control.DATA/'scaler.npz')
    with (control.DATA/'fit.pkl').open('rb') as f: eps=pickle.load(f)
    with (control.DATA/'tune.pkl').open('rb') as f: tune=pickle.load(f)
    for init in C['initializations']: fit(init,eps,tune,norm)
    status(stage='complete',arms=2,test_opened=False)

if __name__=='__main__': main()
