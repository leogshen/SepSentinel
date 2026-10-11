"""Explicit stages for rich-input sequence transfer. No default execution stage."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn
import train_rich_transfer as base
from transfer_sequence import masked_batch,reconstruction_loss
from sepsis_mission_metrics import record_statistics,metrics,bootstrap

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'configs/sepsis_missions.json'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def manifest():
    paths=[Path(__file__),ROOT/'scripts/train_rich_transfer.py',ROOT/'scripts/transfer_sequence.py',
           ROOT/'scripts/prepare_transfer_cinc.py',ROOT/'scripts/sepsis_mission_metrics.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}

def arm_specs(c):
    return [dict(seed=s,initialization=i,nfeatures=n,horizon=c['training_horizon_hours'],
                 name=f"{i}_{n}f_{c['training_horizon_hours']}h_seed{s}")
            for s in c['seeds'] for n in c['feature_counts'] for i in c['initializations']]

def setup(c):
    if c['refractory_hours']!=6:raise ValueError('Calibration sweep currently requires six-hour suppression')
    name=c['run_name']
    if not name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in name):
        raise ValueError('run_name must be a simple directory name')
    if not set(c['feature_counts'])<={10,13,34}:raise ValueError('Invalid feature set')
    if not set(c['initializations'])<={'scratch','source','adapt'}:raise ValueError('Invalid initialization')
    out=ROOT/'results'/name;local=out/'local';local.mkdir(parents=True,exist_ok=True)
    base.OUT=out;base.LOCAL=local;base.DEADLINE=float('inf')
    lock={'config':c,'code_sha256':manifest()};path=out/'implementation_lock.json'
    if path.exists() and json.loads(path.read_text())!=lock:
        raise RuntimeError('Code/config changed. Use a new run_name rather than mixing experiments.')
    if not path.exists():base.dump(path,lock)
    return out,local

def validate_data(out,local):
    path=out/'data_sha256.json'
    files=[local/f'{s}.pkl' for s in ['fit','tune','cal','test']]+[local/'scaler.npz']
    hashes={p.name:digest(p) for p in files}
    if path.exists():
        if json.loads(path.read_text())!=hashes:raise RuntimeError('Prepared data changed')
    else:base.dump(path,hashes)

def target_adapt(model,eps,med,std,seed):
    """Respect each arm's feature mask during target MAE as well as fine-tuning."""
    decoder=nn.Sequential(nn.Linear(96,96),nn.GELU(),nn.Linear(96,34)).cuda()
    opt=torch.optim.AdamW(list(model.encoder.parameters())+list(decoder.parameters()),lr=1e-4)
    rng=np.random.default_rng(seed);losses=[];model.train()
    for step in range(1000):
        items=[]
        for _ in range(32):
            e=eps[int(rng.integers(len(eps)))];end=int(rng.integers(1,len(e['x'])+1));first=max(0,end-72)
            raw=e['x'][first:end].copy();raw[:,~model.active]=np.nan
            items.append(dict(x=raw,t=e['t'][first:end],kind=e['kind']))
        b,y,mask=masked_batch(items,med,std,rng)
        pred=decoder(model.encoder(*b,active=model.active));loss=reconstruction_loss(pred,y,mask)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite adaptation loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters())+list(decoder.parameters()),1.)
        opt.step();losses.append(float(loss.detach()))
    return {'steps':1000,'last100_loss':float(np.mean(losses[-100:])),'active_channels':int(model.active.sum())}

def union_rows(eps,c):
    views=[base.view(eps,h,l) for h,l in c['windows']]
    union=[dict(e,eligible=np.logical_or.reduce([v[i]['eligible'] for v in views])) for i,e in enumerate(eps)]
    return base.rows_for(union)

def predictions(arm,split,eps,rows,local,med,std):
    path=local/f"{arm['name']}_{split}_scores.npz"
    if path.exists():
        z=np.load(path)
        if not np.array_equal(z['rows'],rows):raise RuntimeError('Cached row alignment changed')
        return z['p']
    model=base.RichModel(arm['nfeatures']).cuda()
    model.load_state_dict(torch.load(local/f"{arm['name']}.pt",map_location='cpu',weights_only=True))
    p=base.predict(model,eps,rows,med,std)
    np.savez_compressed(path,rows=rows,p=p);del model;torch.cuda.empty_cache();return p

def freeze_fits(arms,out,local):
    files=[]
    for a in arms:
        p=out/f"{a['name']}.json"
        if not p.exists() or not json.loads(p.read_text()).get('complete'):
            raise RuntimeError(f"Incomplete planned arm: {a['name']}")
        files.extend([p,local/f"{a['name']}.pt"])
    hashes={str(p.relative_to(out)):digest(p) for p in files}
    path=out/'fit_lock.json'
    if path.exists() and json.loads(path.read_text())!=hashes:raise RuntimeError('Frozen fits changed')
    if not path.exists():base.dump(path,hashes)

def calibrate(c,arms,out,local,med,std):
    freeze_fits(arms,out,local)
    eps=base.load('cal');rows=union_rows(eps,c);policies=[]
    for a in arms:
        p=predictions(a,'cal',eps,rows,local,med,std)
        grid=np.unique(np.r_[1.000001,np.quantile(p,np.linspace(0,1,c['calibration_quantiles'])),0])[::-1]
        for h,l in c['windows']:
            cohort=base.view(eps,h,l)
            for gate in c['gates']:
                curve=base.evaluate_scores(cohort,rows,p,grid,gate)
                base.dump(out/f"{a['name']}_{h}_{l}_{gate}_calibration_curve.json",curve)
                for target in c['recall_targets']:
                    key='recall' if gate=='all' else 'subgroup_recall'
                    feasible=[v for v in curve if v[key] is not None and v[key]>=target]
                    policies.append(dict(name=a['name'],window=[h,l],gate=gate,target_recall=target,
                                         policy=feasible[0] if feasible else None))
    # Only commit complete policy lists; evaluation refuses partially calibrated work.
    path=out/'policies.json'
    if path.exists() and json.loads(path.read_text())!=policies:raise RuntimeError('Frozen calibration changed')
    if not path.exists():base.dump(path,policies)
    base.dump(out/'policy_lock.json',{'sha256':digest(path)})

def evaluate(c,arms,out,local,med,std):
    freeze_fits(arms,out,local)
    path=out/'policies.json';lock=json.loads((out/'policy_lock.json').read_text())
    if lock['sha256']!=digest(path):raise RuntimeError('Frozen policies changed')
    policies=json.loads(path.read_text())
    expected=len(arms)*len(c['windows'])*len(c['gates'])*len(c['recall_targets'])
    if len(policies)!=expected:raise RuntimeError('Not all planned policies frozen')
    eps=base.load('test');rows=union_rows(eps,c);ids=np.asarray([e['id'] for e in eps]);results=[]
    for a in arms:
        p=predictions(a,'test',eps,rows,local,med,std)
        for entry in [e for e in policies if e['name']==a['name']]:
            r=dict(entry,exploratory=True)
            if entry['policy'] is None:r['test']=None;results.append(r);continue
            h,l=entry['window'];cohort=base.view(eps,h,l)
            stats=record_statistics(cohort,rows,p,entry['policy']['threshold'],entry['gate'])
            token=f"{a['name']}_{h}_{l}_{entry['gate']}_{entry['target_recall']}"
            np.savez_compressed(local/f'{token}_record_stats.npz',ids=ids,stats=stats)
            r['statistics_file']=f'{token}_record_stats.npz';r['test']=metrics(stats)
            r['ci95']=bootstrap(stats,c['bootstrap_replicates'],c['bootstrap_seed'])
            results.append(r)
        base.dump(out/'evaluation_partial.json',results)
    base.dump(out/'evaluation.json',results)

def summarize(c,out,local):
    results=json.loads((out/'evaluation.json').read_text());paired=[]
    lookup={(r['name'],tuple(r['window']),r['gate'],r['target_recall']):r for r in results}
    for r in results:
        if r['test'] is None:continue
        # Within each seed compare transfer to scratch, and richer inputs to ten channels.
        init,feature,horizon,seed=r['name'].split('_');refs=[]
        if init!='scratch':refs.append('_'.join(['scratch',feature,horizon,seed]))
        if feature!='10f':refs.append('_'.join([init,'10f',horizon,seed]))
        for refname in refs:
            ref=lookup.get((refname,tuple(r['window']),r['gate'],r['target_recall']))
            if not ref or ref['test'] is None:continue
            a=np.load(local/r['statistics_file']);b=np.load(local/ref['statistics_file'])
            if not np.array_equal(a['ids'],b['ids']):raise RuntimeError('Paired record identities differ')
            ci=bootstrap(a['stats'],c['bootstrap_replicates'],c['bootstrap_seed'],b['stats'])
            paired.append(dict(model=r['name'],reference=refname,window=r['window'],gate=r['gate'],
                               target_recall=r['target_recall'],difference_ci95=ci))
    base.dump(out/'paired_comparisons.json',paired)
    aggregate=[]
    groups={}
    for r in results:
        if r['test'] is not None:
            k=(r['name'].rsplit('_seed',1)[0],tuple(r['window']),r['gate'],r['target_recall'])
            groups.setdefault(k,[]).append(r['test'])
    for k,values in groups.items():
        summary={}
        for metric in ['precision','recall','subgroup_recall','event_coverage','alarms_per_patient_day']:
            vals=[v[metric] for v in values if v[metric] is not None]
            summary[metric]={'mean':float(np.mean(vals)),'min':min(vals),'max':max(vals)} if vals else None
        aggregate.append(dict(model=k[0],window=k[1],gate=k[2],target_recall=k[3],n_seeds=len(values),metrics=summary))
    base.dump(out/'seed_summary.json',aggregate)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT)
    sub=parser.add_subparsers(dest='stage',required=True)
    for stage in ['plan','prepare','calibrate','evaluate','summarize']:sub.add_parser(stage)
    train=sub.add_parser('train');train.add_argument('--arm',help='Exact arm name; otherwise all planned arms')
    args=parser.parse_args();c=json.loads(args.config.read_text());arms=arm_specs(c)
    if args.stage=='plan':print(json.dumps({'config':c,'arms':arms},indent=2));return
    out,local=setup(c);torch.set_num_threads(4)
    if args.stage=='prepare':base.prepare();validate_data(out,local);return
    validate_data(out,local)
    if args.stage=='summarize':summarize(c,out,local);return
    norm=np.load(local/'scaler.npz');med,std=norm['median'],norm['std']
    if args.stage=='train':
        if (out/'fit_lock.json').exists():raise RuntimeError('Fits already frozen; use a new run_name')
        chosen=[a for a in arms if not args.arm or a['name']==args.arm]
        if not chosen:raise ValueError('Unknown arm')
        base.adapt=target_adapt
        fit,tune=base.load('fit'),base.load('tune')
        for a in chosen:base.fit_arm(fit,tune,med,std,a['seed'],a['initialization'],a['nfeatures'],a['horizon'])
    elif args.stage=='calibrate':calibrate(c,arms,out,local,med,std)
    elif args.stage=='evaluate':evaluate(c,arms,out,local,med,std)

if __name__=='__main__':main()
