"""Fold-local lab-rich specialization with equal-budget tuning and longer MAE."""
import argparse
import json
import numbers
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
import run_sepsis_cv as cv

control=cv.control
long=cv.long
mae=cv.mae
ROOT=cv.ROOT
SOURCE=cv.OUT
CONFIG=ROOT/'configs/sepsis_lab_specialize_20261014.json'
C=json.loads(CONFIG.read_text())
OUT=ROOT/'results'/C['run_name']
GATE=C['gates'][0]
RAW_LOAD=control.load
METHODS=['rich_scratch','rich_supervised','rich_mae']

def announce(stage,**kw):
    obj=dict(stage=stage,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    control.dump(OUT/'status.json',obj); print(json.dumps(obj),flush=True)

def restricted(eps):
    # Retain histories and records; sample supervised decisions only when the
    # precomputed causal gate holds at the prediction time.
    return [dict(e,eligible=e['eligible']&e['gates'][GATE]) for e in eps]

def load(split):
    eps=RAW_LOAD(split)
    return restricted(eps) if split in ['fit','tune'] else eps

def candidate_arms():
    return [dict(name=f'{method}_lr{lr:g}',method=method,kind='neural',initialization=method,
                 features=34,seed=42,lr=lr,condition='values') for method in METHODS for lr in C['learning_rates']]

def arms():
    chosen=json.loads((control.OUT/'selection.json').read_text())
    return [chosen[m]['arm'] for m in METHODS]+[
        dict(name='broad_scratch',kind='neural',initialization='scratch',features=34,seed=42,condition='values'),
        dict(name='broad_xgb',kind='xgb',features=34,seed=42,condition='values')]

def copy_exact(src,dst):
    if dst.exists(): assert control.sha(src)==control.sha(dst)
    else: shutil.copy2(src,dst)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    source_lock=json.loads((SOURCE/'implementation_lock.json').read_text())
    assert all(control.sha(ROOT/k)==v for k,v in source_lock['code'].items())
    sources={}
    for fold in range(5):
        folder=SOURCE/f'fold_{fold}'
        for rel in ['fit_lock.json','local/mae/local/sourcetargetmae_34f_seed42_recovery.pt',
                    'local/mae/local/sourcetargetmae_34f_seed42_encoder.pt','local/mae/sourcetargetmae_34f_seed42.json']:
            sources[f'fold_{fold}/{rel}']=control.sha(folder/rel)
    lock=dict(config=C,source_cv_lock=control.sha(SOURCE/'implementation_lock.json'),
              fold_manifest=control.sha(SOURCE/'fold_manifest.json'),source_files=sources,
              code={str(p.relative_to(ROOT)):control.sha(p) for p in [Path(__file__),CONFIG]})
    path=OUT/'implementation_lock.json'
    if path.exists(): assert json.loads(path.read_text())==lock,'Specialization implementation changed'
    else: control.dump(path,lock)

def configure(fold):
    norm=cv.configure(fold)
    control.C=C; control.CONFIG=CONFIG
    control.OUT=OUT/f'fold_{fold}'; control.LOCAL=control.OUT/'local'
    control.LOCAL.mkdir(parents=True,exist_ok=True)
    control.base.OUT=control.OUT; control.base.LOCAL=control.LOCAL
    long.WITHHOLD_VALUES=False
    # control.DATA remains the original fold data; normalization remains fit-only.
    return norm

def pretrain(norm,fold):
    mae.OUT=control.LOCAL/'mae'; mae.LOCAL=mae.OUT/'local'; mae.LOCAL.mkdir(parents=True,exist_ok=True)
    mae.C=dict(mae.C,max_steps=C['mae_max_steps'],min_steps=C['mae_min_steps'],patience=C['mae_patience'])
    old=SOURCE/f'fold_{fold}'/'local/mae/local'
    marker=mae.OUT/'resume_origin.json'
    if not marker.exists():
        for name in ['sourcetargetmae_34f_seed42_recovery.pt','sourcetargetmae_34f_seed42_encoder.pt']:
            copy_exact(old/name,mae.LOCAL/name)
        control.dump(marker,{p.name:control.sha(p) for p in old.glob('*.pt')})
    # Deliberately use the broad fold fit/tune pool, not the restricted loader.
    mae.fit('sourcetargetmae',RAW_LOAD('fit'),RAW_LOAD('tune'),norm)

def init_model(a,fold):
    torch.manual_seed(42); model=control.base.RichModel(34).cuda()
    if a['method']=='rich_supervised':
        path=SOURCE/f'fold_{fold}'/'local/values__scratch_34f_seed42.pt'
        locked=json.loads((SOURCE/f'fold_{fold}'/'fit_lock.json').read_text())
        assert control.sha(path)==locked[str(path.relative_to(SOURCE/f'fold_{fold}'))]
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    elif a['method']=='rich_mae':
        path=control.LOCAL/'mae/local/sourcetargetmae_34f_seed42_encoder.pt'
        report=json.loads((path.parent.parent/'sourcetargetmae_34f_seed42.json').read_text()); assert report['complete']
        state=torch.load(path,map_location='cpu',weights_only=True)
        model.encoder.load_state_dict(state,strict=True)
        assert all(torch.equal(v.cpu(),state[k]) for k,v in model.encoder.state_dict().items())
    return model

def stop(epoch,stale):
    return epoch>=C['epochs'] or (epoch>=C['min_epochs'] and stale>=C['patience'])

def train(a,norm,fold):
    fit,tune=load('fit'),load('tune')
    tr=control.base.rows_for(tune,'sample'); ty=control.label(tune,tr); tw=control.weights(tune,tr)
    assert len(np.unique(ty))==2
    model=init_model(a,fold)
    torch.manual_seed(10042); rng=np.random.default_rng(42)
    opt=torch.optim.AdamW(model.parameters(),lr=a['lr'],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=C['lr_factor'],
        patience=C['lr_patience'],threshold=1e-5,threshold_mode='abs',min_lr=C['min_lr'])
    prevalence=np.mean([e['y'][e['eligible']].mean() for e in fit if e['eligible'].any()])
    posweight=torch.tensor(min(20.,(1-prevalence)/max(prevalence,1e-6)),device='cuda')
    history=[]; stale=0; best_epoch=0; steps=0; elapsed_prior=0.
    path=control.LOCAL/f'{a["name"]}.pt'; recovery=control.LOCAL/f'{a["name"]}_recovery.pt'
    if recovery.exists():
        s=torch.load(recovery,map_location='cpu',weights_only=False)
        model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        torch.set_rng_state(s['torch_rng']); torch.cuda.set_rng_state_all(s['cuda_rng']); rng.bit_generator.state=s['numpy_rng']
        history=s['history']; best=s['best']; best_epoch=s['best_epoch']; stale=s['stale']; steps=s['steps']; elapsed_prior=s['elapsed']
    else:
        p=control.base.predict(model,tune,tr,norm['median'],norm['std'])
        best=float(average_precision_score(ty,p,sample_weight=tw))
        long.save_checkpoint(path,model.state_dict())
    initial_ap=best if not history else history[0]['initial_ap']
    start=time.time()
    if not (history and stop(len(history),stale)):
        for epoch in range(len(history)+1,C['epochs']+1):
            model.train(); rows=control.base.rows_for(fit,'one',rng); rng.shuffle(rows); losses=[]
            for pos in range(0,len(rows),32):
                items,y,static=control.base.batch(fit,rows[pos:pos+32],norm['median'],norm['std'])
                prediction=model(long.ORIGINAL_COLLATE(items,norm['median'],norm['std'],active=model.active),static)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(prediction,y,pos_weight=posweight)
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                losses.append(float(loss.detach())); steps+=1
            p=control.base.predict(model,tune,tr,norm['median'],norm['std'])
            ap=float(average_precision_score(ty,p,sample_weight=tw))
            if ap>best+1e-5: best=ap; best_epoch=epoch; stale=0; long.save_checkpoint(path,model.state_dict())
            else: stale+=1
            scheduler.step(ap)
            history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),tune_ap=ap,
                                initial_ap=initial_ap,lr=opt.param_groups[0]['lr'],steps=steps))
            elapsed=elapsed_prior+time.time()-start
            long.save_checkpoint(recovery,dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=rng.bit_generator.state,
                history=history,best=best,best_epoch=best_epoch,stale=stale,steps=steps,elapsed=elapsed))
            control.status('lab_rich_fit',fold=fold,arm=a['name'],**history[-1],best_epoch=best_epoch,stale=stale)
            if stop(epoch,stale): break
    return dict(**a,complete=True,best_tune_ap=best,selected_epoch=best_epoch,history=history,
                hit_epoch_cap=len(history)==C['epochs'],optimizer_steps=steps,
                fit_records_with_lab_decisions=sum(bool(e['eligible'].any()) for e in fit),
                total_minutes=(elapsed_prior+time.time()-start)/60,test_opened=False)

def select(fold):
    chosen={}
    for method in METHODS:
        candidates=[json.loads((control.OUT/f'{a["name"]}.json').read_text()) for a in candidate_arms() if a['method']==method]
        assert all(r['complete'] for r in candidates)
        best=max(candidates,key=lambda r:r['best_tune_ap'])
        arm=dict(name=method,kind='neural',features=34,seed=42,condition='values',initialization=method,lr=best['lr'])
        chosen[method]=dict(arm=arm,candidate=best['name'],tuning_ap=best['best_tune_ap'],
                            all_candidates=[dict(name=r['name'],tuning_ap=r['best_tune_ap']) for r in candidates])
        copy_exact(control.LOCAL/f'{best["name"]}.pt',control.LOCAL/f'{method}.pt')
        control.dump(control.OUT/f'{method}.json',{**best,**arm,'chosen_candidate':best['name']})
    path=control.OUT/'selection.json'
    if path.exists(): assert json.loads(path.read_text())==chosen
    else: control.dump(path,chosen)
    old=SOURCE/f'fold_{fold}'; locked=json.loads((old/'fit_lock.json').read_text())
    for name,original,kind,ext in [('broad_scratch','values__scratch_34f_seed42','neural','.pt'),('broad_xgb','values__xgb_34f_seed42','xgb','.pkl')]:
        model=old/'local'/f'{original}{ext}'; assert control.sha(model)==locked[str(model.relative_to(old))]
        copy_exact(model,control.LOCAL/f'{name}{ext}')
        r=json.loads((old/f'{original}.json').read_text()); control.dump(control.OUT/f'{name}.json',{**r,'name':name,'condition':'values'})
        for split in ['cal','test']:
            copy_exact(old/'local'/f'{original}_{split}_scores.npz',control.LOCAL/f'{name}_{split}_scores.npz')

def audit_coverage():
    reports=[]
    for fold in range(5):
        configure(fold)
        for split in ['fit','tune','cal']:
            eps=RAW_LOAD(split)
            reports.append(dict(fold=fold,split=split,records=len(eps),eligible_decisions=sum(int(e['eligible'].sum()) for e in eps),
                lab_decisions=sum(int((e['eligible']&e['gates'][GATE]).sum()) for e in eps),
                records_with_lab_decisions=sum(bool((e['eligible']&e['gates'][GATE]).any()) for e in eps),
                events=sum(bool((e['y']*e['eligible']).any()) for e in eps),
                events_with_lab_warning=sum(bool((e['y']*e['eligible']*e['gates'][GATE]).any()) for e in eps)))
    control.dump(OUT/'training_coverage.json',reports)
    print(json.dumps(reports[:3]),flush=True)

def aggregate():
    groups={}; fold_metrics=[]; seen=[]; convergence=[]
    for fold in range(5):
        folder=OUT/f'fold_{fold}'
        ids=json.loads((SOURCE/f'fold_{fold}'/'local/record_ids.json').read_text())['test']; seen.extend(ids)
        for r in json.loads((folder/'evaluation.json').read_text()):
            assert r['test'] is not None,'Infeasible policy; do not omit a fold'
            stat=np.load(folder/'local'/r['statistics_file'])['stats']; assert len(stat)==len(ids)
            groups.setdefault((r['name'],r['target_recall']),[]).append(stat)
            fold_metrics.append(dict(fold=fold,name=r['name'],target_recall=r['target_recall'],metrics=r['test']))
        for a in candidate_arms():
            r=json.loads((folder/f'{a["name"]}.json').read_text())
            convergence.append(dict(fold=fold,name=a['name'],epochs=len(r['history']),selected_epoch=r['selected_epoch'],hit_cap=r['hit_epoch_cap']))
        m=json.loads((folder/'local/mae/sourcetargetmae_34f_seed42.json').read_text())
        convergence.append(dict(fold=fold,name='extended_mae',steps=m['steps'],selected_step=m['best_step'],hit_cap=m['hit_cap']))
    assert len(seen)==len(set(seen))==16684
    pooled={k:np.concatenate(v) for k,v in groups.items()}; summaries=[]; pairs=[]
    for (name,target),stat in pooled.items():
        summaries.append(dict(name=name,target_recall=target,metrics=cv.metrics(stat),ci95=cv.bootstrap(stat,C['bootstrap_replicates'],930)))
    for left,right in [('rich_supervised','rich_scratch'),('rich_mae','rich_scratch'),('rich_supervised','broad_scratch'),('rich_mae','broad_xgb')]:
        for target in C['recall_targets']:
            a,b=pooled[(left,target)],pooled[(right,target)]; np.testing.assert_array_equal(a[:,[0,1,4,5]],b[:,[0,1,4,5]])
            ma,mb=cv.metrics(a),cv.metrics(b)
            pairs.append(dict(model=left,reference=right,target_recall=target,
                differences={k:ma[k]-mb[k] for k in ma if isinstance(ma[k],numbers.Real) and isinstance(mb[k],numbers.Real)},
                ci95=cv.bootstrap(a,C['bootstrap_replicates'],930,b)))
    for name,obj in [('out_of_fold_summary',summaries),('fold_metrics',fold_metrics),('paired_comparisons',pairs),('convergence',convergence)]:
        control.dump(OUT/f'{name}.json',obj)

def child(stage,fold,arm=None):
    command=[sys.executable,'-u',str(Path(__file__).resolve()),stage,'--fold',str(fold)]
    if arm: command+=['--arm',arm]
    folder=OUT/f'fold_{fold}'; folder.mkdir(parents=True,exist_ok=True)
    with (folder/f'{stage}_{arm or "all"}.log').open('a') as f:
        subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def queue():
    for fold in range(5):
        announce('extending_mae',fold=fold); child('mae',fold)
        for a in candidate_arms():
            path=OUT/f'fold_{fold}'/f'{a["name"]}.json'
            if path.exists() and json.loads(path.read_text()).get('complete'): continue
            announce('lab_rich_training',fold=fold,arm=a['name']); child('train',fold,a['name'])
        child('select',fold)
        if not (OUT/f'fold_{fold}'/'policy_lock.json').exists(): announce('calibrating',fold=fold); child('calibrate',fold)
    for fold in range(5):
        if not (OUT/f'fold_{fold}'/'evaluation.json').exists(): announce('evaluating',fold=fold); child('evaluate',fold)
    announce('aggregating'); aggregate(); announce('complete',folds=5,supervised_candidates=45)

control.load=load
control.arms=arms

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','queue','mae','train','select','calibrate','evaluate','aggregate'])
    p.add_argument('--fold',type=int,choices=range(5)); p.add_argument('--arm'); args=p.parse_args()
    prepare()
    if args.stage=='prepare': audit_coverage(); return
    if args.stage=='queue':
        try: queue()
        except Exception as exc: announce('failed',error=repr(exc)); raise
        return
    if args.stage=='aggregate': aggregate(); return
    assert args.fold is not None
    norm=configure(args.fold)
    if args.stage=='mae': pretrain(norm,args.fold)
    elif args.stage=='train':
        assert not (control.OUT/'fit_lock.json').exists()
        a=next(a for a in candidate_arms() if a['name']==args.arm)
        control.dump(control.OUT/f'{a["name"]}.json',train(a,norm,args.fold))
    elif args.stage=='select': select(args.fold)
    elif args.stage=='calibrate': control.calibrate(norm)
    else: control.evaluate(norm)

if __name__=='__main__': main()
