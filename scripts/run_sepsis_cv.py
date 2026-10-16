"""Five-fold development-record CV with fold-local normalization and target MAE."""
import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import run_long_sepsis_training as long
import run_long_target_mae as mae
from lab_value_ablation import BLIND_COLUMNS
from sepsis_mission_metrics import metrics, bootstrap

control=long.control
ROOT=long.ROOT
CONFIG=ROOT/'configs/sepsis_cv_20261013.json'
C=json.loads(CONFIG.read_text())
OUT=ROOT/'results'/C['run_name']
INPUT=ROOT/'results/sepsis_missions_adapt_20261010/local'
RAW_TABLE=control.table
RAW_TREE=control.tree_fit
CURRENT_FOLD=None

def announce(stage,**kwargs):
    obj=dict(stage=stage,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs)
    control.dump(OUT/'status.json',obj); print(json.dumps(obj),flush=True)

def partitions(labels,seed=C['split_seed']):
    labels=np.asarray(labels)
    for fold,(rest,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(labels,labels)):
        ft,cal=train_test_split(rest,test_size=.15,stratify=labels[rest],random_state=seed+fold*2)
        fit,tune=train_test_split(ft,test_size=.15/.85,stratify=labels[ft],random_state=seed+fold*2+1)
        yield {k:np.sort(v) for k,v in dict(fit=fit,tune=tune,cal=cal,test=test).items()}

def normalization(eps):
    z=np.concatenate([e['x'][::max(1,len(e['x'])//30)] for e in eps])
    med=np.nanmedian(z,axis=0); std=np.nanstd(z,axis=0)
    med=np.nan_to_num(med); std=np.where(np.isfinite(std)&(std>1e-6),std,1.)
    source=np.load(control.base.OLD/'source_scaler.npz')
    med[:10]=source['median']; std[:10]=source['std']
    return dict(median=med.astype(np.float32),std=std.astype(np.float32))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    code=[Path(__file__),CONFIG,Path(long.__file__),Path(mae.__file__),ROOT/'scripts/run_sepsis_controls.py',
          ROOT/'scripts/train_rich_transfer.py',ROOT/'scripts/transfer_sequence.py',
          ROOT/'scripts/sepsis_mission_metrics.py',ROOT/'scripts/lab_value_ablation.py',ROOT/'scripts/causal_control_features.py']
    lock=dict(config=C,code={str(p.relative_to(ROOT)):control.sha(p) for p in code},
              data={s:control.sha(INPUT/f'{s}.pkl') for s in ['fit','tune','cal','test']},
              source_checkpoint=control.sha(control.base.OLD/'mae_seed42.pt'),
              source_scaler=control.sha(control.base.OLD/'source_scaler.npz'))
    path=OUT/'implementation_lock.json'
    if path.exists(): assert json.loads(path.read_text())==lock,'CV inputs/code changed'
    else: control.dump(path,lock)
    manifest=OUT/'fold_manifest.json'
    if manifest.exists(): return
    records=[]
    for split in ['fit','tune','cal']:
        with (INPUT/f'{split}.pkl').open('rb') as f: records.extend(pickle.load(f))
    ids=[e['id'] for e in records]; assert len(set(ids))==len(ids)
    # Public CinC identifiers are released-record IDs, not verified cross-stay persons.
    assert all(e['person']==e['id'] for e in records)
    with (INPUT/'test.pkl').open('rb') as f: excluded_ids={e['id'] for e in pickle.load(f)}
    assert not set(ids)&excluded_ids
    view=control.base.view(records,12)
    labels=np.asarray([int(bool((e['y']*e['eligible']).any())) for e in view])
    summary=[]; seen=[]
    for fold,split in enumerate(partitions(labels)):
        folder=OUT/f'fold_{fold}'/'local'; folder.mkdir(parents=True,exist_ok=True)
        groups=[set(ix.tolist()) for ix in split.values()]
        assert set.union(*groups)==set(range(len(records)))
        assert all(not groups[i]&groups[j] for i in range(4) for j in range(i))
        for name,ix in split.items():
            with (folder/f'{name}.pkl').open('wb') as f: pickle.dump([records[i] for i in ix],f,protocol=5)
        np.savez(folder/'scaler.npz',**normalization([records[i] for i in split['fit']]))
        control.dump(folder/'record_ids.json',{k:[ids[i] for i in ix] for k,ix in split.items()})
        summary.append(dict(fold=fold,splits={k:dict(records=len(ix),events=int(labels[ix].sum())) for k,ix in split.items()},
            hashes={p.name:control.sha(p) for p in [*(folder/f'{k}.pkl' for k in split),folder/'scaler.npz',folder/'record_ids.json']}))
        seen.extend(split['test'].tolist())
    assert sorted(seen)==list(range(len(records)))
    control.dump(manifest,dict(records=len(records),events=int(labels.sum()),excluded_original_test_records=len(excluded_ids),
                               split_unit='released record; person identity beyond this unavailable',folds=summary))
    announce('prepared',records=len(records),events=int(labels.sum()),folds=5)

def arms():
    result=[]
    for kind,init in [('xgb',None),('neural','scratch'),('neural','sourcetargetmae')]:
        for condition in (['values'] if init=='sourcetargetmae' else ['values','availability']):
            label=init if kind=='neural' else kind
            result.append(dict(name=f'{condition}__{label}_34f_seed42',kind=kind,initialization=init,
                               features=34,seed=42,lr=.0001,condition=condition))
    return result

def configure(fold):
    global CURRENT_FOLD
    CURRENT_FOLD=fold
    control.C=C; control.CONFIG=CONFIG; control.OUT=OUT/f'fold_{fold}'
    control.LOCAL=control.OUT/'local'; control.DATA=control.LOCAL
    control.base.OUT=control.OUT; control.base.LOCAL=control.LOCAL
    torch.set_num_threads(C['threads'])
    entry=json.loads((OUT/'fold_manifest.json').read_text())['folds'][fold]
    assert all(control.sha(control.LOCAL/k)==v for k,v in entry['hashes'].items())
    return np.load(control.LOCAL/'scaler.npz')

def table(split,eps,norm):
    x,rows=RAW_TABLE(split,eps,norm)
    if not long.WITHHOLD_VALUES: return x,rows
    path=control.LOCAL/f'{split}_blind_features.npy'; marker=path.with_suffix('.json')
    if not marker.exists():
        y=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=x.shape)
        for start in range(0,len(x),4096):
            block=np.array(x[start:start+4096],copy=True); block[:,BLIND_COLUMNS]=0
            y[start:start+len(block)]=block
        y.flush(); control.dump(marker,dict(shape=list(x.shape)))
    return np.load(path,mmap_mode='r'),rows

def init_model(a,fit,norm):
    torch.manual_seed(a['seed']); model=control.base.RichModel(34).cuda()
    if a['initialization']=='scratch': return model,None
    path=control.LOCAL/'mae/local/sourcetargetmae_34f_seed42_encoder.pt'
    report=json.loads((path.parent.parent/'sourcetargetmae_34f_seed42.json').read_text())
    assert report['complete']
    state=torch.load(path,map_location='cpu',weights_only=True)
    model.encoder.load_state_dict(state,strict=True)
    assert all(torch.equal(v.cpu(),state[k]) for k,v in model.encoder.state_dict().items())
    return model,dict(steps=report['steps'],selected_step=report['best_step'],encoder_sha256=control.sha(path))

def pretrain(norm):
    mae.OUT=control.LOCAL/'mae'; mae.LOCAL=mae.OUT/'local'; mae.LOCAL.mkdir(parents=True,exist_ok=True)
    # Existing MAE code samples only these fold-local fit and tuning records.
    mae.C=dict(mae.C,threads=C['threads'])
    fit=control.load('fit'); tune=control.load('tune')
    mae.fit('sourcetargetmae',fit,tune,norm)

def fit(a,norm):
    long.WITHHOLD_VALUES=a['condition']=='availability'
    return long.neural_fit(a,norm) if a['kind']=='neural' else RAW_TREE(a,norm)

def fold_summary():
    rows=json.loads((control.OUT/'evaluation.json').read_text())
    control.dump(control.OUT/'metric_summary.json',[{k:r[k] for k in ['name','gate','target_recall','test','ci95'] if k in r} for r in rows])

def aggregate():
    grouped={}; fold_metrics=[]; ids=[]
    for fold in range(5):
        folder=OUT/f'fold_{fold}'; record_ids=json.loads((folder/'local/record_ids.json').read_text())['test']
        ids.extend(record_ids)
        for r in json.loads((folder/'evaluation.json').read_text()):
            assert r['test'] is not None,'Infeasible calibration target: do not silently drop fold'
            key=(r['name'],r['gate'],r['target_recall'])
            stat=np.load(folder/'local'/r['statistics_file'])['stats']
            assert len(stat)==len(record_ids)
            grouped.setdefault(key,[]).append(stat)
            fold_metrics.append(dict(fold=fold,name=key[0],gate=key[1],target_recall=key[2],metrics=r['test']))
    assert len(ids)==len(set(ids))==json.loads((OUT/'fold_manifest.json').read_text())['records']
    pooled={k:np.concatenate(v) for k,v in grouped.items()}
    summaries=[]; comparisons=[]
    for key,stat in pooled.items():
        summaries.append(dict(name=key[0],gate=key[1],target_recall=key[2],out_of_fold_metrics=metrics(stat),
                              conditional_record_bootstrap_ci95=bootstrap(stat,C['bootstrap_replicates'],930)))
    pairs=[('values__xgb_34f_seed42','availability__xgb_34f_seed42'),
           ('values__scratch_34f_seed42','availability__scratch_34f_seed42'),
           ('values__sourcetargetmae_34f_seed42','values__scratch_34f_seed42'),
           ('values__scratch_34f_seed42','values__xgb_34f_seed42')]
    for left,right in pairs:
        for gate in C['gates']:
            for target in C['recall_targets']:
                a,b=pooled[(left,gate,target)],pooled[(right,gate,target)]
                np.testing.assert_array_equal(a[:,[0,1,4,5]],b[:,[0,1,4,5]])
                ma,mb=metrics(a),metrics(b)
                differences={k:ma[k]-mb[k] for k in ma if ma[k] is not None and mb[k] is not None}
                comparisons.append(dict(model=left,reference=right,gate=gate,target_recall=target,
                    point_difference=differences,conditional_record_bootstrap_ci95=bootstrap(a,C['bootstrap_replicates'],930,b)))
    control.dump(OUT/'out_of_fold_summary.json',summaries)
    control.dump(OUT/'fold_metrics.json',fold_metrics)
    control.dump(OUT/'paired_comparisons.json',comparisons)

def child(stage,fold,arm=None):
    command=[sys.executable,'-u',str(Path(__file__).resolve()),stage,'--fold',str(fold)]
    if arm: command+=['--arm',arm]
    with (OUT/f'fold_{fold}'/f'{stage}_{arm or "all"}.log').open('a') as f:
        subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def queue():
    # Training is serialized with the previously requested fine-tuning queue.
    wait_path=ROOT/'results/sepsis_mae_finetune_20261013/status.json'
    announce('waiting_for_finetune')
    while True:
        state=json.loads(wait_path.read_text())
        if state['stage']=='complete': break
        if state['stage']=='failed': raise RuntimeError('Prior fine-tuning failed; inspect before sharing GPU')
        time.sleep(30)
    for fold in range(5):
        folder=OUT/f'fold_{fold}'
        for a in arms():
            report=folder/f'{a["name"]}.json'
            if report.exists() and json.loads(report.read_text()).get('complete'): continue
            if a['initialization']=='sourcetargetmae':
                announce('pretraining',fold=fold); child('mae',fold)
            announce('training',fold=fold,arm=a['name']); child('train',fold,a['name'])
        if not (folder/'policy_lock.json').exists():
            announce('calibrating',fold=fold); child('calibrate',fold)
    # All folds' models/policies are fixed before opening outer-fold outcomes for scoring.
    for fold in range(5):
        if not (OUT/f'fold_{fold}'/'evaluation.json').exists():
            announce('evaluating',fold=fold); child('evaluate',fold)
    announce('aggregating'); aggregate(); announce('complete',folds=5,supervised_fits=25,mae_fits=5)

control.arms=arms
control.table=table
control.init_model=init_model
control.summarize=fold_summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','queue','mae','train','calibrate','evaluate','aggregate'])
    parser.add_argument('--fold',type=int,choices=range(5)); parser.add_argument('--arm')
    args=parser.parse_args()
    prepare()
    if args.stage=='prepare': return
    if args.stage=='queue':
        try: queue()
        except Exception as exc: announce('failed',error=repr(exc)); raise
        return
    if args.stage=='aggregate': aggregate(); return
    assert args.fold is not None
    norm=configure(args.fold)
    if args.stage=='mae': pretrain(norm)
    elif args.stage=='train':
        assert not (control.OUT/'fit_lock.json').exists()
        arm=next(a for a in arms() if a['name']==args.arm)
        start=time.time(); report=fit(arm,norm); report['elapsed_minutes']=(time.time()-start)/60
        control.dump(control.OUT/f'{arm["name"]}.json',report)
    elif args.stage=='calibrate': control.calibrate(norm)
    else: control.evaluate(norm); fold_summary()

if __name__=='__main__': main()
