"""Bounded richer-input transfer experiment. Existing outputs remain unchanged."""
import argparse
import hashlib
import json
import math
import pickle
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score,roc_auc_score
from transfer_sequence import Encoder,collate,masked_batch,reconstruction_loss
from prepare_transfer_cinc import COLS as SHARED

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/transfer_protocol/local'
OUT=ROOT/'results/rich_transfer_20261009'
LOCAL=OUT/'local'
EXTRA=['SBP','MAP','DBP','EtCO2','BaseExcess','HCO3','FiO2','PaCO2','SaO2','AST','BUN',
       'Alkalinephos','Calcium','Chloride','Bilirubin_direct','Glucose','Magnesium','Phosphate',
       'Potassium','TroponinI','Hct','Hgb','PTT','Fibrinogen']
COLS=SHARED+EXTRA
START=time.time()
DEADLINE=float('inf')

def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def status(stage,**kw):
    obj=dict(stage=stage,elapsed_minutes=round((time.time()-START)/60,2),**kw)
    dump(OUT/'status.json',obj);print(obj,flush=True)

def prepare():
    if (OUT/'data_manifest.json').exists():return
    source=ROOT.parent/'data_local/cinc2019_official/training_setB'
    manifest={'columns':COLS,'splits':{},'exploratory':True,'split_unit':'released record'}
    for split in ['fit','tune','cal','test']:
        old=pickle.load(open(OLD/f'cinc_{split}.pkl','rb'));out=[]
        for n,e in enumerate(old):
            df=pd.read_csv(source/(e['id']+'.psv'),sep='|')
            assert np.array_equal(df.ICULOS.to_numpy(),e['t'])
            raw=df[COLS].to_numpy(np.float32)
            # Preserve prior QC and exact values for common channels.
            raw[:,:10]=e['x'];raw[~np.isfinite(raw)]=np.nan
            # Broad physical screening only; no outcome-based exclusions.
            for c,lo,hi in [('SBP',20,350),('MAP',10,250),('DBP',5,200),('SaO2',0,100),
                          ('FiO2',0,100),('Hct',0,100),('Hgb',0,30),('Potassium',0,20)]:
                k=COLS.index(c);bad=(raw[:,k]<lo)|(raw[:,k]>hi);raw[bad,k]=np.nan
            age=np.full_like(raw,np.inf)
            for k in range(34):
                last=np.maximum.accumulate(np.where(np.isfinite(raw[:,k]),e['t'],-np.inf))
                age[:,k]=e['t']-last
            bp=(age[:,10:13]<2).any(1)
            routine=(age[:,[6,7,8]]<24).all(1)
            perf=(age[:,[4,5]]<24).any(1)
            out.append(dict(e,x=raw,kind=np.zeros(34,np.int64),
                            gates={'all':np.ones(len(raw),bool),'bp':bp,
                                   'bp_routine':bp&routine,'bp_routine_perfusion':bp&routine&perf}))
            if n%3000==0:status('preparing',split=split,records=n)
        pickle.dump(out,open(LOCAL/f'{split}.pkl','wb'),protocol=5)
        manifest['splits'][split]={'records':len(out),'rows':sum(len(e['t']) for e in out)}
        if split=='fit':
            z=np.concatenate([e['x'][::max(1,len(e['x'])//30)] for e in out])
            med=np.nanmedian(z,0);std=np.nanstd(z,0)
            med=np.nan_to_num(med);std=np.where(np.isfinite(std)&(std>1e-6),std,1)
            oldnorm=np.load(OLD/'source_scaler.npz');med[:10]=oldnorm['median'];std[:10]=oldnorm['std']
            np.savez(LOCAL/'scaler.npz',median=med.astype(np.float32),std=std.astype(np.float32))
    dump(OUT/'data_manifest.json',manifest)

def load(split):return pickle.load(open(LOCAL/f'{split}.pkl','rb'))

def view(eps,horizon,late=0,gate='all'):
    out=[]
    for e in eps:
        t=e['t'];o=e['onset']
        y=np.zeros(len(t),np.float32) if o is None else ((o-t<=horizon)&(o-t>=-late if late else o-t>=1)).astype(np.float32)
        eligible=(t>=6)&((t+horizon<=t[-1])|(y==1))
        if o is not None:eligible &= t<=o+late if late else t<o
        out.append(dict(e,y=y,eligible=eligible,rich=e['gates'][gate]))
    return out

def rows_for(eps,mode='all',rng=None):
    rng=np.random.default_rng(930) if rng is None else rng;rows=[]
    for i,e in enumerate(eps):
        js=np.flatnonzero(e['eligible'])
        if not len(js):continue
        if mode=='one':js=rng.choice(js,1)
        if mode=='sample':js=np.sort(rng.choice(js,min(8,len(js)),replace=False))
        rows.extend((i,int(j)) for j in js)
    return np.asarray(rows,np.int32)

class RichModel(nn.Module):
    def __init__(self,nfeatures):
        super().__init__();self.encoder=Encoder();self.encoder.feature_id=nn.Embedding(34,32)
        self.head=nn.Sequential(nn.Linear(100,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))
        self.active=np.arange(34)<nfeatures
    def forward(self,b,static):
        z=self.encoder.embedding(*b,active=self.active)
        return self.head(torch.cat([z,static],-1)).squeeze(-1)
    def source_init(self,seed):
        state=torch.load(OLD/f'mae_seed{seed}.pt',map_location='cpu',weights_only=False)['encoder']
        ids=state.pop('feature_id.weight')
        missing=self.encoder.load_state_dict(state,strict=False)
        assert missing.missing_keys==['feature_id.weight'] and not missing.unexpected_keys
        with torch.no_grad():self.encoder.feature_id.weight[:10].copy_(ids)

def batch(eps,rows,med,std):
    items=[];ys=[];static=[]
    for i,j in rows:
        e=eps[i];first=max(0,j-71)
        items.append(dict(x=e['x'][first:j+1],t=e['t'][first:j+1],kind=e['kind']))
        ys.append(e['y'][j]);static.append(e['static'])
    return items,torch.as_tensor(np.asarray(ys),device='cuda'),torch.as_tensor(np.asarray(static),device='cuda')

def predict(model,eps,rows,med,std):
    model.eval();out=[]
    with torch.no_grad():
        for pos in range(0,len(rows),128):
            items,_,s=batch(eps,rows[pos:pos+128],med,std)
            logits=model(collate(items,med,std,active=model.active),s)
            out.append(torch.sigmoid(logits).cpu().numpy())
    p=np.concatenate(out);assert np.isfinite(p).all();return p

def adapt(model,eps,med,std,seed):
    decoder=nn.Sequential(nn.Linear(96,96),nn.GELU(),nn.Linear(96,34)).cuda()
    optimizer=torch.optim.AdamW(list(model.encoder.parameters())+list(decoder.parameters()),lr=1e-4)
    rng=np.random.default_rng(seed);losses=[];model.train()
    for step in range(1000):
        if time.time()>DEADLINE-3600:break
        items=[]
        for _ in range(32):
            e=eps[int(rng.integers(len(eps)))];end=int(rng.integers(1,len(e['x'])+1));first=max(0,end-72)
            items.append(dict(x=e['x'][first:end],t=e['t'][first:end],kind=e['kind']))
        b,y,mask=masked_batch(items,med,std,rng)
        loss=reconstruction_loss(decoder(model.encoder(*b)),y,mask)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        losses.append(float(loss.detach()))
    return {'steps':len(losses),'last100_loss':float(np.mean(losses[-100:])) if losses else None}

def fit_arm(rawfit,rawtune,med,std,seed,init,nfeatures,horizon):
    name=f'{init}_{nfeatures}f_{horizon}h_seed{seed}'
    if (OUT/f'{name}.json').exists():return json.loads((OUT/f'{name}.json').read_text())
    status('training',arm=name);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    fit=view(rawfit,horizon);tune=view(rawtune,horizon)
    model=RichModel(nfeatures).cuda();adaptation=None
    if init!='scratch':model.source_init(seed)
    if init=='adapt':adaptation=adapt(model,rawfit,med,std,seed)
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':3e-4 if init=='scratch' else 1e-4},
                          {'params':model.head.parameters(),'lr':3e-4 if init=='scratch' else 1e-3}],weight_decay=1e-4)
    prevalence=np.mean([e['y'][e['eligible']].mean() for e in fit if e['eligible'].any()])
    weight=min(20,(1-prevalence)/max(prevalence,1e-6))
    tr=rows_for(tune,'sample');ty=np.asarray([tune[i]['y'][j] for i,j in tr]);tw=1/np.bincount(tr[:,0])[tr[:,0]]
    best=-np.inf;stale=0;history=[];ckpt=LOCAL/f'{name}.pt'
    for epoch in range(20):
        if epoch and time.time()>DEADLINE-3600:break
        model.train();rows=rows_for(fit,'one',rng);rng.shuffle(rows);losses=[]
        for pos in range(0,len(rows),32):
            items,y,s=batch(fit,rows[pos:pos+32],med,std)
            pred=model(collate(items,med,std,active=model.active),s)
            loss=nn.functional.binary_cross_entropy_with_logits(pred,y,pos_weight=torch.tensor(weight,device='cuda'))
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            losses.append(float(loss.detach()))
        p=predict(model,tune,tr,med,std)
        score=float(average_precision_score(ty,p,sample_weight=tw))
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),tune_record_weighted_ap=score))
        print(name,history[-1],flush=True)
        if score>best+1e-5:
            best=score;stale=0;torch.save(model.state_dict(),ckpt)
        else:stale+=1
        if stale>=4:break
    report=dict(name=name,seed=seed,initialization=init,nfeatures=nfeatures,horizon=horizon,
                history=history,adaptation=adaptation,complete=True,test_opened=False)
    dump(OUT/f'{name}.json',report);del model,opt;torch.cuda.empty_cache();return report

def evaluate_scores(eps,rows,p,thresholds,gate):
    """Simultaneous threshold sweep with explicit per-record matching and alarm categories."""
    K=len(thresholds);N=len(eps);length=max(len(e['t']) for e in eps)
    score=np.full((length,N),-np.inf,np.float32);valid=np.zeros((length,N),bool)
    warning=np.zeros((length,N),bool);times=np.zeros((length,N),np.float32)
    for i,e in enumerate(eps):
        l=len(e['t']);times[:l,i]=e['t'];warning[:l,i]=e['y'].astype(bool)
        valid[:l,i]=e['eligible']&e['gates'][gate]
    for k,(i,j) in enumerate(rows):score[j,i]=p[k]
    score[~valid]=-np.inf
    events=np.asarray([bool((e['y']*e['eligible']).any()) for e in eps])
    eligible_events=(warning&valid).any(0)
    last=np.full((K,N),-np.inf,np.float32);hit=np.zeros((K,N),bool);alarms=np.zeros((K,N),np.int32)
    early=np.zeros(K,np.int64);duplicate=np.zeros(K,np.int64);noncase=np.zeros(K,np.int64);other=np.zeros(K,np.int64)
    septic=np.asarray([e['onset'] is not None for e in eps])
    onset=np.asarray([e['onset'] if e['onset'] is not None else -np.inf for e in eps])
    for j in range(length):
        fire=(score[j][None,:]>=thresholds[:,None])&(times[j][None,:]-last>=6)&valid[j][None,:]
        correct=fire&warning[j][None,:]
        duplicate+=(correct&hit).sum(1)
        hit|=correct;alarms+=fire
        noncase+=(fire&~septic[None,:]).sum(1)
        # On pre-onset-only views, unmatched septic alarms are too early.
        before=times[j]<onset
        early+=(fire&~warning[j][None,:]&septic[None,:]&before[None,:]).sum(1)
        other+=(fire&~warning[j][None,:]&septic[None,:]&~before[None,:]).sum(1)
        last=np.where(fire,times[j][None,:],last)
    total_hours=sum(int(e['eligible'].sum()) for e in eps);total_events=int(events.sum());gated_events=int(eligible_events.sum())
    output=[]
    for k,threshold in enumerate(thresholds):
        matched=int(hit[k].sum());count=int(alarms[k].sum())
        assert matched<=total_events and matched<=count
        assert count==matched+int(duplicate[k]+early[k]+noncase[k]+other[k])
        output.append(dict(threshold=float(threshold),precision=matched/count if count else None,
            recall=matched/total_events if total_events else None,subgroup_recall=matched/gated_events if gated_events else None,
            events=total_events,eligible_events=gated_events,event_coverage=gated_events/total_events if total_events else None,
            matched=matched,alarms=count,alarms_per_day=24*count/total_hours,
            eligible_hours=int(valid.sum()),total_hours=total_hours,
            false_alarm_categories=dict(too_early=int(early[k]),duplicate=int(duplicate[k]),nonseptic=int(noncase[k]),other=int(other[k]))))
    return output

def tests():
    torch.manual_seed(930);m=RichModel(34).eval();med=np.zeros(34,np.float32);std=np.ones(34,np.float32)
    rng=np.random.default_rng(930);x=rng.normal(size=(12,34)).astype(np.float32)
    a=dict(x=x,t=np.arange(1,13,dtype=np.float32),kind=np.zeros(34,np.int64))
    short=dict(a,x=x[:6],t=a['t'][:6])
    with torch.no_grad():
        full=m.encoder(*collate([a],med,std,'cpu'))[0,:6]
        small=m.encoder(*collate([short],med,std,'cpu'))[0]
        padded=m.encoder(*collate([short,a],med,std,'cpu'))[0,:6]
    assert torch.allclose(full,small,atol=1e-5) and torch.allclose(padded,small,atol=1e-5)
    e=dict(t=np.arange(1,19),y=np.r_[np.zeros(12),np.ones(6)],eligible=np.ones(18,bool),
           onset=19,gates={'all':np.ones(18,bool)})
    rows=np.c_[np.zeros(18,int),np.arange(18)];p=np.ones(18)
    r=evaluate_scores([e],rows,p,np.array([.5]),'all')[0]
    assert r['matched']==1 and r['alarms']==3 and r['false_alarm_categories']['too_early']==2
    # Expanded window: two correct alarms, but one event credit and one duplicate.
    e['y'][6:]=1;r=evaluate_scores([e],rows,p,np.array([.5]),'all')[0]
    assert r['false_alarm_categories']['duplicate']==1 and r['matched']==1
    gpu=RichModel(34).cuda();b=collate([a],med,std)
    loss=gpu(b,torch.zeros((1,4),device='cuda')).square().mean();loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in gpu.parameters() if p.grad is not None)
    del gpu;torch.cuda.empty_cache()
    dump(OUT/'checks.json',dict(causal=True,padding=True,alarm_accounting=True))

def evaluate_all(records,med,std):
    cal,test=load('cal'),load('test');results=[]
    # Same broad scoring population for all arms; labels/windows are applied separately below.
    def union_rows(eps):
        views=[view(eps,h,l) for h,l in [(6,0),(12,0),(6,3)]]
        union=[dict(e,eligible=np.logical_or.reduce([v[i]['eligible'] for v in views])) for i,e in enumerate(eps)]
        return rows_for(union)
    cr=union_rows(cal);qr=union_rows(test)
    for r in records:
        if results and time.time()>DEADLINE:break
        status('scoring',arm=r['name']);m=RichModel(r['nfeatures']).cuda()
        m.load_state_dict(torch.load(LOCAL/f"{r['name']}.pt",map_location='cpu',weights_only=True))
        cp=predict(m,cal,cr,med,std);qp=predict(m,test,qr,med,std)
        np.savez_compressed(LOCAL/f"{r['name']}_predictions.npz",cal_rows=cr,cal_p=cp,test_rows=qr,test_p=qp)
        del m;torch.cuda.empty_cache()
        grid=np.unique(np.r_[1.000001,np.quantile(cp,np.linspace(0,1,151)),0])[::-1]
        for h,late in [(6,0),(12,0),(6,3)]:
            ce,te=view(cal,h,late),view(test,h,late)
            for gate in ['all','bp','bp_routine','bp_routine_perfusion']:
                curve=evaluate_scores(ce,cr,cp,grid,gate)
                policies=[]
                for target in [.3,.6]:
                    metric='recall' if gate=='all' else 'subgroup_recall'
                    choices=[z for z in curve if z[metric] is not None and z[metric]>=target]
                    if choices:policies.append((target,choices[0]))
                if not policies:
                    results.append(dict(name=r['name'],window=[h,late],gate=gate,feasible=False));continue
                testmetrics=evaluate_scores(te,qr,qp,np.array([z['threshold'] for _,z in policies]),gate)
                for (target,policy),testmetric in zip(policies,testmetrics):
                    results.append(dict(name=r['name'],window=[h,late],gate=gate,target_recall=target,
                        calibration=policy,test=testmetric,exploratory=True))
        dump(OUT/'results.json',results)
    return results

def report(results):
    lines=['# Richer-input transfer: exploratory results','',
           'Thresholds were selected on calibration records. This test cohort was previously inspected.',
           'Recall is population event recall; subgroup recall and coverage are reported separately in JSON.',
           '','| Model | Window | Precision | Recall | Alarms/day |','|---|---|---:|---:|---:|']
    for r in results:
        if r.get('gate')!='all' or r.get('target_recall')!=.6:continue
        t=r['test'];p=t['precision']
        lines.append(f"| {r['name']} | {r['window']} | {p:.1%} | {t['recall']:.1%} | {t['alarms_per_day']:.3f} |" if p is not None else f"| {r['name']} | {r['window']} | no alarms | 0 | 0 |")
    (OUT/'REPORT.md').write_text('\n'.join(lines))

def main():
    global DEADLINE
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=float,default=4.7);ap.add_argument('--checks-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(exist_ok=True);LOCAL.mkdir(exist_ok=True);DEADLINE=START+args.hours*3600
    torch.set_num_threads(4);tests()
    if args.checks_only:return
    status('preparation');prepare();norm=np.load(LOCAL/'scaler.npz');med,std=norm['median'],norm['std']
    fit,tune=load('fit'),load('tune')
    plan=[(42,i,34,12) for i in ['scratch','source','adapt']]
    plan += [(42,i,n,12) for n in [10,13] for i in ['scratch','source']]
    plan += [(123,i,34,12) for i in ['scratch','source','adapt']]
    plan += [(42,i,34,6) for i in ['scratch','source','adapt']]
    records=[]
    for seed,init,n,h in plan:
        if records and time.time()>DEADLINE-5400:break
        records.append(fit_arm(fit,tune,med,std,seed,init,n,h))
    dump(OUT/'frozen_arms.json',{'arms':records,'planned':len(plan),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'selection_rule':'fixed priority; stop new fits 90 minutes before budget end; select epochs on tuning AP'})
    del fit,tune
    results=evaluate_all(records,med,std);report(results)
    status('complete',trained_arms=len(records),planned_arms=len(plan))

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);status('failed',error=traceback.format_exc());raise
