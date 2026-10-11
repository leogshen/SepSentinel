"""Reconcile the historical mirror, then seal a record-separated site-B target."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sepsentinel.data.quality import RANGES
from transfer_sequence import FEATURES

OUT=ROOT/'results/transfer_protocol';LOCAL=OUT/'local'
COLS=['HR','O2Sat','Resp','Temp','Lactate','pH','Creatinine','WBC','Platelets','Bilirubin_total']

def fingerprint(x):
    a=np.nan_to_num(np.round(np.asarray(x,dtype=np.float64),3),nan=-999999.)
    a[a==0]=0
    return hashlib.sha256(a.astype('<f8').tobytes()).hexdigest()

def main():
    root=ROOT.parent/'data_local/cinc2019_official'
    manifest=json.loads((root/'download_manifest.json').read_text())
    assert manifest['A']['files']==20336 and manifest['B']['files']==20000
    one=next((root/'training_setB').glob('*.psv'))
    allcols=pd.read_csv(one,sep='|',nrows=0).columns.tolist()
    featurecols=[c for c in allcols if c!='SepsisLabel']
    oldpath=Path.home()/'.cache/kagglehub/datasets/tea340yashjoshi/sepsis-prediction-dataset/versions/1/Dataset.csv'
    old=pd.read_csv(oldpath)
    full={};prefix={}
    for pid,g in old.groupby('Patient_ID'):
        x=g.sort_values('ICULOS')[featurecols].to_numpy()
        full[fingerprint(x)]=str(pid);prefix[fingerprint(x[:8])]=str(pid)
    seen={'A':set(),'B':set()};counts={};splits={k:[] for k in ['fit','tune','cal','test']}
    duplicate_hashes=set();qc={f:0 for f in FEATURES}
    attrition={}
    for site in ['A','B']:
        count={'records':0,'full_mirror_matches':0,'prefix_mirror_matches':0}
        files=sorted((root/f'training_set{site}').glob('*.psv'))
        for i,p in enumerate(files):
            df=pd.read_csv(p,sep='|')
            assert df.columns.tolist()==allcols
            a=df[featurecols].to_numpy();digest=fingerprint(a);pr=fingerprint(a[:8])
            matched=digest in full or pr in prefix
            count['records']+=1
            if digest in full: count['full_mirror_matches']+=1;seen[site].add(full[digest])
            if pr in prefix: count['prefix_mirror_matches']+=1;seen[site].add(prefix[pr])
            if site=='A': continue
            reason=None
            if matched: reason='previously_seen_mirror'
            elif digest in duplicate_hashes: reason='duplicate_record'
            duplicate_hashes.add(digest)
            t=df.ICULOS.to_numpy(dtype=np.float32);labels=df.SepsisLabel.to_numpy()
            if len(t)<6: reason='too_short'
            elif not np.all(np.diff(t)==1): reason='nonunit_clock'
            elif labels[0]==1: reason='left_censored_onset'
            elif np.any(np.diff(labels)<0): reason='nonmonotone_label'
            onset=float(t[np.flatnonzero(labels)[0]]+6) if labels.any() else None
            y=((onset-t>0)&(onset-t<=6)).astype(np.float32) if onset is not None else np.zeros(len(t),np.float32)
            eligible=(t>=6)&((t+6<=t[-1])|(y==1))
            if onset is not None: eligible &= t<onset
            if not eligible.any():reason='no_resolved_eligible_decision'
            if reason:
                attrition[reason]=attrition.get(reason,0)+1;continue
            x=df[COLS].to_numpy(dtype=np.float32)
            for j,f in enumerate(FEATURES):
                lo,hi=RANGES[f];bad=np.isfinite(x[:,j])&((x[:,j]<lo)|(x[:,j]>hi))
                qc[f]+=int(bad.sum());x[bad,j]=np.nan
            age=np.full_like(x,np.inf)
            for j in range(10):
                last=np.maximum.accumulate(np.where(np.isfinite(x[:,j]),t,-np.inf))
                age[:,j]=t-last
            rich=(age[:,[6,7,8]]<24).all(axis=1)&(age[:,[4,5,9]]<24).any(axis=1)
            # (t-24,t] strictly excludes a value exactly 24 hours old.
            rawstatic=df[['Age','Gender']].iloc[0].to_numpy(dtype=np.float32)
            static=np.r_[np.nan_to_num(rawstatic/numbers()),np.isfinite(rawstatic)].astype(np.float32)
            u=int(hashlib.sha256(('cinc_transfer_20260930:'+p.stem).encode()).hexdigest()[:8],16)/2**32
            split='fit' if u<.6 else 'tune' if u<.75 else 'cal' if u<.85 else 'test'
            splits[split].append({'id':p.stem,'person':p.stem,'x':x,'t':t,'y':y,
                                  'eligible':eligible,'rich':rich,'onset':onset,
                                  'static':static,'kind':np.zeros(10,np.int64)})
            if i%5000==0:print(site,i,flush=True)
        counts[site]=count
        print(site,count,flush=True)
    report={'site_counts':counts,'mirror_records':int(old.Patient_ID.nunique()),
            'mirror_records_matched_A':len(seen['A']),'mirror_records_matched_B':len(seen['B']),
            'mirror_unmatched':int(old.Patient_ID.nunique()-len(seen['A']|seen['B'])),
            'attrition':attrition,'invalid_values_removed':qc,
            'target':'official set B, mirror-matched records excluded',
            'split_unit':'released record; repeat-person identity not recoverable',
            'split_rule':'sha256 cinc_transfer_20260930:<record>; 60/15/10/15',
            'endpoint':'assigned onset within next6h; first-positive hour+6; pre-onset only',
            'source_receipt_clock':'not available; Challenge row time',
            'splits':{}}
    for key,eps in splits.items():
        report['splits'][key]={'records':len(eps),'eligible_hours':int(sum(e['eligible'].sum() for e in eps)),
            'events':sum(bool((e['y']*e['eligible']).any()) for e in eps),
            'rich_eligible_hours':int(sum((e['eligible']&e['rich']).sum() for e in eps)),
            'rich_eligible_events':sum(bool((e['y']*e['eligible']*e['rich']).any()) for e in eps)}
        pickle.dump(eps,open(LOCAL/f'cinc_{key}.pkl','wb'),protocol=5)
    (OUT/'cinc_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

def numbers(): return np.asarray([100.,1.],np.float32)

if __name__=='__main__':main()
