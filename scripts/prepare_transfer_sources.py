"""Prepare outcome-independent source histories, preserving previous holdouts."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import time
import numpy as np
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sepsentinel.data.splitting import grouped_patient_split
from sepsentinel.data import sicdb
from sepsentinel.data.quality import RANGES

OUT = ROOT/'results/transfer_protocol'
LOCAL = OUT/'local'
FEATURES = ['heart_rate','spo2','respiratory_rate','temperature',
            'lactate','ph','creatinine','wbc','platelets','bilirubin']

def bucket(person):
    return int(hashlib.sha256(('transfer_source_val:'+person).encode()).hexdigest()[:8],16)/2**32

def package(meta, rows, source, kind):
    result=[]
    columns={f:i for i,f in enumerate(FEATURES)}
    mapping={int(s):g for s,g in rows.groupby('sid',sort=False)}
    for sid,person,length in meta:
        length=int(min(336,np.floor(length)))
        if length<6: continue
        x=np.full((length,len(FEATURES)),np.nan,np.float32)
        g=mapping.get(int(sid))
        if g is None: continue
        for feature, fg in g.groupby('feature'):
            if feature not in columns: continue
            j=columns[feature]
            fg=fg.sort_values('hour')
            ix=fg.hour.to_numpy(dtype=int)-1
            value=fg.value.to_numpy(dtype=np.float32)
            lo,hi=RANGES[feature]
            valid=(ix>=0)&(ix<length)&np.isfinite(value)&(value>=lo)&(value<=hi)
            # Delayed old specimens cannot overwrite more recent known ones.
            measured=fg.measured.to_numpy(dtype=float)
            valid &= measured>=np.maximum.accumulate(np.where(valid,measured,-np.inf))
            x[ix[valid],j]=value[valid]
        if not np.isfinite(x).any(): continue
        result.append({'id':source+':'+str(sid),'person':str(person),'x':x,
                       't':np.arange(1,length+1,dtype=np.float32),
                       'kind':np.asarray(kind,np.int64),'source':source})
    return result

def main():
    LOCAL.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    def log(x): print(f'[{(time.time()-t0)/60:.1f}m] {x}',flush=True)
    report={'features':FEATURES,'outcome_truncation':False,'heldouts_preserved':True}
    old=pickle.load(open(ROOT/'results/mimic31_pooled17.pkl','rb'))
    assignments={str(e['subject_id']):k for k,v in grouped_patient_split(old,random_state=42).items() for e in v}
    del old
    con=duckdb.connect(str(ROOT/'results/prospective_qc/events.duckdb'),read_only=True)
    meta=con.execute('select stay_id,subject_id,los_hours from stays order by stay_id').fetchall()
    trainmeta=[]
    for sid,person,los in meta:
        p=str(person)
        if p not in assignments:
            u=int(hashlib.sha256(('prospective20260923:'+p).encode()).hexdigest()[:8],16)/2**32
            assignments[p]='train' if u<.7 else ('val' if u<.85 else 'test')
        if assignments[p]=='train': trainmeta.append((sid,p,los))
    all_sources={}
    mimic_path=LOCAL/'source_mimic.pkl'
    if mimic_path.exists():
        mimic=pickle.load(open(mimic_path,'rb'))
    else:
        allowed=pd.DataFrame({'stay_id':[m[0] for m in trainmeta]})
        con.register('allowed',allowed)
        feats=','.join("'"+f+"'" for f in FEATURES)
        data=con.execute(f'''select g.stay_id sid,feature,decision_h as "hour",value,measured_h measured
             from gridded g join allowed a using(stay_id) where feature in ({feats})
             order by g.stay_id,decision_h''').df()
        log(f'MIMIC training-person events: {len(data)}')
        mimic=package(trainmeta,data,'mimic',[0]*10)
        pickle.dump(mimic,open(mimic_path,'wb'),protocol=5)
        del data
    con.close()
    all_sources['mimic']=mimic
    log(f'MIMIC histories: {len(mimic)}')
    sic_path=LOCAL/'source_sicdb.pkl'
    if sic_path.exists():
        sic=pickle.load(open(sic_path,'rb'))
    else:
        oldsic=pickle.load(open(ROOT/'results/sicdb_episodes.pkl','rb'))
        people=sorted({e['subject_id'] for e in oldsic})
        sealed=set(np.random.default_rng(42).choice(people,size=int(.3*len(people)),replace=False).tolist())
        allowed_people={int(p.removeprefix('sicdb_p')) for p in people if p not in sealed}
        del oldsic
        con=duckdb.connect()
        con.execute("set memory_limit='8GB'")
        con.execute('set threads=4')
        root=(ROOT.parent/'data_local/sicdb-1.0.8').as_posix()
        cases=pd.read_csv(root+'/cases.csv.gz')
        cases['los_hours']=(cases.TimeOfStay-cases.ICUOffset)/3600
        eligible=cases.loc[cases.PatientID.isin(allowed_people)&cases.AgeOnAdmission.ge(18)&cases.los_hours.ge(6)]
        con.register('eligible',eligible)
        vm={i:f for f in FEATURES[:4] for i in sicdb.VITAL_IDS[f]}
        lm={i:f for f in FEATURES[4:] for i in sicdb.LAB_IDS[f]}
        case=lambda col,m:'case '+col+' '+' '.join(f"when {i} then '{f}'" for i,f in m.items())+' end'
        log('Reading SICdb vitals and labs without sepsis labels')
        vit=con.execute(f'''select d.CaseID sid,{case('DataID',vm)} feature,
          cast(floor((d."Offset"-e.ICUOffset)/3600.0)+1 as integer) as "hour",
          avg(d.Val) as "value", max((d."Offset"-e.ICUOffset)/3600.0) measured
          from read_csv_auto('{root}/data_float_h.csv.gz') d join eligible e using(CaseID)
          where DataID in ({','.join(map(str,vm))}) and d."Offset">=e.ICUOffset
          and (d."Offset"-e.ICUOffset)/3600.0<least(336,e.los_hours)
          group by 1,2,3''').df()
        lab=con.execute(f'''select l.CaseID sid,{case('LaboratoryID',lm)} feature,
          cast(floor((l."Offset"-e.ICUOffset)/3600.0)+1 as integer) as "hour",
          arg_max(LaboratoryValue,l."Offset") as "value",max((l."Offset"-e.ICUOffset)/3600.0) measured
          from read_csv_auto('{root}/laboratory.csv.gz') l join eligible e using(CaseID)
          where LaboratoryID in ({','.join(map(str,lm))}) and l."Offset">=e.ICUOffset
          and (l."Offset"-e.ICUOffset)/3600.0<least(336,e.los_hours)
          group by 1,2,3''').df()
        smeta=[(int(r.CaseID),'sicdb_p'+str(int(r.PatientID)),float(r.los_hours)) for r in eligible.itertuples()]
        sic=package(smeta,pd.concat([vit,lab],ignore_index=True),'sicdb',[1]*4+[0]*6)
        pickle.dump(sic,open(sic_path,'wb'),protocol=5)
        con.close()
        report['sicdb_available_clock']='Hourly block end; laboratory Offset. Separate receipt times unavailable; source reconstruction only.'
    all_sources['sicdb']=sic
    log(f'SICdb histories: {len(sic)}')
    splits={}
    rng=np.random.default_rng(20260930)
    norm=[]
    for source,eps in all_sources.items():
        splits[source]={k:[e for e in eps if (bucket(e['person'])<.1)==(k=='val')] for k in ['train','val']}
        report[source]={k:{'episodes':len(v),'people':len({e['person'] for e in v}),'hours':sum(len(e['x']) for e in v)} for k,v in splits[source].items()}
        train=splits[source]['train']
        indices=rng.choice(len(train),size=min(4000,len(train)),replace=False)
        rows=np.concatenate([train[i]['x'] for i in indices])
        norm.append(rows[rng.choice(len(rows),size=min(200000,len(rows)),replace=False)])
    x=np.concatenate(norm)
    median=np.nanmedian(x,axis=0).astype(np.float32)
    std=np.nanstd(x,axis=0).astype(np.float32);std=np.maximum(std,1e-3)
    np.savez(LOCAL/'source_scaler.npz',median=median,std=std)
    pickle.dump(splits,open(LOCAL/'source_splits.pkl','wb'),protocol=5)
    report['normalization']={'median':median.tolist(),'std':std.tolist(),'training_only':True}
    report['elapsed_minutes']=(time.time()-t0)/60
    (OUT/'source_manifest.json').write_text(json.dumps(report,indent=2))
    log('Source preparation complete')

if __name__=='__main__': main()
