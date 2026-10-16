"""Independent patient-flag audit from frozen thresholds and cached hourly scores."""
import json
import pickle
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/sepsis_cv_20261013'

def audit_fold(fold):
    folder=OUT/f'fold_{fold}'
    with (folder/'local/test.pkl').open('rb') as f: eps=pickle.load(f)
    reports=[]
    for r in json.loads((folder/'evaluation.json').read_text()):
        if r['gate']!='all' or r['target_recall']!=.5: continue
        z=np.load(folder/'local'/f'{r["name"]}_test_scores.npz')
        rows=z['rows']; scores=z['p']; threshold=r['policy']['threshold']
        flags=np.zeros(len(eps),bool); events=[]; expected=[]
        for i,e in enumerate(eps):
            t=e['t']; onset=e['onset']
            warning=np.zeros(len(t),bool) if onset is None else ((onset-t>=1)&(onset-t<=12))
            eligible=(t>=6)&((t+12<=t[-1])|warning)
            if onset is not None: eligible &= t<onset
            events.append(bool(np.any(warning&eligible)))
            expected.extend((i,int(j)) for j in np.flatnonzero(eligible))
        np.testing.assert_array_equal(rows,np.asarray(expected))
        flags[np.unique(rows[scores>=threshold,0])]=True
        events=np.asarray(events)
        tp=int(np.sum(flags&events)); fp=int(np.sum(flags&~events)); n=int(events.sum())
        counts=r['test']['counts']
        assert tp==counts['flagged_event_records'] and tp+fp==counts['flagged_records'] and n==counts['event']
        assert tp/(tp+fp)==r['test']['patient_precision']
        reports.append(dict(fold=fold,name=r['name'],septic_flagged=tp,nonseptic_flagged=fp,event_records=n,
                            records=len(eps),patient_precision=tp/(tp+fp),patient_recall=tp/n,
                            threshold=threshold,timely_recall=r['test']['recall']))
    return reports

def main():
    rows=[r for fold in range(5) for r in audit_fold(fold)]
    groups={}
    for r in rows: groups.setdefault(r['name'],[]).append(r)
    result=[]
    for name,rr in groups.items():
        tp=sum(r['septic_flagged'] for r in rr); fp=sum(r['nonseptic_flagged'] for r in rr)
        result.append(dict(name=name,septic_flagged=tp,nonseptic_flagged=fp,pooled_precision=tp/(tp+fp),
            mean_fold_precision=float(np.mean([r['patient_precision'] for r in rr])),
            fold_precisions=[round(r['patient_precision'],4) for r in rr],
            fold_recalls=[round(r['patient_recall'],4) for r in rr]))
    previous=[]
    path=ROOT/'results/sepsis_long_training_20261012/evaluation.json'
    for r in json.loads(path.read_text()):
        if r['gate']=='all' and r['target_recall']==.5 and r['name'].startswith('scratch_34f_lr0.0001_'):
            m=r['test']; c=m['counts']
            previous.append(dict(name=r['name'],patient_precision=m['patient_precision'],patient_recall=m['patient_recall'],
                timely_recall=m['recall'],septic_flagged=c['flagged_event_records'],
                nonseptic_flagged=c['flagged_records']-c['flagged_event_records'],events=c['event'],nonseptic=c['nonseptic']))
    report=dict(independent_score_reconstruction_passed=True,folds=rows,pooled=result,previous_scratch=previous)
    (OUT/'patient_precision_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(pooled=result,previous_scratch=previous),indent=2))

if __name__=='__main__': main()
