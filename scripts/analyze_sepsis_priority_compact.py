"""Read-only analysis of saved predictions and outcomes; never fit or infer."""
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/sepsis_missions_adapt_20261010'
results=json.loads((OUT/'evaluation.json').read_text())
summary=json.loads((OUT/'seed_summary.json').read_text())
pairs=json.loads((OUT/'paired_comparisons.json').read_text())
eps=pickle.load(open(OUT/'local/test.pkl','rb'))
print('DISCRIMINATION on identical eligible next-12h decisions')
for init in ['scratch','source','adapt']:
    for n in [10,13,34]:
        vals=[]
        for seed in [42,123,456]:
            name=f'{init}_{n}f_12h_seed{seed}';z=np.load(OUT/'local'/f'{name}_test_scores.npz');rows=z['rows'];p=z['p'];keep=[];ys=[]
            for i,j in rows:
                e=eps[i];t=e['t'][j];o=e['onset'];y=o is not None and 1<=o-t<=12
                ok=t>=6 and (t+12<=e['t'][-1] or y) and (o is None or t<o)
                keep.append(ok);ys.append(y)
            keep=np.asarray(keep);y=np.asarray(ys)[keep];pp=p[keep]
            vals.append([roc_auc_score(y,pp),average_precision_score(y,pp)])
        print(init,n,'mean/min/max',np.mean(vals,0).round(4),np.min(vals,0).round(4),np.max(vals,0).round(4),'n',len(y),'positives',y.sum())
print('PATIENT METRICS AND COUNTS, ALL GATE TARGET .6, 12h')
for init in ['scratch','source','adapt']:
    for n in [10,13,34]:
        rr=[r for r in results if r['name'].startswith(f'{init}_{n}f_') and r['gate']=='all' and r['target_recall']==.6 and r['window']==[12,0]]
        print(init,n,'patient precision/recall',np.mean([[r['test']['patient_precision'],r['test']['patient_recall']] for r in rr],0).round(4))
        if n==34:print('counts summed across seeds', {k:sum(r['test']['counts'][k] for r in rr) for k in ['matched','alarms','too_early','duplicate','nonseptic_alarms','other']})
print('OPERATING CURVE 34 source, averaged seeds, all gate')
for window in [[6,0],[12,0],[6,3]]:
    for r in summary:
        if r['model']=='source_34f_12h' and r['gate']=='all' and r['window']==window:
            print(window,r['target_recall'],{k:round(r['metrics'][k]['mean'],4) for k in ['precision','recall']})
print('PAIR CIs 34 vs10 SCRATCH, 12h .6')
for r in pairs:
    if r['model'].startswith('scratch_34f') and r['reference'].startswith('scratch_10f') and r['window']==[12,0] and r['gate']=='all' and r['target_recall']==.6:
        print(r['model'],r['difference_ci95']['precision'],r['difference_ci95']['recall'])
print('MOST SPECIFIC GATE: all inputs per seed, 12h .6')
for r in results:
    if '34f' in r['name'] and r['window']==[12,0] and r['gate']=='bp_routine_perfusion' and r['target_recall']==.6:
        print(r['name'],{k:r['test'][k] for k in ['precision','recall','subgroup_recall','event_coverage','decision_coverage']},r['test']['counts']['eligible_event'])
print('CALIBRATION to TEST source34 seed42,12h all .6')
r=next(r for r in results if r['name']=='source_34f_12h_seed42' and r['window']==[12,0] and r['gate']=='all' and r['target_recall']==.6)
print('cal',r['policy'],'test',r['test'],'CI',r['ci95'])
