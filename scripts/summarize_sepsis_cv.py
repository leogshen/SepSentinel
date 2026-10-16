"""Recover CV reporting without modifying the frozen training implementation."""
import json
import numbers
from pathlib import Path
import numpy as np
import run_sepsis_cv as cv

def differences(a,b):
    return {k:a[k]-b[k] for k in a if isinstance(a[k],numbers.Real) and isinstance(b[k],numbers.Real)}

def main():
    control=cv.control; out=cv.OUT
    lock=json.loads((out/'implementation_lock.json').read_text())
    assert all(control.sha(cv.ROOT/k)==v for k,v in lock['code'].items())
    groups={}; folds=[]; ids=[]; artifacts={}
    for fold in range(5):
        folder=out/f'fold_{fold}'
        fit_lock=json.loads((folder/'fit_lock.json').read_text())
        assert all(control.sha(folder/k)==v for k,v in fit_lock.items())
        assert control.sha(folder/'policies.json')==json.loads((folder/'policy_lock.json').read_text())['sha256']
        record_ids=json.loads((folder/'local/record_ids.json').read_text())['test']
        ids.extend(record_ids)
        rows=json.loads((folder/'evaluation.json').read_text())
        assert len(rows)==len(cv.arms())*len(cv.C['gates'])*len(cv.C['recall_targets'])
        for path in [folder/'evaluation.json',folder/'policies.json',folder/'fit_lock.json']:
            artifacts[str(path.relative_to(out))]=control.sha(path)
        for r in rows:
            assert r['test'] is not None,'Infeasible calibration policy; cannot silently omit a fold'
            path=folder/'local'/r['statistics_file']
            artifacts[str(path.relative_to(out))]=control.sha(path)
            stat=np.load(path)['stats']; assert len(stat)==len(record_ids)
            assert cv.metrics(stat)==r['test'],'Stored metrics and record statistics disagree'
            key=(r['name'],r['gate'],r['target_recall'])
            groups.setdefault(key,[]).append(stat)
            folds.append(dict(fold=fold,name=key[0],gate=key[1],target_recall=key[2],metrics=r['test']))
    assert len(ids)==len(set(ids))==json.loads((out/'fold_manifest.json').read_text())['records']
    pooled={k:np.concatenate(v) for k,v in groups.items()}
    summaries=[]; comparisons=[]
    for key,stat in pooled.items():
        summaries.append(dict(name=key[0],gate=key[1],target_recall=key[2],out_of_fold_metrics=cv.metrics(stat),
            conditional_record_bootstrap_ci95=cv.bootstrap(stat,cv.C['bootstrap_replicates'],930)))
    pairs=[('values__xgb_34f_seed42','availability__xgb_34f_seed42'),
           ('values__scratch_34f_seed42','availability__scratch_34f_seed42'),
           ('values__sourcetargetmae_34f_seed42','values__scratch_34f_seed42'),
           ('values__scratch_34f_seed42','values__xgb_34f_seed42')]
    for left,right in pairs:
        for gate in cv.C['gates']:
            for target in cv.C['recall_targets']:
                a,b=pooled[(left,gate,target)],pooled[(right,gate,target)]
                np.testing.assert_array_equal(a[:,[0,1,4,5]],b[:,[0,1,4,5]])
                comparisons.append(dict(model=left,reference=right,gate=gate,target_recall=target,
                    point_difference=differences(cv.metrics(a),cv.metrics(b)),
                    conditional_record_bootstrap_ci95=cv.bootstrap(a,cv.C['bootstrap_replicates'],930,b)))
    control.dump(out/'out_of_fold_summary.json',summaries)
    control.dump(out/'fold_metrics.json',folds)
    control.dump(out/'paired_comparisons.json',comparisons)
    control.dump(out/'aggregation_recovery.json',dict(reason='Original summary subtracted nested counts dictionaries; scalar differences now filter numeric values.',
        training_implementation_unchanged=True,code_sha256=control.sha(Path(__file__)),input_artifacts=artifacts))
    cv.announce('complete',folds=5,supervised_fits=25,mae_fits=5,aggregation_recovered=True)
    for r in summaries:
        if r['gate']=='all' and r['target_recall']==.5:
            m=r['out_of_fold_metrics']
            print(r['name'], {k:round(m[k],4) for k in ['patient_precision','patient_recall','precision','recall','alarms_per_patient_day']},flush=True)

if __name__=='__main__':
    assert differences(dict(x=.5,counts={'a':2},missing=None),dict(x=.3,counts={'a':1},missing=None))=={'x':.2}
    main()
