"""Analytic Bernoulli random-alert baselines under the finite-alarm evaluator."""
import itertools
import json
import pickle
from pathlib import Path
import numpy as np
from sepsis_mission_metrics import record_statistics, FIELDS

ROOT=Path(__file__).resolve().parents[1]
CV=ROOT/'results/sepsis_cv_20261013'
MODELS=ROOT/'results/sepsis_lab_specialize_20261014'
OUT=MODELS/'random_baselines.json'
GATE='bp_routine_perfusion'

def compact(split,fold,gate):
    with (CV/f'fold_{fold}'/'local'/f'{split}.pkl').open('rb') as f: eps=pickle.load(f)
    out=[]
    for e in eps:
        t=e['t']; o=e['onset']
        warning=np.zeros(len(t),bool) if o is None else ((o-t>=1)&(o-t<=12))
        eligible=(t>=6)&((t+12<=t[-1])|warning)
        if o is not None: eligible &= t<o
        allowed=eligible&e['gates'][gate]
        out.append((t[allowed],warning[allowed],bool((warning&eligible).any()),int(eligible.sum())))
    return out

def expected_record(t,warning,q):
    """Exact expected alarms, probability of >=1 timely alarm, probability of flag."""
    n=len(t)
    if not n: return 0.,0.,0.
    alarm=np.zeros(n); nohit_alarm=np.zeros(n)
    blocked=blocked_nohit=hit=0.; left=0
    for j in range(n):
        while left<j and t[j]-t[left]>=6:
            blocked-=alarm[left]; blocked_nohit-=nohit_alarm[left]; left+=1
        alarm[j]=q*max(0.,1.-blocked)
        possible_first_hit=q*max(0.,1.-hit-blocked_nohit)
        if warning[j]: hit+=possible_first_hit
        else: nohit_alarm[j]=possible_first_hit
        blocked+=alarm[j]; blocked_nohit+=nohit_alarm[j]
    return float(alarm.sum()),float(hit),float(1.-(1.-q)**n)

def totals(records,q):
    d=dict(records=len(records),events=0,covered_events=0,eligible_hours=0,hours=0,
           positive_hours=0,alarms=0.,matched=0.,flagged=0.,flagged_events=0.,
           accessible_records=0,accessible_event_records=0)
    for t,y,event,hours in records:
        a,m,f=expected_record(t,y,q)
        d['events']+=int(event); d['covered_events']+=int(y.any()); d['eligible_hours']+=len(t)
        d['hours']+=hours; d['positive_hours']+=int(y.sum()); d['alarms']+=a; d['matched']+=m
        d['flagged']+=f; d['flagged_events']+=f*event
        d['accessible_records']+=int(len(t)>0); d['accessible_event_records']+=int(event and len(t)>0)
    return d

def measures(d):
    div=lambda a,b: a/b if b else None
    return dict(patient_precision=div(d['flagged_events'],d['flagged']),patient_recall=div(d['flagged_events'],d['events']),
        alarm_precision=div(d['matched'],d['alarms']),timely_population_recall=div(d['matched'],d['events']),
        timely_subgroup_recall=div(d['matched'],d['covered_events']),alarms_per_patient_day=div(24*d['alarms'],d['hours']),
        expected_counts=d)

def alarm_total(records,q):
    total=0.
    for t,_,_,_ in records:
        p=np.zeros(len(t)); blocked=0.; left=0
        for j in range(len(t)):
            while left<j and t[j]-t[left]>=6: blocked-=p[left]; left+=1
            p[j]=q*max(0.,1.-blocked); blocked+=p[j]
        total+=float(p.sum())
    return total

def match_rate(records,target):
    assert target<=alarm_total(records,1.)+1e-8
    lo,hi=0.,1.
    for _ in range(28):
        mid=(lo+hi)/2
        if alarm_total(records,mid)<target: lo=mid
        else: hi=mid
    return (lo+hi)/2

def validate():
    # Enumerate all random decisions; compare the analytic recurrence to the
    # existing evaluator, including suppression across a warning-window boundary.
    for times in [[6,7,8,12,13,19],[6,8,9,10,11,12]]:
        t=np.asarray(times,float); y=np.asarray([0,0,0,1,1,1],bool)
        e=dict(t=t,y=y,eligible=np.ones(len(t),bool),gates={'all':np.ones(len(t),bool)},onset=20.)
        rows=np.array([(0,j) for j in range(len(t))])
        for q in [0.,.17,.63,1.]:
            exact=np.zeros(3)
            for bits in itertools.product([0,1],repeat=len(t)):
                p=q**sum(bits)*(1-q)**(len(t)-sum(bits))
                stat=record_statistics([e],rows,np.asarray(bits,float),.5,'all')[0]
                exact+=p*stat[[FIELDS.index(k) for k in ['alarms','matched','flagged_records']]]
            np.testing.assert_allclose(expected_record(t,y,q),exact,atol=1e-12)

def main():
    validate(); print('Exact random-alarm recurrence agrees with exhaustive evaluator checks.',flush=True)
    models=json.loads((MODELS/'out_of_fold_summary.json').read_text())
    actual={r['name']:r['metrics'] for r in models if r['target_recall']==.5}
    names=list(actual); all_test=[]; rich_test=[]; accum={}; probabilities={}
    for fold in range(5):
        full=compact('test',fold,'all'); rich=compact('test',fold,GATE)
        all_test.extend(full); rich_test.extend(rich)
        for gate,test in [('all',full),(GATE,rich)]:
            fit=compact('fit',fold,gate)
            q=sum(int(y.sum()) for _,y,_,_ in fit)/sum(len(t) for t,_,_,_ in fit)
            key='hourly_prevalence_'+gate
            accum.setdefault(key,[]).append(totals(test,q)); probabilities.setdefault(key,[]).append(q)
        cal=compact('cal',fold,GATE)
        policies=json.loads((MODELS/f'fold_{fold}'/'policies.json').read_text())
        for name in names:
            policy=next(r['policy'] for r in policies if r['name']==name and r['target_recall']==.5)
            q=match_rate(cal,policy['alarms'])
            key='calibration_alarm_burden_matched_'+name
            accum.setdefault(key,[]).append(totals(rich,q)); probabilities.setdefault(key,[]).append(q)
        print('Completed fold',fold,flush=True)
    baselines={key:dict(probabilities_by_fold=probabilities[key],**measures({k:sum(d[k] for d in ds) for k in ds[0]})) for key,ds in accum.items()}
    full=totals(all_test,0.); rich=totals(rich_test,0.)
    prevalence=full['events']/full['records']
    rich_prevalence=rich['accessible_event_records']/rich['accessible_records']
    visit=dict(all_visits=dict(records=full['records'],events=full['events'],prevalence=prevalence,
        random_precision=prevalence,random_recall_if_guess_probability_equals_prevalence=prevalence),
        lab_accessible_visits=dict(records=rich['accessible_records'],events=rich['accessible_event_records'],
                                  random_precision=rich_prevalence),
        note='One independent guess per visit. Constant prevalence scores alone produce all-or-no alerts under a fixed threshold.')
    report=dict(visit_baseline=visit,random_hourly_baselines=baselines,actual_models_at_calibration_target_0_5=actual,
        method='Exact expectations of counts under independent Bernoulli hourly guesses; precision is ratio of expected counts, not expectation of a finite-sample precision ratio. Six-hour suppression and one match/event. Guess probabilities estimated on fit or calibrated to model alarm count on calibration only; held-out outcomes not used to choose probabilities.',
        lab_warning_coverage=dict(events=rich['events'],covered_events=rich['covered_events']),
        validation='Exhaustive enumeration agrees with existing evaluator for two irregular timelines at four probabilities.')
    OUT.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(visit=visit,baselines={k:{a:b for a,b in v.items() if a!='expected_counts'} for k,v in baselines.items()}),indent=2))

if __name__=='__main__': main()
