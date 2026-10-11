"""Record-level alarm accounting and paired uncertainty; no training dependencies."""
import numpy as np

FIELDS=['event','eligible_event','matched','alarms','hours','eligible_hours',
        'nonseptic','disturbed_nonseptic','too_early','duplicate','nonseptic_alarms','other',
        'positive_hours','true_positive_hours','flagged_hours','flagged_records','flagged_event_records']

def record_statistics(eps,rows,p,threshold,gate,refractory=6):
    rows=np.asarray(rows);p=np.asarray(p)
    if len(rows)!=len(p) or not np.isfinite(p).all():raise ValueError('Invalid prediction alignment or scores')
    if len(np.unique(rows,axis=0))!=len(rows):raise ValueError('Duplicate decision rows')
    if len(rows)>1 and np.any(np.lexsort((rows[:,1],rows[:,0]))!=np.arange(len(rows))):
        raise ValueError('Predictions must be in record/time order')
    result=np.zeros((len(eps),len(FIELDS)),float);byid=[[] for _ in eps]
    for (i,j),score in zip(rows,p):byid[i].append((j,score))
    for i,e in enumerate(eps):
        eligible=np.asarray(e['eligible'],bool);gated=eligible&e['gates'][gate]
        positive=np.asarray(e['y'],bool)
        event=bool((positive&eligible).any());gated_event=bool((positive&gated).any())
        last=-np.inf;hit=False;alarms=early=duplicate=noncase=other=flagged=tp=0
        for j,score in byid[i]:
            if not gated[j] or score<threshold:continue
            flagged+=1;tp+=int(positive[j]);t=e['t'][j]
            if t-last<refractory:continue
            last=t;alarms+=1
            if positive[j]:
                if hit:duplicate+=1
                hit=True
            elif e['onset'] is None:noncase+=1
            elif t<e['onset']:early+=1
            else:other+=1
        assert alarms==int(hit)+early+duplicate+noncase+other
        result[i]=[event,gated_event,hit,alarms,eligible.sum(),gated.sum(),
                   e['onset'] is None,bool(alarms) and e['onset'] is None,
                   early,duplicate,noncase,other,(positive&gated).sum(),tp,flagged,
                   bool(flagged),bool(flagged) and event]
    return result

def ratio(a,b):return float(a/b) if b else None

def metrics(stats):
    d=dict(zip(FIELDS,np.asarray(stats).sum(0)))
    return dict(precision=ratio(d['matched'],d['alarms']),
        recall=ratio(d['matched'],d['event']),subgroup_recall=ratio(d['matched'],d['eligible_event']),
        event_coverage=ratio(d['eligible_event'],d['event']),
        decision_coverage=ratio(d['eligible_hours'],d['hours']),
        alarms_per_patient_day=ratio(24*d['alarms'],d['hours']),
        control_stays_disturbed=ratio(d['disturbed_nonseptic'],d['nonseptic']),
        hourly_precision=ratio(d['true_positive_hours'],d['flagged_hours']),
        hourly_recall=ratio(d['true_positive_hours'],d['positive_hours']),
        patient_precision=ratio(d['flagged_event_records'],d['flagged_records']),
        patient_recall=ratio(d['flagged_event_records'],d['event']),
        counts={k:int(v) for k,v in d.items()})

KEYS=['precision','recall','subgroup_recall','event_coverage','alarms_per_patient_day','control_stays_disturbed']

def bootstrap(stats,n=2000,seed=930,reference=None):
    if reference is not None and reference.shape!=stats.shape:raise ValueError('Paired records must align')
    rng=np.random.default_rng(seed);values=np.full((n,len(KEYS)),np.nan)
    for b in range(n):
        ix=rng.integers(0,len(stats),len(stats));m=metrics(stats[ix])
        ref=metrics(reference[ix]) if reference is not None else None
        for j,k in enumerate(KEYS):
            if m[k] is not None and (ref is None or ref[k] is not None):
                values[b,j]=m[k]-(ref[k] if ref is not None else 0)
    return {k:np.nanquantile(values[:,j],[.025,.975]).tolist() if np.isfinite(values[:,j]).any() else None
            for j,k in enumerate(KEYS)}
