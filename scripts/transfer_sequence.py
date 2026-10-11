"""Small causal, observation-aware clinical sequence encoder and MAE tools."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn

FEATURES=['heart_rate','spo2','respiratory_rate','temperature','lactate','ph',
          'creatinine','wbc','platelets','bilirubin']

def inputs(raw, times, median, std, active=None):
    """Rebuild every carried value/age from the supplied (possibly masked) raw events."""
    x=np.asarray(raw,np.float32).copy(); t=np.asarray(times,np.float32)
    if active is not None: x[:,~np.asarray(active,bool)]=np.nan
    obs=np.isfinite(x); T,F=x.shape
    rows=np.arange(T)[:,None]
    last=np.maximum.accumulate(np.where(obs,rows,-1),axis=0)
    ix=np.maximum(last,0)
    filled=x[ix,np.arange(F)]
    ever=last>=0
    filled=np.where(ever,filled,median)
    age=np.where(ever,t[:,None]-t[ix],336.)
    value=np.clip((filled-median)/std,-10,10)
    return np.stack([value,obs.astype(float),ever.astype(float),
                     np.log1p(np.clip(age,0,336))/math.log(337)],axis=-1).astype(np.float32)

def collate(items,median,std,device='cuda',active=None):
    n=len(items);length=max(len(e['x']) for e in items);F=len(median)
    a=np.zeros((n,length,F,4),np.float32)
    times=np.zeros((n,length),np.float32)
    lengths=[]; kinds=[]
    for i,e in enumerate(items):
        l=len(e['x']);lengths.append(l)
        a[i,:l]=inputs(e['x'],e['t'],median,std,active)
        times[i,:l]=e['t']
        kinds.append(e.get('kind',np.zeros(F,np.int64)))
    return (torch.as_tensor(a,device=device),torch.as_tensor(times,device=device),
            torch.as_tensor(lengths,device=device),torch.as_tensor(np.stack(kinds),device=device))

class Encoder(nn.Module):
    def __init__(self,width=96):
        super().__init__()
        self.feature_id=nn.Embedding(10,32)
        self.kind=nn.Embedding(4,32)
        self.value=nn.Sequential(nn.Linear(4,32),nn.GELU(),nn.Linear(32,32))
        self.pool=nn.Linear(32,width)
        self.time_gap=nn.Linear(2,width)
        layer=nn.TransformerEncoderLayer(width,4,width*4,.1,activation='gelu',batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(width)
        self.register_buffer('frequency',torch.exp(torch.arange(0,width,2)*(-math.log(10000)/width)))
    def forward(self,x,t,lengths,kind,active=None):
        b,T,F,_=x.shape
        # A nonlinear interaction MUST precede pooling. Purely additive feature
        # IDs followed by a mean lose the binding between analyte and value.
        v=torch.nn.functional.gelu(self.value(x)+self.feature_id.weight[None,None,:,:]+self.kind(kind)[:,None,:,:])
        if active is None: h=v.mean(2)
        else:
            w=torch.as_tensor(active,dtype=x.dtype,device=x.device)
            h=(v*w[None,None,:,None]).sum(2)/w.sum().clamp_min(1)
        h=self.pool(h)
        angle=t[:,:,None]*self.frequency[None,None,:]
        pos=torch.stack((angle.sin(),angle.cos()),dim=-1).flatten(-2)
        gap=torch.cat([torch.zeros_like(t[:,:1]),t[:,1:]-t[:,:-1]],dim=1)
        timepair=torch.stack([torch.log1p(t.clamp_min(0))/math.log(337),torch.log1p(gap.clamp_min(0))/math.log(337)],dim=-1)
        h=h+pos+self.time_gap(timepair)
        causal=torch.ones(T,T,device=x.device,dtype=torch.bool).triu(1)
        padding=torch.arange(T,device=x.device)[None,:]>=lengths[:,None]
        h=self.temporal(h,mask=causal,src_key_padding_mask=padding)
        return self.norm(h)
    def embedding(self,*args,active=None):
        h=self(*args,active=active)
        return h[torch.arange(len(h),device=h.device),args[2]-1]

class MAE(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder()
        self.decoder=nn.Sequential(nn.Linear(96,96),nn.GELU(),nn.Linear(96,10))
    def forward(self,*args): return self.decoder(self.encoder(*args))

class SepsisModel(nn.Module):
    def __init__(self,active):
        super().__init__();self.encoder=Encoder()
        self.head=nn.Sequential(nn.Linear(100,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,1))
        self.active=np.asarray(active,bool)
    def forward(self,*args,static):
        z=self.encoder.embedding(*args,active=self.active)
        return self.head(torch.cat([z,static],dim=-1)).squeeze(-1)

def hide_observations(raw,rng):
    """40% real targets, with spans and channel-drop candidates preferred."""
    obs=np.isfinite(raw);T,F=raw.shape
    count=int(np.ceil(.4*obs.sum()))
    mask=np.zeros_like(obs)
    if not count: return raw.copy(),mask
    # Budget: half spans, quarter channel masks, quarter independent points.
    for goal,mode in [(int(count*.5),'span'),(int(count*.75),'channel'),(count,'point')]:
        attempts=0
        while mask.sum()<goal and attempts<100:
            candidate=np.zeros_like(mask)
            if mode=='point': candidate=obs.copy()
            elif mode=='channel': candidate[:,int(rng.integers(F))]=True
            else:
                start=int(rng.integers(T));width=int(rng.integers(6,13))
                candidate[start:min(T,start+width),int(rng.integers(F))]=True
            candidates=np.flatnonzero(candidate&obs&~mask)
            if len(candidates):
                take=min(goal-int(mask.sum()),len(candidates))
                mask.flat[rng.choice(candidates,size=take,replace=False)]=True
            attempts+=1
    remaining=np.flatnonzero(obs&~mask)
    if mask.sum()<count:
        mask.flat[rng.choice(remaining,size=count-int(mask.sum()),replace=False)]=True
    x=raw.copy();x[mask]=np.nan
    return x,mask

def masked_batch(items,median,std,rng,device='cuda'):
    hidden=[]; masks=[]
    for e in items:
        x,m=hide_observations(e['x'],rng);hidden.append(dict(e,x=x));masks.append(m)
    batch=collate(hidden,median,std,device)
    n,T,F,_=batch[0].shape
    y=np.zeros((n,T,F),np.float32);m=np.zeros((n,T,F),bool)
    for i,e in enumerate(items):
        l=len(e['x']);y[i,:l]=np.nan_to_num(np.clip((e['x']-median)/std,-10,10))
        m[i,:l]=masks[i]
    return batch,torch.as_tensor(y,device=device),torch.as_tensor(m,device=device)

def reconstruction_loss(pred,y,mask):
    element=torch.nn.functional.huber_loss(pred,y,reduction='none')
    n=mask.sum(1)
    channel=(element*mask).sum(1)/n.clamp_min(1)
    patient=(channel*(n>0)).sum(1)/(n>0).sum(1).clamp_min(1)
    return patient.mean()

class SourceSampler:
    def __init__(self,data,seed):
        self.rng=np.random.default_rng(seed)
        self.sources=[]
        for name in sorted(data):
            persons={}
            for e in data[name]:persons.setdefault(e['person'],[]).append(e)
            self.sources.append(list(persons.values()))
    def batch(self,n):
        out=[]
        for i in range(n):
            persons=self.sources[int(self.rng.integers(len(self.sources)))]
            eps=persons[int(self.rng.integers(len(persons)))]
            e=eps[int(self.rng.integers(len(eps)))];T=len(e['x'])
            length=min(T,int(self.rng.integers(6,169)))
            end=int(self.rng.integers(length,T+1));start=end-length
            out.append(dict(e,x=e['x'][start:end],t=e['t'][start:end]))
        return out

def verify_contract():
    """Meaningful corruption, causality and padding checks before any training."""
    torch.manual_seed(8);rng=np.random.default_rng(8)
    x=rng.normal(size=(20,10)).astype(np.float32);t=np.arange(1,21,dtype=np.float32)
    m=np.zeros(10,np.float32);s=np.ones(10,np.float32)
    hidden=x.copy();hidden[5,0]=np.nan;hidden[6:,0]=np.nan
    assert np.all(inputs(hidden,t,m,s)[5:,0,0]==x[4,0])
    a,mask=hide_observations(x,rng)
    assert np.isnan(a[mask]).all() and mask.sum()==80
    model=Encoder().eval()
    short={'x':x[:10],'t':t[:10]};long={'x':x,'t':t}
    with torch.no_grad():
        full=model(*collate([long],m,s,'cpu'))[0,:10]
        prefix=model(*collate([short],m,s,'cpu'))[0]
        padded=model(*collate([short,long],m,s,'cpu'))[0,:10]
        swapped=x.copy();swapped[:,[0,3]]=swapped[:,[3,0]]
        other=model(*collate([{'x':swapped,'t':t}],m,s,'cpu'))[0]
    assert torch.allclose(full,prefix,atol=1e-5), 'future input changed prefix'
    assert torch.allclose(padded,prefix,atol=1e-5), 'padding changed valid states'
    assert not torch.allclose(other,model(*collate([long],m,s,'cpu'))[0],atol=1e-5), 'feature/value binding lost'
    return {'hidden_observations_removed':True,'carry_recomputed':True,'causal_prefix_invariance':True,'padding_invariance':True,'feature_value_binding':True}
