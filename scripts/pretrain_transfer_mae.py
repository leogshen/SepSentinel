"""Three local source MAE fits, resumable per seed, no target data loaded."""
from pathlib import Path
import argparse
import copy
import json
import pickle
import time
import numpy as np
import torch
from transfer_sequence import MAE,SourceSampler,masked_batch,reconstruction_loss,verify_contract

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/transfer_protocol'
LOCAL=OUT/'local'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='42,123,456');ap.add_argument('--steps',type=int,default=20000)
    args=ap.parse_args()
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    checks=verify_contract();(OUT/'sequence_contract_checks.json').write_text(json.dumps(checks,indent=2))
    data=pickle.load(open(LOCAL/'source_splits.pkl','rb'))
    norm=np.load(LOCAL/'source_scaler.npz');median,std=norm['median'],norm['std']
    train={k:v['train'] for k,v in data.items()};val={k:v['val'] for k,v in data.items()}
    valid=SourceSampler(val,930).batch(256)
    for seed in map(int,args.seeds.split(',')):
        destination=LOCAL/f'mae_seed{seed}.pt';record=OUT/f'mae_seed{seed}.json'
        if record.exists() and json.loads(record.read_text()).get('complete'): continue
        torch.manual_seed(seed);np.random.seed(seed)
        model=MAE().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        sampler=SourceSampler(train,seed);rng=np.random.default_rng(seed)
        best=float('inf');stale=0;history=[];start=time.time()
        report={'seed':seed,'gpu':torch.cuda.get_device_name(0),'steps_limit':args.steps,'batch_size':32,
                'target_records_used':0,'parameters':sum(p.numel() for p in model.parameters()),'complete':False}
        for step in range(1,args.steps+1):
            model.train()
            items=sampler.batch(32);batch,y,mask=masked_batch(items,median,std,rng)
            pred=model(*batch);loss=reconstruction_loss(pred,y,mask)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if step==10: print(f'seed {seed} 10-step timing {(time.time()-start)/10:.3f}s/update',flush=True)
            if step%500!=0 and step!=args.steps: continue
            model.eval();vrng=np.random.default_rng(930);losses=[];baseline=[]
            numer=np.zeros(10);denom=np.zeros(10)
            with torch.no_grad():
                for i in range(0,len(valid),32):
                    b,vy,vm=masked_batch(valid[i:i+32],median,std,vrng)
                    vp=model(*b);vl=reconstruction_loss(vp,vy,vm)
                    losses.append(float(vl));baseline.append(float(reconstruction_loss(b[0][...,0],vy,vm)))
                    err=torch.nn.functional.huber_loss(vp,vy,reduction='none')
                    numer+=(err*vm).sum((0,1)).cpu().numpy();denom+=vm.sum((0,1)).cpu().numpy()
            score=float(np.mean(losses))
            row={'step':step,'validation_huber':score,'carry_huber':float(np.mean(baseline)),
                 'by_channel_huber':(numer/np.maximum(denom,1)).tolist(),'minutes':(time.time()-start)/60}
            history.append(row)
            if score<best-1e-5:
                best=score;stale=0
                torch.save({'encoder':model.encoder.state_dict(),'model':model.state_dict(),'step':step,'seed':seed},destination)
            else: stale+=1
            report.update(history=history,best_validation=best,elapsed_minutes=(time.time()-start)/60)
            record.write_text(json.dumps(report,indent=2))
            print(f'seed {seed} step {step}: MAE {score:.4f}, carry {np.mean(baseline):.4f}, {row["minutes"]:.1f}m',flush=True)
            if stale>=5: break
        report.update(complete=True,elapsed_minutes=(time.time()-start)/60,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        record.write_text(json.dumps(report,indent=2))
        del model,optimizer;torch.cuda.empty_cache()
        print(f'seed {seed} complete',flush=True)

if __name__=='__main__': main()
