"""Single-GPU queue: additional measurements first, then source transfer, then adaptation."""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/'results/sepsis_priority_20261010'
NAMES=['sepsis_missions_measurements_20261010','sepsis_missions_source_20261010','sepsis_missions_adapt_20261010']
DATA=['fit.pkl','tune.pkl','cal.pkl','test.pkl','scaler.npz']

def status(stage,**extra):
    obj=dict(stage=stage,updated_local=time.strftime('%Y-%m-%d %H:%M:%S'),**extra)
    p=QUEUE/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
    print(obj,flush=True)

def wait_process(pid):
    # A process handle refers to this process instance even if its PID is later reused.
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if not handle:
        if ctypes.get_last_error()==87:return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        status('waiting_for_existing_experiment',pid=pid)
        while True:
            rc=kernel.WaitForSingleObject(handle,30000)
            if rc==0:return
            if rc!=258:raise ctypes.WinError(ctypes.get_last_error())
    finally:kernel.CloseHandle(handle)

def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def copy_identical(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if hashfile(source)!=hashfile(target):raise RuntimeError(f'Existing artifact differs: {target}')
    else:shutil.copy2(source,target)

def run(config,stage,arm=None):
    label=config.stem+'_'+stage+('_'+arm if arm else '')
    status(stage,phase=config.stem,arm=arm)
    command=[sys.executable,'-u',str(ROOT/'scripts/sepsis_missions.py'),'--config',str(config),stage]
    if arm:command+=['--arm',arm]
    with (QUEUE/f'{label}.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def seed_data(out,source):
    local=out/'local';local.mkdir(parents=True,exist_ok=True)
    if not (out/'data_manifest.json').exists():
        for name in DATA:copy_identical(source/'local'/name,local/name)
        copy_identical(source/'data_manifest.json',out/'data_manifest.json')
        (out/'reused_data.json').write_text(json.dumps({'source':str(source),
            'sha256':{name:hashfile(local/name) for name in DATA}},indent=2))

def reuse_fits(out,source):
    local=out/'local'
    for p in sorted(source.glob('*_seed*.json')):
        r=json.loads(p.read_text())
        if not isinstance(r,dict) or not r.get('complete') or 'initialization' not in r:continue
        name=r['name'];copy_identical(p,out/p.name)
        copy_identical(source/'local'/f'{name}.pt',local/f'{name}.pt')
        for split in ['cal','test']:
            scores=source/'local'/f'{name}_{split}_scores.npz'
            if scores.exists():copy_identical(scores,local/scores.name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int);args=ap.parse_args()
    QUEUE.mkdir(exist_ok=True)
    base=json.loads((ROOT/'configs/sepsis_missions.json').read_text())
    phases=[]
    for index,name in enumerate(NAMES):
        config=dict(base,run_name=name,initializations=['scratch','source','adapt'][:index+1])
        p=ROOT/'configs'/f'{name}.json'
        if p.exists() and json.loads(p.read_text())!=config:raise RuntimeError('Priority configuration changed')
        p.write_text(json.dumps(config,indent=2));phases.append((p,ROOT/'results'/name,config))
    if args.wait_pid:wait_process(args.wait_pid)
    previous=ROOT/'results/rich_transfer_20261009'
    for index,(config,out,c) in enumerate(phases):
        if (out/'seed_summary.json').exists() and (out/'paired_comparisons.json').exists():
            previous=out;continue
        seed_data(out,previous)
        run(config,'prepare')
        if index:reuse_fits(out,previous)
        init=['scratch','source','adapt'][index]
        order=[10,13,34] if index==0 else [34,13,10]
        for seed in c['seeds']:
            for n in order:
                arm=f'{init}_{n}f_{c["training_horizon_hours"]}h_seed{seed}'
                p=out/(arm+'.json')
                if p.exists() and json.loads(p.read_text()).get('complete'):continue
                run(config,'train',arm)
        for stage in ['calibrate','evaluate','summarize']:run(config,stage)
        previous=out
    status('complete',phases=NAMES)

if __name__=='__main__':
    try:main()
    except Exception:
        QUEUE.mkdir(exist_ok=True);status('failed',error=traceback.format_exc());raise
