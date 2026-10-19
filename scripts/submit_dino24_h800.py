#!/usr/bin/env python3
"""Submit the 24 fully validated DINO probes as one-GPU H800 jobs."""
import argparse, hashlib, json, os, re, shlex, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/main_v2_probe_v2_816_20260917/single_gpu_dino24_20260923_r1'
PLAN=OUT/'plan.json'; RUNNER=ROOT/'runtime/main_v2_probe_v2_dino24_r1/run_single.py'
SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'; PY='/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python'
RUN_ID='main_v2_probe_v2_dino24_20260923_r1'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n'); os.replace(t,p)

def main():
    a=argparse.ArgumentParser(); a.add_argument('--submit',action='store_true'); args=a.parse_args()
    plan=json.loads(PLAN.read_text()); assert plan['status']=='FROZEN_NOT_SUBMITTED' and len(plan['jobs'])==24
    ledger_path=OUT/'submissions.json'; ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        'schema':'DINO24_H800_PRODUCTION_SUBMISSIONS_V1','resource_pool':'h800','jobs':{}}
    for row in plan['jobs']:
        key=row['unique_key']
        if key in ledger['jobs'] and ledger['jobs'][key].get('state')!='SUBMISSION_FAILED': continue
        command=shlex.join([PY,str(RUNNER),'--item',row['item_path'],'--item-sha256',row['item_sha256'],'--run-id',RUN_ID])
        name=f"bq-dino24-p{row['ordinal']:02d}-{row['item_sha256'][:6]}"
        if not args.submit:
            print(json.dumps({'key':key,'name':name,'command':command})); continue
        record={**row,'job_name':name,'command':command,'state':'SUBMITTING','submitted_utc':datetime.now(timezone.utc).isoformat()}
        ledger['jobs'][key]=record; save(ledger_path,ledger)
        argv=[SCO,'acp','jobs','create','--workspace-name=aigc','--aec2-name=h800',f'--job-name={name}',
              '--priority=HIGHEST','--quota-type=reserved','--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617',
              '--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm','--training-framework=pytorch',
              '--worker-nodes=1','--worker-spec=N4lS.Iq.I80.1',f'--command={command}']
        p=subprocess.run(argv,text=True,capture_output=True,timeout=120); ids=list(dict.fromkeys(re.findall(r'pt-[a-z0-9]+',p.stdout+'\n'+p.stderr)))
        record.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,job_id=ids[0] if len(ids)==1 else None,
                      state='SUBMITTED' if p.returncode==0 and len(ids)==1 else 'SUBMISSION_FAILED')
        save(ledger_path,ledger); print(json.dumps({'unique_key':key,'job_id':record['job_id'],'state':record['state']}),flush=True)
        if record['state']!='SUBMITTED': raise RuntimeError(f'submission failed for {key}')

if __name__=='__main__': main()
