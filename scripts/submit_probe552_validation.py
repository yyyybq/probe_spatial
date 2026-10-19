#!/usr/bin/env python3
"""Submit the user's 40 validations as eight isolated one-GPU groups, once."""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/main_v2_probe_v2_816_20260917/probe552_validation_40_20260923_r1'
RUNNER=ROOT/'scripts/run_probe552_validation.py'
SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'
PYTHON='/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--bundle',type=Path,default=OUT/'bundle.json')
    parser.add_argument('--tag',default='r1')
    parser.add_argument('--aec2-name',choices=('zoetrope','h800'),default='zoetrope')
    parser.add_argument('--skip-progress-refresh',action='store_true')
    parser.add_argument('--runner',type=Path,default=RUNNER)
    parser.add_argument('--shard',type=int,action='append',dest='shards')
    a=parser.parse_args()
    OUT=a.bundle.resolve().parent
    assert re.fullmatch(r'[a-z0-9-]{1,16}',a.tag)
    bundle=json.loads((OUT/'bundle.json').read_text())
    assert bundle['cases_count'] > 0 and bundle['shard_count'] > 0
    path=OUT/'submissions.json'
    ledger=json.loads(path.read_text()) if path.exists() else dict(schema='PROBE552_VALIDATION_SUBMISSIONS_V1',jobs={})
    shards=sorted(set(a.shards)) if a.shards else list(range(bundle['shard_count']))
    if any(shard < 0 or shard >= bundle['shard_count'] for shard in shards):
        raise RuntimeError('requested shard is outside the bundle')
    for shard in shards:
        previous=ledger['jobs'].get(str(shard))
        if previous and not (previous.get('state')=='SUBMISSION_FAILED' and not previous.get('job_id')):
            continue
        runner=a.runner.resolve()
        command=shlex.join([PYTHON,str(runner),'--shard',str(shard),'--bundle',str(OUT/'bundle.json'),'--bundle-sha256',sha(OUT/'bundle.json'),'--runner-sha256',sha(runner)])
        name=f'bq-p552-val{bundle["cases_count"]}-s{shard:02d}-0923-{a.tag}'
        argv=[SCO,'acp','jobs','create','--workspace-name=aigc',f'--aec2-name={a.aec2_name}',f'--job-name={name}',
            '--priority=HIGHEST','--quota-type=reserved','--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617',
            '--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm','--training-framework=pytorch',
            '--worker-nodes=1','--worker-spec=N4lS.Iq.I80.1',f'--command={command}']
        if not a.submit:
            print(json.dumps(dict(shard=shard,name=name,command=command,cases=sum(c['shard']==shard for c in bundle['cases']))))
            continue
        # Persist intent before contacting scheduler. An interrupted/ambiguous
        # request is reconciled by exact job name, never blindly resubmitted.
        record=dict(shard=shard,name=name,state='SUBMITTING',command=command,submitted_utc=datetime.now(timezone.utc).isoformat())
        if previous:
            record['previous_failed_attempts']=previous.get('previous_failed_attempts',[])+[previous]
        ledger['jobs'][str(shard)]=record;save(path,ledger)
        p=subprocess.run(argv,text=True,capture_output=True,timeout=120)
        text=p.stdout+'\n'+p.stderr
        ids=list(dict.fromkeys(re.findall(r'pt-[a-z0-9]+',text)))
        record.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,job_id=ids[0] if len(ids)==1 else None,
                      state='SUBMITTED' if p.returncode==0 and len(ids)==1 else 'SUBMISSION_FAILED')
        save(path,ledger)
        if not a.skip_progress_refresh:
            subprocess.run([PYTHON,str(ROOT/'scripts/update_probe552_progress.py')],check=True)
        print(json.dumps(dict(shard=shard,job_id=record['job_id'],state=record['state'])),flush=True)
        if record['state']!='SUBMITTED':
            raise RuntimeError(f'shard {shard} submission failed; inspect ledger before retry')

if __name__=='__main__':
    main()
