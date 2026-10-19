#!/usr/bin/env python3
"""Move exact never-started H800 jobs from one submission ledger to Zoetrope."""
import argparse, concurrent.futures, json, os, re, subprocess, tempfile, threading
from datetime import datetime, timezone
from pathlib import Path

SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'; WORKSPACE='aigc'
LOCK=threading.Lock(); JOB_RE=re.compile(r'job id\s*:\s*(pt-[a-z0-9]+)',re.I)

def now(): return datetime.now(timezone.utc).isoformat()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f'.{path.name}.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally: Path(tmp).unlink(missing_ok=True)
def live_jobs():
    p=subprocess.run([SCO,'acp','jobs','list',f'--workspace-name={WORKSPACE}','--page-size=500','-o','json'],text=True,capture_output=True,check=True,timeout=180)
    return {j['name']:j for j in json.loads(p.stdout)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-ledger',type=Path,required=True); ap.add_argument('--migration-ledger',type=Path,required=True)
    ap.add_argument('--name-prefix',required=True); ap.add_argument('--execute',action='store_true'); ap.add_argument('--parallel',type=int,default=8); a=ap.parse_args()
    if not re.fullmatch(r'[a-z0-9-]{1,28}',a.name_prefix): raise ValueError('invalid name prefix')
    source=json.loads(a.source_ledger.read_text())['jobs']; live=live_jobs(); eligible=[]
    for index,(key,record) in enumerate(source.items()):
        job=live.get(record.get('job_id'))
        if job and job.get('state')=='STARTING' and not job.get('start_time') and (job.get('resource_pool') or {}).get('name')=='h800':
            eligible.append((key,record,job,index))
    eligible.sort(key=lambda x:(x[1].get('ordinal',x[1].get('shard',x[3])),x[0]))
    if a.migration_ledger.exists():
        migration=json.loads(a.migration_ledger.read_text())
        if {r['old_job_id'] for r in migration['jobs'].values()} != {j['name'] for _,_,j,_ in eligible}: raise RuntimeError('existing migration scope differs from fresh never-started scope')
    else:
        migration={'schema':'H800_STARTING_TO_ZOETROPE_MIGRATION_V1','created_utc':now(),'source_ledger':str(a.source_ledger.resolve()),'selection_rule':'H800 STARTING with no start_time in fresh scheduler snapshot','resource_pool':'zoetrope','jobs':{}}
        for key,record,job,index in eligible:
            migration['jobs'][key]={'unique_key':key,'ordinal':record.get('ordinal',record.get('shard',index)),'command':record['command'],'old_job_id':job['name'],'old_job_name':job['display_name'],'old_state':job['state'],'old_start_time':job.get('start_time'),'migration_state':'SELECTED'}
        save(a.migration_ledger,migration)
    print(json.dumps({'selected':len(migration['jobs']),'execute':a.execute}))
    if not a.execute:return
    stopping=[r for r in migration['jobs'].values() if r['migration_state']=='SELECTED']
    for off in range(0,len(stopping),20):
        batch=stopping[off:off+20]; p=subprocess.run([SCO,'acp','jobs','stop',f'--workspace-name={WORKSPACE}',*[r['old_job_id'] for r in batch]],text=True,capture_output=True,timeout=180)
        if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
        for r in batch:r.update(migration_state='H800_STOP_REQUESTED',stop_requested_utc=now(),stop_stdout=p.stdout,stop_stderr=p.stderr)
        save(a.migration_ledger,migration)
    def submit(row):
        name=f"{a.name_prefix}-{int(row['ordinal']):03d}-{row['old_job_id'][3:9]}"
        argv=[SCO,'acp','jobs','create',f'--workspace-name={WORKSPACE}','--aec2-name=zoetrope',f'--job-name={name}','--priority=HIGHEST','--quota-type=reserved','--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617','--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm','--training-framework=pytorch','--worker-nodes=1','--worker-spec=N4lS.Iq.I80.1',f"--command={row['command']}"]
        p=subprocess.run(argv,text=True,capture_output=True,timeout=180); m=JOB_RE.search(p.stdout+'\n'+p.stderr)
        return {**row,'new_job_name':name,'new_job_id':m.group(1) if m else None,'submit_returncode':p.returncode,'submit_stdout':p.stdout,'submit_stderr':p.stderr,'submitted_utc':now(),'migration_state':'ZOETROPE_SUBMITTED' if p.returncode==0 and m else 'ZOETROPE_SUBMISSION_FAILED'}
    pending=[(key,row) for key,row in migration['jobs'].items() if row['migration_state'] in {'H800_STOP_REQUESTED','ZOETROPE_SUBMISSION_FAILED'}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,a.parallel)) as pool:
        futures={pool.submit(submit,row):key for key,row in pending}
        for f in concurrent.futures.as_completed(futures):
            key=futures[f]; migration['jobs'][key]=f.result(); migration['updated_utc']=now()
            with LOCK:save(a.migration_ledger,migration)
            print(json.dumps({'key':key,'new_job_id':migration['jobs'][key]['new_job_id'],'state':migration['jobs'][key]['migration_state']}),flush=True)
    failed=[r for r in migration['jobs'].values() if r['migration_state']!='ZOETROPE_SUBMITTED']
    print(json.dumps({'selected':len(migration['jobs']),'submitted':len(migration['jobs'])-len(failed),'failures':len(failed)}))
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
