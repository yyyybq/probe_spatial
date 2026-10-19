#!/usr/bin/env python3
import json,shlex,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from probe_campaign_20260924_followup import *
OUT=BASE/'validation_release_20260924_r1';OUT.mkdir(exist_ok=True);ledgerpath=OUT/'replacements.json'
def desc(jid):return json.loads(sco('describe',jid,'--workspace-name=aigc','-o','json'))
def command(bundle,out,shard,repair=False):
 runner=ROOT/'scripts/run_probe_hybrid_validation.py';release=ROOT/'scripts/probe_campaign_20260924_followup.py'
 args=[PY,str(ROOT/'scripts/run_validation_release_20260924.py'),str(bundle),str(out),str(shard),sha(bundle),sha(runner),sha(release)]
 if repair:args.append(sha(ROOT/'scripts/repair_lingbot_uri8_20260924.py'))
 return shlex.join(args)
if not ledgerpath.exists():
 wan=BASE/'probe_validation_wan_c56_20260924_r2';w=load(wan/'submissions.json');r=load(BASE/'lingbot_uri8_repair_20260924_r1/submission.json');jobs={}
 for shard,v in w['jobs'].items():jobs['wan-'+shard]={'old_job_id':v['job_id'],'command':command(wan/'bundle.json',BASE/'single_gpu_wan_c56_20260924_r2',shard),'state':'SELECTED','ledger':str(wan/'submissions.json'),'shard':shard}
 jobs['lingbot-repair']={'old_job_id':r['job_id'],'command':command(BASE/'probe_validation_lingbot32_20260924_r2/bundle.json',BASE/'single_gpu_lingbot32_20260924_r2','all',True),'state':'SELECTED'}
 save(ledgerpath,{'created_utc':now(),'jobs':jobs})
d=load(ledgerpath)
ids=[r['old_job_id'] for r in d['jobs'].values()]
with ThreadPoolExecutor(max_workers=10) as p:states=dict(zip(ids,p.map(desc,ids)))
stopping=[]
for k,r in d['jobs'].items():
 if r['state']=='SUBMITTED':continue
 j=states[r['old_job_id']]
 if r['state']=='STOP_INTENT' and j['state'] in {'SUSPENDED','STOPPED','CANCELLED'}:continue
 if not never_started(j):r['state']='PROTECTED_STARTED';continue
 r['state']='STOP_INTENT';stopping.append(j['name'])
save(ledgerpath,d)
if stopping:sco('stop','--workspace-name=aigc',*stopping)
for k,r in d['jobs'].items():
 if r['state'] in {'SUBMITTED','PROTECTED_STARTED'}:continue
 assert r['state']=='STOP_INTENT'
 for attempt in range(120):
  j=desc(r['old_job_id'])
  if j['state'] in {'SUSPENDED','STOPPED','CANCELLED'}:break
  time.sleep(2)
 else:raise RuntimeError('stop pending')
 assert not j.get('start_time')
 r['state']='SUBMIT_INTENT';save(ledgerpath,d)
 r['new_job_id']=create('bq-valrelease-'+r['old_job_id'][3:],r['command'],'zoetrope');r['state']='SUBMITTED';r['at']=now();save(ledgerpath,d)
 if 'ledger' in r:
  path=Path(r['ledger']);w=load(path);old=w['jobs'][r['shard']];w['jobs'][r['shard']]={**old,'prior_attempt':old,'job_id':r['new_job_id'],'command':r['command'],'state':'SUBMITTED','submitted_utc':now()};save(path,w)
 else:
  path=BASE/'lingbot_uri8_repair_20260924_r1/submission.json';old=load(path);save(path,{**old,'prior_attempt':old,'job_id':r['new_job_id'],'command':r['command'],'state':'SUBMITTED','at':now()})
 print(k,r['new_job_id'],flush=True)
