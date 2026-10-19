#!/usr/bin/env python3
"""Reconcile current formal VM/VJ jobs and move pending work to H800 once."""
import json,time,collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from probe_campaign_20260924_followup import BASE,load,save,sco,create,now,never_started
OUT=BASE/'vmvj_all_h800_20260924_r1';OUT.mkdir(exist_ok=True)
P=OUT/'migration.json'
def describe(jid):return json.loads(sco('describe',jid,'--workspace-name=aigc','-o','json'))
def current():
 d={}
 for root,filename in [('single_gpu_r7_20260922_r2','production_submissions.json'),('single_gpu_vm24_targetfix_20260923_r1','submissions.json'),('single_gpu_vm72_vj96_20260924_r1','submissions.json')]:
  for k,r in load(BASE/root/filename)['jobs'].items():
   if k.split('__')[0] in {'videomae','vjepa2'}:d[k]={**r,'ledger_root':root}
 root=BASE/'single_gpu_vm72_vj96_20260924_r1'
 for fn in ['h800_to_zoetrope_20260924_r1.json','zoetrope_half_to_h800_20260924_r1.json']:
  for k,r in load(root/fn)['jobs'].items():
   if r.get('new_job_id'):d[k].update(job_id=r['new_job_id'],command=r['command'])
 return d
if not P.exists():
 d=current();ids=[r['job_id'] for r in d.values()]
 with ThreadPoolExecutor(max_workers=12) as pool:js=dict(zip(ids,pool.map(describe,ids)))
 save(OUT/'initial_scheduler.json',{'at':now(),'jobs':js})
 m={'created_utc':now(),'resource_pool':'h800','jobs':{},'running_outside_h800':{},'initial_counts':{}}
 cnt=collections.Counter()
 for k,r in d.items():
  j=js[r['job_id']];pool=j['resource_pool']['name'];cnt[(k.split('__')[0],pool,j['state'])]+=1
  if pool=='h800' or j['state']=='SUCCEEDED':continue
  if j['state']=='RUNNING':m['running_outside_h800'][k]=r;continue
  # SUSPENDED here is a stopped half-migration whose replacement has not yet been created.
  if not never_started(j) and not(j['state']=='SUSPENDED' and not j.get('start_time')):continue
  m['jobs'][k]={'old_job_id':j['name'],'command':j['roles'][0]['startup_script'],'new_job_name':'bq-all-h8-'+j['name'][3:],'state':'SELECTED','ledger_root':r['ledger_root']}
 m['initial_counts']={'|'.join(k):v for k,v in cnt.items()};save(P,m)
 print(json.dumps(m['initial_counts']),flush=True);print('pending selected',len(m['jobs']),'running outside H800',len(m['running_outside_h800']),flush=True)
m=load(P)
for k,r in m['jobs'].items():
 if r['state'] in {'SUBMITTED','COMPLETED_BEFORE_STOP','STARTED_BEFORE_STOP'}:continue
 if r['state']=='SUBMIT_INTENT':raise RuntimeError('reconcile ambiguous create first '+k)
 j=describe(r['old_job_id'])
 if j['state']=='SUCCEEDED':r['state']='COMPLETED_BEFORE_STOP';save(P,m);continue
 if j.get('start_time'):r['state']='STARTED_BEFORE_STOP';save(P,m);continue
 if j['state'] not in {'SUSPENDED','STOPPED','CANCELLED'}:
  assert never_started(j),j['state'];r['state']='STOP_INTENT';save(P,m);sco('stop','--workspace-name=aigc',r['old_job_id'])
  for attempt in range(60):
   j=describe(r['old_job_id'])
   if j['state'] in {'SUSPENDED','STOPPED','CANCELLED'}:break
   time.sleep(2)
  else:raise RuntimeError('stop still pending '+r['old_job_id'])
 assert not j.get('start_time'),'scheduler start race; manual review required'
 r['state']='SUBMIT_INTENT';save(P,m)
 r['new_job_id']=create(r['new_job_name'],r['command'],'h800');r['state']='SUBMITTED';r['submitted_utc']=now();save(P,m)
 print('migrated',sum(v['state']=='SUBMITTED' for v in m['jobs'].values()),'/',len(m['jobs']),flush=True)
print('done',flush=True)
