#!/usr/bin/env python3
"""Take over one existing LingBot target owner only once eight GPUs are allocated."""
import os,sys,json,time,subprocess
from pathlib import Path
from probe_campaign_20260924_followup import ROOT,BASE,load,save,sha,sco,now
OUT=BASE/'lingbot_target_accelerate_20260924_r1'
def main(i):
 p=load(OUT/'plan.json')['owners'][i]
 for b in p['bindings']:assert sha(b['path'])==b['sha256'],b['path']
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).splitlines();assert len(gpu)==8
 runner=p['runner'];batch=p['batch'];py='/mnt/umm/users/yinbaiqiao/.conda/envs/lingbot-video/bin/python'
 subprocess.run([py,runner,'--batch',batch,'--trust-current-tools'],check=True,stdout=subprocess.DEVNULL)
 status=OUT/f'owner{i:02}_status.json'
 def desc():return json.loads(sco('describe',p['old_job_id'],'--workspace-name=aigc','-o','json'))
 j=desc();assert p['old_script'] in j['roles'][0]['startup_script']
 save(status,{'at':now(),'state':'HANDOFF_ALLOCATED','old_job_id':p['old_job_id'],'old_state':j['state']})
 if j['state'] not in {'SUCCEEDED','FAILED','STOPPED','SUSPENDED','CANCELLED'}:
  sco('stop','--workspace-name=aigc',p['old_job_id'])
  for attempt in range(120):
   j=desc()
   if j['state'] in {'SUCCEEDED','FAILED','STOPPED','SUSPENDED','CANCELLED'}:break
   time.sleep(3)
  else:raise RuntimeError('old owner not stopped; refusing concurrent extraction')
 save(status,{'at':now(),'state':'RESUMING_8GPU','old_job_id':p['old_job_id'],'old_state':j['state']})
 env=os.environ.copy();env['GAP_N_GPUS']='8'
 with (OUT/f'owner{i:02}.log').open('a') as f:r=subprocess.run([py,runner,'--batch',batch,'--execute','--trust-current-tools'],env=env,stdout=f,stderr=subprocess.STDOUT)
 save(status,{'at':now(),'state':'PASS' if r.returncode==0 else 'FAILED','returncode':r.returncode})
 return r.returncode
if __name__=='__main__':sys.exit(main(int(sys.argv[1])))
