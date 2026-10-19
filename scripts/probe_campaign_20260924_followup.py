#!/usr/bin/env python3
"""Rebalance the frozen VM/VJ queue and submit individually validated probes."""
import argparse, collections, hashlib, json, os, re, shlex, subprocess, time
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path('/mnt/umm/users/yinbaiqiao/probe_spatial')
BASE=ROOT/'artifacts/main_v2_probe_v2_816_20260917'
SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'
PY='/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python'
def now(): return datetime.now(timezone.utc).isoformat()
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');os.chmod(tmp,0o644);os.replace(tmp,p)
def sco(*args):
    p=subprocess.run([SCO,'acp','jobs',*args],capture_output=True,text=True,timeout=180)
    if p.returncode: raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout
def live():
    jobs={}
    for page in range(1,4):
        rows=json.loads(sco('list','--workspace-name=aigc','--page-size=500',f'--page-token={page}','-o','json'))
        jobs.update({r['name']:r for r in rows})
        if len(rows)<500:break
    return jobs
def create(name,command,pool):
    out=sco('create','--workspace-name=aigc',f'--aec2-name={pool}',f'--job-name={name}',
        '--priority=HIGHEST','--quota-type=reserved','--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617',
        '--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm','--training-framework=pytorch',
        '--worker-nodes=1','--worker-spec=N4lS.Iq.I80.1',f'--command={command}')
    match=re.search(r'job id\s*:\s*(pt-[a-z0-9]+)',out,re.I)
    if not match:raise RuntimeError('ambiguous create response: '+out)
    return match.group(1)
def never_started(j):
    return j.get('state') in {'STARTING','QUEUEING','PENDING'} and not j.get('start_time') and all(s.get('replicas',0)==0 for r in j.get('roles',[]) for s in r.get('resource_spec',[]))
def rebalance():
    root=BASE/'single_gpu_vm72_vj96_20260924_r1';path=root/'zoetrope_half_to_h800_20260924_r1.json'
    if path.exists():
        m=load(path);selected=[(k,None) for k in m['jobs']]
        return resume_rebalance(path,m,selected)
    old=load(root/'h800_to_zoetrope_20260924_r1.json')['jobs'];jobs=live(); groups=collections.defaultdict(list)
    for k,r in old.items():
        j=jobs.get(r['new_job_id'],{})
        if never_started(j) and j.get('resource_pool',{}).get('name')=='zoetrope':groups[k.split('__')[0]].append((k,r))
    selected=[]
    for model,rs in sorted(groups.items()): selected+=sorted(rs)[::2][:len(rs)//2]
    m={'created_utc':now(),'resource_pool':'h800','eligible_by_model':{k:len(v) for k,v in groups.items()},'jobs':{}}
    for k,r in selected:m['jobs'][k]={'old_job_id':r['new_job_id'],'command':r['command'],'new_job_name':'bq-half-h8-'+r['new_job_id'][3:],'state':'SELECTED'}
    save(path,m);print('selected',len(selected),flush=True)
    resume_rebalance(path,m,selected)
def resume_rebalance(path,m,selected):
    def exact_batch(batch):
        from concurrent.futures import ThreadPoolExecutor
        ids=[m['jobs'][k]['old_job_id'] for k,_ in batch]
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows=list(pool.map(lambda jid:json.loads(sco('describe',jid,'--workspace-name=aigc','-o','json')),ids))
        return {r['name']:r for r in rows}
    for off in range(0,len(selected),16):
        batch=selected[off:off+16];fresh=exact_batch(batch);stopping=[]
        for k,_ in batch:
            r=m['jobs'][k];j=fresh.get(r['old_job_id'],{})
            if r['state'] in {'SUBMITTED','PROTECTED_STARTED'}:continue
            if r['state']=='SUBMIT_INTENT':raise RuntimeError('ambiguous submit intent; reconcile exact new name first')
            if r['state']=='STOP_INTENT' and j.get('state') in {'SUSPENDED','STOPPED','CANCELLED'}:continue
            if not never_started(j):r['state']='PROTECTED_STARTED';continue
            stopping.append(r['old_job_id']);r['state']='STOP_INTENT'
        save(path,m)
        if stopping:sco('stop','--workspace-name=aigc',*stopping)
        after=exact_batch(batch)
        for k,_ in batch:
            r=m['jobs'][k]
            if r['state'] in {'PROTECTED_STARTED','SUBMITTED'}:continue
            j=after.get(r['old_job_id'],{})
            if j.get('state') not in {'SUSPENDED','STOPPED','CANCELLED'}:raise RuntimeError('old job not confirmed stopped '+r['old_job_id'])
            if j.get('start_time'):r['state']='STOPPED_AFTER_START_REVIEW';save(path,m);raise RuntimeError('start race detected')
            r['state']='SUBMIT_INTENT';save(path,m)
            r['new_job_id']=create(r['new_job_name'],r['command'],'h800');r['state']='SUBMITTED';r['submitted_utc']=now();save(path,m)
        print('migrated',sum(r['state']=='SUBMITTED' for r in m['jobs'].values()),flush=True)

def prepare(bundle_dir,out,freeze_pending=False):
    bundle_dir=Path(bundle_dir).resolve();out=Path(out).resolve();bundle=load(bundle_dir/'bundle.json');passed=[]
    for e in bundle['cases']:
        p=Path(e['case_path']);rp=p.parent/'result.json'
        r=load(rp) if rp.exists() else {}
        if not freeze_pending and r.get('status')!='PASS':continue
        assert sha(p)==e['case_sha256']
        if not freeze_pending:
            assert e['case_sha256']==r['case_sha256']
            ip=bundle_dir/'groups'/f"{e['shard']:02d}"/'inventory_result.json';inv=load(ip)
            assert inv['status']=='PASS' and inv['bundle_sha256']==sha(bundle_dir/'bundle.json')
        passed.append((e,load(p),rp))
    assert passed
    if (out/'plan.json').exists():raise RuntimeError('production already prepared')
    out.mkdir(parents=True,exist_ok=True);runid='main_v2_'+out.name
    fp16_modes = {str(case.get('environment', {}).get('PROBE_FP16_PASSTHROUGH', '1'))
                  for _, case, _ in passed}
    if len(fp16_modes) != 1:
        raise RuntimeError('production plan cannot mix validated feature-transfer precisions')
    fp16_passthrough = fp16_modes.pop()
    if fp16_passthrough not in {'0', '1'}:
        raise RuntimeError(f'invalid validated feature-transfer precision: {fp16_passthrough!r}')
    template=(ROOT/'runtime/main_v2_probe_v2_wan_ab112_r1/run_single.py').read_text()
    overlay=Path(bundle['overlay']['path']);assert sha(overlay)==bundle['overlay']['sha256']
    template=template.replace('probe_validation_wan_ab112_20260923_r4/overlay/manifest.json',str(overlay.relative_to(BASE)))
    template=template.replace('single_gpu_wan_ab112_20260923_r1',out.name)
    template=template.replace('main_v2_probe_v2_wan_ab112_20260923_r1',runid)
    template=template.replace('main_v2_probe_v2_wan_ab112',runid)
    template=template.replace('53625dd6dcffd039adcf79a2b943a33ed3b3d22b0b4cb6d19e47d6890cfdcbd8',sha(overlay))
    template=template.replace('"PROBE_FP16_PASSTHROUGH": "1"',
                              f'"PROBE_FP16_PASSTHROUGH": "{fp16_passthrough}"')
    runner=out/'run_single.py';runner.write_text(template);jobs=[]
    for idx,(e,c,rp) in enumerate(passed):
        item={**c['experiment'],'execution_manifest':c['full_execution_manifest']}
        p=out/'items'/f'{idx:04d}.jsonl';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(item,sort_keys=True,separators=(',',':'))+'\n')
        jobs.append({'unique_key':e['unique_key'],'model':c['model'],'ordinal':idx,'shard':e['shard'],'item_path':str(p),'item_sha256':sha(p),'case_path':e['case_path'],'case_sha256':e['case_sha256'],'validation_result':str(rp),'validation_result_sha256':sha(rp) if not freeze_pending else None})
    save(out/'plan.json',{'status':'FROZEN_NOT_SUBMITTED','run_id':runid,'runner':str(runner),'runner_sha256':sha(runner),'validation_bundle':str(bundle_dir/'bundle.json'),'validation_bundle_sha256':sha(bundle_dir/'bundle.json'),'feature_transfer_precision':{'PROBE_FP16_PASSTHROUGH':fp16_passthrough},'jobs':jobs})
    print('prepared',len(jobs),out,flush=True)
def submit(out,ready_only=False,pool='zoetrope'):
    import fcntl
    out=Path(out).resolve()
    with (out/'.submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return submit_locked(out,ready_only,pool)
def submit_locked(out,ready_only=False,pool='zoetrope'):
    out=Path(out).resolve();plan=load(out/'plan.json');path=out/'submissions.json';ledger=load(path) if path.exists() else {'resource_pool':pool,'jobs':{}}
    assert sha(plan['runner'])==plan['runner_sha256'] and sha(plan['validation_bundle'])==plan['validation_bundle_sha256']
    for r in plan['jobs']:
        k=r['unique_key']
        if k in ledger['jobs']:continue
        rp=Path(r['validation_result'])
        result=load(rp) if rp.exists() else {}
        if ready_only and result.get('status')!='PASS':continue
        assert result['status']=='PASS' and result['case_sha256']==r['case_sha256']
        if r.get('validation_result_sha256'):assert sha(rp)==r['validation_result_sha256']
        if 'shard' in r:
            inv=load(Path(plan['validation_bundle']).parent/'groups'/f"{r['shard']:02d}"/'inventory_result.json')
            assert inv['status']=='PASS' and inv['bundle_sha256']==plan['validation_bundle_sha256']
        assert sha(r['case_path'])==r['case_sha256'] and sha(r['item_path'])==r['item_sha256']
        name='bq-ready-'+hashlib.sha256((plan['run_id']+k).encode()).hexdigest()[:16]
        command=shlex.join([PY,plan['runner'],'--item',r['item_path'],'--item-sha256',r['item_sha256'],'--run-id',plan['run_id']])
        record={**r,'validation_result_sha256':sha(rp),'job_name':name,'command':command,'state':'SUBMIT_INTENT','submitted_utc':now()};ledger['jobs'][k]=record;save(path,ledger)
        record['job_id']=create(name,command,pool);record['state']='SUBMITTED';save(path,ledger)
        print(k,record['job_id'],flush=True)
    print('submitted',len(ledger['jobs']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['rebalance','prepare','submit']);p.add_argument('--bundle');p.add_argument('--out');p.add_argument('--freeze-pending',action='store_true');p.add_argument('--ready-only',action='store_true');p.add_argument('--pool',choices=['zoetrope','h800'],default='zoetrope');a=p.parse_args()
    if a.action=='rebalance':rebalance()
    elif a.action=='prepare':prepare(a.bundle,a.out,a.freeze_pending)
    else:submit(a.out,a.ready_only,a.pool)
