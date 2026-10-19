"""Frozen, explicit MiniMax handoff. No CogVideoX/sync jobs are stopped here."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/main_v2_probe_v2_816_20260917/upload_recovery_20260923_r1'
RELEASE=ROOT/'runtime/feature_upload_recovery_v1'
SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'
JOBS=['pt-4lma0exb','pt-z7wd113c','pt-ili01g8p','pt-9cozgsr6','pt-d7l8a52q','pt-q0j47v8u','pt-yrla3xen','pt-k6luh9rd']
TERMINAL={'STOPPED','CANCELLED','CANCELED','DELETED','FAILED','SUCCEEDED'}


def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2)+'\n')


def describe(job):
    result=subprocess.run([SCO,'acp','jobs','describe',job,'--workspace-name','aigc','-o','json'],capture_output=True,text=True,timeout=30)
    if result.returncode:
        deletion=BASE/'delete_commands'/f'{job}.json'
        evidence=result.stdout+result.stderr
        if (deletion.exists() and load(deletion)['returncode']==0 and
                'status code: 404' in evidence and 'trainingJobNotFound' in evidence and f'({job})' in evidence):
            return {'name':job,'state':'DELETED','verification':'explicit deletion plus subsequent scheduler 404',
                    'response':evidence,'roles':load(BASE/'scheduler_before'/job/'job.json')['roles']}
        raise RuntimeError(f'describe {job}: {evidence}')
    return json.loads(result.stdout)


def freeze():
    assert not (RELEASE/'pack.json').exists(), 'release already frozen'
    pub_sha=sha(RELEASE/'minimax_publish.py')
    for mode in ('streaming','target'):
        c=load(BASE/'process_control/pt-4lma0exb'/f'upload-canary-{mode}.json')
        assert c['status']=='UPLOAD_CANARY_PASS' and c['publish_sha256']==pub_sha
        log=(BASE/'process_control/pt-4lma0exb'/f'upload-canary-{mode}.log').read_text()
        assert 'cleanup_canary_pass' in log
    jobs={}
    for shard,job in enumerate(JOBS):
        original=load(BASE/'scheduler_before'/job/'job.json')
        paused=load(BASE/'process_control'/job/'paused.json')
        assert paused['after']['state']=='T' and paused['model']=='minimax_h3'
        assert not paused['sync_processes_not_signalled']
        argv=paused['after']['argv']
        assert argv[argv.index('--mode')+1]=='streaming_prefix'
        assert argv[argv.index('--vfm')+1]=='minimax_h3'
        cmd=original['roles'][0]['startup_script']
        assert f'SHARD_INDEX={shard} SHARD_COUNT=8' in cmd
        assert original['resource_pool']['name']=='zoetrope'
        jobs[job]={'shard':shard,'streaming_argv':argv,'original_scheduler':original,
                   'old_command_sha256':paused['after']['command_sha256']}
    pack={'route_sha256':'0a41e79abcf5326300147cd33611f2e141e0185de73bd0451002ccb8309b1206',
          'backend_source_tree_sha256':'7183cd9b6bd7f147b62b1193bdf3588a4a7eef750f856d54a03d48a21116997f',
          'files':{name:sha(RELEASE/name) for name in ('minimax_publish.py','minimax_resume.py')},'jobs':jobs,
          'protocol_changes':[],'operational_changes':['verified remote done-index','verified upload before payload cleanup','disk/backlog backpressure']}
    save(RELEASE/'pack.json',pack)
    print(json.dumps({'pack':str(RELEASE/'pack.json'),'sha256':sha(RELEASE/'pack.json'),'jobs':len(jobs)}))


def stop_one(job):
    assert job in JOBS
    pack=load(RELEASE/'pack.json')
    current=describe(job)
    assert current['roles'][0]['startup_script']==pack['jobs'][job]['original_scheduler']['roles'][0]['startup_script']
    if current['state'] not in TERMINAL and current['state']!='SUSPENDED':
        assert current['state']=='RUNNING'
        subprocess.run(['python3',str(ROOT/'scripts/feature_upload_recovery_remote_20260923.py'),'inspect',job],check=True,timeout=120)
        inspected=load(BASE/'process_control'/job/'inspection.json')
        assert len(inspected['candidates'])==1 and inspected['candidates'][0]['state']=='T'
        assert inspected['candidates'][0]['command_sha256']==pack['jobs'][job]['old_command_sha256']
        assert not inspected['sync_processes_not_signalled'] and not inspected['publisher_diagnostics']
        result=subprocess.run([SCO,'acp','jobs','stop',job,'--workspace-name','aigc'],capture_output=True,text=True,timeout=30)
        save(BASE/'stop_commands'/f'{job}.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        assert result.returncode==0
    for _ in range(18):
        current=describe(job)
        if current['state']=='SUSPENDED':
            workers=subprocess.check_output([SCO,'acp','jobs','get-workers',job,'--workspace-name','aigc'],text=True,timeout=30)
            rows=[line for line in workers.splitlines() if re.search(r'pt-[0-9a-f]+-worker-\d+',line)]
            if rows and all('Deleted' in row for row in rows):
                # stop is resumable on this scheduler. Remove only the saved old
                # job record once its worker is gone; never delete feature files.
                save(BASE/'suspended_handoffs'/f'{job}.json',{'scheduler':current,'workers':workers})
                result=subprocess.run([SCO,'acp','jobs','delete',job,'--workspace-name','aigc'],capture_output=True,text=True,timeout=30)
                save(BASE/'delete_commands'/f'{job}.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
                assert result.returncode==0
                current=describe(job)
        if current['state'] in TERMINAL:
            save(BASE/'terminal_handoffs'/f'{job}.json',{'job_id':job,'state':current['state'],'scheduler':current,'epoch':time.time()})
            print(job,current['state'],flush=True);return
        time.sleep(5)
    raise RuntimeError(f'{job}: no terminal state yet; replacement forbidden')


def submit_one(job):
    assert job in JOBS
    receipt=BASE/'submissions'/f'{job}.json'
    assert not receipt.exists(), 'submission record already exists; reconcile before any retry'
    pack=load(RELEASE/'pack.json'); old=pack['jobs'][job]['original_scheduler']
    assert load(BASE/'terminal_handoffs'/f'{job}.json')['state'] in TERMINAL
    assert describe(job)['state'] in TERMINAL
    role=old['roles'][0]; spec=role['resource_spec'][0]
    assert int(spec['requests']['device'])==1 and int(spec['replicas'])==1
    command=shlex.join(['/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python',str(RELEASE/'minimax_resume.py'),
                        '--old-job',job,'--pack-sha',sha(RELEASE/'pack.json')])
    argv=[SCO,'acp','jobs','create','--workspace-name','aigc','--aec2-name',old['resource_pool']['name'],
          '--job-name',f'bq-mm-uploadresume-s{pack["jobs"][job]["shard"]:02d}-0923-r1','--priority',old['scheduling']['priority'],
          '--quota-type',old['scheduling']['quota_type'].lower(),'--container-image-url',role['image_path'],
          '--storage-mount',','.join(m['id']+':'+m['mount_path'] for m in old['mount']),
          '--training-framework','pytorch','--worker-nodes','1','--worker-spec',spec['name'],'--command',command]
    result=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    record={'old_job':job,'command':argv,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'status':'UNCONFIRMED'}
    save(receipt,record)  # Even an ambiguous failure must prevent blind duplicate retry.
    assert result.returncode==0
    ids=set(re.findall(r'\bpt-[a-z0-9]{8}\b',result.stdout))-set(JOBS)
    assert len(ids)==1, 'ambiguous submission response; query scheduler before retry'
    new=ids.pop(); detail=describe(new)
    assert detail['roles'][0]['startup_script']==command
    record.update(new_job=new,status='SUBMITTED_CONFIRMED',scheduler_state=detail['state'],scheduler=detail)
    save(receipt,record);print(json.dumps({'old':job,'new':new,'state':detail['state']}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','stop','submit']);p.add_argument('jobs',nargs='*');args=p.parse_args()
    if args.action=='freeze': freeze();return
    assert args.jobs and len(set(args.jobs))==len(args.jobs)
    if args.action=='stop':
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(stop_one,args.jobs))
    else:
        for job in args.jobs: submit_one(job)


if __name__=='__main__': main()
