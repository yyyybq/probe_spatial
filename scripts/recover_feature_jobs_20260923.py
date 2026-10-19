"""Prepare explicit immutable recovery releases and submit confirmed single GPUs."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/main_v2_probe_v2_816_20260917/vjepa_qwen_f3r_recovery_20260923'
SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2)+'\n')
def describe(job): return json.loads(subprocess.check_output([SCO,'acp','jobs','describe',job,'--workspace-name','aigc','-o','json'],text=True,timeout=30))


def submit(key,command,template, pool='zoetrope'):
    dest=BASE/'submissions'/f'{key}.json'; assert not dest.exists(), 'reconcile existing submission before retry'
    role=template['roles'][0]
    cmd=[SCO,'acp','jobs','create','--workspace-name','aigc','--aec2-name',pool,'--job-name',key,
         '--priority',template['scheduling']['priority'],'--quota-type',template['scheduling']['quota_type'].lower(),
         '--container-image-url',role['image_path'],'--storage-mount',','.join(m['id']+':'+m['mount_path'] for m in template['mount']),
         '--training-framework','pytorch','--worker-nodes','1','--worker-spec','N4lS.Iq.I80.1','--command',command]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    record={'command':cmd,'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode,'status':'UNCONFIRMED'}; save(dest,record)
    assert r.returncode==0
    ids=set(re.findall(r'\bpt-[a-z0-9]{8}\b',r.stdout)); assert len(ids)==1
    job=ids.pop(); actual=describe(job); assert actual['roles'][0]['startup_script']==command
    record.update(job=job,status='SUBMITTED_CONFIRMED',scheduler=actual); save(dest,record)
    print(json.dumps({'key':key,'job':job,'state':actual['state']}),flush=True)


def vjepa():
    release=ROOT/'runtime/vjepa_upload_recovery_v1'; packpath=release/'pack.json'
    assert not packpath.exists()
    originals={j:describe(j) for j in ('pt-9013u3wc','pt-jyrfya89')}
    assert all(r['state']=='FAILED' for r in originals.values())
    for j,row in originals.items(): save(BASE/'old_jobs'/f'{j}.json',row)
    plan=ROOT/'artifacts/main_v2_probe_v2_816_20260917/atomic_feature_workers_four_models_20260923_r3'
    shardpath=plan/'shards/p528-vjepa2-stream-01.json'; shard=json.loads(shardpath.read_text())
    worklist=Path(shard['worklist_path']); other=json.loads((plan/'worklists/p528-vjepa2-stream-00.json').read_text())
    own=json.loads(worklist.read_text()); scenes={e['scene'] for e in own['entries']}
    assert not scenes & {e['scene'] for e in other['entries']}
    orphan=Path(shard['feature_root'])/'vjepa2/scannetpp/adf4ab4a53/window_0001/prefix_000011/.feature_layer17.sft.tmp-89'
    assert 'adf4ab4a53' in scenes and orphan.stat().st_mtime<__import__('datetime').datetime.fromisoformat('2026-09-23T03:44:00+00:00').timestamp()
    files=[release/'resume_upload.py',release/'sync_loop.sh',shardpath,worklist,
           ROOT/'scripts/windowfix_v2/run_atomic_gap_shard_atomicrace_v2.py',ROOT/'scripts/windowfix_v2/canonical_gap_aoss_adapter_atomicrace_v2.py',
           ROOT/'scripts/windowfix_v2/canonical_gap_aoss_adapter.py',ROOT/'features/canonical_gap_worklist.py',Path(shard['protocol_path'])]
    pack={'old_jobs':originals,'shard':str(shardpath),'files':{str(p):sha(p) for p in files},
          'orphan':{'path':str(orphan),'size':orphan.stat().st_size,'sha256':sha(orphan)},'extractor_launched':False}
    save(packpath,pack)
    subprocess.run(['bash','-n',str(release/'sync_loop.sh')],check=True)
    command=shlex.join(['/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python',str(release/'resume_upload.py'),sha(packpath)])
    submit('bq-vjepa2-uploadfix-s01-0923-v1',command,originals['pt-9013u3wc'])


if __name__=='__main__':
    assert sys.argv[1:] == ['vjepa']; vjepa()
