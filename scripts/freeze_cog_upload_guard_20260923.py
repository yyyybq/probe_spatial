"""Freeze the separate guard after tests and audit current staging ownership."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/main_v2_probe_v2_816_20260917/upload_recovery_20260923_r1'
RELEASE=ROOT/'runtime/cog_upload_backpressure_v1'
SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'
JOBS={'pt-rutyedtm':(0,3),'pt-tkvc8tad':(2,3),'pt-v2ejbno9':(3,3),'pt-3cx4i1b8':(1,1)}


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (RELEASE/'pack.json').exists()
    all_rows=[]
    for page in range(1,21):
        rows=json.loads(subprocess.check_output([SCO,'acp','jobs','list','--workspace-name','aigc','--page-size','500','--page-token',str(page),'-o','json'],text=True,timeout=30))
        all_rows.extend(rows)
        if len(rows)<500: break
    else: raise RuntimeError('scheduler inventory incomplete')
    active=[r for r in all_rows if r.get('state') in ('RUNNING','STARTING','QUEUEING','PENDING','CREATING') and 'cog' in str(r).lower()]
    # Descriptions, not names alone, establish which payload namespace is shared.
    descriptions={}
    for row in active:
        job=row['name']
        detail=json.loads(subprocess.check_output([SCO,'acp','jobs','describe',job,'--workspace-name','aigc','-o','json'],text=True,timeout=30))
        descriptions[job]=detail
        command='\n'.join(r.get('startup_script','') for r in detail['roles'])
        if 'cogvideox-stream-' in command:
            assert job in JOBS, 'unaccounted same-root CogVideoX streaming uploader'
    assert set(JOBS)<=set(descriptions)
    (BASE/'cog_scheduler_audit.json').write_text(json.dumps({'epoch':time.time(),'jobs_scanned':len(all_rows),'descriptions':descriptions},indent=2)+'\n')
    configs={}; tags=[]
    for job,(shard,rev) in JOBS.items():
        source=ROOT/f'artifacts/main_v2_probe_v2_816_20260917/atomic_feature_workers_four_models_20260923_r{rev}/shards/p528-cogvideox-stream-{shard:02d}.json'
        config=json.loads(source.read_text()); binding=config['source_bindings']['aoss_sync_loop']
        observed=sha(Path(binding['path']))
        inspection=json.loads((BASE/'process_control'/job/'inspection.json').read_text())
        opened=[s['open_script_sha256'] for s in inspection['sync_processes_not_signalled'] if s.get('open_script_sha256')]
        assert opened==[observed] and time.time()-inspection['epoch']<600
        if observed!=binding['sha256']:
            # Existing, pre-task drift: only ADS concurrency was made configurable
            # with the same old defaults (1 transfer / 2 listers). Do not rewrite
            # a live script to satisfy an obsolete manifest. Pin its observed SHA.
            assert job=='pt-3cx4i1b8' and binding['sha256']=='8f00d93742fd6872aed80810fb8a27deee4f82a533a4e2aa962c334614e90cad'
            assert observed=='abd1f22746cf6a5503e2e1257c777b9fbb15b9b24ddd6752b07907be43954f89'
        tag=config['worklist_file_sha256'][:24] if rev==3 else None
        tags.append(tag)
        configs[job]={'sync_path':binding['path'],'sync_sha256':observed,'original_manifest_sync_sha256':binding['sha256'],'staging_tag':tag,
                     'shard_path':str(source),'shard_sha256':sha(source)}
    assert len(tags)==len(set(tags)) and tags.count(None)==1
    pack={'guard_sha256':sha(RELEASE/'guard.py'),
          'process_helper_sha256':sha(ROOT/'scripts/feature_upload_recovery_control_20260923.py'),
          'jobs':configs,'science_changes':[],
          'staging_isolation':'three immutable unique SHA tags; one remaining legacy uploader in this model/source root',
          'restart_condition':{'free_TiB':3,'backlog_GiB_per_job':32},
          'pause_condition':{'free_TiB_below':2,'backlog_GiB_per_job_above':64,'upload_stall_seconds':600}}
    (RELEASE/'pack.json').write_text(json.dumps(pack,indent=2)+'\n')
    print(json.dumps({'status':'FROZEN','jobs':list(JOBS),'pack_sha256':sha(RELEASE/'pack.json')}))


if __name__=='__main__': main()
