"""Reuse the existing atomic worker builder after real, SHA-bound canaries pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from recover_feature_jobs_20260923 import BASE,ROOT,describe,submit,save

SOURCE=ROOT/'runtime/f3r_qwen_recovery_v1/source'
PY='/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python'
CAMPAIGN=ROOT/'artifacts/main_v2_probe_v2_816_20260917'
QUEUE=CAMPAIGN/'atomic_queue_f3r_qwen_20260923_r1/queue.jsonl'
QUEUE_SHA='bae7e3edaf7df145b20f24b73b8cd05e34c5af9b253cb31648609f320614e7cc'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=['f3r','qwen2_5_vl']);parser.add_argument('mode',choices=['streaming','target']);args=parser.parse_args()
    assert args.mode=='streaming' or args.model=='f3r', 'Qwen target is already complete'
    canary=BASE/('backend_canaries' if args.mode=='streaming' else 'f3r_target_canary')
    receipt_path=canary/'receipts'/f'{args.model}.json'
    receipt=json.loads(receipt_path.read_text()); assert receipt['status']=='pass' and receipt['sample_index']==0
    assert receipt['backend_provenance_revalidated_after_runs'] and receipt['vfm']==args.model
    assert receipt['gap_binding']['selected_scene_layer_covered']
    assert all(v['comparison']['files_compared']>0 for v in receipt['results'].values())
    assert sha(QUEUE)==QUEUE_SHA
    stage=BASE/f'production_{args.model}_{args.mode}'
    marker=stage/'submitted.json'; assert not marker.exists(), 'already deployed; reconcile rather than resubmit'
    stage.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,HF_HOME=str(ROOT.parent/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    plan=stage/'templates'
    if not (plan/'summary.json').exists():
        cmd=[PY,str(SOURCE/'scripts/windowfix_v2/prepare_sco_gap_jobs.py'),
             '--source-audit',str(CAMPAIGN/'targeted_reextract_worklists_20260922_r2'/f'source_audit_{args.mode}.json'),
             '--protocol',str(ROOT/'protocols/inscene_streaming_canonical_v2.json'),
             '--canonical-source',str(ROOT/'manifests/c_retrain_20260722/inscene_c_h124_prefix24.jsonl'),
             '--backend-canary-dir',str(canary),'--output-dir',str(plan),'--project-root',str(SOURCE),
             '--include-vfm',args.model,'--n-gpus','1','--aec2-name','zoetrope','--worker-spec','N4lS.Iq.I80.1',
             '--runner',str(SOURCE/'scripts/windowfix_v2/run_atomic_gap_shard_v1.py')]
        subprocess.run(cmd,check=True,env=env)
    templates=list((plan/'shards').glob('*.json'))
    assert templates, 'no canary-certified templates; no submission allowed'
    workers=stage/'workers'
    if not (workers/'manifest.json').exists():
        cmd=[PY,str(SOURCE/'scripts/build_probe528_atomic_worker_plan.py'),'--queue',str(QUEUE),'--queue-sha256',QUEUE_SHA,
             '--output',str(workers),'--claim-root',str(BASE/'f3r_qwen_claims')]
        for template in templates:cmd+=['--template',str(template)]
        subprocess.run(cmd,check=True,env=env)
    manifest=json.loads((workers/'manifest.json').read_text())
    expected=1 if args.mode=='target' else 2724
    assert manifest['atomic_tasks']==expected and manifest['unique_owned_tasks']==expected
    assert len(manifest['workers'])==(1 if args.mode=='target' else 2)
    # Assert a strict subset of the already frozen atomic queue, with no new key.
    wanted={r['unique_key'] for r in map(json.loads,QUEUE.read_text().splitlines())
            if r['model']==args.model and r['mode']==('streaming_prefix' if args.mode=='streaming' else 'target_isolated')}
    seen=set()
    for worker in manifest['workers']:
        keys={r['unique_key'] for r in map(json.loads,Path(worker['task_list']).read_text().splitlines())}
        assert not seen&keys;seen|=keys
        result=subprocess.run([PY,str(SOURCE/'scripts/windowfix_v2/run_atomic_gap_shard_v1.py'),'--batch',worker['batch']],env=env,capture_output=True,text=True)
        save(stage/f'{worker["owner"]}_preflight.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        assert result.returncode==0,result.stdout+result.stderr
    assert seen==wanted
    template=describe('pt-9013u3wc')
    deployed=[]
    for i,worker in enumerate(manifest['workers']):
        key=f'bq-{args.model}-recover-{args.mode[:1]}{i:02d}-0923-v1'
        existing=BASE/'submissions'/f'{key}.json'
        if existing.exists():
            saved=json.loads(existing.read_text());assert saved['status']=='SUBMITTED_CONFIRMED'
            assert describe(saved['job'])['state'] not in ('FAILED','DELETED'), 'replacement already failed; diagnose before retry'
        else:
            command=next(a.split('=',1)[1] for a in worker['sco_argv'] if a.startswith('--command='))
            submit(key,command,template)
        deployed.append(json.loads(existing.read_text()))
    save(marker,{'status':'SUBMITTED_CONFIRMED','atomic_tasks':expected,'canary':str(receipt_path),'canary_sha256':sha(receipt_path),
                 'jobs':[d['job'] for d in deployed],'worklist_keys_unchanged':True})
    print(marker.read_text(),flush=True)


if __name__=='__main__':main()
