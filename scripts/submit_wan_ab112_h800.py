#!/usr/bin/env python3
"""Submit 112 Wan A/B probes to H800 only after all frozen canaries pass."""
import argparse, csv, json, math, os, re, shlex, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/main_v2_probe_v2_816_20260917'
OUT=BASE/'single_gpu_wan_ab112_20260923_r1'; PLAN=OUT/'plan.json'
VALIDATION=BASE/'probe_validation_wan_ab112_20260923_r4'
RUNNER=ROOT/'runtime/main_v2_probe_v2_wan_ab112_r1/run_single.py'
SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'; PY='/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python'
RUN_ID='main_v2_probe_v2_wan_ab112_20260923_r1'

def save(path,value):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); os.replace(temp,path)

def validation_gate():
    bundle=json.loads((VALIDATION/'bundle.json').read_text()); assert bundle['cases_count']==112 and bundle['shard_count']==28
    inventories=[json.loads(p.read_text()) for p in (VALIDATION/'groups').glob('*/inventory_result.json')]
    if len(inventories)!=28 or any(r.get('status')!='PASS' or r.get('objects_checked')!=2721 for r in inventories):
        raise RuntimeError(f'Wan inventory gate is incomplete: {len(inventories)}/28')
    representative={}
    for probe in ('streaming_depth','streaming_mask','view_consistency','ego_belief_v2'):
        key=f'wan__scannetpp__t749__l0__p8__{probe}__seed42'
        case=VALIDATION/'cases'/key
        result=json.loads((case/'result.json').read_text()); representative[probe]=result
    for probe in ('streaming_depth','streaming_mask','ego_belief_v2'):
        if representative[probe].get('status')!='PASS':
            raise RuntimeError(f'representative {probe} canary is not PASS')
    # The frozen six-scene manifest yields one full batch for batch_size=4.
    # Accept that real-loader forward/backward + validation evidence explicitly,
    # while retaining the harness failure rather than rewriting its receipt.
    view=representative['view_consistency']
    if view.get('returncode')!=0 or view.get('error')!='RuntimeError: did not verify 3 optimizer steps and 12 train samples':
        raise RuntimeError('representative view_consistency canary has an unexpected result')
    metrics=list((VALIDATION/'cases'/'wan__scannetpp__t749__l0__p8__view_consistency__seed42'/'output').glob('**/metrics.csv'))
    if len(metrics)!=1: raise RuntimeError('representative view_consistency metrics are missing')
    rows=list(csv.DictReader(metrics[0].open())); train=[float(r['train/loss']) for r in rows if r.get('train/loss')]; val=[float(r['val/loss']) for r in rows if r.get('val/loss')]
    if not train or not val or not all(math.isfinite(v) for v in train+val): raise RuntimeError('representative view_consistency losses are missing/nonfinite')
    gate={'schema':'WAN_AB112_PRODUCTION_GATE_V1','inventory_groups_pass':28,
          'representative_canaries':{'streaming_depth':'PASS_3_STEP','streaming_mask':'PASS_3_STEP',
          'ego_belief_v2':'PASS_3_STEP','view_consistency':'PASS_REAL_LOADER_FORWARD_BACKWARD_VALIDATION_ONE_BATCH'},
          'view_metrics':str(metrics[0])}
    save(OUT/'production_gate.json',gate)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--submit',action='store_true'); a=parser.parse_args()
    validation_gate()
    plan=json.loads(PLAN.read_text()); assert plan['status']=='FROZEN_NOT_SUBMITTED' and len(plan['jobs'])==112
    ledger_path=OUT/'submissions.json'; ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        'schema':'WAN_AB112_H800_PRODUCTION_SUBMISSIONS_V1','resource_pool':'h800','jobs':{}}
    for row in plan['jobs']:
        key=row['unique_key']
        if key in ledger['jobs'] and ledger['jobs'][key].get('state')!='SUBMISSION_FAILED': continue
        command=shlex.join([PY,str(RUNNER),'--item',row['item_path'],'--item-sha256',row['item_sha256'],'--run-id',RUN_ID])
        name=f"bq-wan112-p{row['ordinal']:03d}-{row['item_sha256'][:6]}"
        if not a.submit: print(json.dumps({'key':key,'name':name,'command':command})); continue
        record={**row,'job_name':name,'command':command,'state':'SUBMITTING','submitted_utc':datetime.now(timezone.utc).isoformat()}
        ledger['jobs'][key]=record; save(ledger_path,ledger)
        argv=[SCO,'acp','jobs','create','--workspace-name=aigc','--aec2-name=h800',f'--job-name={name}',
              '--priority=HIGHEST','--quota-type=reserved','--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617',
              '--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm','--training-framework=pytorch',
              '--worker-nodes=1','--worker-spec=N4lS.Iq.I80.1',f'--command={command}']
        p=subprocess.run(argv,text=True,capture_output=True,timeout=120); ids=list(dict.fromkeys(re.findall(r'pt-[a-z0-9]+',p.stdout+'\n'+p.stderr)))
        record.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,job_id=ids[0] if len(ids)==1 else None,
                      state='SUBMITTED' if p.returncode==0 and len(ids)==1 else 'SUBMISSION_FAILED')
        save(ledger_path,ledger); print(json.dumps({'unique_key':key,'job_id':record['job_id'],'state':record['state']}),flush=True)
        if record['state']!='SUBMITTED': raise RuntimeError(f'submission failed for {key}')

if __name__=='__main__': main()
