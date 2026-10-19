#!/usr/bin/env python3
"""Reconstruct eight audited missing payloads with frozen science, restore create-only."""
import json,os,sys,sqlite3,subprocess,hashlib,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.restore_probe552_exact_uris as restore
from scripts.probe_campaign_20260924_followup import BASE,ROOT,load,save,sha,now
# Keep failed attempts immutable.  A retry must name a new receipt directory
# through this environment variable, while retaining the exact same frozen
# worklist, extractor and owner rows.
OUT=Path(os.environ.get('LINGBOT_URI8_REPAIR_OUT', BASE/'lingbot_uri8_repair_20260924_r1'))
RUNTIME=ROOT/'runtime/lingbot_stream01_recovery_v1'
spec=importlib.util.spec_from_file_location('frozen_runner',RUNTIME/'runner.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
def prepare():
 OUT.mkdir(exist_ok=True)
 shard=runner.validate_shard(BASE/'atomic_feature_workers_lingbot_20260923_r1/shards/p528-lingbot-stream-00.json',trust_current_tools=True)
 original=load(shard['worklist_path']);w={k:v for k,v in original.items() if k not in {'entries','counts','content_sha256'}}
 w['entries']=[{**e,'layers':[10,20,30,40,47] if e['prefix']==8 else [10,20,30]} for e in original['entries'] if e['scene']=='00777c41d4' and e['window_id']==0 and e['prefix'] in (8,12)]
 w['counts']={'entries':2,'layer_items':8,'scenes':1};w['content_sha256']=restore.canonical(w);save(OUT/'worklist.json',w)
 c=sqlite3.connect(BASE/'probe_validation_lingbot80_20260924_r1/overlay/streaming_owners.sqlite3');c.row_factory=sqlite3.Row
 rows=[dict(x) for x in c.execute("select * from streaming_owners where scene_id='00777c41d4' and window_id=0 and (prefix=8 or (prefix=12 and layer in (10,20,30)))")];assert len(rows)==8
 shard.update(worklist_path=str(OUT/'worklist.json'),worklist_file_sha256=sha(OUT/'worklist.json'),feature_root=str(OUT/'payload/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1'))
 command=runner.extractor_command(shard,0,1)
 # Invoke the exact pinned file, avoiding cwd shadowing of the frozen module.
 command=command[:1]+[str(runner.FROZEN_EXTRACTOR)]+command[3:]
 save(OUT/'plan.json',{'at':now(),'rows':rows,'command':command,'environment':shard['environment'],'worklist_sha256':sha(OUT/'worklist.json'),'extractor_sha256':sha(runner.FROZEN_EXTRACTOR),'original_shard':shard['shard_id'],'science_changes':[]})
 print('prepared exact 8 files')
def run():
 p=load(OUT/'plan.json');assert sha(OUT/'worklist.json')==p['worklist_sha256'];assert sha(runner.FROZEN_EXTRACTOR)==p['extractor_sha256']
 # Revalidate all frozen backend and model bindings before GPU work.
 runner.validate_shard(BASE/'atomic_feature_workers_lingbot_20260923_r1/shards/p528-lingbot-stream-00.json',trust_current_tools=True)
 env=os.environ.copy();env.update(p['environment']);env['PYTHONPATH']=str(ROOT)+':'+env.pop('WINDOWFIX_EXTRA_PYTHONPATH','');env['FEATURE_NAMESPACE']='FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1'
 save(OUT/'status.json',{'at':now(),'status':'EXTRACTING'})
 with (OUT/'extract.log').open('a') as f:subprocess.run(p['command'],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 results=[]
 for row in p['rows']:
  row['local_path']=str(OUT/'payload'/row['source_uri'].split('/InsScene-15K/')[1]);checked=restore.check_local(row)
  if not checked['status'].startswith('PASS'):raise RuntimeError(str(checked))
  payload=Path(row['local_path']).read_bytes();pr=restore.ensure(row['source_uri'],payload,row['payload_sha256']);del payload
  # An existing sidecar can retain the old worklist provenance; require the same scientific identity/payload SHA.
  from botocore.exceptions import ClientError
  uri=row['source_uri']+'.manifest.json';key=uri.split('aoss://baiqiao/')[1]
  try:
   response=restore.s3().get_object(Bucket='baiqiao',Key=key);raw=response['Body'].read();response['Body'].close();meta=json.loads(raw)
   assert meta['sha256']==row['payload_sha256'] and meta['gap_record_raw_ids_sha256']==restore.canonical(json.loads(row['raw_ids_json']))
   assert meta['layer']==row['layer'] and meta['timestep']==row['timestep'] and meta['vfm']=='lingbot'
   sr={'uri':uri,'created':False,'sha256':restore.sha(raw)}
  except ClientError as e:
   if str(e.response.get('Error',{}).get('Code')) not in {'404','NoSuchKey','NotFound'}:raise
   raw=Path(checked['sidecar_path']).read_bytes();sr=restore.ensure(uri,raw,restore.sha(raw))
  results.append({'uri':row['source_uri'],'payload':pr,'sidecar':sr,'status':'PASS_REMOTE_RESTORED_FULL_SHA'})
  save(OUT/'restore_results.json',{'at':now(),'results':results})
 save(OUT/'status.json',{'at':now(),'status':'PASS','count':8})
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare()
 else:
  try:run()
  except Exception as e:save(OUT/'status.json',{'at':now(),'status':'FAIL','error':repr(e)});raise
