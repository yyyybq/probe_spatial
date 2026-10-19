#!/usr/bin/env python3
"""Validate one layer/prefix group, then run each of its production-config canaries."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import sqlite3
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/main_v2_probe_v2_816_20260917'
OUT = BASE / 'probe552_validation_40_20260923_r1'
THREAD = threading.local()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def progress():
    subprocess.run([sys.executable, str(ROOT / 'scripts/update_probe552_progress.py')], check=True)


def header_check(item):
    uri, row = item
    try:
        if not hasattr(THREAD, 'client'):
            from aoss_client.client import Client
            THREAD.client = Client('/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf')
        client = THREAD.client
        path = 'fj:s3://' + uri[len('aoss://'):]
        size = int(client.size(path))
        if size != int(row['payload_size_bytes']):
            raise ValueError(f'object size {size} != {row["payload_size_bytes"]}')
        raw = client.get(path, range='0-4095')
        if not raw or len(raw) < 8:
            raise ValueError('missing or short safetensors header')
        n = struct.unpack('<Q', raw[:8])[0]
        if not 0 < n < 1_000_000:
            raise ValueError(f'invalid header length {n}')
        if len(raw) < n+8:
            raw = client.get(path, range=f'0-{n+7}')
        header = json.loads(raw[8:8+n])
        feat = header['feat']
        shape = feat['shape']
        offset = max(v['data_offsets'][1] for k,v in header.items() if k != '__metadata__')
        if n+8+offset != size:
            raise ValueError('safetensors shape/offset/file size mismatch')
        expected_header = row.get('payload_header_sha256')
        if expected_header and expected_header != 'GAP_RECEIPT_FULL_SHA':
            envelope = row['origin'] in {'gate_pass','metadata_recovery','minimax_contract_v2'}
            observed = hashlib.sha256(raw[:n+8] if envelope else raw[8:n+8]).hexdigest()
            if observed != expected_header:
                raise ValueError('header SHA mismatch')
        if row['mode'] == 'streaming':
            if shape != json.loads(row['tensor_shape_json']):
                raise ValueError(f'shape mismatch: {shape} != {row["tensor_shape_json"]}')
        elif max(row['source_rows']) >= shape[0]:
            raise ValueError('target row index outside payload')
        return dict(source_uri=uri, status='PASS', mode=row['mode'], shape=shape, dtype=feat['dtype'], checked_utc=now())
    except Exception as e:
        return dict(source_uri=uri, status='FAIL', mode=row['mode'], error=f'{type(e).__name__}: {e}', checked_utc=now())


def inventory_gate(bundle, shard):
    path = OUT / 'groups' / f'{shard:02d}' / 'inventory_result.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['bundle_sha256'] != sha(OUT/'bundle.json'):
            raise RuntimeError('inventory belongs to another bundle')
        return result
    model, layer, timestep, prefix = bundle['groups'][shard]
    overlay = json.loads(Path(bundle['overlay']['path']).read_text())
    owners = {}
    for mode in ('streaming','target'):
        b = overlay[mode+'_database']
        if sha(b['path']) != b['sha256']:
            raise RuntimeError('owner DB SHA changed')
        db = sqlite3.connect(Path(b['path']).as_uri()+'?mode=ro&immutable=1', uri=True)
        db.row_factory = sqlite3.Row
        query = f'SELECT * FROM {mode}_owners WHERE vfm=? AND layer=? AND timestep=?'
        args = [model,layer,timestep]
        if mode == 'streaming':
            query += ' AND prefix=?'; args.append(prefix)
        for row in db.execute(query,args):
            d = dict(row); uri = d['source_uri']
            if uri not in owners:
                owners[uri] = dict(d, mode=mode, source_rows=[])
            if mode == 'target':
                owners[uri]['source_rows'].append(d['source_row_index'])
        db.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    failures, count = [], 0
    started = now()
    with (path.parent/'inventory_objects.jsonl').open('w') as sink:
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for result in pool.map(header_check, owners.items()):
                sink.write(json.dumps(result)+'\n'); count += 1
                if result['status'] != 'PASS':
                    failures.append(result)
                if count % 100 == 0:
                    sink.flush()
                    atomic(path.parent/'inventory_progress.json', dict(checked=count,total=len(owners),failed=len(failures),updated_utc=now()))
    result = dict(status='FAIL' if failures else 'PASS', started_utc=started, ended_utc=now(),
        bundle_sha256=sha(OUT/'bundle.json'), objects_checked=count, failed=len(failures),
        failure_examples=failures[:20], objects_report=str(path.parent/'inventory_objects.jsonl'),
        validation_scope='all required feature URIs: remote size/header/shape; full payload SHA checked by real loader on canary samples')
    atomic(path,result)
    return result


def run_case(entry, gate):
    path = Path(entry['case_path']); result_path = path.parent/'result.json'
    existing = json.loads(result_path.read_text()) if result_path.exists() else None
    if existing and existing.get('status') in {'PASS','FAIL'}:
        return existing
    if sha(path) != entry['case_sha256']:
        raise RuntimeError('case SHA changed')
    case = json.loads(path.read_text())
    result = dict(schema='PROBE552_VALIDATION_RESULT_V1', unique_key=case['unique_key'], model=case['model'],
        status='RUNNING', phase='TRAINING_CANARY', started_utc=now(), case_sha256=entry['case_sha256'],
        inventory_result=str(OUT/'groups'/f"{entry['shard']:02d}"/'inventory_result.json'))
    atomic(result_path,result); progress()
    try:
        if gate['status'] != 'PASS':
            raise RuntimeError(f"full feature inventory gate failed: {gate['failed']} objects; {gate['failure_examples'][:1]}")
        if sha(case['manifest']) != case['manifest_sha256'] or sha(case['overlay_manifest']) != case['overlay_manifest_sha256']:
            raise RuntimeError('manifest/overlay identity changed')
        env = dict(os.environ, **case['environment'])
        gpu = subprocess.run([case['command'][0],'-c','import torch; print(torch.cuda.device_count())'],env=env,capture_output=True,text=True)
        if gpu.returncode or gpu.stdout.strip() != '1':
            raise RuntimeError(f'exactly one CUDA device required: {gpu.stdout} {gpu.stderr}')
        log = path.parent/'train.log'
        with log.open('x') as f:
            p = subprocess.run(case['command'], cwd=case['runtime'], env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
        result.update(returncode=p.returncode, train_log=str(log), train_log_sha256=sha(log))
        if p.returncode:
            tail=log.read_text(errors='replace')[-10000:]
            result['error_log_tail']=tail
            raise RuntimeError(f'training canary exited {p.returncode}; see {log}')
        metrics=list((path.parent/'output').glob('**/metrics.csv'))
        if len(metrics)!=1:
            raise RuntimeError(f'expected one metrics CSV, got {len(metrics)}')
        rows=list(csv.DictReader(metrics[0].open()))
        def values(key):
            return [float(r[key]) for r in rows if r.get(key) not in (None,'')]
        train=values('train/loss');val=values('val/loss');samples=values('trainer/total_samples')
        train_steps=[int(r['step']) for r in rows if r.get('train/loss') not in (None,'')]
        if not train or not all(math.isfinite(v) for v in train):
            raise RuntimeError('missing/nonfinite training losses')
        if not val or not all(math.isfinite(v) for v in val):
            raise RuntimeError('missing/nonfinite validation losses')
        logs=log.read_text(errors='replace')
        batch_size_two='data.data_module.batch_size_per_device=2' in case['command']
        metric_steps_ok=set(train_steps)>={0,1,2}
        sample_count_ok=(bool(samples) and max(samples)>=6) or (metric_steps_ok and batch_size_two)
        if 'max_steps=3' not in logs or not metric_steps_ok or not sample_count_ok:
            raise RuntimeError('did not verify 3 optimizer steps and 6 train samples')
        result.update(status='PASS',phase='COMPLETE', train_losses=train, validation_losses=val,
            max_total_samples=max(samples) if samples else 6,
            sample_count_evidence='trainer/total_samples' if samples else '3 metric steps x batch_size_per_device=2',
            metrics_path=str(metrics[0]), metrics_sha256=sha(metrics[0]),
            checks=dict(full_inventory_headers='PASS',real_loader='PASS',forward='PASS',finite_loss='PASS',optimizer_steps=3,validation='PASS'))
    except Exception as e:
        result.update(status='FAIL',phase='COMPLETE',error=f'{type(e).__name__}: {e}')
    result['ended_utc']=now()
    atomic(result_path,result);progress()
    return result


def main():
    global OUT
    p=argparse.ArgumentParser()
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--bundle-sha256',required=True)
    p.add_argument('--runner-sha256',required=True)
    p.add_argument('--bundle',type=Path,default=OUT/'bundle.json')
    a=p.parse_args()
    OUT=a.bundle.resolve().parent
    assert sha(__file__)==a.runner_sha256
    assert sha(OUT/'bundle.json')==a.bundle_sha256
    bundle=json.loads((OUT/'bundle.json').read_text())
    assert 0<=a.shard<bundle['shard_count']
    checksum=bundle['source_checksum']
    assert sha(checksum['path'])==checksum['sha256']
    subprocess.run(['sha256sum','--status','-c',checksum['path']],cwd=bundle['runtime'],check=True)
    entries=[e for e in bundle['cases'] if e['shard']==a.shard]
    claim=OUT/'groups'/f'{a.shard:02d}'/'claim'
    claim.parent.mkdir(parents=True,exist_ok=True)
    claim.mkdir()
    try:
        for e in entries:
            path=Path(e['case_path']).parent/'result.json'
            if not path.exists():
                atomic(path,dict(unique_key=e['unique_key'],status='RUNNING',phase='FULL_FEATURE_INVENTORY',started_utc=now()))
        progress()
        gate=inventory_gate(bundle,a.shard)
        results=[run_case(e,gate) for e in entries]
        atomic(claim.parent/'completed.json',dict(ended_utc=now(),total=len(results),passed=sum(r['status']=='PASS' for r in results),failed=sum(r['status']=='FAIL' for r in results)))
    finally:
        claim.rmdir()


if __name__=='__main__':
    main()
