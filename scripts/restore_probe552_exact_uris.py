#!/usr/bin/env python3
"""Restore only audited missing feature URIs from exact SHA-identical local copies.

No extraction, deletion, index rewriting, or changes to scientific identity.
Uploads are create-only (If-None-Match: *); existing objects must match exactly.
Every restored payload and sidecar is downloaded and SHA checked afterward.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import sqlite3
import struct
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/main_v2_probe_v2_816_20260917'
VALIDATION=BASE/'probe552_validation_40_20260923_r1'
OUT=BASE/'probe552_uri_restore_20260923_r1'
DATA=Path('/mnt/umm/users/yinbaiqiao/InsScene-15K')
PREFIX='aoss://baiqiao/InsScene-15K/FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1/'
THREAD=threading.local()


def now():return datetime.now(timezone.utc).isoformat()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(d):return sha(json.dumps(d,sort_keys=True,separators=(',',':')).encode())
def load(p):return json.loads(Path(p).read_text())


def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(f'.{p.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');os.chmod(tmp,0o644);os.replace(tmp,p)


def check_local(row):
    try:
        p=Path(row['local_path']);side=Path(str(p)+'.manifest.json')
        assert p.is_file() and not p.is_symlink(), 'local payload missing/nonregular'
        assert side.is_file() and not side.is_symlink(), 'local sidecar missing/nonregular'
        raw=p.read_bytes();meta_raw=side.read_bytes();meta=json.loads(meta_raw)
        assert len(raw)==row['payload_size_bytes'], 'payload size changed'
        assert sha(raw)==row['payload_sha256'], 'payload differs from frozen SHA'
        assert meta['sha256']==row['payload_sha256'] and meta['size_bytes']==len(raw), 'sidecar/payload mismatch'
        assert meta['vfm']==row['vfm'] and meta['layer']==row['layer'] and meta['timestep']==row['timestep'], 'model/layer/timestep mismatch'
        assert meta['mode']=='streaming_prefix' and meta['feature_namespace']=='FEAT_STREAMING_CANONICAL_V2_GAP_DELTA_V1'
        assert meta['canonical_source_sha256']=='ebc496eb0587cda372ccf29a7c5244ef454096151f73c8eed9511967ded1902d'
        assert meta['canonical_protocol_sha256']=='215c720c8537f4dfae6243bedcc7339fbb7e3ba8a6ca2dcaf04f49cd4b738e4f'
        assert meta['gap_record_key']==dict(scene=row['scene_id'],window_id=row['window_id'],prefix=row['prefix'])
        assert meta['gap_record_raw_ids_sha256']==canonical(json.loads(row['raw_ids_json'])), 'canonical raw frame IDs differ'
        n=struct.unpack('<Q',raw[:8])[0];h=json.loads(raw[8:8+n]);shape=h['feat']['shape']
        assert shape==json.loads(row['tensor_shape_json'])==meta['tensors']['feat']['shape'], 'shape mismatch'
        assert h['feat']['dtype']=='F16' and meta['tensors']['feat']['dtype']=='torch.float16'
        assert n+8+max(v['data_offsets'][1] for k,v in h.items() if k!='__metadata__')==len(raw)
        return dict(row,status='PASS_LOCAL_EXACT_SHA_AND_IDENTITY',sidecar_path=str(side),sidecar_sha256=sha(meta_raw),
                    sidecar_size_bytes=len(meta_raw),checked_utc=now())
    except Exception as e:return dict(row,status='FAIL_LOCAL',error=f'{type(e).__name__}: {e}',checked_utc=now())


def audit(workers):
    failures=[json.loads(l) for l in gzip.open(VALIDATION/'unavailable_features.jsonl.gz','rt')]
    assert len(failures)==6917 and len({r['source_uri'] for r in failures})==6917
    overlay=load(VALIDATION/'overlay/manifest.json');binding=overlay['streaming_database']
    assert sha(Path(binding['path']).read_bytes())==binding['sha256']
    db=sqlite3.connect(Path(binding['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
    owners={r['source_uri']:dict(r) for r in db.execute('SELECT * FROM streaming_owners')};db.close()
    rows=[]
    for fail in failures:
        row=owners[fail['source_uri']]
        assert row['source_uri'].startswith(PREFIX) and row['vfm'] in {'videomae','aether'}
        assert len(row['payload_sha256'])==64
        row['local_path']=str(DATA/row['source_uri'].split('/InsScene-15K/',1)[1])
        rows.append(row)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(check_local,rows):
            results.append(r)
            if len(results)%200==0:
                write(OUT/'audit_progress.json',dict(checked=len(results),total=len(rows),counts=dict(Counter(x['status'] for x in results)),updated_utc=now()))
    raw=''.join(json.dumps(r,sort_keys=True)+'\n' for r in results).encode()
    path=OUT/'local_audit.jsonl';path.write_bytes(raw)
    summary=dict(created_utc=now(),total=len(rows),counts=dict(Counter(r['status'] for r in results)),
        payload_bytes=sum(r['payload_size_bytes'] for r in results if r['status'].startswith('PASS')),
        inventory=dict(path=str(path),sha256=sha(raw)),source_overlay=dict(path=str(VALIDATION/'overlay/manifest.json'),sha256=sha((VALIDATION/'overlay/manifest.json').read_bytes())))
    write(OUT/'audit_summary.json',summary);print(json.dumps(summary),flush=True)


def s3():
    if not hasattr(THREAD,'client'):
        from aoss_client.client import Client
        c=Client('/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf')
        mixed=c._get_local_client()
        THREAD.client=mixed._ceph_dict['fj']._s3_resource.meta.client
    return THREAD.client


def ensure(uri,raw,expected):
    from botocore.exceptions import ClientError
    assert uri.startswith(PREFIX)
    key=uri[len('aoss://baiqiao/'):]
    client=s3();created=False
    try:
        client.head_object(Bucket='baiqiao',Key=key)
    except ClientError as e:
        if str(e.response.get('Error',{}).get('Code')) not in {'404','NoSuchKey','NotFound'}:raise
        try:
            client.put_object(Bucket='baiqiao',Key=key,Body=raw,IfNoneMatch='*',Metadata={'sha256':expected})
            created=True
        except ClientError as e:
            if str(e.response.get('Error',{}).get('Code')) not in {'412','PreconditionFailed'}:raise
    response=client.get_object(Bucket='baiqiao',Key=key)
    assert response['ContentLength']==len(raw), 'remote size differs; existing object not overwritten'
    body=response['Body'];digest=hashlib.sha256()
    try:
        while True:
            block=body.read(4*1024*1024)
            if not block:break
            digest.update(block)
    finally:body.close()
    assert digest.hexdigest()==expected, 'remote full SHA differs; existing object not overwritten'
    return dict(uri=uri,created=created,remote_sha256=digest.hexdigest(),bytes=len(raw),verified_utc=now())


def restore_one(row):
    receipt=OUT/'objects'/(canonical(row['source_uri'])+'.json')
    if receipt.exists():
        prior=load(receipt)
        if prior.get('status')=='PASS_REMOTE_RESTORED_FULL_SHA':return prior
    result=dict(source_uri=row['source_uri'],model=row['vfm'],layer=row['layer'],prefix=row['prefix'],
                scene_id=row['scene_id'],window_id=row['window_id'],started_utc=now())
    try:
        checked=check_local(row)
        assert checked['status'].startswith('PASS'), checked.get('error')
        raw=Path(row['local_path']).read_bytes();assert sha(raw)==row['payload_sha256']
        side=Path(row['sidecar_path']).read_bytes();assert sha(side)==row['sidecar_sha256']
        result['payload']=ensure(row['source_uri'],raw,row['payload_sha256'])
        del raw
        result['sidecar']=ensure(row['source_uri']+'.manifest.json',side,row['sidecar_sha256'])
        result['status']='PASS_REMOTE_RESTORED_FULL_SHA'
    except Exception as e:result.update(status='FAIL',error=f'{type(e).__name__}: {e}')
    result['ended_utc']=now();write(receipt,result);return result


def restore(workers,limit):
    summary=load(OUT/'audit_summary.json');b=summary['inventory'];raw=Path(b['path']).read_bytes()
    assert sha(raw)==b['sha256']
    rows=[json.loads(l) for l in raw.splitlines() if json.loads(l)['status']=='PASS_LOCAL_EXACT_SHA_AND_IDENTITY']
    if limit:rows=rows[:limit]
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(restore_one,row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results)%50==0 or len(results)==len(rows):
                status=dict(updated_utc=now(),processed=len(results),total=len(rows),counts=dict(Counter(r['status'] for r in results)),
                    original_missing_keys=6917,restoration_scope='exact original URI, payload and sidecar only; no indices/no deletions/no extraction')
                write(OUT/'restore_progress.json',status);print(json.dumps(status),flush=True)
    write(OUT/('canary_summary.json' if limit else 'restore_summary.json'),dict(status,inventory_sha256=b['sha256'],
        errors=[dict(uri=r['source_uri'],error=r.get('error')) for r in results if r['status']=='FAIL']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['audit','restore']);p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='audit':audit(a.workers)
    else:restore(a.workers,a.limit)
