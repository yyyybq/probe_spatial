#!/usr/bin/env python3
"""Audit every Wan owner against the H800-visible local mirror."""
import argparse, concurrent.futures, hashlib, json, sqlite3, struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

LOCAL = Path('/mnt/umm/users/yinbaiqiao/InsScene-15K')
PREFIX = 'aoss://baiqiao/InsScene-15K/'

def local(uri):
    if not uri.startswith(PREFIX): raise ValueError(uri)
    return LOCAL / uri[len(PREFIX):]

def check(row):
    try:
        path=local(row['source_uri']); st=path.stat()
        if st.st_size != row['payload_size_bytes']: raise ValueError(f'size {st.st_size}')
        with path.open('rb') as f:
            raw=f.read(4096)
        n=struct.unpack('<Q',raw[:8])[0]; header=json.loads(raw[8:8+n]); feat=header['feat']
        if feat['shape'] != json.loads(row['tensor_shape_json']): raise ValueError('shape')
        if feat['dtype'] not in ('F16','float16'): raise ValueError('dtype')
        expected=row['payload_header_sha256']
        if expected != 'GAP_RECEIPT_FULL_SHA':
            observed=hashlib.sha256(raw[:n+8] if row['origin']=='gate_pass' else raw[8:n+8]).hexdigest()
            if observed != expected: raise ValueError('header sha')
        return ('PASS',None)
    except Exception as e: return ('FAIL',f'{type(e).__name__}: {e}')

def check_index(item):
    uri, expected=item
    try:
        path=local(uri); raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('sha')
        return ('PASS',None)
    except Exception as e: return ('FAIL',f'{type(e).__name__}: {e}')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--database',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    db=sqlite3.connect(a.database); db.row_factory=sqlite3.Row
    rows=[dict(r) for r in db.execute('select * from streaming_owners order by vfm,scene_id,layer,prefix,window_id')]; db.close()
    assert len(rows)==76188
    indices={}
    for r in rows:
        old=indices.setdefault(r['source_index_uri'],r['source_index_sha256']); assert old==r['source_index_sha256']
    failures=[]; counts=Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
        for row,result in zip(rows,pool.map(check,rows,chunksize=32)):
            counts[result[0]]+=1
            if result[0]=='FAIL' and len(failures)<100: failures.append({'source_uri':row['source_uri'],'error':result[1]})
    index_fail=[]; index_counts=Counter()
    items=sorted(indices.items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for item,result in zip(items,pool.map(check_index,items,chunksize=8)):
            index_counts[result[0]]+=1
            if result[0]=='FAIL' and len(index_fail)<100: index_fail.append({'source_index_uri':item[0],'error':result[1]})
    out={'schema':'WAN_LOCAL_STREAMING_OVERLAY_AUDIT_V1','checked_utc':datetime.now(timezone.utc).isoformat(),
         'database':str(a.database.resolve()),'features':dict(counts),'indices':dict(index_counts),
         'feature_failures':failures,'index_failures':index_fail,
         'status':'PASS' if not failures and not index_fail and counts['PASS']==76188 else 'FAIL'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,sort_keys=True))

if __name__=='__main__': main()
