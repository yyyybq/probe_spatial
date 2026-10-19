#!/usr/bin/env python3
"""Join physical validation failures back to exact scientific feature keys."""
import gzip
import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/main_v2_probe_v2_816_20260917/probe552_validation_40_20260923_r1'


def main():
    bundle=json.loads((OUT/'bundle.json').read_text())
    overlay=json.loads(Path(bundle['overlay']['path']).read_text())
    databases={}
    for mode in ('streaming','target'):
        b=overlay[mode+'_database']
        assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256']
        c=sqlite3.connect(Path(b['path']).as_uri()+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
        databases[mode]=c
    groups={};all_failures=[]
    for shard, group in enumerate(bundle['groups']):
        p=OUT/'groups'/f'{shard:02d}'/'inventory_result.json'
        if not p.exists():continue
        inventory=json.loads(p.read_text())
        failures=[]
        for line in Path(inventory['objects_report']).read_text().splitlines():
            obj=json.loads(line)
            if obj['status']!='FAIL':continue
            mode=obj['mode']
            owners=databases[mode].execute(f'SELECT * FROM {mode}_owners WHERE source_uri=?',(obj['source_uri'],)).fetchall()
            assert owners
            for owner in owners:
                d=dict(owner)
                row=dict(mode=mode,model=d['vfm'],layer=d['layer'],timestep=d['timestep'],scene_id=d['scene_id'],
                    source_uri=obj['source_uri'],checked_utc=obj['checked_utc'],error=obj['error'],
                    status='EXACT_URI_404' if 'NoSuchKey' in obj['error'] else 'PHYSICAL_VALIDATION_FAILED')
                if mode=='streaming':row.update(window_id=d['window_id'],prefix=d['prefix'])
                else:row.update(frame_index=d['target_raw_id'])
                failures.append(row)
        detail=OUT/'physical_failures'/f'group_{shard:02d}.jsonl.gz'
        detail.parent.mkdir(exist_ok=True)
        with gzip.open(detail,'wt') as f:
            for row in failures:f.write(json.dumps(row)+'\n')
        groups[str(shard)]=dict(model=group[0],layer=group[1],timestep=group[2],prefix=group[3],
            status=inventory['status'],checked_utc=inventory['ended_utc'],objects_checked=inventory['objects_checked'],
            failed_objects=inventory['failed'],failed_keys=dict(Counter(r['mode'] for r in failures)),
            failure_details=str(detail),inventory_result=str(p))
        all_failures.extend(failures)
    with gzip.open(OUT/'unavailable_features.jsonl.gz','wt') as f:
        for row in all_failures:f.write(json.dumps(row)+'\n')
    statuses=Counter()
    for case in bundle['cases']:
        p=Path(case['case_path']).parent/'result.json'
        statuses[json.loads(p.read_text())['status'] if p.exists() else 'WAITING']+=1
    summary=dict(updated_utc=datetime.now(timezone.utc).isoformat(),groups=groups,validation_counts=dict(statuses),
        failed_feature_keys=len(all_failures),failure_details=str(OUT/'unavailable_features.jsonl.gz'))
    (OUT/'physical_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='groups'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-dir',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.validation_dir.resolve()
    main()
