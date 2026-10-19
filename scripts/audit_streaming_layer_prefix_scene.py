#!/usr/bin/env python3
"""Physically audit unverified streaming windows and summarize by layer/prefix/scene."""
import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import os
import sqlite3
import struct
import tempfile
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/main_v2_probe_v2_816_20260917'
EVIDENCE = BASE / 'probe_readiness_552_20260923'
OVERLAY = BASE / 'owner_overlay_r7_receipt_union/overlay_manifest.json'
MODELS = {'dino', 'videomae', 'vjepa2'}
THREAD = threading.local()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    os.fchmod(fd, 0o644)
    with os.fdopen(fd, 'w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def check(item):
    key, owner = item
    try:
        if not hasattr(THREAD, 'client'):
            from aoss_client.client import Client
            THREAD.client = Client('/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf')
        path = 'fj:s3://' + owner['source_uri'][len('aoss://'):]
        error = None
        for attempt in range(3):
            try:
                size = int(THREAD.client.size(path))
                raw = THREAD.client.get(path, range='0-4095')
                break
            except Exception as exc:
                error = exc
                if attempt == 2:
                    raise
                time.sleep(0.25 * (attempt + 1))
        if size != int(owner['payload_size_bytes']):
            raise ValueError(f'object size {size} != {owner["payload_size_bytes"]}')
        if not raw or len(raw) < 8:
            raise ValueError('missing or short safetensors header')
        header_length = struct.unpack('<Q', raw[:8])[0]
        if not 0 < header_length < 1_000_000:
            raise ValueError(f'invalid header length {header_length}')
        if len(raw) < header_length + 8:
            raw = THREAD.client.get(path, range=f'0-{header_length + 7}')
        header = json.loads(raw[8:8 + header_length])
        feature = header['feat']
        expected_shape = json.loads(owner['tensor_shape_json'])
        if feature['shape'] != expected_shape:
            raise ValueError(f'shape {feature["shape"]} != {expected_shape}')
        if feature['dtype'] != owner['tensor_dtype']:
            raise ValueError(f'dtype {feature["dtype"]} != {owner["tensor_dtype"]}')
        offset = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        if header_length + 8 + offset != size:
            raise ValueError('safetensors header/data offsets do not match object size')
        expected_header = owner.get('payload_header_sha256')
        if expected_header and expected_header != 'GAP_RECEIPT_FULL_SHA':
            envelope = owner['origin'] in {'gate_pass', 'metadata_recovery', 'minimax_contract_v2'}
            observed = hashlib.sha256(raw[:header_length + 8] if envelope else raw[8:header_length + 8]).hexdigest()
            if observed != expected_header:
                raise ValueError('header SHA mismatch')
        return dict(**key, status='PASS_PHYSICAL_HEADER', source_uri=owner['source_uri'],
                    shape=feature['shape'], dtype=feature['dtype'], checked_utc=now())
    except Exception as exc:
        return dict(**key, status='FAIL_PHYSICAL_HEADER', source_uri=owner.get('source_uri'),
                    error=f'{type(exc).__name__}: {exc}', checked_utc=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
        default=BASE / 'streaming_layer_prefix_scene_audit_20260923_r1')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (out / 'summary.json').exists():
        raise RuntimeError('audit already completed; refusing to replace it')
    out.mkdir(parents=True, exist_ok=True)
    overlay = json.loads(OVERLAY.read_text())
    binding = overlay['streaming_database']
    if sha(binding['path']) != binding['sha256']:
        raise RuntimeError('streaming owner database identity changed')
    database = sqlite3.connect(Path(binding['path']).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    database.row_factory = sqlite3.Row
    evidence_rows = list(csv.DictReader((EVIDENCE / 'experiments_552.csv').open()))
    specs = {}
    for row in evidence_rows:
        if row['model'] not in MODELS:
            continue
        spec = (row['model'], int(row['disk_layer']), int(row['prefix']))
        specs[spec] = dict(model=spec[0], layer=spec[1], prefix=spec[2],
            required=int(row['streaming_required']), receipt_verified=int(row['streaming_verified']),
            receipt_unverified=int(row['streaming_unverified']))
    tasks = []
    required_scenes = defaultdict(set)
    missing_by_scene = Counter()
    for spec, meta in sorted(specs.items()):
        model, layer, prefix = spec
        missing_path = EVIDENCE / 'unverified_keys' / f'{model}__l{layer}__p{prefix}__streaming.jsonl.gz'
        missing = [json.loads(line) for line in gzip.open(missing_path, 'rt')]
        if len(missing) != meta['receipt_unverified']:
            raise RuntimeError(f'{spec}: missing-key file count changed')
        all_rows = database.execute(
            'SELECT scene_id,window_id FROM streaming_owners WHERE vfm=? AND layer=? AND timestep=749 AND prefix=?',
            spec).fetchall()
        required_scenes[spec] = {row['scene_id'] for row in all_rows}
        if len(all_rows) != meta['required'] or len(required_scenes[spec]) != 681:
            raise RuntimeError(f'{spec}: owner universe is incomplete')
        for key in missing:
            rows = database.execute(
                'SELECT * FROM streaming_owners WHERE vfm=? AND layer=? AND timestep=? AND prefix=? AND scene_id=? AND window_id=?',
                (model, layer, key['timestep'], prefix, key['scene_id'], key['window_id'])).fetchall()
            if len(rows) != 1:
                raise RuntimeError(f'{spec}: expected one owner for {key}, got {len(rows)}')
            identity = dict(model=model, layer=layer, prefix=prefix, scene_id=key['scene_id'],
                            window_id=key['window_id'])
            tasks.append((identity, dict(rows[0])))
            missing_by_scene[spec, key['scene_id']] += 1
    database.close()
    started = now()
    failures = []
    pass_by_spec = Counter()
    fail_by_spec = Counter()
    pass_by_scene = Counter()
    fail_by_scene = Counter()
    report = out / 'object_results.jsonl'
    with report.open('x') as sink, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(check, tasks), 1):
            sink.write(json.dumps(result, sort_keys=True) + '\n')
            spec = (result['model'], result['layer'], result['prefix'])
            scene = (spec, result['scene_id'])
            if result['status'] == 'PASS_PHYSICAL_HEADER':
                pass_by_spec[spec] += 1
                pass_by_scene[scene] += 1
            else:
                fail_by_spec[spec] += 1
                fail_by_scene[scene] += 1
                failures.append(result)
            if index % 100 == 0:
                sink.flush()
                atomic(out / 'progress.json', dict(checked=index, total=len(tasks), failed=len(failures), updated_utc=now()))
    by_spec = []
    scene_rows = []
    for spec, meta in sorted(specs.items()):
        incomplete_scenes = 0
        physically_completed_scenes = 0
        for scene_id in sorted(required_scenes[spec]):
            missing = missing_by_scene[spec, scene_id]
            failed = fail_by_scene[spec, scene_id]
            physical_pass = pass_by_scene[spec, scene_id]
            complete = failed == 0 and physical_pass == missing
            if missing and complete:
                physically_completed_scenes += 1
            if not complete:
                incomplete_scenes += 1
            scene_rows.append(dict(model=spec[0], layer=spec[1], prefix=spec[2], scene_id=scene_id,
                receipt_unverified_windows=missing, physical_pass_windows=physical_pass,
                physical_fail_windows=failed, scene_complete=complete))
        physical_pass = pass_by_spec[spec]
        physical_fail = fail_by_spec[spec]
        by_spec.append(dict(**meta, physical_pass=physical_pass, physical_fail=physical_fail,
            verified_after_physical_audit=meta['receipt_verified'] + physical_pass,
            incomplete_scenes=incomplete_scenes,
            scenes_completed_by_physical_audit=physically_completed_scenes,
            layer_prefix_ready=(physical_fail == 0 and physical_pass == meta['receipt_unverified'])))
    for name, rows in [('by_layer_prefix.csv', by_spec), ('by_scene.csv', scene_rows)]:
        with (out / name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = dict(schema='STREAMING_LAYER_PREFIX_SCENE_PHYSICAL_AUDIT_V1', started_utc=started,
        ended_utc=now(), models=sorted(MODELS), owner_overlay={'path': str(OVERLAY), 'sha256': sha(OVERLAY)},
        evidence_snapshot={'path': str(EVIDENCE / 'feature_evidence_snapshot.json'),
                           'sha256': sha(EVIDENCE / 'feature_evidence_snapshot.json')},
        specs=len(specs), objects_checked=len(tasks), objects_passed=len(tasks)-len(failures),
        objects_failed=len(failures), layer_prefix_ready=sum(row['layer_prefix_ready'] for row in by_spec),
        layer_prefix_total=len(by_spec), ready_specs=[row for row in by_spec if row['layer_prefix_ready']],
        blocked_specs=[row for row in by_spec if not row['layer_prefix_ready']],
        failure_examples=failures[:50], object_results=str(report), by_layer_prefix=str(out / 'by_layer_prefix.csv'),
        by_scene=str(out / 'by_scene.csv'),
        acceptance='Receipt-verified windows plus physical size/header/shape verification of every previously unverified window; training canary still required before production.')
    atomic(out / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {'ready_specs', 'blocked_specs', 'failure_examples'}}, indent=2))


if __name__ == '__main__':
    main()
