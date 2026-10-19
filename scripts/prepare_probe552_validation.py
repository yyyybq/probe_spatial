#!/usr/bin/env python3
"""Freeze exact routes and one canary per explicitly requested ready experiment."""
import gzip
import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/main_v2_probe_v2_816_20260917'
OUT = BASE / 'probe552_validation_40_20260923_r1'
EVIDENCE = BASE / 'probe_readiness_552_20260923'
SOURCE = ROOT / 'runtime/main_v2_probe_v2_r7_single_v3/source'
PYTHON = '/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def binding(path):
    return {'path': str(path), 'sha256': sha(path)}


def connect(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    db.row_factory = sqlite3.Row
    return db


def main():
    if (OUT / 'bundle.json').exists():
        raise RuntimeError('validation bundle already frozen')
    queue_path = args.queue.resolve() if args.queue else OUT / 'queue.jsonl'
    if not queue_path.exists() and not args.queue:
        queue_path = OUT / 'queue_40.jsonl'
    queue = [json.loads(l) for l in queue_path.read_text().splitlines()]
    assert queue and len(queue) == len({r['unique_key'] for r in queue})
    queue_models = {r['model'] for r in queue}
    evidence = load(EVIDENCE / 'feature_evidence_snapshot.json')
    rows_by_prefix = {}
    manifest_bindings = {}
    shas = {r['protocol']['canonical_manifest_sha256'] for r in queue}
    for b in load(BASE / 'execution_manifest_r3_receipt_union/manifest.json')['availability_classes']:
        if b['sha256'] in shas:
            assert sha(b['path']) == b['sha256']
            rows_by_prefix[int(b['prefix'])] = [json.loads(l) for l in Path(b['path']).read_text().splitlines()]
            manifest_bindings[int(b['prefix'])] = dict(path=b['path'], sha256=b['sha256'], rows=2721)
    stream_specs = {(r['model'], r['disk_layer'], r['timestep'], r['prefix']) for r in queue}
    required_stream = {}
    required_target = set()
    for model, layer, timestep, prefix in stream_specs:
        for r in rows_by_prefix[prefix]:
            required_stream[model, 'scannetpp', r['scene_id'], layer, timestep, int(r['streaming_window_id']), prefix] = r
    for q in queue:
        if q['requires_target']:
            for r in rows_by_prefix[q['prefix']]:
                for h in (1, 2, 4):
                    required_target.add((q['model'], 'scannetpp', r['scene_id'], q['disk_layer'], q['timestep'], int(r['streaming_window_indices'][q['prefix'] - 1 + h])))
    base_overlay_path = args.base_overlay.resolve()
    base_overlay = load(base_overlay_path)
    base_stream = connect(base_overlay['streaming_database']['path'])
    # Wan's complete streaming route is frozen in its dedicated official
    # overlay rather than the 552 receipt-union overlay.  Target-dependent Wan
    # probes need that exact stream overlay during validation.
    wan_stream = connect(args.wan_stream_overlay) if args.wan_stream_overlay else None
    base_target = connect(base_overlay['target_database']['path'])
    streams, targets = {}, {}
    stream_schema = base_stream.execute("SELECT sql FROM sqlite_master WHERE name='streaming_owners'").fetchone()[0]
    target_schema = base_target.execute("SELECT sql FROM sqlite_master WHERE name='target_owners'").fetchone()[0]
    for model, layer, timestep, prefix in stream_specs:
        stream_db = wan_stream if model == 'wan' and wan_stream is not None else base_stream
        for row in stream_db.execute(
                "SELECT * FROM streaming_owners WHERE vfm=? AND layer=? AND timestep=? AND prefix=?",
                (model, layer, timestep, prefix)):
            d = dict(row)
            k = tuple(d[n] for n in ('vfm', 'source', 'scene_id', 'layer', 'timestep', 'window_id', 'prefix'))
            if k not in required_stream:
                continue
            assert json.loads(d['raw_ids_json']) == required_stream[k]['streaming_indices']
            streams[k] = d
    for model, layer, timestep in {(q['model'], q['disk_layer'], q['timestep']) for q in queue if q['requires_target']}:
        for row in base_target.execute(
                "SELECT * FROM target_owners WHERE vfm=? AND layer=? AND timestep=?",
                (model, layer, timestep)):
            d = dict(row)
            k = tuple(d[n] for n in ('vfm', 'source', 'scene_id', 'layer', 'timestep', 'target_raw_id'))
            if k in required_target and not d['status'].startswith('PENDING'):
                targets[k] = d
    gate_root = ROOT / 'artifacts/wan_aether_streaming_official_r1'
    gate_manifest = load(gate_root / 'overlay/manifest.json')
    gate_path = Path(gate_manifest['route_file']['path'])
    assert sha(gate_path) == gate_manifest['route_file']['sha256']
    aether_gate_uris = {r['source_uri'] for r in (json.loads(l) for l in gzip.open(gate_path, 'rt')) if r['key'][1] == 'aether'}
    for path in sorted((gate_root / 'gate').glob('gate-*/receipts.jsonl.gz')):
        for line in gzip.open(path, 'rt'):
            r = json.loads(line)
            if r.get('source_uri') not in aether_gate_uris or r['status'] != 'REFERENCE_SOURCE_READ_ONLY':
                continue
            e = r['expected']; ev = r['evidence']
            k = (e['vfm'], e['source'], e['scene_id'], e['feature_layer'], e['timestep'], e['window_id'], e['prefix'])
            if k not in required_stream:
                continue
            assert e['raw_ids'] == required_stream[k]['streaming_indices']
            streams[k] = dict(zip(('vfm','source','scene_id','layer','timestep','window_id','prefix'), k),
                unit_id=canonical(k), source_uri=r['source_uri'], source_index_uri=r['source_index_uri'],
                payload_sha256=ev['sidecar'].get('declared_payload_sha256_unverified'),
                payload_size_bytes=ev['safetensors']['object_size_bytes'], payload_header_sha256=ev['safetensors']['header_sha256'],
                source_index_sha256=ev['source_index']['sha256'], tensor_shape_json=json.dumps(e['expected_shape']),
                tensor_dtype=e['expected_dtype'], raw_ids_json=json.dumps(e['raw_ids']), input_length=81,
                valid_length=e['prefix'], pad_mode='repeat_tail', origin='gate_pass', status=r['status'])
    receipt_bindings = []
    for ev in evidence['receipt_evidence']:
        if ev['model'] not in queue_models:
            continue
        path = EVIDENCE / 'receipts' / (ev['sha256'] + '.json')
        assert sha(path) == ev['sha256']
        receipt = load(path)
        assert receipt['status'] == 'PASS_REMOTE_VERIFIED'
        receipt_bindings.append(binding(path))
        model = ev['model']; mode = ev['mode']; timestep = receipt['lineage']['timestep']
        namespace = receipt['lineage']['feature_namespace']
        root = f'aoss://baiqiao/InsScene-15K/{namespace}/{model}/scannetpp/'
        for rel, obj in receipt['objects'].items():
            identity = obj['canonical_unit_identity']
            if mode == 'streaming' and identity.get('kind') == 'streaming_feature':
                g = identity['gap_record_key']
                k = (model, 'scannetpp', g['scene'], identity['layer'], timestep, g['window_id'], g['prefix'])
                if k not in required_stream or 'gap_record_raw_ids_sha256' not in identity:
                    continue
                raw_ids = required_stream[k]['streaming_indices']
                assert canonical(raw_ids) == identity['gap_record_raw_ids_sha256']
                index_rel = g['scene'] + '/prefix_index.npy'
                index_obj = receipt['objects'][index_rel]
                previous = streams.get(k, {})
                streams[k] = dict(zip(('vfm','source','scene_id','layer','timestep','window_id','prefix'), k),
                    unit_id=canonical(k), source_uri=root+rel, source_index_uri=root+index_rel,
                    payload_sha256=obj['payload_sha256'], payload_size_bytes=obj['payload_size_bytes'],
                    payload_header_sha256='GAP_RECEIPT_FULL_SHA', source_index_sha256=index_obj['payload_sha256'],
                    tensor_shape_json=previous.get('tensor_shape_json', json.dumps([2,11,30,45,3072])),
                    tensor_dtype=previous.get('tensor_dtype', 'float16'), raw_ids_json=json.dumps(raw_ids),
                    input_length=identity['input_num_frames'], valid_length=len(raw_ids), pad_mode=identity['temporal_padding'],
                    origin='extraction_receipt_full_sha', status='PASS_REMOTE_VERIFIED')
            elif mode == 'target' and identity.get('kind') == 'target_delta_layer' and identity.get('payload_role') == 'feature':
                layer = identity['layer']; scene = identity['scene']
                index_rel = str(Path(rel).parent / f'target_indices_layer{layer}.npy')
                index_obj = receipt['objects'][index_rel]
                for index, raw_id in enumerate(identity['target_indices']):
                    k = (model, 'scannetpp', scene, layer, timestep, int(raw_id))
                    if k not in required_target:
                        continue
                    targets[k] = dict(zip(('vfm','source','scene_id','layer','timestep','target_raw_id'), k),
                        canonical_row_index=int(raw_id), source_uri=root+rel, source_index_uri=root+index_rel,
                        source_row_index=index, tensor_key='feat', payload_sha256=obj['payload_sha256'],
                        payload_size_bytes=obj['payload_size_bytes'], payload_header_sha256='GAP_RECEIPT_FULL_SHA',
                        index_sha256=index_obj['payload_sha256'], index_size_bytes=index_obj['payload_size_bytes'],
                        origin='gap_delta_exact', status='PASS_REMOTE_VERIFIED')
    assert set(streams) == set(required_stream), len(set(required_stream) - set(streams))
    assert set(targets) == required_target, len(required_target - set(targets))
    overlay_dir = OUT / 'overlay'; overlay_dir.mkdir(parents=True, exist_ok=True)
    databases = {}
    for mode, schema, owners in [('streaming', stream_schema, streams), ('target', target_schema, targets)]:
        path = overlay_dir / (mode + '_owners.sqlite3')
        if path.exists():
            raise RuntimeError(f'refusing existing database: {path}')
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        db.execute("INSERT INTO metadata VALUES ('status','READY')")
        db.execute(schema)
        table = mode + '_owners'
        columns = [r[1] for r in db.execute(f'PRAGMA table_info({table})')]
        for k, owner in sorted(owners.items()):
            if mode == 'target':
                owner['canonical_row_index'] = owner['target_raw_id']
            db.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in columns)})", [owner[n] for n in columns])
        db.commit(); db.close()
        databases[mode + '_database'] = dict(binding(path), rows=len(owners))
    overlay = dict(schema='CANONICAL_V2_PROBE816_MIXED_OWNER_OVERLAY_V1', status='READY', fail_closed=True,
        models=sorted(queue_models), scope=f'EXACT_READY_{len(queue)}_VALIDATION_ONLY', **databases,
        counts=dict(streaming_duplicate_owners=0, streaming_holes=0, target_duplicate_owners=0, target_holes=0),
        queue=binding(queue_path), feature_evidence=binding(EVIDENCE / 'feature_evidence_snapshot.json'),
        receipt_evidence=receipt_bindings, base_overlay=binding(base_overlay_path))
    overlay_path = overlay_dir / 'manifest.json'; write(overlay_path, overlay)
    groups = sorted(stream_specs, key=lambda x: (x[0] != 'videomae', x))
    cases = []
    for q in queue:
        cfg = SOURCE / 'configs/experiment' / (q['experiment_config'] + '.yaml')
        assert sha(cfg) == q['experiment_config_sha256']
        shard = groups.index((q['model'], q['disk_layer'], q['timestep'], q['prefix']))
        case_dir = OUT / 'cases' / q['unique_key']; case_dir.mkdir(parents=True)
        all_rows = rows_by_prefix[q['prefix']]
        selected = []
        batch_overrides = [v for v in q['scientific_overrides']
                           if v.startswith('data.data_module.batch_size_per_device=')]
        assert len(batch_overrides) == 1
        batch_size = int(batch_overrides[0].split('=', 1)[1])
        # A one-epoch canary must contain three full batches, including for
        # Wan C whose frozen batch size is four. Do not weaken the step gate.
        for split, count in [('train', max(6, 3 * batch_size)), ('val', max(2, batch_size))]:
            seen = set()
            for r in all_rows:
                if r['split'] == split and r['scene_id'] not in seen:
                    selected.append(r); seen.add(r['scene_id'])
                    if len(seen) == count:
                        break
            assert len(seen) == count
        manifest_path = case_dir / 'manifest.jsonl'
        manifest_path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in selected))
        output = case_dir / 'output'
        command = [PYTHON, 'vidfm3d/train.py', 'experiment=' + q['experiment_config']] + q['scientific_overrides']
        command += [f'paths.log_dir={output}', 'paths.run_folder_name=run', 'logger=canary_csv',
            'trainer.max_epochs=1', '++trainer.max_steps=3', '++trainer.limit_train_batches=3',
            '++trainer.limit_val_batches=3' if q['probe']=='action_dynamics' else '++trainer.limit_val_batches=1',
            '++trainer.num_sanity_val_steps=0', '++trainer.log_every_n_steps=1', '++trainer.enable_progress_bar=false',
            'data.data_module.num_workers=0', 'data.data_module.num_workers_val=0']
        env = dict(PROBE_OWNER_OVERLAY_MANIFEST=str(overlay_path),
            PROBE_OWNER_LOCAL_MIRROR_ROOT='/mnt/umm/users/yinbaiqiao/InsScene-15K',
            INSCENE_MANIFEST=str(manifest_path), INSCENE_MANIFEST_SHA256=sha(manifest_path), INSCENE_TRUST_EXPANDED_MANIFEST='1',
            INSCENE_DATA_ROOT='/mnt/umm/users/yinbaiqiao/InsScene-15K/data',
            INSCENE_LOCAL_SCENE_ROOT='/mnt/umm/users/yinbaiqiao/InsScene-15K/data/processed_scannetpp_v2/processed_scannetpp_v2',
            INSCENE_STREAMING_FEAT_ROOT='aoss://baiqiao/PROBE816_OWNER_OVERLAY_ONLY',
            INSCENE_TARGET_FEAT_ROOT='aoss://baiqiao/PROBE816_OWNER_OVERLAY_ONLY', INSCENE_TARGET_LAYOUT='flat',
            INSCENE_TARGET_INDEX_ROOT='/mnt/umm/users/yinbaiqiao/InsScene-15K/PROBE816_OWNER_OVERLAY_ONLY',
            INSCENE_PROBE_TASK=q['probe'], INSCENE_METADATA_CACHE_MAX_SCENES='32',
            INSCENE_LABEL_CACHE_ROOT=str(OUT / 'label_cache'), PROBE_FP16_PASSTHROUGH=args.fp16_passthrough,
            AOSS_CACHE_DIR=f'/mnt/local_cache/inscene/probe552_validation/shard{shard}', AOSS_LOG_LOCAL_HITS='0',
            MPLCONFIGDIR=f'/tmp/mpl-probe552-{shard}', OMP_NUM_THREADS='4', WANDB_MODE='offline',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(SOURCE))
        case = dict(schema='PROBE552_VALIDATION_CASE_V1', unique_key=q['unique_key'], model=q['model'],
            probe=q['probe'], layer=q['disk_layer'], prefix=q['prefix'], timestep=q['timestep'], shard=shard,
            runtime=str(SOURCE), source_checksum=binding(SOURCE.parent / 'source.sha256'),
            overlay_manifest=str(overlay_path), overlay_manifest_sha256=sha(overlay_path),
            manifest=str(manifest_path), manifest_sha256=sha(manifest_path), full_execution_manifest=manifest_bindings[q['prefix']],
            command=command, environment=env, samples=selected, experiment=q)
        case_path = case_dir / 'case.json'; write(case_path, case)
        cases.append(dict(unique_key=q['unique_key'], model=q['model'], shard=shard, case_path=str(case_path), case_sha256=sha(case_path)))
    bundle = dict(schema='PROBE552_VALIDATION_BUNDLE_V1', status='FROZEN_NOT_SUBMITTED', cases=cases,
        queue=binding(queue_path), overlay=binding(overlay_path), groups=[list(g) for g in groups],
        cases_count=len(cases), shard_count=len(groups), runtime=str(SOURCE), source_checksum=binding(SOURCE.parent/'source.sha256'))
    write(OUT / 'bundle.json', bundle)
    write(OUT / 'requested.json', dict(requested_utc=evidence['updated_utc'],
        unique_keys=[q['unique_key'] for q in queue]))
    print(json.dumps(dict(cases=len(cases), streaming_owners=len(streams), target_owners=len(targets), shards=len(groups))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--queue',type=Path,
                        help='Frozen JSONL queue to bind into a new validation output directory')
    parser.add_argument('--base-overlay',type=Path,default=BASE/'owner_overlay_r7_receipt_union/overlay_manifest.json')
    parser.add_argument('--wan-stream-overlay',type=Path,
                        help='Wan official streaming_owners.sqlite3 for target-dependent Wan validation')
    parser.add_argument('--fp16-passthrough', choices=('0', '1'), default='1',
                        help='Keep cached features in fp16 through host transfer only when numerically validated')
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    SOURCE=args.source.resolve()
    main()
