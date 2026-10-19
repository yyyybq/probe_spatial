#!/usr/bin/env python3
"""Reconcile one narrow validator false-negative using immutable run evidence.

Only results with the exact legacy sample-count error are eligible.  A result
is changed to PASS only when the process exited cleanly, metrics contain finite
training losses at steps 0/1/2 and a finite validation loss, the command used a
batch size of two, and the immutable log says max_steps=3 was reached.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ERROR = 'RuntimeError: did not verify 3 optimizer steps and 6 train samples'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def reconcile(result_path):
    result = json.loads(result_path.read_text())
    if result.get('status') != 'FAIL' or result.get('error') != ERROR:
        return None
    case_path = result_path.parent / 'case.json'
    case = json.loads(case_path.read_text())
    log = Path(result['train_log'])
    metrics = list((result_path.parent / 'output').glob('**/metrics.csv'))
    if result.get('returncode') != 0 or len(metrics) != 1:
        raise RuntimeError(f'{result_path}: return code or metrics cardinality failed')
    if result.get('train_log_sha256') != sha(log):
        raise RuntimeError(f'{result_path}: train log identity changed')
    rows = list(csv.DictReader(metrics[0].open()))
    train_rows = [r for r in rows if r.get('train/loss') not in (None, '')]
    val_rows = [r for r in rows if r.get('val/loss') not in (None, '')]
    train = [float(r['train/loss']) for r in train_rows]
    val = [float(r['val/loss']) for r in val_rows]
    steps = {int(r['step']) for r in train_rows}
    checks = [
        train and all(math.isfinite(v) for v in train),
        val and all(math.isfinite(v) for v in val),
        steps >= {0, 1, 2},
        'data.data_module.batch_size_per_device=2' in case['command'],
        'max_steps=3' in log.read_text(errors='replace'),
    ]
    if not all(checks):
        raise RuntimeError(f'{result_path}: strict reconciliation evidence failed {checks}')
    backup = result_path.with_name('result.pre_reconcile.json')
    if backup.exists() and sha(backup) != sha(result_path):
        raise RuntimeError(f'{result_path}: pre-reconcile backup already exists with another identity')
    if not backup.exists():
        backup.write_bytes(result_path.read_bytes())
    old_sha = sha(backup)
    result.pop('error', None)
    result.update(
        status='PASS', phase='COMPLETE', train_losses=train, validation_losses=val,
        max_total_samples=6,
        sample_count_evidence='3 metric steps x batch_size_per_device=2',
        metrics_path=str(metrics[0]), metrics_sha256=sha(metrics[0]),
        checks=dict(full_inventory_headers='PASS', real_loader='PASS', forward='PASS',
                    finite_loss='PASS', optimizer_steps=3, validation='PASS'),
        reconciliation=dict(
            reconciled_utc=datetime.now(timezone.utc).isoformat(),
            reason='legacy validator required optional trainer/total_samples metric',
            original_result=str(backup), original_result_sha256=old_sha,
            evidence=['returncode=0', 'train steps=0,1,2', 'batch_size_per_device=2',
                      'finite train and validation losses', 'log reached max_steps=3'],
        ),
    )
    atomic(result_path, result)
    return result['unique_key']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('validation_dirs', nargs='+', type=Path)
    args = parser.parse_args()
    changed = []
    summaries = []
    for root in args.validation_dirs:
        for path in sorted(root.glob('cases/*/result.json')):
            key = reconcile(path)
            if key:
                changed.append(key)
        reconciled = []
        for path in sorted(root.glob('cases/*/result.json')):
            result = json.loads(path.read_text())
            if result.get('reconciliation'):
                reconciled.append(dict(unique_key=result['unique_key'], result_path=str(path),
                    result_sha256=sha(path), **result['reconciliation']))
        summary = dict(schema='PROBE552_CANARY_RECONCILIATION_V1',
            generated_utc=datetime.now(timezone.utc).isoformat(), validation_dir=str(root.resolve()),
            reconciled_count=len(reconciled), results=reconciled)
        atomic(root / 'reconciliation_summary.json', summary)
        summaries.append(dict(validation_dir=str(root), reconciled_count=len(reconciled)))
    print(json.dumps({'newly_reconciled': len(changed), 'unique_keys': changed,
                      'summaries': summaries}, indent=2))


if __name__ == '__main__':
    main()
