#!/usr/bin/env python3
"""Maintain the 552-row experiment ledger from feature evidence and run receipts.

--refresh-features recomputes exact coverage and missing-key files. This script
never submits jobs. Validation workers call it after each state transition.
"""
import argparse
import csv
import fcntl
import io
import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/main_v2_probe_v2_816_20260917'
EVIDENCE = BASE / 'probe_readiness_552_20260923'
OUT = BASE / 'probe552_progress'
VALIDATION = BASE / 'probe552_validation_40_20260923_r1'
ACTIVE_VALIDATION = OUT / 'active_validation.json'
if ACTIVE_VALIDATION.exists():
    VALIDATION = Path(json.loads(ACTIVE_VALIDATION.read_text())['path'])


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.update-', dir=path.parent)
    os.fchmod(fd, 0o644)
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    os.replace(tmp, path)


def load(path, default=None):
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except PermissionError:
        # GPU containers can create receipts as root with mode 0600. Scheduler
        # state still supplies the live status until a readable receipt exists.
        return default


def refresh(refresh_features=False):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.lock').open('a') as lock:
        deadline = time.monotonic() + 120
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
        if refresh_features:
            subprocess.run(['python3', str(EVIDENCE / 'analyze.py')], check=True)
        try:
            old = load(OUT / 'experiments_552.json', {'experiments': []})
        except PermissionError:
            # A previous container writer used mkstemp's 0600 default. The
            # canonical run receipts below are sufficient to rebuild the view.
            old = {'experiments': []}
        previous = {r['unique_key']: r for r in old['experiments']}
        evidence = load(EVIDENCE / 'feature_evidence_snapshot.json')
        feature_rows = list(csv.DictReader((EVIDENCE / 'experiments_552.csv').open()))
        queue = [json.loads(l) for l in (BASE / 'valid7_user_scope_20260922_r4/queue_552_with_aether.jsonl').read_text().splitlines()]
        features = {r['unique_key']: r for r in feature_rows}
        model_evidence = {r['model']: r for r in evidence['models']}
        requested = load(VALIDATION / 'requested.json', {'unique_keys': []})
        requested_keys = set(requested['unique_keys'])
        cases = load(VALIDATION / 'bundle.json', {'cases': []})['cases']
        case_by_key = {r['unique_key']: r for r in cases}
        extra_validation_submission_by_key = {}
        dino_validation_root = BASE / 'probe_validation_dino24_20260923_r1'
        dino_validation_cases = load(dino_validation_root / 'bundle.json', {'cases': []})['cases']
        dino_validation_submissions = load(dino_validation_root / 'submissions.json', {'jobs': {}})['jobs']
        for entry in dino_validation_cases:
            case_by_key[entry['unique_key']] = entry
            requested_keys.add(entry['unique_key'])
            extra_validation_submission_by_key[entry['unique_key']] = dino_validation_submissions.get(str(entry['shard']))
        for extra_name in ('probe_validation_vm72_vj96_20260924_r1', 'probe_validation_lingbot80_20260924_r1', 'probe_validation_lingbot32_20260924_r2', 'probe_validation_cog96_20260924_r1', 'probe_validation_wan_c8_l20_20260925_r2', 'probe_validation_cog12_ego_20260928_r2'):
            extra_root = BASE / extra_name
            ledger = ('validation_submissions.json' if extra_name in {
                'probe_validation_wan_c8_l20_20260925_r2',
                'probe_validation_lingbot32_20260924_r2',
                'probe_validation_cog12_ego_20260928_r2',
            } else 'submissions.json')
            extra_jobs = load(extra_root / ledger, {'jobs': {}})['jobs']
            for entry in load(extra_root / 'bundle.json', {'cases': []})['cases']:
                case_by_key[entry['unique_key']] = entry
                requested_keys.add(entry['unique_key'])
                extra_validation_submission_by_key[entry['unique_key']] = extra_jobs.get(str(entry['shard']))
        physical = load(VALIDATION / 'physical_summary.json', {'groups': {}})['groups']
        physical_by_spec = {(g['model'],g['layer'],g['timestep'],g['prefix']):g for g in physical.values()}
        ling_restore_root = BASE / 'lingbot_uri8_repair_20260924_r1'
        ling_restore = load(ling_restore_root / 'status.json', {})
        if ling_restore.get('status') != 'PASS':
            for ip in (BASE / 'probe_validation_lingbot80_20260924_r1/groups').glob('*/inventory_result.json'):
                inv = load(ip, {})
                if inv.get('status') != 'FAIL':
                    continue
                bundle = load(BASE / 'probe_validation_lingbot80_20260924_r1/bundle.json')
                model, layer, timestep, prefix = bundle['groups'][int(ip.parent.name)]
                physical_by_spec[model,layer,timestep,prefix] = dict(model=model,layer=layer,timestep=timestep,prefix=prefix,
                    failed_keys={'streaming':1,'target':0}, failure_details=str(ip), checked_utc=inv.get('ended_utc'))
        recovery_root=BASE/'probe552_uri_restore_20260923_r1'
        recovery=load(recovery_root/'restore_summary.json') or load(recovery_root/'restore_progress.json')
        history_cases={}
        for history in sorted(BASE.glob('probe552_validation_40_*/bundle.json')):
            if history.parent==VALIDATION:continue
            for entry in load(history,{'cases':[]})['cases']:
                prior=load(Path(entry['case_path']).parent/'result.json')
                if prior:history_cases.setdefault(entry['unique_key'],[]).append(prior)
        submissions = load(VALIDATION / 'submissions.json', {'jobs': {}})['jobs']
        production_root = BASE / 'single_gpu_r7_20260922_r2'
        production_run_id = 'main_v2_probe_v2_r7_single_20260922_r2'
        production_submissions = load(production_root / 'production_submissions.json', {'jobs': {}})['jobs']
        production_state = production_root / 'state' / 'production' / production_run_id
        targetfix_root = BASE / 'single_gpu_vm24_targetfix_20260923_r1'
        targetfix_run_id = 'main_v2_probe_v2_vm24_targetfix_20260923_r1'
        targetfix_submissions = load(targetfix_root / 'submissions.json', {'jobs': {}})['jobs']
        aether_root = BASE / 'single_gpu_aether24_20260923_r1'
        aether_run_id = 'main_v2_probe_v2_aether24_20260923_r1'
        aether_submissions = load(aether_root / 'submissions.json', {'jobs': {}})['jobs']
        vmvj_root = BASE / 'single_gpu_vm72_vj96_20260924_r1'
        vmvj_run_id = 'main_v2_probe_v2_vm72_vj96_20260924_r1'
        vmvj_submissions = load(vmvj_root / 'submissions.json', {'jobs': {}})['jobs']
        vmvj_migration_path = vmvj_root / 'h800_to_zoetrope_20260924_r1.json'
        vmvj_migration = load(vmvj_migration_path, {'jobs': {}})
        for key, replacement in vmvj_migration.get('jobs', {}).items():
            if replacement.get('migration_state') != 'ZOETROPE_SUBMITTED' or key not in vmvj_submissions:
                continue
            original = vmvj_submissions[key]
            vmvj_submissions[key] = {
                **original,
                'job_id': replacement['new_job_id'],
                'job_name': replacement['new_job_name'],
                'migration': {
                    'old_job_id': replacement['old_job_id'],
                    'old_job_name': replacement['old_job_name'],
                    'new_pool': 'zoetrope',
                    'evidence': str(vmvj_migration_path),
                },
            }
        qf_root = BASE / 'single_gpu_qwen_f3r48_20260923_r1'
        qf_run_id = 'main_v2_probe_v2_qwen_f3r48_20260923_r1'
        qf_submissions = load(qf_root / 'submissions.json', {'jobs': {}})['jobs']
        # Thirty-two never-started Zoetrope Qwen/F3R jobs were deliberately
        # stopped and replaced on H800. Reconcile by unique key so the ledger
        # follows the active replacement instead of reporting the old job as
        # SUSPENDED. The item, SHA, command and run ID were preserved.
        qf_migration_path = (BASE / 'status_10models_20260923T1351Z' /
            'zoetrope_to_h800_20260923T1640Z' / 'result.json')
        qf_migration = load(qf_migration_path, {'replacements': []})
        qf_key_by_old_job = {
            record.get('job_id'): key for key, record in qf_submissions.items()
            if record.get('job_id')
        }
        for replacement in qf_migration.get('replacements', []):
            key = qf_key_by_old_job.get(replacement.get('old_job'))
            if not key:
                continue
            original = qf_submissions[key]
            qf_submissions[key] = {
                **original,
                'job_id': replacement['new_job'],
                'job_name': replacement['new_display_name'],
                'migration': {
                    'old_job_id': replacement['old_job'],
                    'old_job_name': replacement['old_display_name'],
                    'new_pool': replacement['pool'],
                    'evidence': str(qf_migration_path),
                },
            }
        dino_root = BASE / 'single_gpu_dino24_20260923_r1'
        dino_run_id = 'main_v2_probe_v2_dino24_20260923_r1'
        dino_submissions = load(dino_root / 'submissions.json', {'jobs': {}})['jobs']
        lingbot_roots = {}
        lingbot_submissions = {}
        for name in ('single_gpu_lingbot48_20260924_r1', 'single_gpu_lingbot32_20260924_r2', 'single_gpu_cog96_20260924_r1', 'single_gpu_cog84_h800_20260928_r1'):
            root = BASE / name
            plan = load(root / 'plan.json', {})
            for key, record in load(root / 'submissions.json', {'jobs': {}})['jobs'].items():
                if record.get('job_id'):
                    lingbot_submissions[key] = record
                    lingbot_roots[key] = (root, plan['run_id'])
        # One full H800 node was intentionally released on 2026-09-28. Six
        # Cog jobs needed a fresh state root on Zoetrope because the stopped
        # H800 workers had left root-owned receipts. Follow their repaired
        # replacement jobs rather than the intentionally suspended originals.
        node_release = load(BASE / 'h800_node_release_20260928_r1/migration.json', {})
        cog_h800 = load(BASE / 'single_gpu_cog84_h800_20260928_r1/submissions.json', {'jobs': {}})['jobs']
        cog_key_by_old_id = {record.get('job_id'): key for key, record in cog_h800.items()}
        restart_root = BASE / 'single_gpu_cog84_zoetrope_restart_20260928_r2'
        restart_plan = load(restart_root / 'plan.json', {})
        for replacement in node_release.get('zoetrope_state_root_repairs', []):
            if replacement.get('state') != 'ZOETROPE_SUBMITTED':
                continue
            key = cog_key_by_old_id.get(replacement.get('old_job_id'))
            if not key or key not in lingbot_submissions or not restart_plan:
                continue
            lingbot_submissions[key] = {
                **lingbot_submissions[key], 'job_id': replacement['new_job_id'],
                'job_name': replacement['new_job_name'],
                'migration_evidence': str(BASE / 'h800_node_release_20260928_r1/migration.json'),
            }
            lingbot_roots[key] = (restart_root, restart_plan['run_id'])
        for migration_path in (vmvj_root / 'zoetrope_half_to_h800_20260924_r1.json', BASE / 'vmvj_all_h800_20260924_r1/migration.json'):
            for key, replacement in load(migration_path, {'jobs': {}})['jobs'].items():
                if replacement.get('state') != 'SUBMITTED' or not replacement.get('new_job_id'):
                    continue
                for catalog in (vmvj_submissions, targetfix_submissions, production_submissions):
                    if key in catalog:
                        catalog[key] = {**catalog[key], 'job_id': replacement['new_job_id'],
                            'job_name': replacement['new_job_name'], 'submitted_utc': replacement.get('submitted_utc'),
                            'migration_evidence': str(migration_path)}
                        break
        current_jobs = load(OUT / 'scheduler_jobs.json', {'observed_utc': None, 'jobs': []})
        scheduler_by_id = {r['name']: r for r in current_jobs['jobs']}
        records, events = [], []
        for q in queue:
            key = q['unique_key']
            f = features[key]
            ready = f['feature_ready'] == 'True'
            m = model_evidence[q['model']]
            extraction_jobs = []
            for j in m['jobs']:
                if j['category'] != 'production':
                    continue
                live = scheduler_by_id.get(j['job_id'])
                extraction_jobs.append(dict(job_id=j['job_id'], name=j['display_name'],
                    state=live['state'] if live else j['state'],
                    observed_utc=current_jobs['observed_utc'] if live else evidence['scheduler_observed_utc']))
            if q['model'] == 'lingbot' and q['requires_target']:
                ling_handoff = load(BASE / 'lingbot_target_accelerate_20260924_r1/plan.json', {'owners': []})
                ling_jobs = load(BASE / 'lingbot_target_accelerate_20260924_r1/submissions.json', {'jobs': {}})['jobs']
                ids = [r['old_job_id'] for r in ling_handoff['owners']] + [r['job_id'] for r in ling_jobs.values() if r.get('job_id')]
                for jid in ids:
                    live = scheduler_by_id.get(jid)
                    if live:
                        extraction_jobs.append(dict(job_id=jid,name=live['display_name'],state=live['state'],observed_utc=current_jobs['observed_utc']))
            stage = 'WAITING_VALIDATION' if ready else 'FEATURE_PENDING'
            validation = dict(status='WAITING' if ready else 'NOT_READY', attempts=[])
            validation['prior_attempts']=history_cases.get(key,[])
            historical_passes=[r for r in validation['prior_attempts'] if r.get('status')=='PASS']
            if ready and historical_passes:
                validation['status']='PASS'
                validation['attempts']=[historical_passes[-1]]
                stage='VALIDATED_READY_TO_TRAIN'
            if key in requested_keys:
                validation['status'] = 'PREPARING'
                stage = 'VALIDATION_PREPARING'
            case = case_by_key.get(key)
            if case:
                attempt = load(Path(case['case_path']).parent / 'result.json')
                submission = extra_validation_submission_by_key.get(key) or submissions.get(str(case['shard']))
                validation.update(case_path=case['case_path'], shard=case['shard'], status='QUEUED')
                stage = 'VALIDATION_QUEUED'
                if submission:
                    validation['job_id'] = submission.get('job_id')
                    live = scheduler_by_id.get(submission.get('job_id'))
                    validation['scheduler_state'] = live['state'] if live else submission.get('state', 'SUBMITTED')
                    if submission.get('returncode') or not submission.get('job_id'):
                        validation['status'] = 'SUBMISSION_FAILED'
                        stage = 'VALIDATION_FAILED'
                    elif live and live['state'] in {'FAILED', 'STOPPED', 'CANCELLED'} and not attempt:
                        validation['status'] = 'JOB_FAILED_NO_RECEIPT'
                        stage = 'VALIDATION_FAILED'
                if attempt:
                    validation['attempts'] = [attempt]
                    validation['status'] = attempt['status']
                    validation['result_path'] = str(Path(case['case_path']).parent / 'result.json')
                    stage = {'RUNNING': 'VALIDATING', 'PASS': 'VALIDATED_READY_TO_TRAIN', 'FAIL': 'VALIDATION_FAILED'}.get(attempt['status'], 'VALIDATING')
                    validation['phase'] = attempt.get('phase')
                    live = scheduler_by_id.get(validation.get('job_id'))
                    if attempt['status'] == 'RUNNING' and live and live['state'] in {'FAILED','STOPPED','CANCELLED','SUCCEEDED'}:
                        validation['status'] = 'JOB_ENDED_WITHOUT_FINAL_RECEIPT'
                        stage = 'VALIDATION_FAILED'
            physical_group = physical_by_spec.get((q['model'],q['disk_layer'],q['timestep'],q['prefix']))
            physical_failed = bool(physical_group and (
                physical_group['failed_keys'].get('streaming',0) or
                (q['requires_target'] and physical_group['failed_keys'].get('target',0))))
            training = load(OUT / 'training' / (key + '.json'), {'status': 'NOT_STARTED'})
            production = lingbot_submissions.get(key) or dino_submissions.get(key) or qf_submissions.get(key) or aether_submissions.get(key) or vmvj_submissions.get(key) or targetfix_submissions.get(key) or production_submissions.get(key)
            if production:
                if key in lingbot_submissions:
                    current_root, current_run_id = lingbot_roots[key]
                    ledger_name = 'submissions.json'
                elif key in dino_submissions:
                    current_root, current_run_id, ledger_name = dino_root, dino_run_id, 'submissions.json'
                elif key in qf_submissions:
                    current_root, current_run_id, ledger_name = qf_root, qf_run_id, 'submissions.json'
                elif key in aether_submissions:
                    current_root, current_run_id, ledger_name = aether_root, aether_run_id, 'submissions.json'
                elif key in vmvj_submissions:
                    current_root, current_run_id, ledger_name = vmvj_root, vmvj_run_id, 'submissions.json'
                elif key in targetfix_submissions:
                    current_root, current_run_id, ledger_name = targetfix_root, targetfix_run_id, 'submissions.json'
                else:
                    current_root, current_run_id, ledger_name = production_root, production_run_id, 'production_submissions.json'
                current_state = current_root / 'state' / 'production' / current_run_id
                training = dict(
                    status='QUEUED', job_id=production.get('job_id'),
                    job_name=production.get('job_name'), submitted_utc=production.get('submitted_utc'),
                    submission_ledger=str(current_root / ledger_name),
                )
                if key in targetfix_submissions and key in production_submissions:
                    old_attempt = production_submissions[key]
                    training['prior_attempts'] = [{
                        'job_id': old_attempt.get('job_id'), 'job_name': old_attempt.get('job_name'),
                        'submitted_utc': old_attempt.get('submitted_utc'), 'status': 'FAILED',
                        'reason': 'superseded after legacy target owner URI returned 404',
                    }]
                live = scheduler_by_id.get(production.get('job_id'))
                if live:
                    training['scheduler_state'] = live['state']
                    training['scheduler_observed_utc'] = current_jobs['observed_utc']
                    training['resource_pool'] = live.get('resource_pool', {}).get('name')
                    if live['state'] == 'RUNNING':
                        training['status'] = 'RUNNING'
                    elif live['state'] in {'FAILED', 'STOPPED', 'CANCELLED'}:
                        training['status'] = 'FAILED'
                state_receipt = load(current_state / 'experiments' / (key + '.json'))
                if state_receipt:
                    training['state_receipt'] = str(current_state / 'experiments' / (key + '.json'))
                    if state_receipt.get('state') in {'STARTING', 'RUNNING'}:
                        training['status'] = 'RUNNING'
                    elif state_receipt.get('state') == 'FAILED':
                        training['status'] = 'FAILED'
                complete_receipt = current_state / 'completed' / (key + '.json')
                if complete_receipt.exists():
                    training['status'] = 'SUCCEEDED'
                    training['completion_receipt'] = str(complete_receipt)
            if training['status'] in {'QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED'}:
                stage = 'TRAINING_' + training['status']
            gaps = EVIDENCE / 'unverified_keys'
            stem = f"{q['model']}__l{q['disk_layer']}__p{q['prefix']}"
            missing_modes = [mode for mode in ('streaming', 'target') if int(f[mode + '_unverified'])]
            rec = dict(unique_key=key, queue_ordinal=q['queue_ordinal'], model=q['model'], layer=q['layer'],
                       disk_layer=q['disk_layer'], timestep=q['timestep'], prefix=q['prefix'], probe=q['probe'], seed=q['seed'],
                       stage=stage, feature_ready=ready and not physical_failed,
                       receipt_coverage_complete=ready, requires_target=q['requires_target'],
                       feature_observed_utc=evidence['updated_utc'], missing_feature_modes=missing_modes,
                       streaming=dict(required=int(f['streaming_required']), verified=int(f['streaming_verified']),
                                      unverified=int(f['streaming_unverified']), details=str(gaps / (stem + '__streaming.jsonl.gz'))),
                       target=dict(required=int(f['target_required']), unverified=int(f['target_unverified']),
                                   details=str(gaps / (stem + '__target.jsonl.gz')) if q['requires_target'] else None),
                       extraction_jobs=extraction_jobs, validation=validation, training=training,
                       experiment_config=q['experiment_config'], config_sha256=q['experiment_config_sha256'])
            if physical_failed:
                rec['physical_feature_validation'] = physical_group
                if not case and rec['stage'] == 'WAITING_VALIDATION':
                    rec['stage']='FEATURE_PENDING'
                    validation['status']='NOT_READY'
                    stage=rec['stage']
                for mode in ('streaming','target'):
                    count=physical_group['failed_keys'].get(mode,0) if mode=='streaming' or q['requires_target'] else 0
                    rec[mode]['physical_failed_keys']=count
                    if count:
                        rec[mode]['unverified']=max(rec[mode]['unverified'],count)
                        rec[mode]['details']=physical_group['failure_details']
                        if mode not in rec['missing_feature_modes']:
                            rec['missing_feature_modes'].append(mode)
                rec['streaming']['verified']=rec['streaming']['required']-rec['streaming']['unverified']
            active = [j for j in extraction_jobs if j['state'] in {'RUNNING', 'STARTING', 'QUEUEING', 'PENDING'}]
            rec['feature_work_status'] = 'READY' if ready else ('ACTIVE_EXTRACTION_OR_PUBLICATION' if active else 'PENDING_NO_ACTIVE_JOB_CONFIRMED')
            if physical_failed:
                rec['feature_work_status']='PHYSICAL_READ_FAILED_NEEDS_ROUTE_RECOVERY'
            recovery_applies=q['model']=='aether' or (q['model']=='videomae' and q['prefix']==24)
            if recovery and recovery_applies:
                complete=recovery['total']==6917 and recovery['counts'].get('PASS_REMOTE_RESTORED_FULL_SHA')==6917
                rec['uri_recovery']=dict(status='RESTORED_FULL_SHA' if complete else 'RESTORING',
                    processed=recovery['processed'],total=6917,updated_utc=recovery['updated_utc'],
                    evidence_path=str(recovery_root/('restore_summary.json' if complete else 'restore_progress.json')))
                if physical_failed:
                    rec['feature_work_status']='RESTORED_WAITING_REVALIDATION' if complete else 'RESTORING_EXACT_URIS'
            old_row = previous.get(key)
            rec['stage_changed_utc'] = old_row['stage_changed_utc'] if old_row and old_row['stage'] == stage else now()
            if not old_row or old_row['stage'] != stage:
                events.append(dict(at=now(), unique_key=key, previous=old_row['stage'] if old_row else None, stage=stage))
            records.append(rec)
        assert len(records) == len({r['unique_key'] for r in records}) == 552
        summary = dict(updated_utc=now(), feature_observed_utc=evidence['updated_utc'], total=552,
                       stages=dict(Counter(r['stage'] for r in records)), feature_ready=sum(r['feature_ready'] for r in records),
                       receipt_coverage_complete=sum(r['receipt_coverage_complete'] for r in records),
                       requested_validation=len(requested_keys), scheduler_observed_utc=current_jobs['observed_utc'])
        payload = dict(schema='PROBE552_PROGRESS_V1', **summary, experiments=records)
        atomic(OUT / 'experiments_552.json', json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
        flat = []
        for r in records:
            flat.append(dict(unique_key=r['unique_key'], model=r['model'], layer=r['layer'], prefix=r['prefix'], probe=r['probe'],
                stage=r['stage'], feature_work_status=r['feature_work_status'],
                streaming_unverified=r['streaming']['unverified'], target_unverified=r['target']['unverified'],
                streaming_physical_failed=r['streaming'].get('physical_failed_keys',0),
                uri_restore_status=r.get('uri_recovery',{}).get('status',''),
                physical_checked_utc=r.get('physical_feature_validation',{}).get('checked_utc',''),
                streaming_missing_details=r['streaming']['details'], target_missing_details=r['target']['details'],
                validation_status=r['validation']['status'], validation_phase=r['validation'].get('phase', ''), validation_job_id=r['validation'].get('job_id', ''),
                validation_error=(r['validation']['attempts'][-1].get('error', '') if r['validation']['attempts'] else ''),
                training_status=r['training']['status'], training_job_id=r['training'].get('job_id', ''),
                feature_observed_utc=r['feature_observed_utc'], stage_changed_utc=r['stage_changed_utc']))
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
        atomic(OUT / 'experiments_552.csv', buf.getvalue())
        atomic(OUT / 'summary.json', json.dumps(summary, indent=2) + '\n')
        with (OUT / 'events.jsonl').open('a') as f:
            for event in events:
                f.write(json.dumps(event) + '\n')
        md = ['# 552 项实验进度', '', f"更新：{summary['updated_utc']}；特征证据：{summary['feature_observed_utc']}", '',
              '[逐项 JSON](experiments_552.json) · [逐项表格 CSV](experiments_552.csv) · [启动窗口与阻塞分析](ETA_REPORT_20260923.md)', '',
              '| 状态 | 数量 |', '|---|---:|']
        md += [f'| {k} | {v} |' for k, v in sorted(summary['stages'].items())]
        md += ['', '状态：FEATURE_PENDING（特征未齐，见 feature_work_status 和未验证键明细）→ WAITING_VALIDATION → VALIDATION_PREPARING/QUEUED → VALIDATING → VALIDATED_READY_TO_TRAIN → TRAINING_QUEUED/RUNNING/SUCCEEDED。失败单列，不自动视为通过。', '',
               'unverified 表示尚无验证证据，不等于文件不存在或需要重新提取。训练状态仅依据独立正式训练记录，不把 canary 当正式实验。', '',
               '刷新状态：`python3 scripts/update_probe552_progress.py`',
               '刷新特征覆盖及精确缺口：`python3 scripts/update_probe552_progress.py --refresh-features`', '',
               '验证 worker 在每项开始/完成时自动刷新此表；正式训练进展见 TRAINING_QUEUED/RUNNING/SUCCEEDED。',
               '持续监控：`python3 scripts/watch_probe552_progress.py`，每分钟更新任务状态，每十分钟更新特征证据。',
               '特征验证中的当前 404/失败证据优先于历史完成记录；feature_ready 不会仅凭旧 receipt 恢复为 true。', '']
        atomic(OUT / 'README.md', '\n'.join(md))
        print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-features', action='store_true')
    args = parser.parse_args()
    refresh(refresh_features=args.refresh_features)
