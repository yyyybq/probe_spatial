#!/usr/bin/env python3
"""Move exactly one eight-GPU CogVideoX node from H800 to Zoetrope."""
import argparse
import collections
import concurrent.futures
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/mnt/umm/users/yinbaiqiao/probe_spatial')
BASE = ROOT / 'artifacts/main_v2_probe_v2_816_20260917'
SCO = '/mnt/umm/users/yinbaiqiao/.sco/bin/sco'
LEDGERS = (
    ('cog_formal', BASE / 'single_gpu_cog84_h800_20260928_r1/submissions.json'),
    ('cog_validation', BASE / 'probe_validation_cog12_ego_20260928_r2/validation_submissions.json'),
    ('lingbot_validation', BASE / 'probe_validation_lingbot32_20260924_r2/validation_submissions.json'),
)
OUT = BASE / 'h800_node_release_20260928_r2/migration.json'
SCHEDULER_SNAPSHOT = Path('/tmp/all_scheduler_for_node_20260928.json')
REPAIR_ROOT = BASE / 'single_gpu_cog84_zoetrope_restart_20260928_r2'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(value):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name('.' + OUT.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(OUT)


def run(*args):
    completed = subprocess.run([SCO, 'acp', 'jobs', *args], text=True, capture_output=True, timeout=180)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    return completed.stdout


def host_for(job_id):
    text = run('get-workers', job_id, '--workspace-name=aigc')
    for line in text.splitlines():
        fields = [field.strip() for field in line.split('|')]
        if len(fields) >= 6 and re.fullmatch(r'\d+(?:\.\d+){3}', fields[3]):
            return fields[3], fields[-2]
    return None, None


def describe(job_id):
    return json.loads(run('describe', job_id, '--workspace-name=aigc', '-o', 'json'))


def candidates():
    submitted = []
    for source, ledger in LEDGERS:
        records = json.loads(ledger.read_text())['jobs']
        submitted.extend((source, key, record) for key, record in records.items()
                         if record.get('state') == 'SUBMITTED' and record.get('job_id'))
    def inspect_one(item):
        source, key, record = item
        if record.get('state') != 'SUBMITTED' or not record.get('job_id'):
            return None
        job = describe(record['job_id'])
        if job.get('resource_pool', {}).get('name') != 'h800' or job.get('state') != 'RUNNING':
            return None
        host, worker_state = host_for(record['job_id'])
        if host and worker_state == 'Running':
            return dict(source=source, unique_key=key, job_id=record['job_id'],
                        job_name=record.get('job_name', record['job_id']), command=record['command'], host_ip=host)
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        return [row for row in pool.map(inspect_one, submitted) if row]


def scheduler_candidates():
    rows = json.loads(SCHEDULER_SNAPSHOT.read_text())
    jobs = [job for job in rows if job.get('state') == 'RUNNING'
            and job.get('resource_pool', {}).get('name') == 'h800'
            and job.get('display_name', '').startswith('bq-')]
    def inspect_one(job):
        host, worker_state = host_for(job['name'])
        if host and worker_state == 'Running':
            return dict(source='scheduler', unique_key=job['display_name'], job_id=job['name'],
                        job_name=job['display_name'], command=job['roles'][0]['startup_script'], host_ip=host)
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        return [row for row in pool.map(inspect_one, jobs) if row]


def inspect(include_scheduler=False):
    grouped = collections.defaultdict(list)
    records = scheduler_candidates() if include_scheduler else candidates()
    for record in records:
        grouped[record['host_ip']].append(record)
    report = dict(schema='H800_PROBE_NODE_MIGRATION_V1', inspected_utc=now(),
                  source='scheduler_snapshot' if include_scheduler else 'campaign_ledgers',
                  source_ledgers=[str(path) for _, path in LEDGERS],
                  hosts={host: rows for host, rows in sorted(grouped.items())},
                  eligible_hosts=sorted(host for host, rows in grouped.items() if len(rows) == 8))
    save(report)
    print(json.dumps(dict(hosts={host: len(rows) for host, rows in grouped.items()},
                          eligible_hosts=report['eligible_hosts']), sort_keys=True))


def migrate(host):
    if not OUT.exists():
        raise RuntimeError('run inspect before migrate')
    report = json.loads(OUT.read_text())
    records = report['hosts'].get(host, [])
    if len(records) not in {7, 8}:
        raise RuntimeError(f'{host} does not contain a complete releasable 7- or 8-task allocation')
    # Recheck the assignment immediately before stopping to protect against a
    # scheduler move or an already-completed task.
    fresh = {row['job_id']: (describe(row['job_id']), host_for(row['job_id'])) for row in records}
    if any(job.get('state') != 'RUNNING' or job.get('resource_pool', {}).get('name') != 'h800'
           or worker[0] != host or worker[1] != 'Running' for job, worker in fresh.values()):
        raise RuntimeError('worker placement changed; refusing to stop a non-exact node')
    report.update(status='STOP_INTENT', selected_host=host, selected_jobs=records, stop_intent_utc=now())
    save(report)
    run('stop', '--workspace-name=aigc', *(row['job_id'] for row in records))
    report.update(status='STOP_REQUESTED', stop_requested_utc=now())
    save(report)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        stopped = [describe(row['job_id']) for row in records]
        if all(job.get('state') in {'SUSPENDED', 'STOPPED', 'CANCELLED'} for job in stopped):
            break
        time.sleep(5)
    else:
        raise RuntimeError('selected jobs did not all enter a terminal stopped state')
    # The selected node is released from this campaign only when no selected
    # worker remains Running there. The exact post-stop evidence is retained.
    post = {row['job_id']: dict(state=describe(row['job_id']).get('state'), worker=host_for(row['job_id'])) for row in records}
    if any(value['worker'][1] == 'Running' for value in post.values()):
        raise RuntimeError('a selected worker still runs after stop')
    report.update(status='NODE_RELEASED_CONFIRMED', released_utc=now(), post_stop=post)
    save(report)
    for row in records:
        entry = next(item for item in report['selected_jobs'] if item['job_id'] == row['job_id'])
        entry['migration_state'] = 'SUBMIT_INTENT'
        save(report)
        name = 'bq-cog-h8free-z-' + row['job_id'][3:]
        output = run('create', '--workspace-name=aigc', '--aec2-name=zoetrope', f'--job-name={name}',
                     '--priority=HIGHEST', '--quota-type=reserved',
                     '--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617',
                     '--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm',
                     '--training-framework=pytorch', '--worker-nodes=1', '--worker-spec=N4lS.Iq.I80.1',
                     f"--command={row['command']}")
        match = re.search(r'job id\s*:\s*(pt-[a-z0-9]+)', output, re.I)
        if not match:
            raise RuntimeError('ambiguous Zoetrope submission: ' + output)
        entry.update(migration_state='ZOETROPE_SUBMITTED', new_job_id=match.group(1), new_job_name=name,
                     submitted_utc=now())
        save(report)
    report.update(status='COMPLETE', completed_utc=now())
    save(report)
    print(json.dumps(dict(released_host=host, migrated=[r['new_job_id'] for r in report['selected_jobs']])))


def repair_cog():
    if not OUT.exists() or not (REPAIR_ROOT / 'plan.json').exists():
        raise RuntimeError('migration evidence and fresh restart plan are required')
    report = json.loads(OUT.read_text())
    source = json.loads(LEDGERS[0][1].read_text())['jobs']
    plan = json.loads((REPAIR_ROOT / 'plan.json').read_text())
    plan_by_key = {row['unique_key']: row for row in plan['jobs']}
    runner = Path(plan['runner'])
    repairs = report.setdefault('zoetrope_state_root_repairs', [])
    repaired_old = {row['old_job_id'] for row in repairs if row.get('new_job_id')}
    for old in report['selected_jobs']:
        if old['job_id'] in repaired_old or old['job_id'] not in {row['job_id'] for row in source.values()}:
            continue
        key = next(key for key, row in source.items() if row['job_id'] == old['job_id'])
        item = plan_by_key[key]
        command = ' '.join([str('/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python'), str(runner),
                            '--item', item['item_path'], '--item-sha256', item['item_sha256'],
                            '--run-id', plan['run_id']])
        record = dict(old_job_id=old['job_id'], failed_zoetrope_job_id=old['new_job_id'], unique_key=key,
                      fresh_runner=str(runner), fresh_run_id=plan['run_id'], command=command,
                      state='SUBMIT_INTENT', submitted_utc=now())
        repairs.append(record); save(report)
        name = 'bq-cog-h8free-r2-' + old['job_id'][3:]
        output = run('create', '--workspace-name=aigc', '--aec2-name=zoetrope', f'--job-name={name}',
                     '--priority=HIGHEST', '--quota-type=reserved',
                     '--container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617',
                     '--storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm',
                     '--training-framework=pytorch', '--worker-nodes=1', '--worker-spec=N4lS.Iq.I80.1',
                     f'--command={command}')
        match = re.search(r'job id\s*:\s*(pt-[a-z0-9]+)', output, re.I)
        if not match:
            raise RuntimeError('ambiguous fresh-root submission: ' + output)
        record.update(state='ZOETROPE_SUBMITTED', new_job_id=match.group(1), new_job_name=name)
        save(report)
    print(json.dumps(dict(repaired=[row.get('new_job_id') for row in repairs])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('inspect', 'migrate', 'repair-cog'))
    parser.add_argument('--host')
    parser.add_argument('--include-scheduler', action='store_true')
    args = parser.parse_args()
    if args.action == 'inspect':
        inspect(args.include_scheduler)
    elif args.action == 'migrate':
        if not args.host:
            parser.error('--host is required for migrate')
        migrate(args.host)
    else:
        repair_cog()


if __name__ == '__main__':
    main()
