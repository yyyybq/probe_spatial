"""Use the platform's normal worker-login CLI for explicitly scoped control."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import pty
import re
import select
import shlex
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/main_v2_probe_v2_816_20260917/upload_recovery_20260923_r1'
SCO = '/mnt/umm/users/yinbaiqiao/.sco/bin/sco'
COG = {'pt-rutyedtm', 'pt-tkvc8tad', 'pt-v2ejbno9', 'pt-3cx4i1b8'}
MINI = {'pt-4lma0exb', 'pt-z7wd113c', 'pt-ili01g8p', 'pt-9cozgsr6',
        'pt-d7l8a52q', 'pt-q0j47v8u', 'pt-yrla3xen', 'pt-k6luh9rd'}


def control(job, action):
    assert job in COG | MINI
    model = 'cogvideox' if job in COG else 'minimax_h3'
    folder = OUT / 'scheduler_before' / job
    folder.mkdir(parents=True, exist_ok=True)
    details = json.loads(subprocess.check_output([SCO, 'acp', 'jobs', 'describe', job,
                         '--workspace-name', 'aigc', '-o', 'json'], text=True, timeout=30))
    # Keep the scheduler receipt before any signals, including exact command/resources.
    if not (folder / 'job.json').exists():
        (folder / 'job.json').write_text(json.dumps(details, indent=2)+'\n')
    assert details['state'] == 'RUNNING', f'{job} is not RUNNING; nothing signalled'
    cmd = '\n'.join(r.get('startup_script', '') for r in details['roles'])
    assert ('cogvideox-stream-' in cmd if model == 'cogvideox' else 'main_v2_minimax_bf16_v3_extractor/run_production_shard_v3.sh' in cmd)
    workers = subprocess.check_output([SCO, 'acp', 'jobs', 'get-workers', job,
                                      '--workspace-name', 'aigc'], text=True, timeout=30)
    names = sorted(set(re.findall(r'pt-[0-9a-f]+-worker-\d+', workers)))
    assert len(names) == 1, f'expected one worker: {names}'
    worker = names[0]
    (folder / 'worker.txt').write_text(workers)
    master, slave = pty.openpty()
    process = subprocess.Popen([SCO, 'acp', 'jobs', 'exec', job, '--workspace-name', 'aigc',
                                '--worker-name', worker], stdin=slave, stdout=slave, stderr=slave,
                               start_new_session=True)
    os.close(slave)
    output = b''
    sent = False
    request_id=uuid.uuid4().hex
    deadline = time.monotonic() + 55
    argv = ['/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python', str(ROOT / 'scripts/feature_upload_recovery_control_20260923.py'), action,
            '--job', job, '--worker', worker, '--model', model,'--request-id',request_id]
    if action.startswith('upload-canary-'):
        argv += ['--publish-sha', hashlib.sha256((ROOT/'runtime/feature_upload_recovery_v1/minimax_publish.py').read_bytes()).hexdigest()]
        deadline = time.monotonic() + 600
    try:
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.25)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
                (folder / (action + '_login.log')).write_bytes(output)
                if not sent and re.search(rb'[#\$] ', output):
                    os.write(master, ('exec ' + shlex.join(argv) + '\n').encode())
                    sent = True
                if any(s in output for s in (b'"status": "EXTRACTOR_PAUSED_SYNCS_UNTOUCHED"', b'"status": "EXTRACTOR_RESUMED"', b'"status": "UPLOAD_CANARY_PASS"', b'"status": "PAUSED_AT_COMMITTED_BOUNDARY"', b'"status": "PAUSED_WITH_PENDING_TEMP"', b'"status": "COG_GUARD_ACTIVE"')):
                    break
                if b'command terminated with exit code' in output:
                    break
                if action=='inspect' and b'"action": "inspect"' in output:
                    break
            if process.poll() is not None:
                break
    finally:
        if process.poll() is None:
            process.terminate()  # only the local login client, never a job process
        process.wait(timeout=10)
        os.close(master)
    (folder / (action + '_login.log')).write_bytes(output)
    expected = OUT / 'process_control' / job / {'pause':'paused.json', 'resume':'resumed.json', 'inspect':'inspection.json'}.get(action,action+'.json')
    assert expected.exists(), f'{job}: control not confirmed; inspect {folder}/{action}_login.log'
    result = json.loads(expected.read_text())
    assert result['worker'] == worker and result.get('request_id')==request_id, 'stale control receipt is not current success'
    print(json.dumps({'job':job, 'worker':worker, 'status':result.get('status', 'INSPECTED'),
                      'extractor_pid':result.get('after', {}).get('pid'),
                      'syncs_untouched':result['sync_processes_not_signalled']}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['inspect', 'pause', 'resume', 'settle-pause', 'upload-canary-streaming', 'upload-canary-target', 'install-cog-guard'])
    p.add_argument('jobs', nargs='+')
    args = p.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(control, job, args.action):job for job in args.jobs}
        failures = []
        for future, job in futures.items():
            try:
                future.result()
            except Exception as exc:
                failures.append(job)
                print(json.dumps({'job':job, 'error':str(exc)}), flush=True)
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
