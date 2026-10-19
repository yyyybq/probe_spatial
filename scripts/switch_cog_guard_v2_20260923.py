"""Use scoped worker login to switch the four CogVideoX guards in place."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import pty
import re
import select
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SCO = '/mnt/umm/users/yinbaiqiao/.sco/bin/sco'
JOBS = {'pt-rutyedtm', 'pt-3cx4i1b8', 'pt-tkvc8tad', 'pt-v2ejbno9'}
BASE = ROOT/'artifacts/main_v2_probe_v2_816_20260917/upload_recovery_20260923_r1/cog_guard_v2'


def switch(job, pack_sha):
    assert job in JOBS
    details = json.loads(subprocess.check_output(
        [SCO, 'acp', 'jobs', 'describe', job, '--workspace-name', 'aigc', '-o', 'json'],
        text=True, timeout=30,
    ))
    assert details['state'] == 'RUNNING'
    workers = subprocess.check_output(
        [SCO, 'acp', 'jobs', 'get-workers', job, '--workspace-name', 'aigc'],
        text=True, timeout=30,
    )
    names = sorted(set(re.findall(r'pt-[0-9a-f]+-worker-\d+', workers)))
    assert len(names) == 1
    worker = names[0]
    folder = BASE/job
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'worker_query.txt').write_text(workers)
    status_path = folder/'status.json'
    if status_path.exists():
        existing = json.loads(status_path.read_text())
        if (
            existing.get('pack_sha256') == pack_sha
            and existing.get('worker') == worker
            and existing.get('phase') == 'EXTRACTING'
            and time.time() - existing.get('epoch', 0) < 600
        ):
            return existing

    master, slave = pty.openpty()
    login = subprocess.Popen(
        [SCO, 'acp', 'jobs', 'exec', job, '--workspace-name', 'aigc', '--worker-name', worker],
        stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
    )
    os.close(slave)
    output = b''
    sent = False
    prepare_argv = [
        '/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python',
        str(ROOT/'runtime/cog_upload_backpressure_v2/switch_guard.py'),
        '--job', job, '--pack-sha', pack_sha, '--prepare-only',
    ]
    guard_argv = [
        'nohup', '/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python',
        str(ROOT/'runtime/cog_upload_backpressure_v2/guard.py'),
        '--job', job, '--pack-sha', pack_sha,
    ]
    command = '( ' + ' '.join(shlex.quote(value) for value in prepare_argv)
    command += ' >>' + shlex.quote(str(folder/'prepare.log')) + ' 2>&1'
    command += ' && exec ' + ' '.join(shlex.quote(value) for value in guard_argv)
    command += ' >>' + shlex.quote(str(folder/'guard.log')) + ' 2>&1 </dev/null ) &\n'
    command += 'disown\nexit\n'
    deadline = time.monotonic() + 150
    try:
        while time.monotonic() < deadline:
            if select.select([master], [], [], .25)[0]:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
                (folder/'switch_login.log').write_bytes(output)
                if not sent and re.search(rb'[#\$] ', output):
                    os.write(master, command.encode())
                    sent = True
                if b'command terminated with exit code' in output:
                    break
            if login.poll() is not None:
                break
    finally:
        if login.poll() is None:
            login.terminate()
        login.wait(timeout=10)
        os.close(master)
    (folder/'switch_login.log').write_bytes(output)
    assert sent, f'{job}: remote command was not sent'
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if status_path.exists():
            value = json.loads(status_path.read_text())
            if value['pack_sha256'] == pack_sha and value['worker'] == worker:
                assert value['phase'] == 'EXTRACTING'
                return value
        time.sleep(2)
    raise RuntimeError(f'{job}: no fresh v2 status; inspect guard.log')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack-sha', required=True)
    parser.add_argument('jobs', nargs='+')
    args = parser.parse_args()
    assert set(args.jobs) <= JOBS and len(args.jobs) == len(set(args.jobs))
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(switch, job, args.pack_sha): job for job in args.jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(json.dumps({'job': job, 'error': str(exc)}), flush=True)
    print(json.dumps({'results': results}, indent=2), flush=True)
    if len(results) != len(args.jobs):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
