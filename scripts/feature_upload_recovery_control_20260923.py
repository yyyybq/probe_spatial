"""Narrow process-level pause/resume. Never signal a scheduler job or uploader."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

BASE = Path('/mnt/umm/users/yinbaiqiao/probe_spatial/artifacts/main_v2_probe_v2_816_20260917/upload_recovery_20260923_r1')


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    assert os.uname().machine in ('x86_64', 'aarch64')
    # The cluster's conda builds omit the Python wrappers even on a modern
    # kernel. Use the same Linux pidfd syscall, never fall back to a reused PID.
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open failed')
    return fd


def pidfd_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    libc = ctypes.CDLL(None, use_errno=True)
    rc = libc.syscall(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))
    if rc < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_send_signal failed')


def process(pid):
    p = Path('/proc') / str(pid)
    argv = [x.decode(errors='replace') for x in (p / 'cmdline').read_bytes().split(b'\0') if x]
    stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': int(pid), 'argv': argv, 'state': stat[0], 'ppid': int(stat[1]),
            'start_ticks': stat[19], 'command_sha256': hashlib.sha256(b'\0'.join(x.encode() for x in argv)).hexdigest()}


def field(argv, name):
    return argv[argv.index(name)+1] if name in argv else None


def pending_writes(pid, syncs):
    paths = set()
    for fd in (Path('/proc')/str(pid)/'fd').iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if '/InsScene-15K/FEAT_STREAMING_' in target and '.tmp-' in target:
            paths.add(target)
    for s in syncs:
        try:
            env = dict(x.split(b'=',1) for x in (Path('/proc')/str(s['pid'])/'environ').read_bytes().split(b'\0') if b'=' in x)
            root = Path(env[b'FEATURE_ROOT'].decode())
            listing = Path(env[b'LOG_DIR'].decode())/'atomic_temps.txt'
            for rel in listing.read_text().splitlines():
                if rel and not Path(rel).is_absolute() and '..' not in Path(rel).parts and (root/rel).exists():
                    paths.add(str(root/rel))
        except (OSError, KeyError):
            continue
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect', 'pause', 'resume', 'settle-pause', 'upload-canary-streaming', 'upload-canary-target', 'install-cog-guard'])
    parser.add_argument('--job', required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--model', required=True, choices=['cogvideox', 'minimax_h3'])
    parser.add_argument('--publish-sha')
    parser.add_argument('--request-id')
    args = parser.parse_args()
    assert socket.gethostname() == args.worker, 'worker identity mismatch'
    assert args.job.startswith('pt-') and '/' not in args.job
    folder = BASE / 'process_control' / args.job
    folder.mkdir(parents=True, exist_ok=True)
    candidates, syncs, publishers = [], [], []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            row = process(p.name)
        except (OSError, ValueError, IndexError):
            continue
        argv = row['argv']
        if any(x.endswith('/minimax_publish.py') for x in argv):
            detail={k:row[k] for k in ('pid','ppid','state','start_ticks')}
            try:
                detail['wchan']=(p/'wchan').read_text()
                detail['io']=(p/'io').read_text()
                detail['fds']=[os.readlink(f) for f in (p/'fd').iterdir()]
            except OSError:
                pass
            publishers.append(detail)
        if '-m' in argv and field(argv, '-m') == 'features.run_inscene15k' and field(argv, '--vfm') == args.model and field(argv, '--mode') == 'streaming_prefix':
            candidates.append(row)
        elif any('sync_loop' in x or x.endswith('ads-cli') for x in argv):
            # Never persist uploader argv: it may contain inline credentials.
            saved={k:row[k] for k in ('pid', 'ppid', 'state', 'start_ticks')}
            if any('sync_loop' in x for x in argv):
                script_fd=p/'fd/255'
                try:
                    saved['open_script_sha256']=hashlib.sha256(script_fd.read_bytes()).hexdigest()
                    saved['open_script_path']=os.readlink(script_fd)
                except OSError:
                    saved['open_script_sha256']=None
            syncs.append(saved)
    report = {'epoch': time.time(), 'job': args.job, 'worker': args.worker, 'model': args.model,
              'action': args.action, 'candidates': candidates, 'sync_processes_not_signalled': syncs,
              'publisher_diagnostics':publishers, 'request_id':args.request_id}
    if args.action == 'install-cog-guard':
        assert args.model=='cogvideox' and len(candidates)==1 and candidates[0]['state'] in ('T','t')
        release=BASE.parents[2]/'runtime/cog_upload_backpressure_v1'
        pack=release/'pack.json'; pack_sha=hashlib.sha256(pack.read_bytes()).hexdigest()
        config=json.loads(pack.read_text()); assert args.job in config['jobs']
        assert hashlib.sha256((release/'guard.py').read_bytes()).hexdigest()==config['guard_sha256']
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==config['process_helper_sha256']
        directory=BASE/'cog_guard'/args.job; directory.mkdir(parents=True,exist_ok=True)
        status_path=directory/'status.json'
        if status_path.exists():
            previous=json.loads(status_path.read_text())
            if time.time()-previous['epoch']<90:
                raise RuntimeError('guard heartbeat exists; inspect instead of duplicate launch')
        started=time.time()
        with (directory/'guard.log').open('ab') as log:
            guard=subprocess.Popen([sys.executable,str(release/'guard.py'),'--job',args.job,'--pack-sha',pack_sha],
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        for _ in range(60):
            if guard.poll() is not None: raise RuntimeError('guard exited; inspect guard.log')
            if status_path.exists():
                state=json.loads(status_path.read_text())
                if state['epoch']>=started and state['guard_pid']==guard.pid:
                    report.update(status='COG_GUARD_ACTIVE',guard=state)
                    (folder/'install-cog-guard.json').write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps(report)); return
            time.sleep(.5)
        raise RuntimeError('guard startup unconfirmed; inspect before any retry')
    if args.action == 'settle-pause':
        assert args.model=='cogvideox' and len(candidates)==1
        row=candidates[0]; assert row['state'] in ('T','t')
        old=json.loads((folder/'paused.json').read_text())
        assert row['start_ticks']==old['after']['start_ticks'] and row['command_sha256']==old['after']['command_sha256']
        fd=pidfd_open(row['pid'])
        report['pending_before']=pending_writes(row['pid'],syncs)
        try:
            deadline=time.monotonic()+15
            pending=report['pending_before']
            while pending and time.monotonic()<deadline:
                pidfd_signal(fd,signal.SIGCONT)
                time.sleep(0.2)
                pidfd_signal(fd,signal.SIGSTOP)
                for _ in range(50):
                    if process(row['pid'])['state'] in ('T','t'): break
                    time.sleep(0.05)
                pending=pending_writes(row['pid'],syncs)
            report['pending_after']=pending
        finally:
            pidfd_signal(fd,signal.SIGSTOP)
            os.close(fd)
        report['after']=process(row['pid'])
        report['status']='PAUSED_AT_COMMITTED_BOUNDARY' if not pending else 'PAUSED_WITH_PENDING_TEMP'
        (folder/'settle-pause.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report)); return
    if args.action.startswith('upload-canary-'):
        assert args.model == 'minimax_h3'
        assert len(candidates)==1 and candidates[0]['state'] in ('T','t'), 'canary requires paused extractor'
        source = BASE.parents[2] / 'runtime/feature_upload_recovery_v1/minimax_publish.py'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==args.publish_sha
        mode = args.action.removeprefix('upload-canary-')
        assert args.job == 'pt-4lma0exb', 'first-shard canary only'
        assert not publishers, 'another canary publisher is still active'
        with (folder/(args.action+'.log')).open('a') as log:
            subprocess.run([sys.executable,str(source),'--mode',mode,'--shard','0','--limit','1','--canary','--node-canary'],check=True,stdout=log,stderr=subprocess.STDOUT)
        report['status']='UPLOAD_CANARY_PASS'
        report['publish_sha256']=args.publish_sha
        (folder / (args.action+'.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report)); return
    if args.action == 'inspect':
        (folder / 'inspection.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report)); return
    if args.action == 'pause':
        assert len(candidates) == 1, f'expected one streaming extractor, got {len(candidates)}; nothing signalled'
        row = candidates[0]
        assert not (folder / 'paused.json').exists(), 'pause record already exists; inspect before retry'
        (folder / 'before_pause.json').write_text(json.dumps(report, indent=2)+'\n')
        pidfd = pidfd_open(row['pid'])
        try:
            assert process(row['pid'])['start_ticks'] == row['start_ticks']
            pidfd_signal(pidfd, signal.SIGSTOP)
            for _ in range(50):
                after = process(row['pid'])
                if after['state'] in ('T', 't'):
                    break
                time.sleep(0.1)
            assert after['state'] in ('T', 't'), 'stop not confirmed'
        finally:
            os.close(pidfd)
        report['after'] = after
        report['status'] = 'EXTRACTOR_PAUSED_SYNCS_UNTOUCHED'
        (folder / 'paused.json').write_text(json.dumps(report, indent=2)+'\n')
    else:
        old = json.loads((folder / 'paused.json').read_text())
        row = old['after']; current = process(row['pid'])
        assert old['worker'] == args.worker and old['model'] == args.model
        assert current['start_ticks'] == row['start_ticks'] and current['command_sha256'] == row['command_sha256']
        assert current['state'] in ('T', 't'), 'extractor not paused'
        assert (BASE / 'resume_approved.json').exists(), 'isolated upload canary approval missing'
        approval = json.loads((BASE / 'resume_approved.json').read_text())
        assert approval.get('status') == 'PASS' and args.job in approval.get('jobs', [])
        pidfd = pidfd_open(row['pid'])
        try:
            assert process(row['pid'])['start_ticks'] == row['start_ticks']
            pidfd_signal(pidfd, signal.SIGCONT)
        finally:
            os.close(pidfd)
        report['status'] = 'EXTRACTOR_RESUMED'
        (folder / 'resumed.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
