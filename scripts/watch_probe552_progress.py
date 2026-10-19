#!/usr/bin/env python3
"""Keep the user-requested 552 progress view current; never submit work."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from update_probe552_progress import OUT, atomic, now

SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'watcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(OUT/'watcher.json',json.dumps(dict(pid=os.getpid(),started_utc=now(),status='RUNNING',
        task_poll_seconds=60,feature_poll_seconds=600))+'\n')
    def stop(*_):
        atomic(OUT/'watcher.json',json.dumps(dict(pid=os.getpid(),stopped_utc=now(),status='STOPPED'))+'\n')
        sys.exit(0)
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    next_features=time.monotonic()+600
    while True:
        try:
            jobs=[]
            # The VM/VJEPA2 wave added 168 production jobs. Keep one thousand
            # rows so earlier Aether, F3R and Qwen replacements are still
            # visible while this wave occupies the newest rows.
            for page in range(1,11):
                p=subprocess.run([SCO,'acp','jobs','list','--workspace-name','aigc','--page-size','100','--page-token',str(page),'-o','json'],
                    capture_output=True,text=True,timeout=45,check=True)
                rows=json.loads(p.stdout);jobs.extend(rows)
                if len(rows)<100:break
            atomic(OUT/'scheduler_jobs.json',json.dumps(dict(observed_utc=now(),jobs=jobs))+'\n')
        except Exception as e:
            atomic(OUT/'scheduler_last_error.json',json.dumps(dict(at=now(),error=str(e)))+'\n')
        try:
            update_features=time.monotonic()>=next_features
            command=[sys.executable,str(Path(__file__).with_name('update_probe552_progress.py'))]
            if update_features:command.append('--refresh-features')
            subprocess.run(command,check=True)
            if update_features:next_features=time.monotonic()+600
            atomic(OUT/'watcher_heartbeat.json',json.dumps(dict(pid=os.getpid(),updated_utc=now(),status='RUNNING'))+'\n')
        except Exception as e:
            atomic(OUT/'watcher_last_error.json',json.dumps(dict(at=now(),error=str(e)))+'\n')
            next_features=max(next_features,time.monotonic()+60)
        time.sleep(60)


if __name__=='__main__':main()
