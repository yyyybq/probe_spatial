"""Start a detached guard through a normal shell, then verify fresh heartbeat.

The interactive exec transport can reap children on disconnect. nohup + disown
and redirected stdin avoid treating an open terminal as a service supervisor.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import select
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/main_v2_probe_v2_816_20260917/upload_recovery_20260923_r1'
RELEASE=ROOT/'runtime/cog_upload_backpressure_v1'
SCO='/mnt/umm/users/yinbaiqiao/.sco/bin/sco'


def install(job):
    pack=RELEASE/'pack.json'; pack_sha=hashlib.sha256(pack.read_bytes()).hexdigest()
    assert job in json.loads(pack.read_text())['jobs']
    saved=json.loads((BASE/'process_control'/job/'paused.json').read_text())
    worker=saved['worker']
    info=json.loads(subprocess.check_output([SCO,'acp','jobs','describe',job,'--workspace-name','aigc','-o','json'],text=True,timeout=30))
    assert info['state']=='RUNNING'
    workers=subprocess.check_output([SCO,'acp','jobs','get-workers',job,'--workspace-name','aigc'],text=True,timeout=30)
    assert worker in workers and 'Running' in workers
    folder=BASE/'cog_guard'/job
    audit=BASE/'cog_guard_deployments'/job; audit.mkdir(parents=True,exist_ok=True)
    heartbeat=folder/'status.json'
    if heartbeat.exists():
        state=json.loads(heartbeat.read_text())
        if time.time()-state['epoch']<180:
            assert state['pack_sha256']==pack_sha
            print(json.dumps({'job':job,'status':'ALREADY_ACTIVE','guard':state}),flush=True); return
    before=time.time()
    master,slave=pty.openpty()
    login=subprocess.Popen([SCO,'acp','jobs','exec',job,'--workspace-name','aigc','--worker-name',worker],
                           stdin=slave,stdout=slave,stderr=slave,start_new_session=True)
    os.close(slave); output=b''; sent=False
    command='mkdir -p '+shlex.quote(str(folder))+'\n'+shlex.join(['nohup','/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python',str(RELEASE/'guard.py'),
                        '--job',job,'--pack-sha',pack_sha])
    command+=' >>'+shlex.quote(str(folder/'guard.log'))+' 2>&1 </dev/null &\ndisown\nexit\n'
    try:
        deadline=time.monotonic()+90
        while time.monotonic()<deadline and login.poll() is None:
            if select.select([master],[],[],.25)[0]:
                try: chunk=os.read(master,65536)
                except OSError: break
                if not chunk: break
                output+=chunk
                if not sent and re.search(rb'[#\$] ',output):
                    os.write(master,command.encode());sent=True
    finally:
        if login.poll() is None: login.terminate()
        login.wait(timeout=10);os.close(master)
        (audit/'detached_install.log').write_bytes(output)
    assert sent, 'remote login failed; no guard command sent'
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if heartbeat.exists():
            state=json.loads(heartbeat.read_text())
            if state['epoch']>=before and state['pack_sha256']==pack_sha and state['worker']==worker:
                record={'epoch':time.time(),'job':job,'status':'CONFIRMED','guard':state,'scheduler':info}
                (audit/'deployment.json').write_text(json.dumps(record,indent=2)+'\n')
                print(json.dumps({'job':job,'status':'CONFIRMED','guard':state}),flush=True);return
        time.sleep(5)
    raise RuntimeError(f'{job}: no fresh heartbeat; inspect, do not blindly relaunch')


if __name__=='__main__':
    import sys
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(install,sys.argv[1:]))
