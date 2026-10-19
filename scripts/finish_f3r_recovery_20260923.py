"""One-off continuation: deploy each frozen F3R subset only after its canary passes."""
import json
from pathlib import Path
import subprocess
import time

from recover_feature_jobs_20260923 import BASE, ROOT, describe, save


def main():
    pending = {
        'streaming': ('backend_canaries', 'pt-494m9kh2'),
        'target': ('f3r_target_canary', 'pt-l0xzkzuf'),
    }
    deadline = time.monotonic() + 3 * 3600
    while pending:
        if time.monotonic() >= deadline:
            raise RuntimeError('Canary wait exceeded three hours; no unverified deployment')
        for mode, (folder, job) in list(pending.items()):
            receipt = BASE / folder / 'receipts/f3r.json'
            if receipt.exists():
                evidence = json.loads(receipt.read_text())
                if evidence.get('status') != 'pass':
                    raise RuntimeError(f'{mode} canary did not pass: {receipt}')
                subprocess.run([
                    '/usr/bin/python3',
                    str(ROOT / 'scripts/deploy_f3r_qwen_after_canary_20260923.py'),
                    'f3r', mode,
                ], check=True, cwd=str(ROOT))
                del pending[mode]
            else:
                state = describe(job)['state']
                if state not in ('CREATING', 'STARTING', 'QUEUEING', 'PENDING', 'RUNNING'):
                    raise RuntimeError(f'{mode} canary {job} is {state} without PASS receipt')
        save(BASE / 'f3r_continuation_status.json', {
            'status': 'WAITING_CANARY' if pending else 'SUBMITTED_CONFIRMED',
            'pending_modes': list(pending),
            'checked_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        })
        print((BASE / 'f3r_continuation_status.json').read_text(), flush=True)
        if pending:
            time.sleep(30)


if __name__ == '__main__':
    main()
