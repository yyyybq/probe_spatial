"""Read-only two-point upload/cleanup measurement; writes only diagnostic JSON."""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'InsScene-15K'
OUT = ROOT / 'artifacts/main_v2_probe_v2_816_20260917/upload_diagnosis_20260923'
SCALE = {'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3, 'TiB': 1024**4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--previous')
    args = parser.parse_args()
    previous = json.loads((OUT / (args.previous + '.json')).read_text()) if args.previous else None
    now = time.time()
    result = {'epoch': now, 'logs': {}, 'models': {}, 'errors': []}
    v = os.statvfs(DATA)
    result['filesystem'] = {'total_bytes': v.f_blocks*v.f_frsize,
                            'available_bytes': v.f_bavail*v.f_frsize,
                            'free_bytes': v.f_bfree*v.f_frsize}
    grouped = defaultdict(lambda: {'uploaded_bytes': 0, 'deleted_payloads': 0,
                                  'deleted_sidecars': 0, 'upload_rounds': 0})
    for mode in ('STREAMING', 'TARGET'):
        base = DATA / f'FEAT_{mode}_CANONICAL_V2_GAP_DELTA_V1/.windowfix_v2_gap_logs'
        for path in base.glob('*/*/2026092*/aoss_publish/runner.log'):
            key = str(path)
            model = path.relative_to(base).parts[0]
            try:
                size = path.stat().st_size
                offset = previous['logs'].get(key, {}).get('size', size) if previous else size
                # Only bytes appended during this observation, not old progress bars.
                if size < offset:
                    result['errors'].append({'path': key, 'error': 'log truncated; delta excluded'})
                    offset = size
                with path.open('rb') as f:
                    f.seek(offset)
                    chunk = f.read(max(0, size-offset)).decode(errors='replace').replace('\x00', '')
                upload = re.findall(r'copiedSize:\s*([\d.]+)\s*(TiB|GiB|MiB|KiB|B)', chunk)
                deleted = re.findall(r'\{"already_absent":\s*\d+,\s*"payloads":\s*(\d+),\s*"sidecars":\s*(\d+)\}', chunk)
                g = grouped[model + '/' + mode.lower()]
                g['uploaded_bytes'] += sum(float(n)*SCALE[u] for n,u in upload)
                g['upload_rounds'] += len(upload)
                g['deleted_payloads'] += sum(int(n) for n,_ in deleted)
                g['deleted_sidecars'] += sum(int(n) for _,n in deleted)
                result['logs'][key] = {'size': size, 'mtime': path.stat().st_mtime}
            except OSError as exc:
                result['errors'].append({'path': key, 'error': str(exc)})
    result['models'] = dict(grouped)
    if previous:
        dt = now-previous['epoch']
        result['interval_seconds'] = dt
        result['filesystem_net_growth_bytes'] = previous['filesystem']['free_bytes']-result['filesystem']['free_bytes']
        for v in grouped.values():
            v['upload_MiB_per_second'] = v['uploaded_bytes']/1024**2/dt
            v['deleted_payloads_per_second'] = v['deleted_payloads']/dt
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (args.label + '.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='logs'}, indent=2))


if __name__ == '__main__':
    main()
