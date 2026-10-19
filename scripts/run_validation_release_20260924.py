#!/usr/bin/env python3
"""Run SHA-bound canaries and submit only inventory+canary PASS experiments."""
import sys,subprocess,json,hashlib
from pathlib import Path
from probe_campaign_20260924_followup import ROOT,BASE,PY,load,sha

def main():
 bundle=Path(sys.argv[1]);out=Path(sys.argv[2]);shardarg=sys.argv[3]
 assert sha(bundle)==sys.argv[4]
 runner=ROOT/'scripts/run_probe_hybrid_validation.py';assert sha(runner)==sys.argv[5]
 release=ROOT/'scripts/probe_campaign_20260924_followup.py';assert sha(release)==sys.argv[6]
 if len(sys.argv)>7:
  repair=ROOT/'scripts/repair_lingbot_uri8_20260924.py';assert sha(repair)==sys.argv[7]
  subprocess.run([PY,str(repair)],check=True)
  assert load(BASE/'lingbot_uri8_repair_20260924_r1/status.json')['status']=='PASS'
 shards=range(load(bundle)['shard_count']) if shardarg=='all' else [int(shardarg)]
 for i in shards:
  subprocess.run([PY,str(runner),'--bundle',str(bundle),'--shard',str(i),'--bundle-sha256',sha(bundle),'--runner-sha256',sha(runner)],check=True)
  subprocess.run([PY,str(release),'submit','--out',str(out),'--ready-only'],check=True)
if __name__=='__main__':main()
