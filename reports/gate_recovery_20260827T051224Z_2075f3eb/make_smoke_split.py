#!/usr/bin/env python3
"""Write the exact five-scene recovery-only split manifest."""

import hashlib
import json
import os
from pathlib import Path


out = Path("/mnt/umm/users/yinbaiqiao/probe_spatial/reports/gate_recovery_20260827T051224Z_2075f3eb/manifests/strict_b2_smoke_split.json")
canonical = Path("/mnt/umm/users/yinbaiqiao/probe_spatial/configs/splits/inscene15k_v1.json")
scenes = [
    "scannetpp/00777c41d4",
    "scannetpp/00a231a370",
    "scannetpp/00dd871005",
    "scannetpp/020312de8d",
    "scannetpp/027cd6ea0f",
]
payload = {
    "schema_version": 1,
    "purpose": "gate_recovery_strict_b2_smoke",
    "canonical_split_manifest": str(canonical),
    "canonical_split_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
    "splits": {"train": scenes},
}
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out)
