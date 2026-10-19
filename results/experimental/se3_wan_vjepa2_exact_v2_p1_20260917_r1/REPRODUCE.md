# Reproduce

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -m analyses.se3_protocol_v2.wan_vjepa2_p1 all --device cuda
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python -m pytest -q tests/test_wan_vjepa2_p1.py
```

The command verifies all frozen parent hashes and PASS receipts before reading predictions/checkpoints. It does not rerun parent training or extraction.
