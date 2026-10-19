# Manual runtime diagnostics

`ddp_smoke.py` checks NCCL initialization, broadcast, and all-reduce without
Lightning or a dataset:

```bash
torchrun --standalone --nproc_per_node=4 scripts/diagnostics/ddp_smoke.py
```

Run it only in a GPU allocation with the requested device count. It is an
operator diagnostic, not a default CI test or scientific result. The file was
moved from the temporary-looking root path `tmp_ddp_smoke.py` during the
2026-10-11 release cleanup; communication behavior is unchanged.
