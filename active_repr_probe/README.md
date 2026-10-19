# Active Spatial representation probing

This is an independent frozen-representation pipeline. It does not modify
VAGEN training, the checkpoint registry, or any shared feature cache.

## Frozen MVP contract

- Main input: exactly three contiguous RGB observations plus task text.
- No pose, distance, GT relation, generated answer, or history action is part
  of the main representation input.
- Taps: visual terminal, middle LLM layer at the last valid prompt token, and
  final norm at that same token.
- The state bank stores actual environment branches for each candidate action.
  `distance_decrease` and `task_score_improve` are separate labels.
- A label that lacks environment evidence is `UNAVAILABLE`; it is not replaced
  with a geometric approximation.

## Commands

Run on the intended GPU/runtime host, after the exact paths have been mounted:

```bash
python3 -m active_repr_probe.audit \
  --vagen-root /mnt/umm/users/yinbaiqiao/VAGEN-Lite \
  --output-dir artifacts/active_repr_probe/audit

python3 -m active_repr_probe.build_state_bank \
  --vagen-root /mnt/umm/users/yinbaiqiao/VAGEN-Lite \
  --env-config /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/env_config_h800_7b_6types.yaml \
  --output-dir artifacts/active_repr_probe/state_bank_canary \
  --episodes 24 --steps-per-episode 6 \
  --enable-visibility-labels --enable-collision-labels

python3 -m active_repr_probe.canary \
  --audit artifacts/active_repr_probe/audit/checkpoints.json \
  --state-bank artifacts/active_repr_probe/state_bank_canary/states.jsonl \
  --output artifacts/active_repr_probe/canary_status.json --strict
```

Only after this preflight reports `READY`, run one exact checkpoint on at most
48 samples:

```bash
python3 -m active_repr_probe.extract_features \
  --state-bank artifacts/active_repr_probe/state_bank_canary/states.jsonl \
  --checkpoint /absolute/exact/checkpoint \
  --backend qwen2_5_vl --output artifacts/active_repr_probe/features/qwen_canary.pt \
  --limit 48
```

Run the full approximately-2,000 state MVP only after all five checkpoint
canaries have matching sample-id hashes and valid tap manifests. No command in
this directory starts RL or a layer sweep.
