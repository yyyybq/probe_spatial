# A1 relative-camera-movement probe

This experiment is the protocol-matched counterpart of the existing streaming
A1 depth/camera probe. It keeps the frozen features, trainable `BackbonePA`,
depth head, data, optimizer, schedule, and loss weighting unchanged. The only
task substitution is:

```text
absolute camera pose [B,S,9]  ->  adjacent relative movement [B,S-1,9]
```

For world-to-camera extrinsics `E`, edge `i` maps camera-`i` coordinates into
camera-`i+1` coordinates:

```text
M[i] = E[i+1] @ inverse(E[i])
```

The 9-D encoding contains the first two rows of the relative rotation followed
by relative translation. This is exactly the encoding used by the C-series
`path_actions`; movement is a prediction target here, never an input.

## Train

Use the movement counterpart of any existing streaming A1 config. For example:

```bash
python vidfm3d/train.py \
  experiment=inscene15k_streaming/streaming_movement_dino_v1 \
  prefix_len=8 \
  streaming_feat_root=/path/to/dino_prefix_8
```

Available configs cover Aether, CogVideoX, DINO, F3R, LingBot, MiniMax-H3,
VideoMAE, V-JEPA2, Wan, and the direct-VLM route. Each inherits its matching
`streaming_depth_*` config and changes only the job identity, active camera
head, and camera/movement loss weights.

`streaming_movement` is registered as a streaming-only task in the current
canonical-v2 matrix builder. Historical Probe816 queues remain immutable; a
new matrix build contains movement as an additional task instead.

The frozen execution runtime is `runtime/a1_movement_canonical_v2`. Its real
canonical DINO p8 canary completed `train -> checkpoint -> eval` on one H800;
the SHA-bound PASS receipt is under
`artifacts/a1_movement_canonical_v1/validation/dino-p8-real-20260922-r6/RESULT.json`.

## Evaluate

```bash
python vidfm3d/eval_a1_streaming.py \
  experiment=inscene15k_streaming/streaming_movement_dino_v1 \
  "+ckpt_path='/path/to/epoch=99-step=100.ckpt'" \
  prefix_len=8 \
  streaming_feat_root=/path/to/dino_prefix_8
```

The common `adjacent_movement_*` metrics are emitted by both direct-movement
and absolute-pose evaluation. Absolute-pose predictions are first converted to
the same ordered `S-1` adjacent edges, so these values—not the historical
all-pairs camera AUC—are the controlled comparison:

- `RotErr_deg`, `Rac_5`, `Rac_15`
- `TransAngle_deg`, `Tac_5`, `Tac_15`
- `TransL2`, `TransMagErr`
- `Auc_3`, `Auc_5`, `Auc_10`, `Auc_30`

Translation direction is sign-sensitive. Near-stationary ground-truth edges
are assessed by translation magnitude/L2 rather than an undefined direction.
