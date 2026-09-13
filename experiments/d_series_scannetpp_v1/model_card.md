# Frozen backbone and probe card

All three backbones are frozen. Exact checkpoint revisions and machine-readable preprocessing are in `models.yaml`.

V-JEPA2 uses `facebook/vjepa2-vitl-fpc64-256` at revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`, final encoder layer 23 after layer normalisation. Its 64-frame, 256-square input produces `[32,16,16,1024]` tokens in time-height-width order. Isolated-target D1 repeats only the target RGB and is separate from the D3 oracle native-target reference. Native D3 masks prefix patches before encoder self-attention, retains absolute RoPE positions, and predicts distinct tubelets 4 and 5 for h2 and h4. The teacher reference may see the label clip; the predictor may not.

Wan uses `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` at revision `0fad780a534b6463e45facd96134c9f345acfa5b`, not an I2V or 14B substitution. It uses layer 20, 81 frames at 480x832, empty text, UniPC with flow shift 3, and the pre-specified argument 749. In this repository 749 is an **index into `scheduler.timesteps`**, not a raw diffusion timestep. Noise is deterministic per sample. Future/target captions and pixels cannot construct conditioning. The formal t749 cache is distinct from existing t499/t999 assets and cannot silently fall back to them.

DINOv2 uses `facebook/dinov2-large` (ViT-L/14, without register tokens) at revision `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`. Every prefix frame is encoded independently after bicubic 420x728 resize and checkpoint image normalisation; CLS is dropped, `[30,52,1024]` patches are stacked over time. It is a static visual control and has no native temporal predictor.

Spatial D1 uses a shared token-wise shallow depth head followed by fixed spatial restoration. Its weak control is token-wise linear. Pooled D1 maps the pooled latent to a small learned grid and lightweight decoder; it does not use a giant `C -> H*W*hidden` layer. D2 uses the same external predictor family across backbones with only the input projection changing. Its context-only baseline is independently trained at comparable capacity. D3 linear adaptation is a spatially shared token-wise affine map fit on train pairs only; retrained readouts match target-readout capacity.

Every checkpoint receipt must include protocol and manifest hash, backbone revision, branch/reference space, seed, train/dev scene hashes, train-only latent/depth/motion statistics, parameter count, selected epoch, and source run. Historical C1 and tiny-pilot checkpoints fail this lineage gate and are not formal checkpoints.
