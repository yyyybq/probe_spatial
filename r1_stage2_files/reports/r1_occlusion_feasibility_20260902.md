# R1 Occlusion Feasibility Check

## Result

`R1-v1 occlusion = unchanged existing projection/depth proxy`

## Evidence

- Official Active Spatial HTTP renderer path is:
  `examples/train/active_spatial/start_gs_render_http_service.sh`
- HTTP handler returns only encoded RGB images.
- Worker calls `GaussianRenderer.render_image_from_cam_param()` and encodes the returned image bytes.
- `GaussianRenderer.render_image_from_cam_param()` calls `gsplat.rendering.rasterization(...)` with default `render_mode="RGB"` and discards the second and third return values:
  `render_colors, _, _ = rasterization(...)`.
- Installed gsplat signature supports `render_mode` values including `D`, `ED`, `RGB+D`, `RGB+ED`, and `segmented=False`, so alpha/depth may be feasible with a versioned renderer API change.
- No current endpoint exposes depth, alpha, per-object render, instance/object-ID masks, or true visible target pixel fraction.

## Decision

Do not block H1 FOV/projective repair on occlusion. Do not claim a true observation-aligned visible-fraction oracle until a versioned renderer API exposes and validates the required channels.
