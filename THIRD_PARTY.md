# Third-party source notices

Local inventory checked on 2026-10-11. This file records evidence in the
checkout; it does not replace upstream licenses or assign new licenses.

| Source | Local evidence | Release status |
| --- | --- | --- |
| VidFM3D-derived project | Root `LICENSE`: CC BY-NC 4.0; original author notices retained | The old README's MIT description was inconsistent with this file and has been removed |
| CroCo and its subcomponents | `features/f3r/f3r/croco/LICENSE` and `NOTICE` | Preserve both files and source headers |
| Fast3R / DUSt3R | Copyright/license references in `features/f3r/f3r/` source headers | No separate root LICENSE was found for this vendored tree; recover the notice for the actual imported revision before redistributing it |
| VGGT | Meta copyright/license references in `vidfm3d/vggt/` source headers | No separate LICENSE was found in the vendored tree; recover the notice for the actual imported revision |
| Aether | Vendored source under `features/aether/aether/` | No separate LICENSE was found in this subtree; establish source revision and redistribution terms |
| Model weights and external datasets | Loaded/downloaded separately by backend and data scripts | Not covered by the root code license; retain their own access and usage requirements |

The root license is retained unchanged. Missing upstream notices are an open
release blocker; a generic attribution paragraph is not a substitute for the
actual license belonging to each imported revision. Historical Open-Sora and
V-JEPA deletions already existed before this audit and were not reverted.
