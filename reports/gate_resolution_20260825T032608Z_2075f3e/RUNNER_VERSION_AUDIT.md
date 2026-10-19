# Runner version audit

Gate tag: `gate_resolution_20260825T032608Z_2075f3e`  
Audit time: 2026-08-25 UTC  
Verdict: **PASS for cache-consuming DINO p8 gates; unresolved for any path that requires new cache extraction or V-JEPA2 predictor-target semantics. No interface was reconstructed.**

## Current identity and lineage

- Current `features/run_inscene15k.py`: 82,812 bytes, SHA256 `ee95d1d4d2ba554a919d2668d7be555222dfd1a0ba440a5a83d7a67d1eba9486`, Git blob `4b01d7a7f880441ddca0b928874a677045c76fda`.
- Recovery source from dangling tree `e7fc7f857a2dd27eed3626c64d18f083bd1c0d87`: 79,853 bytes, SHA256 `3272dc357751d22ee94b871ffc15b9076ee711c1d10bf6ffb446124a60ccc819`, blob `0bc9214a60d97977546c8d7827e7cc37fc0d7865`.
- `HEAD` version: 65,871 bytes, SHA256 `4a7f9a8e24550aa4131bd63551d3f0e8db7bda4035da22ff011af3d9ae667bc2`, blob `3c9ce6a12ef38e497e6bde534a5f19da11e3fcd4`.
- The current file is the recovered source plus traceable, uncommitted cache-completion changes: local/AOSS done-index checks, local CogVideoX-I2V snapshot recognition, cache-check-only mode, and nonzero exit when scenes fail. The exact diff is `RUNNER_RECOVERY_TO_CURRENT.diff`.
- These post-recovery changes do not add `--streaming-window-obs-len` or `predictor_target`; the current file is not represented as a recovered newer original.

## Actual supported interface

AST extraction found 36 CLI entries; the machine-readable inventory is `RUNNER_CLI_AST.json`.

- Extraction modes: `normal`, `shuffled`, `target_isolated`, `streaming_prefix`, `context_segment`, `streaming_target`.
- VFM backends: `wan`, `cogvideox`, `vjepa`, `vjepa2`, `dino`, `videomae`, `aether`, `opensora`, `f3r`, `minimax_h3`, `lingbot`.
- Motion-uniform controls: `--temporal-sampling`, `--streaming-motion-step`, `--streaming-rotation-weight`, `--streaming-window-stride`, and `--streaming-max-windows-per-scene`.
- Streaming-prefix selection: `--prefix-min-len`, `--prefix-max-len`, `--prefix-stride`, and `--prefix-lengths`.
- Target selection: `--target-from-streaming-windows`, `--target-prefix-lengths`, and `--target-horizons`.
- The current motion-uniform streaming-prefix window length is derived from the maximum requested prefix/model length. Target-isolated selection independently derives the required observation count from `prefix + horizon`.
- The current `target_isolated` V-JEPA2 path repeats the isolated target RGB frame and calls the ordinary V-JEPA2 featurizer. There is no predictor-target mode or symbol.

The exact CLI flag `--streaming-window-obs-len` is absent. The exact symbol `predictor_target` is absent.

## Original-source search

The following evidence sources were searched without writing into or restoring the runner:

- All reachable Git refs, branch history, tags, reflog-visible history, and the sole worktree.
- `git fsck --full --unreachable --no-reflogs`: 494 unreachable objects were enumerated.
- All 2,161 Git objects were classified; 1,410 text-sized blobs (at most 500 KiB) were content-scanned. Twenty distinct runner/runner-patch candidates were identified. None contains either missing interface.
- All dangling-tree runner candidates, including the verified recovery tree.
- Cursor Remote Local History. Its runner timeline contains 70,933-byte, 0-byte, 0-byte, and 79,853-byte versions; none contains either missing interface.
- Cursor project records and Codex sessions. Earlier `streaming_window_obs_len` hits are the dataset constructor/method, not a runner flag. Exact flag and `predictor_target` hits occur only in the later recovery request, audit commands, and reports; no original implementation body was found.
- Current repository source bundles, historical launcher scripts, Hydra/config artifacts, report launch bundles, tar/zip candidates, and scheduler-resolved startup commands. The scheduler commands use the live mounted worktree and do not contain an archived newer runner.

Evidence files: `ALL_GIT_BLOB_RUNNER_STRING_HITS.csv`, `ALL_GIT_RUNNER_CANDIDATE_BLOBS.csv`, `GIT_ALL_REFS_RUNNER_STRING_HITS.txt`, `GIT_FSCK_UNREACHABLE.txt`, `CURSOR_CODEX_RUNNER_STRING_FILES.txt`, `CODEX_SESSION_RUNNER_STRING_CONTEXTS.txt`, and the prior recovery runner-contract audit.

Conclusion: **no evidence-backed original implementation of either missing interface was found.** No file has been labeled `RECONSTRUCTED_INTERFACE`, because no reconstruction was performed.

## Dependency decision for this run

| Planned path | Needs new runner extraction? | Missing-interface decision |
|---|---:|---|
| Compute-node exact-object gate | No | Consume known AOSS objects; not blocked by either missing interface. |
| DINO p8 real DataLoader and strict-B2 smoke | No | Uses existing streaming/target caches plus RGB/mask/depth geometry; not blocked. |
| Strict B2 standard/strict/query-only/direction/no-update/oracle on an existing cache-common set | No | Not blocked by `predictor_target`; feature provenance and common IDs still must pass. |
| C1/C2 visual-only, action-only, matched shuffle using existing streaming and target caches | No | Not blocked by runner-version gap. |
| Copy baseline when an exact isolated anchor already exists | No | Not blocked; prove object identity and target-index provenance. |
| Copy baseline requiring a new isolated anchor cache | Yes | Blocked until the extraction semantics are proven. DINO can use the existing `target_isolated` encoder path if a gated re-extraction is truly necessary; V-JEPA2 predictor-target semantics remain unresolved. |
| New cross-prefix cache extraction intended to guarantee one shared full observation window | Yes | Blocked until an evidence-backed interface is recovered or a separately labeled reconstruction and regression tests are approved. |
| New V-JEPA2 predictor-target cache extraction | Yes | Blocked; original implementation was not recovered. |

Existing cache consumption is therefore allowed to proceed through gates. Before a formal all-prefix comparison, compute-node inspection must verify `prefix_index.npy`/manifest window provenance and exact raw target IDs. A failure there would block only the affected cache/model path and would not retroactively justify silently reconstructing the historical runner.

