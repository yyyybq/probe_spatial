# Probing Spatial — Project Progress Report

Last audited: 2026-08-16 UTC. Repository: `/mnt/umm/users/yinbaiqiao/probe_spatial`.

This report is based on the current working tree, committed git history, untracked experiment artifacts, configs, code, logs, result CSV/JSON/Markdown files, and session reports. Important caveat: the working tree is dirty and contains many untracked scripts, manifests, feature-extraction backends, and logs. Therefore, "current project state" is newer than `origin/main` and should not be inferred from committed history alone.

## 1. Executive Summary

`probe_spatial` studies what kinds of spatial, temporal, egocentric, and action-conditioned information are encoded in frozen video / vision foundation model features. The method is diagnostic probing: extract frozen VFM features from InsScene-15K ScanNet++ video windows, train lightweight probe heads, and evaluate whether the probes can recover geometry, view overlap, hidden-object position, or future target features under camera actions.

The project began as an extension of VidFM3D's A1-style pixel-aligned 3D probing, then evolved into a broader Spatial Diagnostic Suite: A1 depth/camera/mask, A2 view consistency, A3 abnormal temporal order, B1/B2 egocentric belief, and C1/C2/C3 action-conditioned dynamics. The most important methodological shift was the move to a shared streaming-prefix protocol over ScanNet++ only. This happened because Infinigen frames were discovered to be independent rendered views rather than valid temporal trajectories; old temporal streaming results that include Infinigen are historical only.

Current strongest findings:

- Geometry-specialized or discriminative visual features are ahead of video-generation features on most measured probes. F3R is strongest on A1 geometry and A2 view consistency; DINO is strongest on C1/C2/C3 retrieval-style action tasks.
- Direct metric geometry is not the same as action-conditioned feature predictability. F3R dominates depth/camera/view consistency, but DINO and Qwen2.5-VL dominate C action-conditioned retrieval.
- B2 hidden-object localization remains hard. Across the 7/20 A/B table, best models are around 55-56 degrees mean angular error, with all models still far from precise object permanence.
- Longer prefixes often do not help C tasks. In the 7/28 prefix robustness report, DINO/Qwen/VideoMAE/Wan generally degrade from prefix 8 to prefix 24 on C1/C2; best-prefix reporting is therefore optimistic and should be paired with prefix AUC.

Current stage:

- A/B/C default-layer streaming pipeline is largely operational and has reportable validation metrics.
- C-series target leakage/zero-target bugs were found, fixed, and C was retrained/evaluated with manifest-driven AOSS data loading.
- Layer sweeps and model expansion are ongoing: Wan/CogVideoX A/B layer sweeps have many provisional rows; VideoMAE/Qwen C results were added; LingBot/MiniMax-H3 extraction/probing infrastructure is present, but final scientific metrics are not recoverable from current artifacts.

Largest unresolved question:

- Whether the apparent C-series success, especially DINO/Qwen retrieval strength, reflects spatial action understanding rather than representation-specific retrieval shortcuts, target-feature smoothness, or prefix/window biases. Current evidence is strong for decodable action-conditioned target identity, but not yet causal proof of 3D reasoning.

## 2. Research Motivation

The repository asks whether frozen VFMs encode useful 3D and temporal structure beyond surface appearance. The original VidFM3D-style question was: can frozen video features support direct prediction of depth, camera pose, and instance identity? This project extends that into more diagnostic questions:

- Does a representation support online, streaming spatial inference rather than full-clip access?
- Can it compare viewpoints and identify overlapping visual regions?
- Can it localize an object after it leaves view?
- Can it use camera motion to predict future feature states?
- Do different model families encode different spatial signals?

The motivation is not just to rank models. The experiments aim to separate several forms of "spatial understanding": pixel-aligned metric geometry, object-conditioned egocentric memory, and action-conditioned future prediction.

Evidence: `README.md`, `EXPERIMENT_PROTOCOL.md`, `TRAINING_LOGIC_AUDIT.md`, `vidfm3d/models/probe_ext_module.py`, probe heads in `vidfm3d/models/components/`.

## 3. Core Research Questions

1. What spatial information is linearly or shallowly decodable from frozen VFM features?
2. Are view overlap, depth/camera pose, and object permanence encoded by the same models/layers, or do they separate by model family?
3. Does increasing streaming context length improve temporal/spatial reasoning, or does it introduce noise and optimization difficulty?
4. Can action-conditioned probes predict isolated future target features better than no-action, shuffled-action, or copy baselines?
5. Are B1/B2 hidden-object results genuine object permanence, or artifacts of mask/query construction, pose priors, and dataset biases?
6. Which results survive protocol fixes: ScanNet++ only, exact target-isolated features, scene/sample manifests, no old invalid C checkpoints?

## 4. Project Evolution

### Phase 1: VidFM3D-style A1 probing

Question -> Can frozen VFM features support direct 3D pixel-aligned readouts?

Method -> Use DPT/pixel-aligned heads to predict depth, camera pose, and instance identity.

Observation -> Historical Wan/V-JEPA2 A1 results showed V-JEPA2 slightly better than Wan, but camera AUC was very low overall. Later streaming default-layer A1 results expanded to 8 models; F3R became the clear best on depth and camera pose.

Conclusion -> Frozen features contain recoverable geometry, but camera pose remains difficult and strongly model-dependent.

Remaining issue -> A1 is a direct readout task, not evidence of object permanence or action-conditioned reasoning.

Evidence: `PROGRESS_REPORT.md`, `logs/eval_results/results_table.csv`, `docs/session_handoff_20260720_c_retrain_ab_eval.md`.

### Phase 2: Spatial Diagnostic Suite A/B/C

Question -> Which spatial abilities beyond A1 can be probed?

Method -> Add A2 view consistency, A3 shuffled temporal abnormal detection, B1/B2 hidden-object localization, C1 action dynamics, C2 path integration, and C3 counterfactual action probes.

Observation -> Early non-streaming results produced useful diagnostics but also exposed confounds: A2 class imbalance, B1 GT-mask leakage, A3 dtype/seed leakage, and C-series target-loading failures.

Conclusion -> The suite is useful, but each task needs strict controls and careful metric choice.

Evidence: `PROGRESS_REPORT.md`, `EXPERIMENT_PROTOCOL.md`, `TRAINING_LOGIC_AUDIT.md`.

### Phase 3: Streaming protocol and temporal validity correction

Question -> Can these probes be made valid for temporal inference?

Method -> Use ScanNet++ only; extract independent streaming prefixes at lengths 8/12/16/24; build `target_isolated` future features for C; use fixed objects across B prefix sweeps.

Observation -> Infinigen frames are independent views, so old temporal results involving Infinigen are invalid for memory/action claims.

Conclusion -> Current temporal evidence should come from ScanNet++ streaming windows only.

Remaining issue -> Old reports contain historical numbers that must not be mixed with current streaming evidence.

Evidence: `TEMPORAL_DATA_VALIDITY.md`, `README.md`, `TRAINING_LOGIC_AUDIT.md`.

### Phase 4: C-series repair, AOSS manifests, and final C evaluation

Question -> Are C1/C2/C3 valid after fixing zero-target and data-loading issues?

Method -> Build manifest-driven C datasets, exclude invalid scenes, retrain 72 C jobs for 6 VFMs, rescue interrupted jobs, evaluate all checkpoints, then add VideoMAE/Qwen extra C results.

Observation -> 72/72 C eval summaries were completed on 2026-07-24; 96 total rows including VideoMAE/Qwen were available by 2026-07-28. DINO is strongest by C prefix AUC and best-prefix metrics; Qwen2.5-VL is close on C1/C2.

Conclusion -> C-series is no longer blocked by the old target bug. Final C metrics are reportable as validation results, but interpretation remains correlational.

Evidence: `docs/c_retrain_session_report_20260724.md`, `logs/inscene15k_streaming/results/c_eval_20260724_1903/`, `logs/inscene15k_streaming/results/c_eval_with_extra_20260728/`.

### Phase 5: Layer/model expansion and recovery work

Question -> Do later/earlier layers change conclusions, especially for Wan/CogVideoX and new backends?

Method -> Add layer specs and extraction support for DINO, VideoMAE, Aether, F3R, Qwen/BAGEL, MiniMax-H3, LingBot; launch A/B layer sweeps and recovery jobs.

Observation -> Wan/CogVideoX A/B layer sweep results exist, but many rows are provisional. LingBot/MiniMax-H3 configs, extractors, job builders, and AOSS sync logs exist, but final peer-level metrics were not found in stable summary artifacts.

Conclusion -> Layer/model expansion is implemented and partially evaluated, but not all new model claims are validated.

Evidence: `vidfm3d/utils/feature_layers.py`, `features/minimax_h3/`, `features/lingbot/`, `logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.md`, `logs/probing_jobs_minimax_h3_*`, `logs/probing_jobs_probe_fill_*`.

## 5. Current Methodology

Pipeline:

```text
ScanNet++ scenes
  -> motion-normalized temporal windows
  -> streaming_prefix feature caches at p = 8,12,16,24
  -> optional target_isolated future feature caches
  -> lightweight frozen-feature probe heads
  -> validation metrics from eval_diag / task-specific evaluators
```

Datasets:

- Primary current temporal dataset: InsScene-15K ScanNet++ only.
- Historical/non-temporal data: Infinigen may appear in older reports, but not valid for temporal memory/action conclusions.
- C retrain manifests: 681 included scenes from 683 candidates; excluded `a003a6585e` and `c24f94007b`; sample counts p8=2724, p12=2724, p16=2723, p24=2721 with val n=276.

Models/backbones currently represented:

- Main reported matrix: `dino`, `vjepa2`, `f3r`, `videomae`, `aether`, `cogvideox`, `wan`, `qwen2_5_vl`.
- Implemented/expanding: `minimax_h3`, `lingbot`, `bagel`, `qwen2_5_vl_3b`, `opensora`.

Default feature layers:

- Wan: layer 20 at t=749, 1536 channels.
- CogVideoX: layer 20 at t=749, 3072 channels.
- V-JEPA2 ViT-L: layer 23, 1024 channels.
- DINO: layer 0 / last hidden state, 1024 channels.
- VideoMAE: layer 23, 1024 channels.
- Aether: layer 1 at t=749, 3072 channels.
- F3R: layer 24, 1024 channels.
- Qwen2.5-VL: layer -1 visual/merger output, 3584 channels.

Evidence: `vidfm3d/utils/feature_layers.py`.

Probe tasks:

- A1 `streaming_depth` and `streaming_mask`: depth loss, camera pose AUC, identity loss.
- A2 `view_consistency`: pairwise view overlap; final evaluation reports balanced accuracy, AUROC, PR-AUC, overlap MAE.
- A3 `abnormal`: legacy non-streaming shuffled-context temporal order control.
- B1 `ego_belief`: masked object tokens from visible history plus final global token; predicts azimuth/elevation/log-distance.
- B2 `ego_belief_v2`: object query plus all prefix patch tokens; predicts 16x8 az/el bin and log-distance.
- C1 `action_dynamics`: context prefix + 9D relative camera action -> target isolated feature.
- C2 `path_integration`: recurrent action sequence -> waypoint target features.
- C3 `counterfactual`: observed context + alternative actions -> intervention target features.

Evidence: `vidfm3d/models/probe_ext_module.py`, `probe_*` modules, `configs/experiment/inscene15k_streaming/*.yaml`.

## 6. Experiments and Results

### Core Experiment Table

| Experiment | Question | Setup | Main Result | Interpretation | Status |
| --- | --- | --- | --- | --- | --- |
| A1 streaming depth/camera/mask | Which models expose direct metric geometry? | 8 models x 4 prefixes, ScanNet++ streaming, default layers | F3R mean depth 0.1347 and camera AUC-30 0.3852; DINO best mask loss 1.2563 | F3R has strongest geometry readout; DINO has strong identity/semantic separability | ✅ Confirmed |
| A2 view consistency | Can features identify overlapping views? | 8 models x 4 prefixes, balanced acc | F3R 0.9827, DINO 0.9597, Qwen 0.9403 mean balanced acc | View overlap is highly decodable from F3R/DINO/Qwen; generation models lag | ✅ Confirmed |
| B2 hidden-object query localization | Can object query recover hidden-object direction? | 8 models, 4 prefixes; angle table from 7/20 handoff | VideoMAE 55.5 deg, F3R 56.5 deg, V-JEPA2 59.4 deg; all weak | Hidden-object spatial belief exists but remains coarse; top-1 alone is misleading | 🟡 Preliminary |
| C1 action dynamics | Does action help predict future target feature? | 8 models x 4 prefixes, target-isolated, fixed manifests | DINO best R@1 0.4783; Qwen best 0.4529; Wan 0.1522 | Strong action-conditioned retrieval in DINO/Qwen; many diffusion/3D models weak | ✅ Confirmed |
| C2 path integration | Can probes roll through action horizons? | C horizons 1,2,4; primary global R@1 | DINO best 0.1944; Qwen 0.1787; VideoMAE 0.1135; V-JEPA2 0.0894 | Some representations support multi-step target retrieval, but performance drops with longer prefixes | ✅ Confirmed |
| C3 counterfactual | Does alternative action change future prediction consistently? | intervention validity, 8 models x 4 prefixes | DINO best 0.6120; V-JEPA2 0.5559; VideoMAE 0.5417 | DINO/V-JEPA2/VideoMAE show strongest counterfactual margins/validity | ✅ Confirmed |
| A3 abnormal temporal order | Are temporal order signals present? | Legacy non-streaming Wan/V-JEPA2, shuffled cache | Wan v1 pair_acc 86.3%, ctrl 16.5%; V-JEPA2 old 97.5% invalid pending fix | Wan encodes temporal order under fixed dtype/seed controls; V-JEPA2 old result cannot be trusted | 🟡 Preliminary |
| Wan/Cog layer sweep A/B | Which layers matter for A/B? | 2026-07-25 campaign, 223 jobs; many provisional rows | Wan B2 best formal p12/l15 50.62 deg; Cog B2 best formal p8/l0 62.35 or p16/l10 64.61 | Middle Wan layers help B2; many rows not true best | 🟡 Preliminary |
| LingBot/MiniMax-H3 expansion | Extend model matrix to new video generators | Feature specs, extractors, configs, job lists, AOSS sync logs | No final stable summary metrics found | Implemented infrastructure, not validated scientifically yet | ⚪ Implemented but not tested |

### A/B Default-Layer Summary

Evidence: `logs/eval_results/results_table.csv` and `docs/session_handoff_20260720_c_retrain_ab_eval.md`.

Mean over prefixes:

| Model | A1 Depth ↓ | A1 Camera AUC-30 ↑ | A1 Mask ↓ | A2 Bal-Acc ↑ | B2 Angle ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| F3R | 0.1347 | 0.3852 | 1.4804 | 0.9827 | 56.45 deg (n=3) |
| DINO | 0.1425 | 0.0386 | 1.2563 | 0.9597 | 69.01 deg |
| V-JEPA2 | 0.2687 | 0.1849 | 3.0291 | 0.9240 | 59.39 deg |
| VideoMAE | 0.2888 | 0.2124 | 4.2575 | 0.9210 | 55.53 deg |
| Qwen2.5-VL | 0.2193 | 0.0269 | 2.8522 | 0.9403 | 71.36 deg |
| Wan | 0.3599 | 0.0603 | 4.8688 | 0.8175 | 66.44 deg |
| CogVideoX | 0.4375 | 0.0302 | 5.7245 | 0.7918 | 70.80 deg |
| Aether | 0.5413 | 0.0198 | 7.9184 | 0.7238 | 62.46 deg |

Note: `logs/eval_results/results_table.csv` records B2 top-1 in the metric column; the B2 angle table was reconstructed in `docs/session_handoff_20260720_c_retrain_ab_eval.md` from per-run `val_summary.json`. Treat B2 as less cleanly packaged than A1/A2/C.

### C-Series Default + Extra Summary

Evidence: `logs/inscene15k_streaming/results/c_eval_with_extra_20260728/c_series_final_metrics_with_extra.md` and `c_series_prefix_robustness.md`.

Best prefix per model:

| Probe | Best models | Main result |
| --- | --- | --- |
| C1 action_dynamics | DINO, Qwen2.5-VL | DINO p8 R@1 0.4783; Qwen p8 R@1 0.4529 |
| C2 path_integration | DINO, Qwen2.5-VL, VideoMAE | DINO p8 R@1 0.1944; Qwen p8 0.1787; VideoMAE p8 0.1135 |
| C3 counterfactual | DINO, V-JEPA2, VideoMAE | DINO p8 validity 0.6120; V-JEPA2 p24 0.5559; VideoMAE p8 0.5417 |

Prefix AUC ranking:

- C1: DINO 0.4556, Qwen 0.4343, V-JEPA2 0.1255, Wan 0.1119.
- C2: DINO 0.1396, Qwen 0.1316, V-JEPA2 0.0823, VideoMAE 0.0599.
- C3: DINO 0.5936, V-JEPA2 0.5452, VideoMAE 0.5154, F3R 0.4927.

Prefix trend:

- Most C1/C2 leaders do worse at p24 than p8. DINO C2 drops by -0.0954 and Qwen C2 by -0.0918.
- V-JEPA2 is the clearest positive p24-p8 case for C3 (+0.0303).
- This argues against reporting only best prefix.

## 7. Key Findings

### Finding 1: F3R is the strongest direct-geometry representation in the current A1/A2 suite.

Evidence: `logs/eval_results/results_table.csv` shows F3R mean depth 0.1347, camera AUC-30 0.3852, A2 balanced accuracy 0.9827. It is first on depth, camera, and view consistency.

Interpretation: F3R's 3D/multiview training objective likely yields more explicit metric and view-overlap information than generic video generators.

Confidence: High for validation-set readout performance; medium for general scientific claim because no multi-seed or held-out test publication artifact was found.

### Finding 2: DINO and Qwen2.5-VL dominate action-conditioned C retrieval despite weaker A1 camera geometry.

Evidence: `c_series_prefix_robustness.md` shows DINO C1 AUC 0.4556 and Qwen C1 AUC 0.4343; DINO C2 AUC 0.1396 and Qwen C2 AUC 0.1316. In A1 camera, DINO is only 0.0386 and Qwen 0.0269.

Interpretation: C-series success may rely on stable feature identity/retrieval structure and action-conditioned correspondence, not simply metric camera-pose decodability.

Confidence: High for observed metric separation; medium-low for causal explanation.

### Finding 3: Video generation backbones are not consistently best at spatial probing.

Evidence: Wan/CogVideoX/Aether trail F3R/DINO/V-JEPA2/VideoMAE/Qwen on A1/A2 means and on most C prefix-AUC rankings. For example, Aether A2 mean 0.7238 and C2 AUC 0.0015; CogVideoX C2 AUC 0.0115.

Interpretation: Generative video modeling alone does not guarantee easily decodable metric/spatial/action structure in the probed layers.

Confidence: Medium-high for these backbones/layers; layer sweeps are incomplete for some families.

### Finding 4: B2 hidden-object localization has signal but remains coarse and metric-sensitive.

Evidence: B2 angle table in `docs/session_handoff_20260720_c_retrain_ab_eval.md` reports best mean angular errors around 55-56 degrees; Wan early B2 v1 vs ctrl improved from 59.3 deg to 47.1 deg in historical mixed-source/non-current results in `PROGRESS_REPORT.md`.

Interpretation: Object-conditioned representations contain some direction signal, but current probes are far from precise object permanence. Top-1 bin accuracy can disagree with angular error and should be secondary.

Confidence: Medium. The current angle summary is report-level rather than a single canonical CSV, and old mixed-source B2 numbers are historical.

### Finding 5: Longer context is not automatically better.

Evidence: `c_series_prefix_robustness.md` shows negative p24-p8 for DINO/Qwen/VideoMAE/Wan on C1/C2; DINO C2 falls from 0.1944 to 0.0990.

Interpretation: Longer prefixes may add irrelevant or harder-to-optimize context, and fixed-shape prefix-specific jobs can favor shorter, cleaner context windows.

Confidence: High for observed C metrics; interpretation needs controls for sample difficulty and prefix-window composition.

### Finding 6: Several early positive results were invalidated by confounds, but the project responded by hardening protocols.

Evidence: `PROGRESS_REPORT.md` documents A3 dtype and seed leakage; `docs/session_handoff_20260720_c_retrain_ab_eval.md` documents C zero-target bug; `TEMPORAL_DATA_VALIDITY.md` invalidates Infinigen temporal conclusions; `docs/c_retrain_session_report_20260724.md` documents manifest/AOSS repair and successful C retraining/eval.

Interpretation: The project has an unusually clear audit trail of negative findings and protocol corrections; current results are stronger because of those corrections.

Confidence: High.

## 8. Negative Results and Failed Attempts

### A2 accuracy inflation

Old A2 results had high raw overlap accuracy, but the validation/test set was heavily negative. `PROGRESS_REPORT.md` reports about 84.3% no-overlap baseline in the older setup. Current A2 uses balanced accuracy/AUROC/PR-AUC, which is a protocol improvement.

Status: Confound identified and mitigated in current eval.

### A3 dtype leakage and seed leakage

The abnormal temporal-order control initially learned dtype and split-reused random noise patterns rather than temporal order. After fixes, Wan ctrl pair_acc dropped to 16.5% with mean_delta near 0, while Wan v1 reached 86.3%. V-JEPA2 old 97.5% is not clean evidence.

Status: Wan fixed result is useful; V-JEPA2 A3 needs rerun under fixed protocol.

### B1 GT-mask leakage / task-design weakness

B1 with masks and pose-era historical variants performed well, but ablations showed nomask approximately collapsed to pose-only/control. Current B1 no longer gets pose, and B2 was introduced to remove spatial mask input to the head.

Status: B1 is useful as an upper-bound/diagnostic but should not be sold as mask-free object permanence.

### C zero-target bug

Old C checkpoints were invalid because target features were missed and targets became zero/fallback. `docs/session_handoff_20260720_c_retrain_ab_eval.md` marks old C results invalid; `docs/c_retrain_session_report_20260724.md` records the fixed retrain/eval.

Status: Resolved for current C results; old C checkpoints must not be used.

### Aggressive AOSS retraining launch

An aggressive C launch with too many slots and batch size 8 produced CUDA OOM, especially for 3072-channel VFMs. The final safe config used one process per GPU, batch size 4, and manifests.

Status: Operational negative result; use conservative defaults.

### Wan/Cog A/B layer sweep recovery

The 2026-07-25 A/B layer campaign had many failed/retry/interrupted jobs. By 2026-08-03, 213/223 had evaluable results, but many were provisional latest checkpoints rather than true best.

Status: Useful exploratory evidence; not all formal.

## 9. Caveats and Experimental Risks

- Train/val/test protocol: `EXPERIMENT_PROTOCOL.md` says at least seeds 41/42/43 and frozen val-selected settings should be evaluated once on test. Current artifacts are mainly validation summaries; no final held-out test table was found.
- Single-seed risk: Most reported rows appear seed 42 / one run per setting. Seed variance is not yet quantified.
- Best-prefix selection bias: C best-prefix tables are optimistic. Prefix AUC and p24-p8 should be preferred for model-level claims.
- B2 metric packaging risk: Main CSV records B2 top-1, while stronger B2 conclusions use angle values reconstructed in reports. A canonical B2 angle CSV should be produced.
- Probe-capacity confound: Transformer heads can be powerful, especially B2 all-patch attention. Linear/MLP controls exist in protocol but are not part of the streaming sweep.
- Dataset/window bias: ScanNet++ windows, object visibility selection, and prefix sampling may introduce object/category/viewpoint priors. B2 ctrl still showed non-random signal historically.
- Feature retrieval shortcut: C metrics evaluate predicted feature retrieval. High R@1 may reflect feature-space smoothness or identity cues, not necessarily explicit physical simulation.
- AOSS/environment fragility: environment variables can override Hydra roots; setting `INSCENE_STREAMING_FEAT_ROOT` to a parent path breaks training.
- Dirty worktree: many current scripts/manifests/results are untracked. Reproducibility depends on preserving these artifacts.
- New model expansion: LingBot/MiniMax-H3 are configured, but final result evidence is missing in the audited summaries.
- Data inventory gap: the C retrain audit resolved the working set to 681 valid scenes from 683 visible candidates; the identity of the "original 684th scene" is not recoverable from current repository artifacts.

## 10. Open Questions

1. Do C1/C2/C3 improvements survive no-action, shuffled-action, copy, and feature-smoothness baselines under the same manifest and prefix-AUC reporting?
2. Are DINO/Qwen C results evidence of spatial action understanding, or mostly high-quality target-feature retrieval?
3. Why do longer prefixes hurt many C tasks? Is this due to harder sample composition, optimization, model memory limits, or irrelevant context?
4. Can B2 hidden-object localization be made substantially more precise without GT spatial mask leakage?
5. Do layer sweeps change the main model ranking once only formal best checkpoints are used?
6. Are findings stable across seeds and final held-out test split?

## 11. Recommended Next Experiments

### P0 — Must Do

Experiment: canonical test-set evaluation for A/B/C.

- Question: Which current validation findings survive the frozen test protocol?
- Hypothesis: F3R remains strongest on A1/A2; DINO/Qwen remain strongest on C.
- Minimal experiment: choose settings by current val results, run one held-out test evaluation with exact commit/dirty status, manifest hashes, feature roots, seeds, and checkpoints recorded.
- Expected evidence: one canonical `results/test_*.csv` and markdown report.
- Success criterion: rankings remain qualitatively stable.
- Failure interpretation: current validation results overfit settings/prefix/layer selection.

Experiment: C baseline battery on current fixed manifests.

- Question: Does action input causally matter?
- Hypothesis: True action-conditioned models beat no-action, shuffled-action, and last-observation-copy baselines by more than noise.
- Minimal experiment: for top models DINO/Qwen/V-JEPA2/Wan, report C1/C2 R@1 and mean rank against no-action, shuffled-action, and copy baselines already logged in raw C CSV where available.
- Expected evidence: baseline-normalized deltas, not just absolute R@1.
- Success criterion: positive action gain across prefixes, not only best prefix.
- Failure interpretation: C success is likely retrieval/copy/feature-prior driven.

Experiment: canonical B2 angle summary.

- Question: How strong is hidden-object localization under current streaming ScanNet++ protocol?
- Hypothesis: B2 has weak but nonzero object-conditioned direction signal.
- Minimal experiment: generate a canonical CSV with `mean_ang_err_deg`, `az/el`, `top1/top3`, log-distance, n, prefix, model, layer, checkpoint.
- Expected evidence: one auditable B2 angle table replacing report-only extraction.
- Success criterion: real models beat controls on angular error and log-distance.
- Failure interpretation: current B2 story is metric/report fragile.

### P1 — Strongly Recommended

Experiment: multi-seed verification for top and ambiguous settings.

- Question: Are main findings robust to random seed?
- Minimal experiment: seeds 41/42/43 for F3R A1/A2, DINO/Qwen C1/C2/C3, best B2 settings.
- Success criterion: confidence intervals do not invert rankings.

Experiment: prefix difficulty audit.

- Question: Why does p24 often underperform p8?
- Minimal experiment: compare sample overlap, motion magnitude, target distance, valid horizon counts, and object visibility distributions by prefix.
- Success criterion: determine whether p24 drop is data difficulty or representation/context limitation.

Experiment: formalize Wan/Cog layer sweep.

- Question: Are middle layers reliably better?
- Minimal experiment: rerun/evaluate only missing formal best rows, remove provisional rows from headline claims.
- Success criterion: one formal-only layer-sweep summary.

### P2 — Optional

Experiment: extend finalized metrics to MiniMax-H3/LingBot/BAGEL/Qwen2.5-VL-3B.

- Question: Do newer video/VLM backbones change the model-family story?
- Minimal experiment: complete feature audit, training, and eval tables matching the C/A/B schemas.
- Success criterion: no missing summaries, no zero-valid rows, and canonical result CSVs.

Experiment: readout-capacity controls.

- Question: Are Transformer probe heads too strong?
- Minimal experiment: compare linear/MLP/Transformer for B2 and C top settings.
- Success criterion: finding remains with lower-capacity readouts or is honestly reclassified as nonlinear probe evidence.

## 12. Current Project Status

| Component | Status | Evidence | Next Step |
| --- | --- | --- | --- |
| Core codebase | Functional but dirty | `git status --short` shows modified tracked files and many untracked scripts/logs/manifests | Preserve/commit reproducibility-critical scripts/manifests |
| Streaming protocol | Current default | `README.md`, `TEMPORAL_DATA_VALIDITY.md`, configs under `inscene15k_streaming` | Keep ScanNet++ only for temporal claims |
| A1/A2 default results | Reportable validation | `logs/eval_results/results_table.csv` | Produce canonical test table |
| B2 default results | Preliminary/reportable with caveat | `docs/session_handoff_20260720_c_retrain_ab_eval.md`, per-run summaries referenced there | Generate canonical angle CSV |
| A3 abnormal | Legacy partial | `PROGRESS_REPORT.md` | Rerun V-JEPA2/final A3 if needed |
| C1/C2/C3 default+extra | Reportable validation | `logs/inscene15k_streaming/results/c_eval_with_extra_20260728/` | Add baseline-normalized C table and test eval |
| C manifests/AOSS | Operational | `manifests/c_retrain_20260722/*.meta.json`, `docs/c_retrain_session_report_20260724.md` | Track manifests and audit sidecars |
| Wan/Cog layer sweep | In progress | `logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/` | Separate formal from provisional results |
| VideoMAE/Qwen C extra | Completed validation | `c_eval_with_extra_20260728` validation JSON, raw/best CSVs | Add to canonical main result table |
| LingBot/MiniMax-H3 | Implemented but not validated | `features/lingbot/`, `features/minimax_h3/`, configs/jobs/logs | Find or run final eval summaries |

## 13. Important Files / Entry Points

Training and evaluation:

- `vidfm3d/train.py`: Hydra/Lightning training entry.
- `vidfm3d/eval_diag.py`: final diagnostic evaluation entry for A2/B/C and summary writing.
- `eval_b2.py`, `eval_c1_metrics.py`, `eval_pertask_v3.py`: task-specific/legacy evaluators.
- `scripts/run_streaming_probe_sweep.sh`: local streaming sweep launcher.
- `scripts/launch_c_retrain_40gpu.sh`: C-series multi-node retraining launcher, untracked but used.
- `scripts/run_c_eval_slots.py`: C-series evaluation slot launcher, untracked but used.

Dataset and feature extraction:

- `vidfm3d/data/components/inscene15k_dataset.py`: central InsScene-15K loader, streaming-prefix, manifests, B/C target construction, AOSS path support.
- `vidfm3d/utils/aoss_fs.py`: local/AOSS filesystem abstraction.
- `vidfm3d/utils/temporal_sampling.py`: motion-normalized temporal window sampling.
- `features/run_inscene15k.py`: feature extraction for Wan/CogVideoX/DINO/VideoMAE/Aether/F3R and streaming modes.
- `features/minimax_h3/`, `features/lingbot/`: newer feature backends.

Probe modules:

- `vidfm3d/models/probe_ext_module.py`: unified Lightning module for A2/A3/B1/B2/C1/C2/C3/SAE dispatch.
- `vidfm3d/models/components/probe_view_consistency.py`: A2 head.
- `vidfm3d/models/components/probe_ego_belief.py`: B1 head.
- `vidfm3d/models/components/probe_ego_belief_v2.py`: B2 query-token head.
- `vidfm3d/models/components/probe_action_dynamics.py`: C1 head.
- `vidfm3d/models/components/probe_path_integration.py`: C2 head.
- `vidfm3d/models/components/probe_counterfactual.py`: C3 head.

Configs:

- `configs/experiment/inscene15k_streaming/`: current default streaming configs.
- `configs/experiment/inscene15k_ext/`: legacy/non-streaming or direct-VLM extension configs.
- `configs/splits/inscene15k_v1.json`: scene split manifest.
- `vidfm3d/utils/feature_layers.py`: layer/channel/file naming registry.

Results and reports:

- `logs/eval_results/results_table.csv`: A/B default validation table.
- `logs/eval_results/results_summary.md`: A/B default markdown summary.
- `logs/inscene15k_streaming/results/c_eval_20260724_1903/`: fixed 6-model C final eval.
- `logs/inscene15k_streaming/results/c_eval_with_extra_20260728/`: C final eval with VideoMAE/Qwen and prefix robustness.
- `logs/probing_jobs_20260725_ab/recovery_20260730/perf_analysis/evaluation_results_all_probes.md`: Wan/Cog A/B layer-sweep status.
- `docs/c_retrain_session_report_20260724.md`: C retrain/eval repair story.
- `docs/session_handoff_20260720_c_retrain_ab_eval.md`: A/B consolidation and C bug handoff.
- `PROGRESS_REPORT.md`: historical progress report; useful but contains superseded/non-current results.
- `EXPERIMENT_PROTOCOL.md`, `TRAINING_LOGIC_AUDIT.md`, `TEMPORAL_DATA_VALIDITY.md`: protocol and caveat sources.

## Self-Review

- Hypotheses vs conclusions: Causal interpretations are marked as interpretation/speculation; observed metrics are separated from explanations.
- Negative results: A2 imbalance, A3 leakage, B1 mask leakage, C zero-target bug, AOSS OOM, and provisional layer-sweep status are included.
- Evidence: Major claims point to concrete files or result artifacts.
- Findings: The report emphasizes F3R geometry, DINO/Qwen C retrieval, B2 weakness, prefix non-monotonicity, and protocol hardening rather than only listing runs.
- Bottleneck: The top bottlenecks are canonical test evaluation, C baseline interpretation, and B2 canonical angle packaging.
