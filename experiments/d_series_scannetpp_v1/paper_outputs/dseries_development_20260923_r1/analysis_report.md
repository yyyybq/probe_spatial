# Predictive spatial structure: development report

**Status: development-only.** The frozen protocol has 612 train and 69 development ScanNet++ scenes, 2,721 prefixes, and horizons 2 and 4. There is no historically unused held-out test split; no result below should be presented as final test evidence.

## Core scientific message

The question is not merely whether an encoder contains spatial information, but whether a predictive video representation keeps that information in a form a fixed downstream spatial reader can use. We separate three capabilities: reading target-view layout from real features (D1), using the camera-motion condition to predict a useful future latent (D2), and transferring a reader from a real to V-JEPA2's native predicted future latent (D3).

## What the completed experiments establish

**D1: spatial structure is sample-specific, not a layout prior.** For Wan2.1, the spatial readout reaches MAE `0.1547±0.0007` at h2 and `0.1517±0.0007` at h4, versus `0.2406` and `0.2377` for the train-layout prior. The paired target-minus-prior MAE differences are `-0.0859 [-0.0936,-0.0781]` (h2) and `-0.0860 [-0.0935,-0.0783]` (h4). Target features also beat a within-scene wrong target and a copied context state. Thus the probe reads sample-specific target layout rather than a scene-average or temporal-copy baseline.

**D2: the motion condition contributes beyond context alone.** For Wan's external predictor, true camera motion has lower MAE than identity motion (`-0.0058` at h2; `-0.0058` at h4), matched-horizon shuffled motion (`-0.0065`; `-0.0064`), and an independently trained same-family context-only predictor (`-0.0028`; `-0.0045`). All of these MAE confidence intervals exclude zero. This is a modest but repeatable conditional-prediction effect; it is not evidence of intervening on a physical action.

**D3: V-JEPA2's native predictor retains spatial structure, but shifts the readout interface.** A spatial reader trained on real V-JEPA2 future states has MAE `0.1824/0.1827` (h2/h4) on real features, `0.2459/0.2427` on predicted features, and `0.2890/0.2974` on the copied last context. Frozen transfer is therefore substantially better than copying context but worse than reading the real target. Training an equal-capacity reader directly on predicted latents recovers to `0.2233/0.2271`. The appropriate conclusion is partial preservation plus an interface shift, not either perfect preservation or information-theoretic loss.

## Scope and next reporting step

The evidence is now complete for Wan D1/D2 and V-JEPA2 D1/D2/D3 at three seeds. The attached tables provide h2/h4 metrics, seed variation, and paired scene-cluster confidence intervals for Wan; the existing V-JEPA2 report provides the D3 transfer intervals. A combined final figure/table should keep D1, D2, and D3 distinct instead of ranking models with a single score.

## Evidence sources

- Wan D1/D2 three-seed report: `../../diagnostics/dseries_wan_d1_d2_three_seed_20260923_zoetrope_r2/report.md`
- V-JEPA2 D1/D3 three-seed report: `../../diagnostics/dseries_vjepa2_d1_d3_three_seed_20260917_zoetrope_r1/report.md`
- All raw records are linked by hashes in each report's `summary.json`.
