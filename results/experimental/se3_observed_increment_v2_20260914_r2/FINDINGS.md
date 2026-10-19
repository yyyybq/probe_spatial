# Incremental findings

## New-scene confirmation

DINO-L0 is the only runnable four-core-model cell. On 118 tuples from 100 scenes outside the legacy 69, two-step gain versus copy is 0.2863 (95% CI [0.2397, 0.3308]). Correct-action gain versus the learned without-condition predictor is 0.1952 (CI [0.1605, 0.2295]). These are distinct claims.
Normalized observed-order margin is -0.0009 (CI [-0.0079, 0.0072]); tie-aware accuracy is 0.4660. Control-adjusted comparisons are in `paired_bootstrap.csv`.
This confirms or weakens only the DINO cell; it is not four-model confirmation.

## Real-history contrast

Unavailable under the required contract. There are 1731 exact V-JEPA2 L23 p8 context states in the frozen A/B/C design and no complete non-canonical matched store. Existing eval dumps contain one original-window source plus isolated endpoints, so they cannot implement a recursive same-role A/B/C comparison. `vjepa2_real_history_missing_contexts.csv` is the exact supplement list.

## Sparse layer contrast

- V-JEPA2 L0: gain vs copy 0.1238; gain vs without-condition -0.0828; normalized order -0.0036; tie-aware accuracy 0.4742.
- V-JEPA2 L17: gain vs copy 0.1280; gain vs without-condition -0.0018; normalized order -0.0084; tie-aware accuracy 0.4121.
- V-JEPA2 L23: gain vs copy 0.0777; gain vs without-condition 0.0366; normalized order -0.0113; tie-aware accuracy 0.3773.

This is an availability-frozen L0/L17/L23 contrast on 169 validity-intersection tuples, not a complete early/middle/late sweep; the predeclared middle L11 cell is absent.

## Local protocol corrections

The additive A→B→C and B→A paths are independently accumulated and checked against a float64 roundoff bound. They are never assigned to the same tensor. Composite support is described only as an empirical componentwise min–max range check, never as distribution-in certification.
