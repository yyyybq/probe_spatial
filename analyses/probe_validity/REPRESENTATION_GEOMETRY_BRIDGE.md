# Representation Geometry / Cosine Illusion Bridge

Status: `DESCRIPTIVE_REUSE_NOT_NEW_CONFIRMATION`

## Scope

This bridge reuses the historical P12 C1 common-set summary, not a new model
run: 276 validation samples from 69 scenes, prefix 12, and the selected default
layer for each model. Its source is `results/paper_validity_20260821T200224Z_2075f3e/c_baselines_summary.csv`
(`sha256=bd8e6ea824327462781e279619f814b243638cc9702e7f019371a2b702dacc13`). The historical summary was produced by
`analyses/paper_validity/run_paper_validity.py`; it uses the same Wan P12/L20
checkpoint and C1 aggregate values as the strict construct-validity audit
(Wan full R@1 0.1522; copy R@1 0.1196). This is development-set, post-hoc
mechanistic context and does not alter any predeclared gate or paper status.

`raw copy cosine` is the mean cosine from the last observed feature to the
future target. `temporal displacement = 1 - raw copy cosine`. `centered`
means the global target mean was subtracted before L2 normalization and cosine
retrieval; it removes a shared feature direction but is not whitening.

| Model | Raw copy cosine | Raw full cosine | Temporal displacement | Centered copy cosine | Centered full cosine | Copy R@1 | Full R@1 | Full - shuffle R@1 | Full - copy R@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vjepa2 L23 | 0.6635 | 0.8826 | 0.3365 | 0.1870 | 0.4432 | 0.1993 | 0.1304 | +0.0688 | -0.0688 |
| videomae L23 | 0.8587 | 0.9869 | 0.1413 | 0.1177 | 0.4784 | 0.1739 | 0.0942 | +0.0254 | -0.0797 |
| wan L20 | 0.9789 | 0.9955 | 0.0211 | 0.3127 | 0.6708 | 0.1196 | 0.1522 | +0.0543 | +0.0326 |
| cogvideox L20 | 0.9234 | 0.9985 | 0.0766 | 0.0542 | 0.4770 | 0.0254 | 0.0326 | +0.0072 | +0.0072 |
| dino L0 | 0.8212 | 0.8567 | 0.1788 | 0.5791 | 0.6281 | 0.5290 | 0.4529 | +0.1884 | -0.0761 |

## What The Recovered Record Says

- Wan, not DINO, has the highest P12 raw temporal cosine among the four current
  core models: 0.9789 for Wan,
  corresponding to only 0.0211 mean raw
  cosine displacement. Its centered copy cosine remains 0.3127,
  so this is not solely a global-mean artifact.
- DINO is likely the remembered "highest" result for two different quantities:
  it has the highest P12 full retrieval R@1 (0.4529) and
  centered copy cosine (0.5791). It
  does *not* have the largest raw temporal cosine; Wan does. Centered full
  prediction alignment is instead highest for Wan (0.6708).
- Wan's raw cosine is close to saturation (copy 0.9789,
  full 0.9955), making raw cosine changes
  intrinsically hard to interpret.
  Yet its retrieval comparison is not a pure-copy tie: full minus copy is
  +0.0326 R@1, exactly the strict audit's
  point estimate. The confidence interval crosses zero, so the valid conclusion
  remains weak action-sensitive evidence, not an established non-copy transition.

## Discriminability And Actionability

Using global full R@1 only as a model-level *retrieval-discriminability proxy*
and legacy full-minus-shuffle R@1 as an *actionability proxy*, their correlation
over these five models is Pearson 0.9886 and
Spearman 0.9000. Centered copy cosine versus
the same actionability proxy is Pearson 0.9544
and Spearman 0.9000. These are only
five non-independent model-level points, share the same readout and candidate
pool, and are visibly influenced by DINO; they are descriptive covariation, not
evidence that feature geometry causes action sensitivity.

## Consequence For Wan's Copy Gate

The recovered record supports a mixed explanation. Wan's feature space is
unusually temporally smooth in raw cosine, which can make an MSE/cosine-trained
predictor look close to the target even when it mostly persists state. However,
centering leaves nonzero state persistence and the action-conditioned predictor
still has a positive, albeit imprecise, retrieval advantage over copy. Thus the
copy-gate failure cannot be reduced either to "no dynamics" or to a cosmetic
cosine artifact. The current evidence supports: `HIGH_TEMPORAL_SMOOTHNESS +
WEAK_UNRESOLVED_INCREMENTAL_ACTION_SIGNAL`.

## Not Available In The Recovered Record

The old result deliberately records `whitened_cosine = NaN`: the raw target
tensors required to fit a train-only whitening transform were not retained in
that packaged analysis. It also does not contain random-pair or same-scene
target-pair cosine distributions, covariance spectra/effective rank, pairwise
prediction separation under matched actions, or matched-action ground-truth
target separation. Those quantities should not be reconstructed from aggregate
cosines. A full geometry study would require the frozen raw target/prediction
tensors, a train-fitted centering/whitening transform, and scene-clustered
uncertainty; it would remain development evidence until an independent split
exists.
