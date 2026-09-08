# Step 7 — Structured-error robustness under distribution shift

Step 7 is deliberately **out of distribution** relative to fitting. The action-coordinate and direct-output ExtraTrees selectors are trained only from the frozen Step-5 Oxford-IIIT Pet IID deletion/addition training cases. The Step-5 hyperparameters are reused without refitting to Step-7 validation outcomes:

- action-coordinate `min_samples_leaf = 20`;
- direct-output `min_samples_leaf = 8`;
- direct-output temperature `0.005`;
- validation-best fixed action `area_connected16`;
- estimator seeds `101–105`.

The entire official Oxford-IIIT Pet test partition (3,669 resized masks at 128×128) is then evaluated under nine predeclared structured errors. None of these structured cases is used for fitting or tuning.

## Primary structured errors

1. `boundary_expand2`: radius-2 binary dilation (systematic over-segmentation).
2. `boundary_contract2`: radius-2 binary erosion (systematic under-segmentation).
3. `clustered_fp03`: spatially correlated false-positive field affecting exactly 3% of eligible background pixels.
4. `clustered_fn08`: spatially correlated false-negative field affecting exactly 8% of eligible foreground pixels.
5. `holes2_r4`: two radius-4 interior deletions, preferentially centered at pixels with distance at least five from background.
6. `fragment_cut2`: a two-pixel horizontal or vertical cut through the central half of the object bounding box.
7. `bridge_spur12`: a thin 12-pixel outward corridor from a boundary point terminating in a radius-2 false-positive blob.
8. `occlusion25`: a localized rectangle with side lengths 25% of the object bounding-box dimensions, centered on a foreground pixel.
9. `mixed_structured`: correlated additions, correlated deletions, and a smaller localized occlusion in one case.

A secondary `component_dropout` diagnostic is applied only to held-out masks with at least two natural 8-connected foreground components of area at least five pixels. It is excluded from the nine-condition headline mean because it has a restricted eligibility set.

## Statistical unit

The clean held-out mask is the cluster. For each method, the nine primary structured conditions are averaged within mask before population inference. The four frozen primary loss contrasts are:

- action-coordinate selector minus raw structured input;
- direct-output selector minus raw structured input;
- action-coordinate selector minus validation-best fixed morphology;
- direct-output selector minus action-coordinate selector.

Each uses 5,000 clean-mask bootstrap resamples. Family-wise 95% percentile intervals apply a Bonferroni adjustment across those four comparisons.

## Execution

The preferred frozen input is the successful Step-5 GitHub Actions artifact:

```bash
python code/run_structured_step7.py \
  --step5-artifact step5-oxford-pet-results.zip \
  --output-dir results/step7_structured \
  --n-boot 5000
```

The script verifies the exact Step-5 artifact SHA-256 before accepting it. `--step5-cache-dir` is also supported for a verified working copy of the three Step-5 NPZ caches.

The full case metrics are compressed. Lightweight summaries, paired intervals, adverse-case accounting, seed summaries, protocol metadata, and SHA-256 checksums are retained in the compact repository package.
