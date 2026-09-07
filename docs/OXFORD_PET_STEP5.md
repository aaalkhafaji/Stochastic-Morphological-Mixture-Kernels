# Step 5 — Oxford-IIIT Pet higher-resolution mask validation

## Purpose

This campaign tests whether the stochastic morphological output-space framework continues to operate on real, human-annotated natural-image silhouettes well beyond the 28×28 MNIST/Fashion-MNIST grids used in the frozen baseline.

The estimator receives **binary masks, not RGB photographs**, so the official Oxford-IIIT Pet `annotations.tar.gz` archive is sufficient. The RGB image archive is not required for the primary quantitative experiment.

## Upstream source

Official dataset page: https://www.robots.ox.ac.uk/~vgg/data/pets/

Official annotations archive: https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

Expected archive MD5: `95a8c909bbe2e81eed6a22bccdf3f68f` (also used by torchvision).

License stated by Oxford: Creative Commons Attribution-ShareAlike 4.0 International; copyright remains with the original image owners.

The repository does **not** redistribute the Oxford annotation archive.

## Frozen primary protocol

- official `trainval.txt` supplies the train/validation pool;
- official `test.txt` supplies held-out test masks;
- deterministic breed-stratified selection over all 37 breeds;
- primary analysis uses **all official samples**: each breed-specific `trainval` pool is deterministically split 50:50 into fitting/validation subsets, matching the original paper's approximate 50-train/50-validation convention, and the entire official `test.txt` partition is retained;
- no held-out test mask is subsampled in the primary analysis;
- 128×128 aspect-ratio-preserving nearest-neighbor letterbox representation;
- primary trimap conversion: labels 1 and 3 are foreground; label 2 is background;
- six primary corruption conditions exactly match the frozen baseline manuscript;
- train/validation: one preassigned condition per clean mask;
- test: all six conditions per clean mask (about 22,000 held-out corrupted cases for the complete official test partition);
- five fixed estimator seeds, 101–105;
- tuning uses train/validation only;
- confidence intervals resample clean test-mask clusters after averaging corruptions and fitted seeds within each mask.

No result-based sample exclusion is permitted.

## Primary methods reported

1. corrupted input;
2. validation-best fixed morphology from the 31-action bank;
3. action-coordinate conditional selector;
4. direct-output selector from Step 2;
5. direct-output Gibbs sampled estimator (reported through exact expected metrics over its finite output law);
6. target-aware per-case bank oracle, diagnostic only.

The modern learned-morphology comparisons remain reserved for Step 8 so that Step 5 isolates the resolution/domain question.

## Run locally

From the repository root:

```bash
python code/run_oxford_pet_step5.py --download-annotations
```

If you already downloaded the official archive:

```bash
python code/run_oxford_pet_step5.py \
  --annotations-archive /path/to/annotations.tar.gz
```

The script verifies the official MD5 before extracting.

## Output

The script creates `results/step5_oxford_pet/` containing:

- exact selected IDs and protocol metadata;
- compact extracted candidate-score/features caches;
- raw held-out metric table;
- model/tuning record;
- clean-mask-clustered summary;
- paired 2,000-resample bootstrap comparisons;
- a generated LaTeX result table.

These outputs should be frozen and hashed before their numerical values are inserted into the manuscript.

## GitHub Actions

`.github/workflows/step5-oxford-pet.yml` provides a manual workflow that downloads only the official annotations, runs the campaign, and uploads a ZIP artifact with the numerical Step-5 outputs. The dataset archive itself is not committed to GitHub.
