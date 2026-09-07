# Reproducibility protocol and data dictionary

## Environment and execution

Reference software: Python 3.12; exact package versions in `requirements.txt`.
CPU-only execution uses one BLAS/OpenMP thread and `n_jobs=1` for the forests.
The wrapper sets thread limits before launching any numerical script. Install
packages in a fresh virtual environment and execute from the package root.
Model deserialization uses joblib; use the supplied, checksum-verified model files.

`python code/check_package.py` checks the entire frozen release without third-party
packages. `python code/reproduce.py --level quick --workdir ../smm-quick` copies
this package to a fresh directory, performs the original mathematical checks,
and replays two representative cases per dataset for all 15 forests. The smoke
check regenerates the morphological outputs and verifies frozen-pair invariance.
It is a packaging check, not a new empirical experiment.

The analysis level executes the following scripts in the copied package:

```bash
python code/audit_math.py
python code/smoke_check.py
python code/verify_release.py
python code/summarize.py
python code/audit_kernel_theory.py
python code/validate_kernel_quotient.py
python code/plot_topology_sharpness.py
python code/build_supplement_guide.py
```

`verify_release.py` replays all five seeds for each dataset over every retained
test case, independently regenerates selected inputs and filter scores, and
refits seed 101 from each complete training bank. It also remeasures CPU runtime.
The full level first deletes only copied `results/*_bank.npz` caches and calls
`python code/run_campaign.py --stage all`. Existing caches otherwise are reused
by the extractor. This is why the wrapper explicitly removes them for full runs.
The scripts write results into their own copied root, not the frozen archive.

For optional reconstruction from original source archives, first create a work
copy with the wrapper, then run `python code/prepare_data.py --cache downloads`
inside it. The script generates synthetic masks and downloads the four image
archives recorded in `data/source_manifest.json`. Compare that manifest with the
frozen original, then rerun the full level from this copy. Source hashes are not
silently replaced in the original release. No network is used by the ordinary
quick, analysis, or full levels after dependency installation.

## Frozen design

The operator bank has 31 fixed actions. `morphology.py:NAMES` and
`results/environment.json` specify the order. Candidate outputs and five global
statistics of the input and each output produce 160 inference features. Targets
are training/validation labels only; they do not enter inference features.
Forests use 96 ExtraTrees estimators with all features, no bootstrap, and one job.
Leaf size is selected from 2, 8, and 20 with validation seed 99; final seeds are
101, 102, 103, 104, and 105. See the retained selection JSON for every choice.

| Dataset | Shape | Training | Validation | Scored test |
|---|---|---:|---:|---:|
| Synthetic shapes | 48 x 48 | 1,000 | 300 | 499 |
| MNIST binary masks | 28 x 28 | 2,000 | 500 | 1,000 |
| Fashion-MNIST binary masks | 28 x 28 | 2,000 | 500 | 1,000 |

Public pixels are binarized at intensity >=128. Train/validation images come
from official training partitions. Public test IDs used in development and
identical binary masks are quarantined. The prospective shapes test has 500
stored masks; test ID 173 duplicates validation and is excluded, leaving 499.
The original bytes and exclusion ID remain available for traceability.

Corruption probabilities are deletion then addition, applied according to
`run_campaign.py:corrupt`; consult that function for the precise pixel law.

| Index | CSV name | Deletion | Addition | Use |
|---:|---|---:|---:|---|
| 0 | balanced04 | .04 | .04 | Primary |
| 1 | balanced10 | .10 | .10 | Primary |
| 2 | salt04 | .01 | .04 | Primary |
| 3 | salt10 | .01 | .10 | Primary |
| 4 | pepper04 | .04 | .01 | Primary |
| 5 | pepper10 | .10 | .01 | Primary |
| 6 | shift20 | .20 | .20 | Diagnostic |
| 7 | clean | 0 | 0 | Diagnostic |

There are 14,994 primary cases and 4,998 diagnostic cases. Training and
validation use one corruption per image; final test images use all eight.
Repeated corruption cases are not treated as independent source images.

## Metrics and inference rules

The composite loss is 0.4(1-Dice) + 0.2(binary MSE) + 0.2 times each of two
capped, target-normalized Betti errors, exactly as defined in `morphology.py`.
The separately reported Betti error columns are raw absolute errors, not the
capped terms used in the loss. Foreground uses 8-connectivity; background uses
4-connectivity with a permanent exterior frame. Beta0 counts foreground
components; beta1 counts bounded background components, not foreground graph cycles.
All footprint pipelines are extended and then cropped at the pipeline end.

The conditional selector chooses a minimum predicted-risk action. Action Gibbs
weights all action labels. The output quotient assigns a class the minimum
predicted risk among labels producing that exact mask, applies a uniform-reference
Gibbs law over distinct masks, and lifts it uniformly within each class. The
exploratory temperature grid is .005, .02, .1; validation selects .005 for each
of the 15 frozen forests. The new construction was developed after inspecting
the retained test set, so its holdout evaluation is exploratory despite
validation-only temperature selection.

Sampled-method metrics are exact categorical expectations, not selected random
draws. Mean-mask results are separately scored. Aggregate PSNR is
-10 log10(mean binary MSE), not mean per-image PSNR. The 2,000-resample bootstrap
clusters the six corruptions and five model seeds by clean image. The six paired
control comparisons use Bonferroni-adjusted descriptive percentile intervals.
These intervals are conditional on this design, not distribution-free certificates.

## Array dictionary

| File/field | Meaning |
|---|---|
| `data/*_clean.npz: train, val, test` | Boolean masks, image x height x width |
| `train_ids, val_ids, test_ids` | Original partition row IDs; synthetic source indices |
| `source_train, source_test` | Selected unthresholded public image pixels |
| `pilot_test_ids` | Public rows quarantined after development use |
| `test_excluded` | Synthetic test indices excluded from scoring |
| `results/*_bank.npz: features` | Case x 160 inference feature array |
| `scores` | Case x 31 actions x 7 metrics |
| `image_index` | Index into the corresponding clean split array |
| `condition` | Integer index in the eight-condition table above |
| `shape` | Height, width |
| `outputs` | Candidate masks packed on the final width axis, big bit order |
| `noisy` | Noisy mask flattened then packed, big bit order |
| `*_predictions.npz` | Seed-keyed selected actions and categorical probabilities |
| `*_quotient_predictions.npz` | Risks, quotient probabilities, TV stress, class counts |

Metric axis order: loss, mse, dice, iou, beta0_error, beta1_error, topology_exact.
For a bank row `i`, decode outputs using
`np.unpackbits(bank['outputs'][i], axis=-1, count=W)`. Decode the input using
`np.unpackbits(bank['noisy'][i], count=H*W).reshape(H,W)`.
Map raw `image` or bank `image_index` through the corresponding `*_ids` array to
obtain the original source ID. Test shape indices skip 173. Primary campaign CSVs
use named conditions; quotient CSVs use numeric indices. Do not join these two
condition columns without converting them.

## Expected agreement

With the pinned stack, discrete actions and masks should agree exactly. Float32
bank/probability storage allows small rounding residuals. Existing replay checks
use the tolerances explicitly recorded in the code (probabilities 1e-7; sampled
score recomputation 2e-6). Use all available raw digits for comparisons, not the
rounded manuscript table. Runtime measurements, PDF timestamps, and compressed
container bytes need not match. The archive SHA-256 values certify the released
bytes, not future regeneration metadata.
