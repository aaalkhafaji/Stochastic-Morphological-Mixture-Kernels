# Release notes — v2.1.0

This repository snapshot accompanies the JVCIR submission *Stochastic
Morphological pi-Mixture Kernels: Identifiability, Topological Stability, and
Output-Space Learning* by Adnan H. Abdulwahid and Ram C. Neupane.

## Included

- Executable Python code for the empirical campaign and mathematical checks.
- Frozen clean inputs for Shapes, MNIST, and Fashion-MNIST-derived experiments.
- All fitted models used in the reported study.
- Frozen candidate banks, predictions, metrics, validation outputs, and summaries.
- Manuscript figures in PDF and PNG.
- Editable and compiled supplementary guide.
- Citation metadata and cryptographic checksums.

## Release status

This is an unpublished research release prepared for journal submission. No
article DOI or repository DOI is invented in the files. Add persistent identifiers
only after GitHub/Zenodo mint or expose them.

The project materials remain subject to `LICENSE_NOTICE.md`; this packaging step
does not select or grant a new open-source license.

## Compact GitHub packaging

The GitHub-distribution ZIP was reduced to below 25 MB by excluding fitted model binaries and large cached result arrays. Those frozen artifacts remain in the companion Zenodo archive and are enumerated in `ZENODO_ASSETS.csv`. Clean input arrays and all source code remain on GitHub so a full campaign can be regenerated.

## Strengthening branch — Step 5 staged validation

The strengthening branch now includes the successfully executed Oxford-IIIT Pet higher-resolution campaign. GitHub Actions processed the official annotation archive and all 3,669 official held-out test masks at 128x128 under six frozen corruption conditions. The action-coordinate selector remains competitive with the validation-best fixed morphology, while the direct-output learner is adverse under the two prespecified paired comparisons; all adverse conditions are retained. Lightweight result/provenance files are included in the compact repository, while large caches are reserved for the next Zenodo archival update. The frozen v2.1.0 baseline results remain unchanged.


## Strengthening branch — Step 6 genuine segmentation errors

The DAVIS 2016 campaign has now been executed successfully on official pre-computed outputs from six unsupervised source methods. The held-out analysis uses 20 video sequences, 1,376 unique ground-truth frames, and 8,236 source-method/frame cases with 5,000 sequence-clustered bootstrap resamples. Both learned selectors significantly reduce the manuscript composite loss relative to the raw segmentations under the predeclared family-wise contrasts, while region J and boundary F do not improve. The action-coordinate selector improves 17/20 sequences and all six source methods; the direct-output selector improves 17/20 sequences and five of six methods. Adverse sequences and the strong resizing/topology caveat are retained.

### Strengthening Step 7
- Added a predeclared structured-error Oxford-IIIT Pet distribution-shift campaign.
- Reuses the exact Step-5 IID training distribution, fixed action, hyperparameters, and five estimator seeds; no structured-error case enters fitting/tuning.
- Adds nine primary structured errors plus an eligibility-restricted natural component-dropout diagnostic.
- Adds clean-mask-clustered bootstrap inference with four predeclared family-wise loss contrasts and explicit adverse-condition accounting.

Step 7 completed successfully on all 3,669 held-out Oxford masks across nine predeclared structured errors (33,021 cases), without structured-error refitting or retuning. The direct-output selector reduces mean loss from 0.1587 to 0.1010 and improves aggregate Dice, IoU, exact topology agreement, and boundary F. The action-coordinate selector improves raw loss slightly but is significantly worse than the validation-best fixed morphology. Adverse structured families and the component-dropout diagnostic are retained.

## Strengthening branch — Step 8 modern/fair baselines

Step 8 completed successfully on the frozen Oxford protocol. Public SoftMorph2 was retrieved at runtime at the recorded upstream commit, and the independent product-logic implementation matched the public 2D code with maximum absolute discrepancy 0.0. A regularized empirical 3x3 W-operator and an intentionally advantaged condition-aware fixed-bank control were also evaluated.

Across 22,014 IID held-out cases, both learned selectors significantly outperform SoftMorph2 and the W-operator under the eight-comparison family-wise plan, although the condition-aware fixed control remains descriptively best. Across 33,021 structured-OOD cases with no retuning, the direct-output selector significantly outperforms SoftMorph2 and the W-operator, while SoftMorph2 significantly outperforms the action-coordinate selector. Unfavorable directions are retained.

## Strengthening branch — Step 9 theorem-targeted stress tests

- Added `run_theorem_stress_step9.py` with exact/finite checks for replication sensitivity, topology sharpness, incidence rank, Bayesian reversal, and Wasserstein composition.
- Enumerated all 2,304 Hamming-one pairs on `3x3`, all 512 `3x3` inputs for the 31-action incidence diagnostic, all 130,816 unordered row pairs for the Bayesian-error identity, and all 16 `2x2` states for exact finite transport.
- Verified the sharp replication constant through 256 duplicate copies and 4,500 additional random partitioned categorical laws.
- Added separate figures for replication sensitivity, Bayes error versus row-law TV, and exact composition contraction.
- These checks support theorem consequences only; they do not claim global contraction or identifiability of the trained high-resolution adaptive kernel.
