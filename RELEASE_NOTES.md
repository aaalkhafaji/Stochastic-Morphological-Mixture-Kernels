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
