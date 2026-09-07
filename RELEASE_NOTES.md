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

The working strengthening branch adds an execution-ready Oxford-IIIT Pet higher-resolution validation protocol, external-data provenance checks, a preprocessing topology audit, clean-mask-clustered bootstrap statistics, and a GitHub Actions execution workflow. No Oxford benchmark performance claim is included in this snapshot because the official annotation archive was not executed in the authoring environment. The frozen v2.1.0 baseline results remain unchanged.
