# Stochastic Morphological pi-Mixture Kernels

**Strengthening campaign working branch - Step 4 (derived from compact release v2.1.0)**

Companion manuscript: *Stochastic Morphological pi-Mixture Kernels: Identifiability, Topological Stability, and Output-Space Learning*, Adnan H. Abdulwahid and Ram C. Neupane (2026). This is an unpublished research package prepared for JVCIR submission.

This working branch is intentionally kept compact and is **not yet the archival public release**. Steps 2--4 add direct output-space learning, a broad retraining invariance stress suite, and a global oracle/excess-risk decomposition with finite numerical checks while preserving the frozen v2.1.0 baseline. The eventual GitHub package will remain below 25 MB. It contains the complete source code, clean input arrays, lightweight numerical summaries/checks, figures, documentation, LaTeX supplement source, and citation metadata. Large fitted-model binaries and cached per-case result banks are stored only in the companion Zenodo archival snapshot.

## Repository / archive split

- **GitHub:** code, clean inputs, configuration, documentation, figures, lightweight summaries, and reproducibility scripts.
- **Zenodo:** the complete frozen archival snapshot, including fitted `.joblib` models, candidate banks, prediction arrays, and large compressed metric tables.
- `ZENODO_ASSETS.csv` records every artifact omitted from this compact GitHub package, with its size and SHA-256 digest.

Public baseline repository: `https://github.com/aaalkhafaji/Stochastic-Morphological-Mixture-Kernels`. Frozen baseline archive: `https://doi.org/10.5281/zenodo.22637861`. Campaign changes should not overwrite that frozen release; they will become a later version only after the full 12-step audit.

## Step-2 direct output-space learner

`code/output_space_learning.py` implements a scalar cost learner on canonical distinct-output rows `phi(x,z)`. `code/test_output_space_invariance.py` retrains that learner after action permutation, exact duplication, and permutation+duplication and checks that the canonical training table, fitted costs, and output law are unchanged to tolerance `1e-12`.

## Step-3 invariance stress suite

`code/stress_test_output_space_invariance.py` expands the deterministic construction check to four image sizes, three fitting seeds, duplicate multiplicities through 64 copies, multiple duplicated operators, random permutations, multi-action duplication, unequal multiplicities, and output-preserving class refinements. The recorded stress run performs 3,136 canonical signature checks, 288 independently retrained nonbase comparisons, 6,912 output-law comparisons, and 288 direct API checks. All recorded discrepancies are exactly zero; see `results/step3_invariance_stress_summary.json` and `results/step3_invariance_stress_detail.csv`.

## Step-4 oracle/excess-risk audit

The manuscript now separates excess risk into finite-bank approximation, output-cost estimation, and entropy/reference terms. `code/audit_oracle_risk.py` independently checks the pointwise/general-reference bound, the uniform-reference specialization, the exact global decomposition, hard output selection, and monotonicity under bank enlargement on 2,500 random finite problems (19,692 conditional inputs). All bound violations are zero to numerical precision and the largest exact-decomposition residual is `1.11e-16`; see `results/step4_oracle_risk_checks.json`.

The archived `code/kernel_quotient.py` remains as the post-hoc action-risk quotient baseline.

## Quick validation

Use Python 3.12. From the extracted repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python code/check_package.py
python code/reproduce.py --level quick --workdir ../smm-quick
```

The `quick` level runs the self-contained mathematical audit and does not require the omitted model/result binaries.

## Full regeneration from GitHub alone

The clean input arrays remain in `data/`, so the large fitted models and result banks can be regenerated rather than downloaded:

```bash
python code/reproduce.py --level full --workdir ../smm-full
```

This creates a new work directory, rebuilds the campaign, refits models, and regenerates analyses. CPU time depends on the machine.

## Exact replay from the archived frozen artifacts

For exact byte-level replay of the originally frozen fitted models and cached result arrays, download the companion **`JVCIR_Zenodo_Archive_v2.1.0.zip`** after it is deposited on Zenodo. Extract/copy the archived `models/` and heavy `results/` assets into a working copy of this repository, then run:

```bash
python code/reproduce.py --level analysis --workdir ../smm-analysis
```

The excluded-file hashes in `ZENODO_ASSETS.csv` make the GitHub/Zenodo split auditable.

## Contents

| Path | Purpose |
|---|---|
| `code/` | Morphology, learning, quotient kernels, campaign, analyses, and checks |
| `data/` | Clean masks/source arrays, IDs, hashes, and upstream license notice |
| `models/README.md` | Explains where archived fitted models live and how to regenerate them |
| `results/` | Lightweight CSV/JSON/LaTeX summaries and numerical checks |
| `figures/` | Manuscript/supplement figures in PDF and PNG |
| `docs/` | Reproducibility, validation, GitHub, LaTeX, and Zenodo guidance |
| `Supplementary_Guide.pdf` | Reader-facing supplementary overview |
| `JVCIR_Supplementary_Guide.tex` | Editable LaTeX source for the guide |
| `CITATION.cff` | Citation metadata |
| `.zenodo.json` | Zenodo release metadata |
| `ZENODO_ASSETS.csv` | Manifest of large artifacts stored in the Zenodo snapshot |
| `SHA256SUMS` | Checksums for this compact GitHub release |
| `LICENSE_NOTICE.md` | Project rights status |
| `THIRD_PARTY_NOTICES.md` | Data attribution and upstream notices |

## GitHub upload

Extract this ZIP and commit/upload the **contents**, not the ZIP file itself. The repository source package is deliberately below 25 MB, and no individual file approaches GitHub's ordinary per-file limits. Do not commit regenerated `*.joblib`, large `*.npz`, or compressed per-case metric files; `.gitignore` and the Zenodo split are intended to keep those out of Git history.

## Citation and generative-AI disclosure

Use `CITATION.cff` for citation metadata. The frozen baseline manuscript contains the authors' earlier AI declaration. Because this strengthening working branch uses AI assistance in scientific theorem/code development, that declaration must be reconsidered before any revised submission so that the final disclosure accurately reflects the actual workflow and the publisher policy then in force.

## Step-5 Oxford-IIIT Pet higher-resolution campaign

`code/run_oxford_pet_step5.py` is the frozen Step-5 protocol for an independent natural-image mask domain. It uses only the official Oxford-IIIT Pet trimap annotation archive, preserves aspect ratio by nearest-neighbor letterboxing to 128×128, retains the entire official test partition, evaluates all six predeclared primary corruptions, and reports clean-mask-clustered bootstrap intervals. The external Oxford archive is **not** redistributed here; see `data/OXFORD_IIIT_PET_NOTICE.md` and `docs/OXFORD_PET_STEP5.md`.

The official Step-5 campaign has now been executed through the manual GitHub Actions workflow. The run used 1,840 fitting, 1,840 validation, and all 3,669 official test masks at 128x128; every held-out mask was evaluated under all six frozen corruption conditions (22,014 test cases). The action-coordinate selector achieved mean composite loss 0.0162 versus 0.0164 for the validation-best fixed morphology, while the direct-output selector was adverse at 0.0201. The two predeclared paired clean-mask comparisons confirm higher direct-output loss relative to both baselines. Lightweight summaries/provenance are frozen under `results/step5_oxford_pet/`; large caches remain outside the compact GitHub tree for archival deposition.

## Step 6: DAVIS 2016 genuine segmentation-error validation

Step 6 tests the morphology framework on actual algorithm-produced segmentation
errors from the official DAVIS 2016 benchmark.  It uses the pre-computed masks
of the six unsupervised methods NLC, FST, SAL, TRC, MSG, and CVOS and the
official dense ground-truth annotations.  No synthetic corruption is introduced
in this campaign, and statistical inference is clustered by video sequence.

The easiest execution route is GitHub Actions:

1. Open **Actions**.
2. Choose **Step 6 DAVIS 2016 genuine-error validation**.
3. Click **Run workflow** on `main`.
4. After success, download the artifact **step6-davis2016-results**.

The workflow downloads the official DAVIS 2016 data and pre-computed result
archives, verifies the official MD5 published for `DAVIS-data.zip`, records
SHA-256 hashes for both downloaded archives, selectively extracts only masks and
split files (RGB frames are not used by the estimator), runs the predeclared
analysis, and freezes checksums for the compact result artifact.
