# Stochastic Morphological pi-Mixture Kernels

**Compact GitHub reproducibility repository, version 2.1.0**

Companion manuscript: *Stochastic Morphological pi-Mixture Kernels: Identifiability, Topological Stability, and Output-Space Learning*, Adnan H. Abdulwahid and Ram C. Neupane (2026). This is an unpublished research package prepared for JVCIR submission.

This GitHub package is intentionally kept **below 25 MB**. It contains the complete source code, clean input arrays, lightweight numerical summaries/checks, figures, documentation, LaTeX supplement source, and citation metadata. Large fitted-model binaries and cached per-case result banks are stored only in the companion Zenodo archival snapshot.

## Repository / archive split

- **GitHub:** code, clean inputs, configuration, documentation, figures, lightweight summaries, and reproducibility scripts.
- **Zenodo:** the complete frozen archival snapshot, including fitted `.joblib` models, candidate banks, prediction arrays, and large compressed metric tables.
- `ZENODO_ASSETS.csv` records every artifact omitted from this compact GitHub package, with its size and SHA-256 digest.

The Zenodo DOI and GitHub repository URL should be inserted into `CITATION.cff`, `.zenodo.json`, and this README after the records are actually created; no identifier is invented here.

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

## Citation and generative-AI declaration

Use `CITATION.cff` and add the eventual repository URL/article DOI after publication. The manuscript declaration states that AI-assisted tools were used only for grammar correction, language readability, and formatting support; they were not used to generate scientific content, mathematical derivations, simulations, data, figures, results, analysis, interpretations, or conclusions. All content was reviewed, verified, and approved by the authors, who take full responsibility for the accuracy and integrity of the manuscript.
