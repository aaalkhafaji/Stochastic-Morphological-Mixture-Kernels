# Results included on GitHub

This directory retains lightweight CSV/JSON/LaTeX summaries and mathematical checks. Large cached candidate banks, prediction arrays, and compressed per-case metric tables are intentionally excluded so the GitHub repository package remains below 25 MB.

For exact replay using the frozen precomputed artifacts, use the companion Zenodo archival snapshot. For a clean recomputation from the supplied input arrays, run:

```bash
python code/reproduce.py --level full --workdir ../smm-full
```

The excluded-file names, sizes, and SHA-256 digests are listed in `ZENODO_ASSETS.csv`.


Step-3 deterministic representation-invariance verification is stored in
`step3_invariance_stress_summary.json` and `step3_invariance_stress_detail.csv`.
These are construction checks, not independent benchmark evidence.

Step-4 finite oracle/excess-risk verification is stored in
`step4_oracle_risk_checks.json`. It checks the written pointwise/global bounds,
the exact excess-risk decomposition, hard-selector bound, and bank-enlargement
monotonicity on randomly generated finite conditional-risk problems. It is a
mathematical sanity check, not benchmark evidence.
