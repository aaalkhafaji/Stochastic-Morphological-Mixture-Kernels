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


Step-5 Oxford-IIIT Pet higher-resolution evidence is stored under
`step5_oxford_pet/`. The compact tree retains protocol IDs, preprocessing audit,
selection metadata, per-condition and macro-breed summaries, clustered confidence
intervals, paired comparisons, generated LaTeX, and provenance. Large NPZ caches
and the compressed per-case metric table are excluded from Git history; their
full-artifact hashes are retained for the next Zenodo archival update.


Step-6 DAVIS 2016 genuine-segmentation-error evidence is stored under
`step6_davis2016/`. The directory preserves the exact successful GitHub Actions
result files, including provenance, protocol sequence/case tables, source-method
summaries, sequence-clustered confidence intervals, paired contrasts, resize
audit, and compressed per-case metrics. `GITHUB_ACTIONS_ARTIFACT.txt` records
the SHA-256 of the original downloaded artifact.

## Step 7 structured-error distribution shift

`step7_structured/` contains the frozen 33,021-case Oxford structured-error campaign: protocol/provenance, raw compressed case metrics, seed summaries, condition summaries, clean-mask-clustered intervals, four family-wise loss contrasts, adverse accounting, the component-dropout diagnostic, and generated LaTeX. No Step-7 structured case enters fitting or tuning.
