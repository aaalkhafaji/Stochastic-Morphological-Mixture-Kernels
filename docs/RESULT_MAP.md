# Map from manuscript evidence to files

| Manuscript evidence | Frozen files | Generator/check |
|---|---|---|
| Main dataset metric tables | `results/summary.csv`, `*_test_metrics.csv.gz` | `summarize.py` |
| Paired control comparisons | `results/paired_comparisons.csv` | `summarize.py` |
| Primary-comparison figure | `figures/primary_comparison.*` | `summarize.py` |
| Noise/clean-input diagnostics | `results/condition_metrics.csv`, `figures/noise_diagnostics.*` | `summarize.py` |
| Selected qualitative examples | `results/qualitative_case_ids.json`, `figures/qualitative_cases.*` | `summarize.py` |
| Full fixed-bank results | `results/all_candidate_metrics.csv`, `figures/candidate_bank.*` | `summarize.py` |
| Gradient variance | `results/gradient_variance.csv`, `figures/gradient_variance.*` | `summarize.py` |
| CPU runtime table | `results/runtime.csv` | `verify_release.py`, then `summarize.py` |
| Output-space exploratory table | `results/quotient_summary.csv`, `*_quotient_metrics.csv.gz` | `validate_kernel_quotient.py` |
| Topology sharpness figure | `figures/topology_sharpness.*` | `plot_topology_sharpness.py` |
| Original finite checks | `results/mathematical_checks.json` | `audit_math.py` |
| Expanded graph/entropy/transport checks | `results/kernel_theory_checks.json` | `audit_kernel_theory.py` |
| Step-4 oracle/excess-risk checks | `results/step4_oracle_risk_checks.json` | `audit_oracle_risk.py` |
| Step-5 Oxford higher-resolution validation | `results/step5_oxford_pet/*` | `run_oxford_pet_step5.py` / GitHub Actions |
| Step-6 DAVIS genuine-error validation | `results/step6_davis2016/*` | `run_davis_step6.py` / GitHub Actions |
| Split identity and all saved model replays | `results/release_verification.json` | `verify_release.py` |
| Written proof/derivation inventory | `docs/proof_inventory.json` | Complete proofs in the main manuscript |

All generators live in `code/`. There are six baseline figures, each with a PDF and PNG. Steps 5 and 6 add independent Oxford and DAVIS result tables plus machine-readable summaries rather than new figures.
Generated manuscript tables are retained in `results/generated_results.tex` and
`results/generated_quotient.tex`. Full mathematical proofs remain in the main
manuscript; the numerical-check files support rather than replace them.

## Step 7 structured-error robustness

- Protocol and definitions: `results/step7_structured/STEP7_PROTOCOL_FROZEN.json`, `docs/STRUCTURED_ERROR_STEP7.md`.
- Aggregate structured-error results: `results/step7_structured/step7_summary.csv`.
- Per-error-family results: `results/step7_structured/step7_condition_summary.csv`.
- Clean-mask-clustered intervals: `results/step7_structured/step7_clustered_ci.csv`.
- Four predeclared loss contrasts: `results/step7_structured/step7_paired_comparisons.json`.
- Adverse-condition/image accounting: `results/step7_structured/step7_adverse_audit.json`.
- Natural component-dropout diagnostic: `results/step7_structured/step7_component_dropout.csv`.
- Frozen input/protocol metadata: `results/step7_structured/step7_provenance.json`.

- Step-7 generated manuscript table: `results/step7_structured/generated_step7_structured.tex`; supplement table: `results/generated_step7_supp.tex`.

## Step 8 modern/fair baselines

- Aggregate IID/OOD summaries: `results/step8_baselines/step8_summary.csv`.
- Clean-mask-clustered intervals: `results/step8_baselines/step8_clustered_ci.csv`.
- Eight predeclared family-wise loss contrasts: `results/step8_baselines/step8_paired_comparisons.json`.
- SoftMorph public-code equivalence check: `results/step8_baselines/softmorph_equivalence.json`.
- Validation-only selections and upstream commit: `results/step8_baselines/step8_selection.json`.
- Condition-aware fixed selections: `results/step8_baselines/step8_condition_aware_selection.json`.
- Provenance and no-retuning declaration: `results/step8_baselines/step8_provenance.json`.
- Independent completion audit: `results/step8_baselines/STEP8_COMPLETION_AUDIT.json`.
- Generated manuscript table: `results/generated_step8_baselines.tex`.

## Step 9 theorem-targeted finite checks

- `results/step9_theorem_stress/step9_summary.json`: gate-level summary.
- `step9_replication_cardinality.csv`: sharp replication constants and `m` versus `M` cardinality.
- `step9_topology_hamming1_3x3.csv`, `step9_topology_random_multiflip.csv`: exact/random topology-bound checks.
- `step9_identifiability_gram.csv`, `step9_identifiability_prefix.csv`: incidence-rank diagnostics.
- `step9_bayes_pairwise.csv`: equal-prior Bayes error versus row-law TV for all unordered pairs.
- `step9_transport_iterates.csv`: exact contraction coefficients through eight compositions.
- `figures/step9_replication_bound.*`, `step9_bayes_tv_error.*`, `step9_transport_contraction.*`: visual diagnostics.

## Step 10 (execution-ready; no results claimed yet)
- Workflow: `.github/workflows/step10-generalization-calibration.yml`
- Runner: `code/run_step10_generalization.py`
- Frozen design: `docs/STEP10_GENERALIZATION_CALIBRATION.md`
- Planned artifact: `step10-generalization-calibration-results`
- Scope: Oxford-to-DAVIS zero-refit transfer, direct-output Gibbs calibration/selective risk, and computational scaling.
