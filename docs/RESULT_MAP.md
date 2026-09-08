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
