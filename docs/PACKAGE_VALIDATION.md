# Preparation validation record

On 6 September 2026, the documented **analysis** reproduction level completed
successfully from a new copy of this package. It regenerated all summaries and
figures, replayed all 15 saved forests on the full retained test set, matched
fresh seed-101 refits on all three datasets, reran both mathematical-check suites,
and regenerated the exploratory output-space analysis and the four-page guide.

Every compared table agreed within 1e-12; exact recorded maximum differences
are in `results/submission_preparation_checks.json`. Expanded kernel-theory
checks agreed exactly. The original check suite's singular values differ by
roundoff below 1e-12, with the same rank and all checks passing. The frozen release
retains the original data, model, result, and figure bytes; new runtime measurements
remain in the validation work copy and do not silently replace the manuscript's
reported timings.

This preparation did not rerun complete candidate-bank extraction or every model
fit from scratch. That capability is supplied as the **full** reproduction level.
The original campaign's outputs are retained. No independent new benchmark or
external proof certification is claimed.


The subsequent LaTeX conversion retains the guide's text and numerical tables
in a standalone source file. Its compiled four-page PDF has been visually
checked. The archive now contains that PDF under `Supplementary_Guide.pdf`, the
editable `JVCIR_Supplementary_Guide.tex`, and LaTeX build instructions. The original
ReportLab generator remains available; the computation code, data, fitted models,
and numerical result records are unchanged by this format conversion.
