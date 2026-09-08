# Step 8 — Modern and fair baselines (COMPLETE)

## Status
**DONE / PASS — comparison-completeness gate satisfied.**

The frozen GitHub Actions artifact `step8-modern-baseline-results.zip` has SHA-256
`1fdcddb32406ae2c097439d7d1f84e690715b08bbe6812da65b027b8d0215840`. Its internal `SHA256SUMS` verifies with zero mismatches. Independent
recomputation from all raw case rows agrees with the archived aggregate summaries to
within 3.88e-7 (the largest discrepancy is only CSV rounding on very large Betti-error
means); all eight archived paired point estimates agree to within 2.89e-12.

## Frozen scientific question
Is the reported behavior specific to the stochastic/output-space construction, or can it
be matched by strong fixed, learned-local, or contemporary differentiable-morphology
post-processing under the same binary-mask information budget?

## Baselines and fairness
All trainable/tunable Step-8 baselines use only the Step-5 Oxford-IIIT Pet training and
validation masks at 128x128. No RGB images or held-out targets are supplied. The
structured-OOD benchmark is never used for tuning.

- **Condition-aware fixed morphology (IID only):** one action from the same 31-action
  bank is selected separately for each known IID corruption condition on validation
  data. It is intentionally advantaged because it receives the corruption label.
- **SoftMorph2:** public upstream source is retrieved at runtime, with commit
  `19729d10cc8d012e13154a470f07a094d683c2d0` frozen in provenance. The independent
  vectorized product-logic implementation agrees with the public 2D code with maximum
  absolute discrepancy 0.0 (required tolerance 2e-6).
- **Empirical 3x3 W-operator:** a regularized translation-invariant binary operator over
  all 512 local 3x3 neighborhoods, with regularization and threshold selected on
  validation only. This is a W-operator control aligned with the operator class used by
  discrete morphological neural networks; it is not represented as a reimplementation
  of the full DMNN lattice-descent architecture.

SoftMorph validation selects closing, 4-connectivity, one iteration, sigma=0.75,
threshold=0.5 (validation loss 0.099214). The W-operator selects alpha=1 and threshold
0.6 (validation loss 0.161106).

## IID Oxford results
Across 22,014 held-out IID cases:
- condition-aware fixed morphology: loss **0.01574**;
- action-coordinate selector: **0.01619**;
- validation-best fixed morphology: **0.01639**;
- direct-output selector: **0.02006**;
- SoftMorph2: **0.09752**;
- empirical W-operator: **0.15709**.

The condition-aware fixed control is descriptively best, but it receives the true
corruption-condition label. The eight predeclared family-wise comparisons show that both
learned selectors have substantially lower IID loss than SoftMorph2 and the empirical
W-operator:
- direct minus SoftMorph2: **-0.07746**, family-wise 95% CI **[-0.07904, -0.07596]**;
- action minus SoftMorph2: **-0.08132**, CI **[-0.08272, -0.08001]**;
- direct minus W-operator: **-0.13703**, CI **[-0.13882, -0.13511]**;
- action minus W-operator: **-0.14089**, CI **[-0.14248, -0.13933]**.

These results do not overturn the Step-5 finding that the current direct-output learner
is inferior to the strongest same-bank IID controls.

## Structured-OOD results
Across the frozen 33,021 Step-7 structured cases, with no Step-8 retuning:
- direct-output selector: loss **0.10104**;
- validation-best fixed morphology: **0.15108**;
- SoftMorph2: **0.15195**;
- action-coordinate selector: **0.15385**;
- raw input: **0.15870**;
- empirical W-operator: **0.16448**.

The predeclared family-wise contrasts are:
- direct minus SoftMorph2: **-0.05091**, CI **[-0.05217, -0.04965]**;
- direct minus W-operator: **-0.06343**, CI **[-0.06474, -0.06218]**;
- action minus SoftMorph2: **+0.00190**, CI **[+0.00118, +0.00258]**;
- action minus W-operator: **-0.01062**, CI **[-0.01120, -0.01008]**.

Thus SoftMorph2 significantly outperforms the action-coordinate selector under structured
shift, while the representation-invariant direct-output selector significantly
outperforms both external baselines. Its OOD advantage also appears in Dice (0.96564),
IoU (0.93556), and exact resized-target topology agreement (0.52874), versus SoftMorph2
(0.96251, 0.92985, 0.38385).

## Interpretation and retained limitations
Step 8 supplies a fair modern/differentiable morphology comparison without granting the
external controls RGB information or structured-OOD tuning. It supports two bounded
claims: (i) the Oxford IID performance of the same-bank selectors is not reproduced by
these two external morphology controls under the matched post-processing protocol; and
(ii) the Step-7 direct-output OOD result remains strong against SoftMorph2 and a learned
local W-operator.

It does **not** establish universal superiority over SoftMorph as an end-to-end trainable
component, nor over full DMNN/BiMoNN architectures. SoftMorph2 is evaluated here as a
validation-selected post-processing operator on binary masks, and the W-operator control
does not reproduce the full DMNN optimization/architecture. The condition-aware fixed
control remains slightly better than the learned selectors on IID data and is
intentionally advantaged by condition labels. All unfavorable directions are retained.

## Gate
The predeclared comparison-completeness gate passes:
1. public SoftMorph2 provenance is frozen;
2. independent/public-code equivalence passes at discrepancy 0.0;
3. all selections are validation-only;
4. both IID and structured-OOD evaluations complete;
5. all eight family-wise contrasts are reported, including the significant OOD win of
   SoftMorph2 over action-coordinate selection.
