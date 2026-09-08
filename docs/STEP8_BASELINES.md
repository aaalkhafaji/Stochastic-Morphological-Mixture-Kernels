# Step 8 — Modern and fair baselines (execution-ready)

## Scientific question
Is the reported gain specific to the proposed stochastic/output-space construction, or can it be matched by strong fixed, learned-local, or contemporary differentiable-morphology post-processing under the same information budget?

## Frozen comparison protocol
Step 8 is frozen before Step-8 outcomes are inspected.

### Training/tuning information
All trainable/tunable Step-8 baselines use only the successful Step-5 Oxford-IIIT Pet train/validation masks at 128x128. They receive **binary masks only**—no RGB images, class labels, test targets, or corruption labels (except the explicitly named condition-aware fixed control).

### Primary held-out tests
1. **Oxford IID:** the full Step-5 held-out test, 3,669 clean masks × 6 predeclared IID corruptions = 22,014 cases.
2. **Oxford structured OOD:** the Step-7 nine-family test, 3,669 masks × 9 structured corruptions = 33,021 cases. No Step-8 baseline is retuned on structured errors.

## Baselines
### Existing controls
- Raw input.
- Validation-best fixed 31-action morphology.
- Action-coordinate selector.
- Direct-output selector.
- Oracle diagnostic (not a deployable comparator).

### New strong same-bank control
**Condition-aware fixed morphology (IID only).** For each of the six known IID corruption labels, select one of the same 31 actions using validation loss only. This control is intentionally advantaged because it receives the corruption-condition label; the proposed learned selectors do not.

### Contemporary differentiable morphology
**SoftMorph2 validation-selected post-processing.** Step 8 retrieves the public SoftMorph2 2D code at runtime and records its Git commit. The authors' source is not redistributed. A local vectorized product-logic implementation is required to agree with the downloaded public implementation to max absolute discrepancy <=2e-6 before the benchmark is allowed to run.

The predeclared SoftMorph candidate family is small and fixed:
- erosion, dilation, opening, closing;
- 4- or 8-connectivity;
- 1 or 2 iterations;
- direct binary input or a fixed Gaussian relaxation sigma=0.75;
- threshold 0.5;
- product fuzzy logic.

One global configuration is selected on Oxford validation composite loss and frozen for IID test and Step-7 structured OOD.

### Learned local binary morphology
**Regularized empirical 3x3 W-operator.** The 512 possible zero-padded 3x3 binary neighborhoods are fitted by empirical foreground frequency on the Step-5 training pairs. Rare/unseen patterns use an identity-centered regularization prior. Alpha in {1,10,100} and threshold in {0.4,0.5,0.6} are selected on validation only. This is a direct translation-invariant binary image-operator control, aligned with the W-operator class emphasized by modern discrete morphological neural networks, but is not labeled as an implementation of the DMNN lattice-descent algorithm.

## Statistics
All method/condition repetitions are averaged within clean held-out mask before inference. Step 8 reports 5,000 clean-mask-clustered bootstrap intervals. Eight predeclared pairwise loss comparisons are family-wise corrected:
- Direct vs SoftMorph, Action vs SoftMorph, Direct vs W3, Action vs W3 on IID;
- the same four comparisons on structured OOD.

## Gate
Step 8 passes as a **comparison-completeness** step if:
1. public SoftMorph2 is retrieved and provenance is recorded;
2. the independent product-logic implementation agrees with the public code within 2e-6;
3. all new baseline choices are validation-only;
4. IID and structured OOD tests execute fully;
5. all eight predeclared comparisons and adverse results are reported, regardless of direction.

A win by the proposed method is *not* required for the gate; unfavorable modern-baseline results must remain in the paper.
