# Step 9 — Theorem-targeted numerical stress tests (COMPLETE)

## Status
**DONE / PASS.** These checks target exact finite consequences of the theorems; they are not new empirical-performance benchmarks.

## 1. Replication sensitivity and cardinality
For duplicate counts r = 1, 2, 4, 8, 16, 32, 64, 128, 256, the sharp categorical TV constant from Proposition 5 is attained numerically to maximum residual **1.110e-16**. Across 4,500 additional random categorical distributions with random output partitions, the maximum positive violation is **0.0**. In the sharp two-output construction, the distinct-output cardinality remains M=2 while the action-label count reaches m=258; hence the uniform-output penalty tau log M is representation stable whereas the action-coordinate tau log m term grows with redundant labels.

## 2. Topology sensitivity
All **2304** unordered Hamming-one pairs on the full 3x3 binary state space were enumerated. The maximum observed one-flip |Δβ0| is **3**, and the explicit proof constructions achieve |Δβ0|=|Δβ1|=3 at d_H=1. A further **30000** random multi-flip (mask, k) checks on 5x5, 8x8, and 12x12 grids produce no violation of |Δβk|≤3 d_H.

## 3. Finite incidence-rank diagnostics
On the complete 3x3 state space (512 inputs), the paper's 31-action bank induces exactly **18** distinct deterministic maps and the full incidence matrix has rank **18**. Thus the 31 label weights are not globally identifiable on this finite grid. The rank deficit here is completely explained by exact map coincidences: after collapsing identical maps, the 18-map bank has full rank 18. Adding an exact duplicate label produces 32 action coordinates while the rank remains 18, numerically illustrating the theorem's distinction between operator labels and identifiable transition laws.

## 4. Bayesian reversal and residual ambiguity
For a fixed four-action lossy morphology kernel on all 512 3x3 masks with uniform prior, the Bayesian reverse recovers the prior marginal with maximum residual **2.168e-19**. Across all **130816** unordered row pairs, the equal-prior identity 0.5(1-TV) matches direct optimal classification error with residual **0.0**. Yet H H_mu^← differs from identity by as much as **0.9953**, and **130312** row pairs have overlapping supports, so marginal recovery is not samplewise inversion.

## 5. Transport and composition
On the full 2x2 state space, the fixed morphological kernel K=0.6 I + 0.4 O_(5x5) has exact Wasserstein coefficient κ(K)=**0.6**. Exact finite optimal transport gives κ(K^2)=**0.36** and κ(K^n)=0.6^n through n=8, with maximum residual **6.939e-17**. The topology iterate bound has no positive numerical violation. In contrast, deterministic action **O_square2** has κ=**4.0**, demonstrating that a morphological operator need not be contractive.

## Gate
All Step-9 gates pass exactly or to floating-point tolerance.
