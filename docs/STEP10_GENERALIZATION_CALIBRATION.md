# Step 10 — Generalization, calibration, selective risk, and scaling

## Status
**EXECUTION-READY.** Protocol and code are frozen before Step-10 outcomes are inspected.

## Primary questions
1. **Cross-domain transfer:** train only on the frozen Oxford-IIIT Pet Step-5 IID corruption protocol, then evaluate without refitting on official held-out DAVIS 2016 genuine segmentation errors.
2. **Calibration:** evaluate the direct-output Gibbs law as an uncertainty distribution on Oxford IID and on cross-domain DAVIS.
3. **Selective risk:** test whether high Gibbs confidence identifies lower-regret cases at coverages 100%, 80%, 60%, 40%, and 20%.
4. **Computational scaling:** measure candidate generation plus representation-invariant direct-output scoring over image sizes 64, 128, 256, and 480 and nested banks of 7, 15, 31, and 63 actions.

## Frozen training and transfer protocol
The only fitted models are the five Step-5 Oxford ExtraTrees models, reconstructed from the frozen Step-5 training cache with the already-selected leaf sizes and per-seed temperatures. No DAVIS frame, target, sequence, or source method is used for fitting, temperature choice, or calibration adjustment.

The DAVIS target is the same official held-out validation partition and the same six unsupervised source methods used in Step 6: NLC, FST, SAL, TRC, MSG, and CVOS. The Step-10 question differs from Step 6: Step 6 fitted on DAVIS training sequences, whereas Step 10 deliberately transfers the Oxford-trained model unchanged.

## Calibration definitions
For each case, the five frozen direct-output Gibbs laws are averaged on the same distinct-output support. Let `q` be this ensemble law and let `O*` be the set of candidate outputs attaining the smallest realized composite loss.

Reported quantities include:
- confidence `max_z q(z)`;
- top-choice correctness `1[argmax q in O*]`;
- probability mass on the optimal set `q(O*)`;
- optimal-set NLL `-log q(O*)`;
- Brier score against a uniform target on tied optimal outputs;
- normalized entropy;
- MAP regret and Gibbs expected regret above the candidate-bank oracle.

Reliability uses ten equal-width confidence bins and also reports a ten-bin adaptive ECE. No temperature is recalibrated on test data.

## Selective risk
Cases are sorted by ensemble confidence. At predeclared coverages 1.0, 0.8, 0.6, 0.4, and 0.2, Step 10 reports top-choice accuracy, MAP loss, MAP regret, and Gibbs expected regret. Non-monotone behavior is retained if observed.

## Scaling protocol
Scaling is a computational benchmark only. It does not claim predictive accuracy for the expanded 63-action bank. The 31-action manuscript bank is nested inside a deterministic 63-action morphology bank. For each image size and bank size, the benchmark records median candidate-generation time, direct-output pair-feature/prediction time, total time, materialized candidate-bank memory, and empirical log-log slopes.

## Gate
Step 10 passes as an evidence-completeness step if:
- the full Oxford-to-DAVIS transfer executes without target leakage;
- calibration/reliability/selective-risk tables are produced for IID and DAVIS transfer;
- all four image sizes and four bank sizes execute;
- adverse calibration, transfer, or scaling behavior is retained rather than filtered.

No threshold requires the transferred model to outperform the DAVIS-trained Step-6 model; such a requirement would encourage cherry-picking and is therefore explicitly excluded.
