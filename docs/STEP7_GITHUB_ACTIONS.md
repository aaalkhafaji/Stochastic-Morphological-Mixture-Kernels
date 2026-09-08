# Step 7 GitHub Actions execution

This repository state is execution-ready for the predeclared Step 7 structured-error robustness campaign.

## Frozen input

The workflow downloads the successful Step 5 Oxford-IIIT Pet GitHub Actions artifact by artifact ID `10035125626` and verifies the exact artifact SHA-256:

`889e7df9af279974d46595444bf768c627602d188ca8c93d0f65b30739f16e96`

The full Step 7 runner refuses a mismatched artifact.

## Run

1. Push this repository update to the default branch.
2. Confirm the `Package integrity` workflow is green.
3. Open **Actions** -> **Step 7 Structured-error robustness**.
4. Choose **Run workflow**, branch `main`.
5. Wait for the `structured-errors` job to finish.
6. Download the artifact **step7-structured-error-results** and preserve it unchanged for audit/integration.

## Predeclared full campaign

The full run uses all 3,669 held-out Oxford-IIIT Pet test masks at 128x128, nine structured-error families, five frozen estimator seeds, and 5,000 clean-mask-clustered bootstrap resamples. The models are trained only on the frozen Step 5 IID deletion/addition training distribution; Step 7 structured cases are not used for fitting or tuning.

The workflow first runs an offline corruption smoke test and a fast real-data compatibility preflight on the frozen Step 5 artifact. It then executes the single authoritative full fit/evaluation and verifies every result file against the run-generated `SHA256SUMS` before artifact upload.
