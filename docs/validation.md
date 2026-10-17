# Release validation - 17 October 2026

## Checks completed

1. All 13 vendored dependency files parsed; packaged hashes were checked against
   `source_provenance.json`. Research functions retain their original definitions;
   workstation launch blocks and the WORKSPACE default were the packaging changes.
2. Four automated tests passed: artifact hashing, nonoverlapping exact allocation
   for 80/50/300 on a 1000-sample fixture, missing-channel rejection, and a two-epoch
   PMBNN run saving test HR and six-parameter history.
3. Both pretrained Module 1A exports produced exactly equal raw predictions,
   smoothed predictions and posteriors to the original pickle objects on the same
   120-sample local input. Compression changed storage only.
4. Both sensor modes passed a four-module integration smoke check on 120 local
   samples: Module 1A -> Module 1B -> PM-EKF -> two-epoch personalized PMBNN.
   Proxy/VO2 outputs and primary HR metrics were finite. Input recordings and
   smoke-test outputs were not committed.
5. All four model artifacts have SHA256 hashes. Text was scanned for personal
   workstation paths and credentials; participant files were excluded.
6. The actual `run_pipeline.py` command completed with a temporary 120-sample
   ACC+GYRO recording and two calibration epochs. Subject-level outputs and the
   four metrics' grouped median/min/max columns were verified.

These are software and serialization checks. Full LOSO training and the complete
5000-epoch cohort benchmark were not rerun for this packaging task. No new
performance or generalization claim follows from the short smoke checks.

## Actual validation environment

Windows, Python 3.9.13; NumPy 1.26.4; pandas 1.5.3; scikit-learn 1.0.2;
SciPy 1.10.1; joblib 1.1.0; PyTorch 2.1.0+cpu; matplotlib 3.5.2;
SymPy 1.10.1; openpyxl 3.0.10; threadpoolctl 3.6.0.

The host's old threadpoolctl 2.2.0 had a library-discovery incompatibility.
The checks and proxy fitting used an isolated 3.6.0 helper without modifying
the host environment. The supplied requirement pins include that working version.
A fresh Conda installation and Linux execution were not tested during packaging.

## Pretrained-model provenance

The existing intensity classifiers were trained on development participants
1, 3, 5, 6, 7, 8, 9 and 10. Each frozen Module 1B ensemble was newly fitted with
the same eight participants, saved LOSO intensity results and established model
settings. This adds persistence to the former runtime-only proxy fitting step.
Independent recordings were used only for software smoke tests, not model fitting.
The resting-value defaults embedded in the artifacts are cohort aggregates.
