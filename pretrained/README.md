# Frozen wrist models

Each sensor mode has two compressed joblib files:

* `<mode>_module1a.joblib`: existing all-eight-participant intensity ensemble,
  fitted scalers, temporal transition matrix and selected smoothing parameters.
* `<mode>_module1b.joblib`: newly exported fitted metabolic-proxy regressors,
  scaler, ordered feature names and cohort-average demographic fallback values.

Modes: `dot_wrist_with_gyr` and `e4_compatible_no_gyr`.
Module 1B training uses the saved development LOSO intensity predictions and the
pelvis/bilateral-thigh kinetic teacher, with the unchanged established settings.
It is newly fitted because the original workflow did not serialize Module 1B.
No independent-dataset records are used to fit these weights.
Joblib artifacts are stored using Git LFS. Run `git lfs pull` after cloning
if the local files are small text pointers instead of the actual weights.

The files contain sklearn 1.0.2 estimators and require the documented environment.
Load only trusted files. `manifest.json` contains the SHA256 digest and size for
each artifact. Hashes are checked before inference. Compression is lossless and
does not quantize model parameters. No participant recordings are included.

These final models are for independent-data use. They are not held-out predictions
for the eight development participants used during fitting.
