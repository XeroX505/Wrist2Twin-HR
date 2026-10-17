# Wrist2Twin-HR

Research implementation of a modular wrist-motion-to-heart-rate pipeline.
The repository contains the four module interfaces, the established numerical
implementation, pretrained wrist models, and one-command orchestration.
Participant recordings and manuscript files are not distributed.

## Modules

| Interface | Purpose | Learned artifacts |
| --- | --- | --- |
| `module1a_intensity.py` | Wrist features to raw/posterior/temporally smoothed intensity | Point ensemble, window classifier, scalers, transition matrix and smoothing configuration |
| `module1b_metabolic_proxy.py` | Wrist features and predicted intensity to metabolic observation | Extra Trees and histogram-gradient-boosting regressors, scaler and ordered feature schema |
| `module2_pm_ekf.py` | Metabolic observation and predicted intensity to oxygen uptake | Physiological EKF; no pretrained neural weights |
| `module3_pmbnn.py` | Personalized oxygen-uptake-to-HR calibration and testing | Subject-specific neural fit and six bounded physiological parameters |

`legacy/` contains the exact dependency closure of the established research code.
Only standalone workstation launch blocks were removed and the WORKSPACE default
was made portable. `source_provenance.json` records original and packaged hashes.
Only the four interfaces and `run_pipeline.py` are supported release entry points;
other historical functions in `legacy/` are retained to preserve imports.

## Installation

The supplied artifacts were validated with **Python 3.9.13** and
**scikit-learn 1.0.2**. Use an isolated environment for this legacy research stack:

```bash
conda env create -f environment.yml
conda activate wrist2twin
python run_pipeline.py --action validate
python -m unittest discover -s tests -v
```

The shipped artifacts use **scikit-learn 1.0.2**. Load them in that version.
See [docs/validation.md](docs/validation.md) for the tests actually performed.
Use only trusted joblib/pickle files: deserialization can execute code.
The manifest checks file integrity, not the trustworthiness of an unknown file.
Pretrained artifacts are stored with **Git LFS** because the proxy ensembles are
large. Install Git LFS before cloning, or run `git lfs install` and `git lfs pull`
inside an existing clone. GitHub's ZIP download may not include the model payloads.
The loader's SHA256 check detects missing LFS payloads before inference.
The established implementation runs on CPU. The shell launcher is suitable for
a scheduled HPC CPU job; it does not request a partition or a GPU implicitly.

## One-command use

Copy `config.example.json` to `config.local.json`, then edit only the paths and
the requested dataset/sensor configuration. Absolute Linux and Windows paths
are supported; relative paths are resolved from this repository directory.
`config.both_sensors.example.json` supplies the two-mode/four-dataset layout
for reproducing the broader sensor comparison. Remove unwanted datasets before running.

```bash
bash run_pipeline.sh config.local.json
# Equivalent command on any platform:
python run_pipeline.py --config config.local.json --action run
```

Windows PowerShell:

```powershell
.\run_pipeline.ps1 -Config config.local.json
```

The default is ACC+GYRO, 5000 epochs and the primary activity-wise **80:20**
calibration/test design. The optional sensitivity scenarios are `50` and `300`.
`300` denotes 300 training samples at 1 Hz, allocated proportionally across
activities by the established split routine. Very short recordings or many tiny
activity groups may not permit this allocation. Inspect `n_train`, `n_test` and
the saved `original_index` before using such recordings.

Both model configurations are supplied:

* `dot_wrist_with_gyr`: velocity, acceleration and gyroscope features.
* `e4_compatible_no_gyr`: velocity and acceleration features; no gyroscope.

For ACC-only external experiments, set `modes` to `["e4_compatible_no_gyr"]`,
and use PPG_DaLiA and Stress_Aerobic paths with that mode in `datasets`.
For ACC+GYRO, use BITS2_ADL_no_existing_health_issues and PAMAP2. Dataset names
are labels for outputs, not a request to download data. Inclusion/exclusion of
particular recordings is a dataset-preparation decision; the runner processes
all eligible files in each configured folder except explicit `exclude_records`
IDs (case-insensitive). The examples exclude Stress S11 and PAMAP2 subject109-protocol,
as in the primary comparison. Other-health-conditions BITS recordings are not
included in the example configurations.

## Two distinct evaluation pathways

**Development cohort reproduction.** Set `development_enabled` to `true` and
provide the development Data folder. Module 1A is trained with leave-one-subject-out
(LOSO) splitting. The saved development pathway feeds *measured pelvis and bilateral
thigh kinetic observations*, together with LOSO wrist intensity, into Module 2.
Module 3 is calibrated separately for each participant. It is not an end-to-end
held-out wrist-regressor evaluation of Module 1B. All-participant pretrained
artifacts must not be used to report LOSO performance on their training subjects.

**Independent wrist-only pathway.** Set `development_enabled` to `false`.
The provided Module 1A/1B artifacts are frozen models fitted using all eight
development participants. Independent wrist recordings pass through both
modules, then Module 2 and personalized Module 3 calibration. No development
recordings are needed for this inference/calibration pathway. Population-average
demographic defaults are embedded in the Module 1B artifact; available independent
demographics override the supported fields.

## Important numerical definitions

* Module 1A: point/window posterior weights 0.55/0.45; transition regularization
  70; Viterbi decoding, centred 31-s rolling mode and 20-s minimum segments.
* Module 1B: pelvis/left-thigh/right-thigh velocity teacher, sex-specific segment
  mass fractions, oxygen energy equivalent 19.6, log1p regression, mean of the two
  inverse-transformed regressors, clipping to [0, 0.12] L/s, 8-s EMA.
* Movement-efficiency coefficient: 0.012 for moderate-high, 0.060 otherwise.
  `MODHIGH_PROXY_SCALE = 1.0`; there is no manual high-intensity amplitude boost.
* Module 2 uses a fixed HR surrogate of 70 bpm, not the participant's measured HR,
  and adds the resting oxygen-uptake value to the EKF exercise component.
* Module 3: MSE + 10000 x differential-equation loss, RMSprop, neural HR fit as
  primary output, six bounded lambdas logged every 25 epochs and at the final epoch.
* Metabolic observations are L/s; final oxygen uptake is ml/s; HR is bpm.

This is the established **offline** protocol. Whole-record robust scaling,
window posteriors and temporal decoding use recording context. VO2-derived
features are computed on the full record before personalized splitting.
The split groups identical activity labels in first-occurrence order, keeping
within-label sample order; repeated bouts with the same label are grouped together.
Train/test rows are concatenated without interpolation or transition samples.
Physics differences are computed only within training activity groups.

The secondary white-box HR trajectory is initialized from measured HR at the
start of each test activity group, matching the original implementation. It is
an initialized diagnostic reconstruction, not an independent uninitialized HR
forecast. The primary neural HR test metric does not use those test HR values.

## Inputs and outputs

See [docs/data_format.md](docs/data_format.md) for processed input channels,
demographic column names and the development folder layout.

Each independent recording produces:

* `module1_outputs.csv`: raw/smoothed intensity, posteriors and wrist proxy.
* `module2_outputs.csv`: EKF oxygen uptake and physiological state trajectories.
* `module3/part3_predictions_per_activity_80.csv`: held-out HR, neural HR,
  diagnostic white-box HR, EKF VO2 and original row indices.
* Training rows, final lambdas, epoch history, metrics and HR plots.

`subject_metrics.csv` collects four primary HR metrics per recording and scenario.
`group_metrics.csv` gives recording count and median/min/max for RMSE, MAE, R2 and r.
Multiple recordings from one person remain separate rows: `n_recordings` is not
necessarily the number of unique participants. `run_config.json` records settings.
Use a fresh output directory for each run to avoid mixing experimental results.

## Retraining

```bash
python run_pipeline.py --config config.local.json --action train
```

This recomputes LOSO intensity, fits the all-participant Module 1A models and
trains/saves Module 1B using the resulting out-of-fold intensity predictions.
It is substantially slower than using the frozen artifacts. No measured VO2 or
measured HR is used as a Module 1A/1B fitting target. Some legacy data-building
helpers read VO2 for unused diagnostic fields; it is not a model input or target.

The repository's Module 1A weights are losslessly repackaged existing models.
Module 1B was previously retrained inside Module 2 without being saved; the
distributed Module 1B artifacts were newly fitted with the same model settings,
development data and saved LOSO intensities. They are not recovered historical
serialized weights. See [pretrained/README.md](pretrained/README.md).
