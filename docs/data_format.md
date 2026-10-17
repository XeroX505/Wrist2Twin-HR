# Processed data contract

The pretrained models expect the same processed **1-Hz** feature semantics as
training. Raw high-rate device acceleration cannot be passed in directly.
Velocity extraction, gravity handling and physical units must match the original
preprocessing; wrist devices with different preprocessing may have distribution
shift even when the column names match. No raw dataset download is performed.

## Independent datasets

The legacy enumerator reads files directly inside a dataset folder with suffix
`_E4_ACC_HR_1Hz.csv`. This suffix is historical and also applies to processed
ACC+GYRO files. A record identifier is the filename prefix before the first
underscore. It must be unique in the dataset folder. For multiple sessions, use
unique prefixes, with matching demographic identifiers.

Required CSV columns:

| Columns | Meaning and units |
| --- | --- |
| `HR_bpm` | Measured heart rate for personalized calibration and held-out evaluation |
| `vel_x_mps`, `vel_y_mps`, `vel_z_mps` | Processed triaxial wrist velocity, m/s |
| `acc_x_mps2`, `acc_y_mps2`, `acc_z_mps2` | Processed wrist acceleration, m/s2 |
| `gyr_x_rad_s`, `gyr_y_rad_s`, `gyr_z_rad_s` | Wrist angular velocity, rad/s; ACC+GYRO only |
| `activity`, `markers` or `marker` | Activity identifier; sparse markers are forward filled |
| `t` (optional) | Increasing timestamps of processed 1-Hz samples; otherwise row order is used |

Optional `total_velocity_mps` and `total_gyration_rad_s` override axis norms.
Numeric input must be finite; perform missing-value processing explicitly before
evaluation. The release runner rejects absent/nonfinite channels instead of
silently filling missing wrist sensors with zeros.
Some prepared PAMAP2 recordings contain time gaps after removal of unlabelled
intervals. The runner warns about gaps and retains the established row-wise
1-s computation, without filling the missing intervals. This reproduces the
existing workflow, not a continuous-time treatment of the gaps. Singular
`marker` is accepted for the prepared Stress_Aerobic data.

`subject_info_summary.csv` in the dataset folder may contain `record_id`,
`subject_id`, `participant`, or `id` (first matching column), plus `weight_kg`,
`height_cm`, and `sex` (male/female). Unavailable values use the saved development
defaults. Muscle percentage and resting VO2 retain those population defaults,
matching the original external demographic adapter. These assumptions should be
reported when interpreting results.

## Development cohort

```text
Data/
  Data Storage file.xlsx
  Participant 1/ramon_windowed_data.xlsx
  Participant 3/ramon_windowed_data.xlsx
  ...
```

The metadata workbook includes `Participant_ID`, `Weight (kg)`, `Height (cm)`,
`Muscle (%)`, `Sex` and `RMR VO2`. Recordings include `t`, `markers`, `hr`,
`vo2` (ml/s), `vco2`, and the processed IMU columns used in training.
Legacy identifiers map to anatomy as follows: DOT6 = wrist, DOT7 = pelvis,
DOT8 = left thigh, DOT9 = right thigh.

Wrist features use `velocity_DOT6`, `acceleration_DOT6`,
`linear_velocity_{x,y,z}_DOT6`, `filtered_{x,y,z}_DOT6`, and, with gyroscope,
`gyration_DOT6`, `filtered_gyr_{x,y,z}_DOT6`. Teacher construction requires
`velocity_DOT7`, `velocity_DOT8`, `velocity_DOT9`.

Development labels are the original protocol strings. Moderate: `mopping`,
`stacking shelves`, `treadmill 3km`, `treadmill 5km`. Moderate-high: `stairs`,
`cycling`. Other labels map to low in the established helper. Check activity
spelling before retraining; a new label is not automatically assigned correctly.

Data are deliberately excluded from Git. Do not commit metadata workbooks,
participant-level CSVs, raw files, manuscripts or output directories.
