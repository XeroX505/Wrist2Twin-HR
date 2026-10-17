import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hybrid_parts_common import load_info, load_subject, regression_metrics
from models import EKFModel


X0_BY_INTENSITY = {
    0: np.array([100.0, 41.5, 0.145, 0.61, 0.08], dtype=float),
    1: np.array([107.5, 38.0, 0.12, 0.66, 0.275], dtype=float),
    2: np.array([115.0, 34.0, 0.095, 0.70, 0.55], dtype=float),
}


def ensure_ekf_columns(df):
    out = df.copy()
    for sensor in [5, 6, 7, 8, 9, 10]:
        for prefix in ["velocity", "acceleration", "gyration"]:
            col = f"{prefix}_DOT{sensor}"
            if col not in out.columns:
                out[col] = 0.0
    return out


def run_ekf_vo2(data_root, participant_id, part1_dir, output_dir, constant_hr=70.0):
    info = load_info(data_root)
    df = load_subject(data_root, participant_id)
    part1 = pd.read_csv(os.path.join(part1_dir, "part1_intensity_mu_timeseries.csv"))
    intensity = part1["intensity_pred"].to_numpy(dtype=int)
    mu = part1["mu"].to_numpy(dtype=float)
    ekf = segmented_ekf(df, info.loc[participant_id], intensity, mu, constant_hr)
    os.makedirs(output_dir, exist_ok=True)
    ekf.to_csv(os.path.join(output_dir, "part2_ekf_outputs.csv"), index=False)
    metrics = regression_metrics(df["vo2"].to_numpy(), ekf["ekf_vo2_ml_s"].to_numpy())
    metrics.update(
        {
            "participant_id": participant_id,
            "ekf_vo2_min": float(ekf["ekf_vo2_ml_s"].min()),
            "ekf_vo2_mean": float(ekf["ekf_vo2_ml_s"].mean()),
            "ekf_vo2_max": float(ekf["ekf_vo2_ml_s"].max()),
            "true_vo2_min": float(df["vo2"].min()),
            "true_vo2_mean": float(df["vo2"].mean()),
            "true_vo2_max": float(df["vo2"].max()),
        }
    )
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "part2_ekf_vo2_summary.csv"), index=False)
    plot_ekf_vo2(df, ekf, os.path.join(output_dir, "part2_ekf_vo2_full_dataset.png"))
    return ekf, metrics


def segmented_ekf(df, info_row, intensity, mu, constant_hr):
    full = ensure_ekf_columns(df)
    mass = float(info_row["Weight (kg)"])
    sex = str(info_row["Sex"]).lower()
    mass_torso = mass * (0.6664 if sex.startswith("m") else 0.6314)
    mass_leg = mass * (0.1668 if sex.startswith("m") else 0.1843)
    ke = 0.5 * (
        mass_torso * full["velocity_DOT7"].astype(float).to_numpy() ** 2
        + mass_leg * full["velocity_DOT8"].astype(float).to_numpy() ** 2
        + mass_leg * full["velocity_DOT9"].astype(float).to_numpy() ** 2
    )
    mp_o2 = ke / 1000.0 / np.clip(mu, 0.005, 0.2) / 19.6
    mp_co2 = 0.8 * mp_o2
    outputs = []
    starts = np.r_[0, np.where(intensity[1:] != intensity[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(df)]
    for start, end in zip(starts, ends):
        seg = full.iloc[start:end].copy().reset_index(drop=True)
        seg["t"] = np.arange(len(seg))
        seg["hr"] = constant_hr
        params = {
            "mass": mass,
            "height": float(info_row["Height (cm)"]),
            "percentage_muscle": float(info_row["Muscle (%)"]),
            "sex": sex,
            "const_GPO2": 2.4666666666666667,
            "const_GPCO2": 0.031666666666666666,
            "const_mu": 0.06,
            "approach_nr": 2,
            "mu_correction_factors": [1, 1, 1, 1],
        }
        model = EKFModel(params, seg)
        model.controller_mode = "delayed"
        model.RQ_mode = "constant"
        x0 = X0_BY_INTENSITY[int(intensity[start])].copy()
        x_est, p_est, vo2co2 = model.run_ekf(
            x0,
            np.eye(5) * 0.01,
            seg["hr"].to_numpy(),
            [mp_o2[start:end], mp_co2[start:end]],
            seg["t"].to_numpy(),
            np.eye(5) * 0.01,
            np.eye(2) * 0.005,
        )
        vo2_l_min = np.asarray(vo2co2)[:, 0] * 60.0
        # Data VO2 unit is ml/s. EKF VO2 is L/min, converted to ml/s and
        # added to measured resting metabolic rate from the storage file.
        vo2_ml_s = float(info_row["RMR VO2"]) + vo2_l_min * 1000.0 / 60.0
        outputs.append(
            pd.DataFrame(
                {
                    "ekf_vo2_ml_s": vo2_ml_s,
                    "ekf_vo2_l_min": vo2_l_min,
                    "pa_o2": x_est[:, 0],
                    "pa_co2": x_est[:, 1],
                    "cv_o2": x_est[:, 2],
                    "cv_co2": x_est[:, 3],
                    "vta": x_est[:, 4],
                    "intensity_pred": intensity[start:end],
                    "mu": mu[start:end],
                    "mp_o2_obs": mp_o2[start:end] * 60.0,
                }
            )
        )
    return pd.concat(outputs, ignore_index=True)


def plot_ekf_vo2(df, ekf, path):
    plt.figure(figsize=(14, 5))
    plt.plot(df["t"], df["vo2"], color="black", linewidth=1.2, label="True VO2")
    plt.plot(df["t"], ekf["ekf_vo2_ml_s"], color="#1f77b4", linewidth=1.1, label="EKF VO2")
    last = None
    for t, act in zip(df["t"], df["activity"]):
        if act != last:
            plt.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
            last = act
    plt.xlabel("Time (s)")
    plt.ylabel("VO2 (ml/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
