import os
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from hybrid_parts_common import load_info, load_subject, participant_ids
from models import EKFModel
from part2_ekf_vo2 import X0_BY_INTENSITY, ensure_ekf_columns


def regression_metrics(y, pred):
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    corr = np.nan
    if np.std(y) > 1e-8 and np.std(pred) > 1e-8:
        corr = float(np.corrcoef(y, pred)[0, 1])
    return {"r2": float(r2_score(y, pred)), "rmse": rmse, "mae": mae, "corr": corr}


def load_sensor_prediction(sensor_root, sensor_set, pid):
    path = os.path.join(sensor_root, sensor_set, f"participant_{pid}_timeseries.csv")
    pred = pd.read_csv(path)
    mu = 0.02 * pred["prob_modhigh"].to_numpy(dtype=float) + 0.06 * (1.0 - pred["prob_modhigh"].to_numpy(dtype=float))
    return pred["intensity_pred"].to_numpy(dtype=int), mu


def dot789_mp_o2(df, info_row, mu):
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
    return ke / 1000.0 / np.clip(mu, 0.005, 0.2) / 19.6


def dot6_direct_mp_o2(df, info_row, mu, effective_mass_fraction=0.05):
    full = ensure_ekf_columns(df)
    mass = float(info_row["Weight (kg)"])
    effective_mass = mass * effective_mass_fraction
    ke = 0.5 * effective_mass * full["velocity_DOT6"].astype(float).to_numpy() ** 2
    return ke / 1000.0 / np.clip(mu, 0.005, 0.2) / 19.6


def dot6_teacher_features(df):
    full = ensure_ekf_columns(df)
    out = pd.DataFrame(
        {
            "v6": full["velocity_DOT6"].astype(float),
            "a6": full["acceleration_DOT6"].astype(float),
            "g6": full["gyration_DOT6"].astype(float),
        }
    )
    out["v6_abs"] = out["v6"].abs()
    out["v6_sq"] = out["v6"] ** 2
    out["a6_abs"] = out["a6"].abs()
    out["g6_abs"] = out["g6"].abs()
    out["power_va"] = out["v6_abs"] * out["a6_abs"]
    out["power_vg"] = out["v6_abs"] * out["g6_abs"]
    for col in ["v6_abs", "v6_sq", "a6_abs", "g6_abs", "power_va", "power_vg"]:
        out[f"{col}_ema_5s"] = out[col].ewm(span=5, adjust=False).mean()
        out[f"{col}_ema_30s"] = out[col].ewm(span=30, adjust=False).mean()
        out[f"{col}_std_30s"] = out[col].rolling(30, min_periods=1).std().fillna(0.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_dot6_teacher(subjects, info, train_pids, sensor_root):
    xs, ys = [], []
    for pid in train_pids:
        df = subjects[pid]
        _, mu789 = load_sensor_prediction(sensor_root, "DOT789", pid)
        teacher = dot789_mp_o2(df, info.loc[pid], mu789)
        xs.append(dot6_teacher_features(df))
        ys.append(np.log1p(np.clip(teacher, 0.0, None)))
    X = pd.concat(xs).to_numpy(dtype=np.float32)
    y = np.concatenate(ys)
    scaler = StandardScaler().fit(X)
    model = HistGradientBoostingRegressor(max_iter=260, learning_rate=0.045, l2_regularization=0.05, random_state=81)
    model.fit(scaler.transform(X), y)
    return scaler, model


def predict_dot6_teacher_mp_o2(df, scaler, model):
    X = dot6_teacher_features(df).to_numpy(dtype=np.float32)
    pred = np.expm1(model.predict(scaler.transform(X)))
    return np.clip(pred, 0.0, None)


def run_custom_ekf(df, info_row, intensity, mp_o2, constant_hr=70.0, r_value=0.005, n_value=0.01):
    full = ensure_ekf_columns(df)
    mass = float(info_row["Weight (kg)"])
    sex = str(info_row["Sex"]).lower()
    mp_co2 = 0.8 * np.asarray(mp_o2, dtype=float)
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
        x_est, _, vo2co2 = model.run_ekf(
            x0,
            np.eye(5) * 0.01,
            seg["hr"].to_numpy(),
            [mp_o2[start:end], mp_co2[start:end]],
            seg["t"].to_numpy(),
            np.eye(5) * n_value,
            np.eye(2) * r_value,
        )
        vo2_l_min = np.asarray(vo2co2)[:, 0] * 60.0
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
                    "mp_o2_obs": mp_o2[start:end] * 60.0,
                }
            )
        )
    return pd.concat(outputs, ignore_index=True)


def plot_vo2(df, ekf, title, path):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(df["t"], df["vo2"], color="black", linewidth=1.15, label="True VO2")
    ax.plot(df["t"], ekf["ekf_vo2_ml_s"], color="#0072B2", linewidth=1.05, label="EKF VO2")
    last = None
    for t, act in zip(df["t"], df["activity"]):
        if act != last:
            ax.axvline(t, color="0.75", linestyle="--", linewidth=0.75, alpha=0.55)
            last = act
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("VO2 (ml/s)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run(data_root="Data", sensor_root="outputs_sensor_intensity_comparison", output_dir="outputs_dot6_ekf_vo2_comparison"):
    info = load_info(data_root)
    pids = participant_ids(data_root, info)
    subjects = {pid: load_subject(data_root, pid) for pid in pids}
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    r_grid = [0.001, 0.005, 0.02, 0.05]
    for pid in pids:
        subj_dir = os.path.join(output_dir, f"participant_{pid}")
        os.makedirs(subj_dir, exist_ok=True)
        train_pids = [p for p in pids if p != pid]
        teacher_scaler, teacher_model = fit_dot6_teacher(subjects, info, train_pids, sensor_root)
        df = subjects[pid]
        intensity789, mu789 = load_sensor_prediction(sensor_root, "DOT789", pid)
        intensity6, mu6 = load_sensor_prediction(sensor_root, "DOT6", pid)
        source_defs = {
            "DOT789_baseline": (intensity789, dot789_mp_o2(df, info.loc[pid], mu789)),
            "DOT6_direct": (intensity6, dot6_direct_mp_o2(df, info.loc[pid], mu6)),
            "DOT6_teacher_scaled": (intensity6, predict_dot6_teacher_mp_o2(df, teacher_scaler, teacher_model)),
        }
        for source_name, (intensity, mp_o2) in source_defs.items():
            for r_value in r_grid:
                row = {
                    "participant_id": pid,
                    "source": source_name,
                    "r_value": r_value,
                    "true_vo2_min": float(df["vo2"].min()),
                    "true_vo2_mean": float(df["vo2"].mean()),
                    "true_vo2_max": float(df["vo2"].max()),
                }
                try:
                    ekf = run_custom_ekf(df, info.loc[pid], intensity, mp_o2, r_value=r_value)
                    metrics = regression_metrics(df["vo2"].to_numpy(), ekf["ekf_vo2_ml_s"].to_numpy())
                    row.update(
                        {
                            "ekf_vo2_min": float(ekf["ekf_vo2_ml_s"].min()),
                            "ekf_vo2_mean": float(ekf["ekf_vo2_ml_s"].mean()),
                            "ekf_vo2_max": float(ekf["ekf_vo2_ml_s"].max()),
                            "error": "",
                            **metrics,
                        }
                    )
                    if r_value == 0.005:
                        ekf.to_csv(os.path.join(subj_dir, f"{source_name}_ekf_outputs_default_r.csv"), index=False)
                        plot_vo2(df, ekf, f"P{pid} {source_name} EKF VO2, R={r_value}", os.path.join(subj_dir, f"{source_name}_vo2_default_r.png"))
                except Exception as exc:
                    row.update(
                        {
                            "ekf_vo2_min": np.nan,
                            "ekf_vo2_mean": np.nan,
                            "ekf_vo2_max": np.nan,
                            "r2": np.nan,
                            "rmse": np.nan,
                            "mae": np.nan,
                            "corr": np.nan,
                            "error": str(exc),
                        }
                    )
                rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, "dot6_ekf_vo2_comparison_summary.csv"), index=False)
    ok = summary.dropna(subset=["rmse"])
    best = ok.loc[ok.groupby(["participant_id", "source"])["rmse"].idxmin()].sort_values(["participant_id", "source"])
    best.to_csv(os.path.join(output_dir, "dot6_ekf_vo2_oracle_best_r_by_subject.csv"), index=False)
    agg = summary.groupby(["source", "r_value"])[["rmse", "mae", "corr", "r2"]].agg(["median", "mean", "min", "max"])
    agg.to_csv(os.path.join(output_dir, "dot6_ekf_vo2_aggregate_by_r.csv"))
    return summary, best
