import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimize_wrist_loso_part12 import build_items, predict_intensity, train_intensity, window_features, wrist_features


MODES = ["e4_compatible_no_gyr", "dot_wrist_with_gyr"]


def e4_file_to_dot6_dataframe(path):
    raw = pd.read_csv(path)
    n = len(raw)
    df = pd.DataFrame()
    df["t"] = np.arange(n, dtype=int)
    df["unix_t"] = raw["t"].to_numpy() if "t" in raw.columns else np.arange(n, dtype=int)
    df["activity"] = "external"
    df["markers"] = "external"
    df["hr"] = raw["HR_bpm"].astype(float).interpolate(limit_direction="both").to_numpy() if "HR_bpm" in raw.columns else np.nan
    df["HR_bpm"] = df["hr"]
    df["vo2"] = 0.0
    df["vco2"] = 0.0

    vx = raw["vel_x_mps"].astype(float).to_numpy() if "vel_x_mps" in raw.columns else np.zeros(n)
    vy = raw["vel_y_mps"].astype(float).to_numpy() if "vel_y_mps" in raw.columns else np.zeros(n)
    vz = raw["vel_z_mps"].astype(float).to_numpy() if "vel_z_mps" in raw.columns else np.zeros(n)
    ax = raw["acc_x_mps2"].astype(float).to_numpy() if "acc_x_mps2" in raw.columns else np.zeros(n)
    ay = raw["acc_y_mps2"].astype(float).to_numpy() if "acc_y_mps2" in raw.columns else np.zeros(n)
    az = raw["acc_z_mps2"].astype(float).to_numpy() if "acc_z_mps2" in raw.columns else np.zeros(n)
    gx = raw["gyr_x_rad_s"].astype(float).to_numpy() if "gyr_x_rad_s" in raw.columns else np.zeros(n)
    gy = raw["gyr_y_rad_s"].astype(float).to_numpy() if "gyr_y_rad_s" in raw.columns else np.zeros(n)
    gz = raw["gyr_z_rad_s"].astype(float).to_numpy() if "gyr_z_rad_s" in raw.columns else np.zeros(n)

    df["linear_velocity_x_DOT6"] = vx
    df["linear_velocity_y_DOT6"] = vy
    df["linear_velocity_z_DOT6"] = vz
    if "total_velocity_mps" in raw.columns:
        df["velocity_DOT6"] = raw["total_velocity_mps"].astype(float).to_numpy()
    else:
        df["velocity_DOT6"] = np.sqrt(vx**2 + vy**2 + vz**2)

    df["filtered_x_DOT6"] = ax
    df["filtered_y_DOT6"] = ay
    df["filtered_z_DOT6"] = az
    df["acceleration_DOT6"] = np.sqrt(ax**2 + ay**2 + az**2)

    if "total_gyration_rad_s" in raw.columns:
        df["gyration_DOT6"] = raw["total_gyration_rad_s"].astype(float).to_numpy()
    else:
        df["gyration_DOT6"] = np.sqrt(gx**2 + gy**2 + gz**2)
    df["filtered_gyr_x_DOT6"] = gx
    df["filtered_gyr_y_DOT6"] = gy
    df["filtered_gyr_z_DOT6"] = gz
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def external_item_from_df(df, mode):
    feat = wrist_features(df, mode)
    dummy_y = np.zeros(len(df), dtype=int)
    wx, _, ranges = window_features(feat, dummy_y)
    return {"df": df, "feat": feat, "wx": wx, "ranges": ranges}


def list_external_files(dataset_dir, suffix="_E4_ACC_HR_1Hz.csv"):
    if not os.path.isdir(dataset_dir):
        return []
    files = []
    for name in sorted(os.listdir(dataset_dir)):
        if name.endswith(suffix):
            subject_id = name.split("_")[0]
            files.append((subject_id, os.path.join(dataset_dir, name)))
    return files


def train_and_save_model(data_root, mode, model_dir):
    print(f"Training final internal model: {mode}")
    _, pids, items = build_items(data_root, mode)
    train_items = [items[pid] for pid in pids]
    model = train_intensity(train_items)
    payload = {
        "mode": mode,
        "participant_ids": pids,
        "model": model,
        "description": (
            "Final Part1 wrist-intensity model trained on all internal subjects. "
            "Use e4_compatible_no_gyr as the primary external-E4 model because E4 has no gyroscope."
        ),
    }
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{mode}_best_intensity_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    with open(os.path.join(model_dir, f"{mode}_selected_temporal_parameters.txt"), "w", encoding="utf-8") as f:
        for k, v in model["params"].items():
            if k != "trans":
                f.write(f"{k}: {v}\n")
    return payload, path


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_external_prediction(pred_df, path, title):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.3]})
    axes[0].plot(pred_df["t"], pred_df["intensity_pred"], color="#0072B2", linewidth=1.05, label="Smoothed intensity")
    axes[0].plot(pred_df["t"], pred_df["intensity_raw"], color="#D55E00", linewidth=0.65, alpha=0.45, label="Raw intensity")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title(title)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(pred_df["t"], pred_df["prob_low"], label="Low", linewidth=0.9)
    axes[1].plot(pred_df["t"], pred_df["prob_moderate"], label="Moderate", linewidth=0.9)
    axes[1].plot(pred_df["t"], pred_df["prob_modhigh"], label="Moderate-High", linewidth=0.9)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Posterior")
    axes[1].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def predict_external_for_model(payload, dataset_name, subject_id, file_path, output_root):
    mode = payload["mode"]
    model = payload["model"]
    df = e4_file_to_dot6_dataframe(file_path)
    item = external_item_from_df(df, mode)
    raw, smooth, prob = predict_intensity(model, item)
    pred = pd.DataFrame(
        {
            "t": df["t"],
            "unix_t": df["unix_t"],
            "HR_bpm": df["HR_bpm"],
            "velocity_DOT6": df["velocity_DOT6"],
            "acceleration_DOT6": df["acceleration_DOT6"],
            "intensity_raw": raw,
            "intensity_pred": smooth,
            "prob_low": prob[:, 0],
            "prob_moderate": prob[:, 1],
            "prob_modhigh": prob[:, 2],
            "mu": 0.02 * prob[:, 2] + 0.06 * (1.0 - prob[:, 2]),
        }
    )
    out_dir = os.path.join(output_root, mode, dataset_name, subject_id)
    os.makedirs(out_dir, exist_ok=True)
    pred.to_csv(os.path.join(out_dir, "external_intensity_predictions.csv"), index=False)
    plot_external_prediction(pred, os.path.join(out_dir, "external_intensity_predictions.png"), f"{dataset_name} {subject_id} {mode}")
    return {
        "mode": mode,
        "dataset": dataset_name,
        "subject_id": subject_id,
        "n": len(pred),
        "velocity_median": float(pred["velocity_DOT6"].median()),
        "velocity_q95": float(pred["velocity_DOT6"].quantile(0.95)),
        "low_fraction": float(np.mean(pred["intensity_pred"] == 0)),
        "moderate_fraction": float(np.mean(pred["intensity_pred"] == 1)),
        "modhigh_fraction": float(np.mean(pred["intensity_pred"] == 2)),
        "mean_prob_low": float(pred["prob_low"].mean()),
        "mean_prob_moderate": float(pred["prob_moderate"].mean()),
        "mean_prob_modhigh": float(pred["prob_modhigh"].mean()),
        "mean_mu": float(pred["mu"].mean()),
    }


def run(data_root="Data", weee_dir="WEEE", ppg_dir="PPG_DaLiA", output_root="outputs_saved_intensity_models_external", mode="both"):
    os.makedirs(output_root, exist_ok=True)
    model_dir = os.path.join(output_root, "saved_models")
    modes = MODES if mode == "both" else [mode]
    model_payloads = []
    for m in modes:
        payload, model_path = train_and_save_model(data_root, m, model_dir)
        print(f"Saved model: {model_path}")
        model_payloads.append(payload)

    datasets = [("WEEE", weee_dir), ("PPG_DaLiA", ppg_dir)]
    rows = []
    for payload in model_payloads:
        for dataset_name, dataset_dir in datasets:
            for subject_id, file_path in list_external_files(dataset_dir):
                print(f"Predicting {dataset_name} {subject_id} with {payload['mode']}")
                rows.append(predict_external_for_model(payload, dataset_name, subject_id, file_path, output_root))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_root, "external_intensity_prediction_summary.csv"), index=False)
    with open(os.path.join(output_root, "README_saved_models_external_prediction.md"), "w", encoding="utf-8") as f:
        f.write("# Saved internal wrist-intensity models and external predictions\n\n")
        f.write("The script trains final Part1 intensity models on all internal subjects, saves both model objects as pickle files, and applies them directly to WEEE and PPG_DaLiA E4 wrist data.\n\n")
        f.write("For external E4 validation, `e4_compatible_no_gyr` is the primary model because E4 does not provide gyroscope channels. `dot_wrist_with_gyr` is also exported for comparison; its external gyroscope inputs are set to zero.\n\n")
        f.write("## External prediction summary\n\n")
        if len(summary):
            f.write(summary.groupby(["mode", "dataset"])[["low_fraction", "moderate_fraction", "modhigh_fraction", "mean_mu"]].agg(["median", "mean"]).round(4).to_markdown())
            f.write("\n")
    return summary
