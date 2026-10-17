import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from compare_dot6_ekf_vo2 import regression_metrics, run_custom_ekf
from hybrid_parts_common import load_info, load_subject, participant_ids
from optimize_wrist_loso_part12 import wrist_features
from part3_pinn_hr import train_predict_pinn
from train_saved_intensity_models_external import e4_file_to_dot6_dataframe, list_external_files


CHRONOLOGICAL_SCENARIOS = ["300", "20", "50", "80"]
PER_ACTIVITY_SCENARIOS = ["20", "50", "80"]
PHYS_WEIGHT = 1000.0
MODHIGH_PROXY_SCALE = 1


def dot789_mp_o2(df, info_row, mu):
    mass = float(info_row["Weight (kg)"])
    sex = str(info_row["Sex"]).lower()
    mass_torso = mass * (0.6664 if sex.startswith("m") else 0.6314)
    mass_leg = mass * (0.1668 if sex.startswith("m") else 0.1843)
    v7 = df["velocity_DOT7"].astype(float).to_numpy() if "velocity_DOT7" in df.columns else np.zeros(len(df))
    v8 = df["velocity_DOT8"].astype(float).to_numpy() if "velocity_DOT8" in df.columns else np.zeros(len(df))
    v9 = df["velocity_DOT9"].astype(float).to_numpy() if "velocity_DOT9" in df.columns else np.zeros(len(df))
    ke = 0.5 * (mass_torso * v7**2 + mass_leg * v8**2 + mass_leg * v9**2)
    return ke / 1000.0 / np.clip(mu, 0.005, 0.2) / 19.6


def safe_run_ekf(df, info_row, intensity, mp_o2):
    mp_o2 = np.nan_to_num(np.asarray(mp_o2, dtype=float), nan=0.0, posinf=0.12, neginf=0.0)
    intensity = np.asarray(intensity, dtype=int)
    mp_o2 = mp_o2 * np.where(intensity == 2, MODHIGH_PROXY_SCALE, 1.0)
    mp_o2 = np.clip(mp_o2, 0.0, 0.12)
    last_error = None
    for r_value, n_value in [(0.005, 0.01), (0.01, 0.01), (0.02, 0.02), (0.05, 0.05), (0.10, 0.10)]:
        try:
            ekf = run_custom_ekf(df, info_row, intensity, mp_o2, r_value=r_value, n_value=n_value)
            ekf.attrs["r_value"] = r_value
            ekf.attrs["n_value"] = n_value
            return ekf, r_value, n_value
        except np.linalg.LinAlgError as exc:
            last_error = exc
            print(f"EKF singular matrix with r={r_value}, n={n_value}; retrying with larger covariance.")
    raise last_error


def read_internal_intensity(root, participant_id):
    candidates = [
        os.path.join(root, f"participant_{participant_id}", "part1", "part1_intensity_timeseries.csv"),
        os.path.join(root, f"participant_{participant_id}", "part1", "part1_intensity_temporal_corrected_timeseries.csv"),
        os.path.join(root, f"participant_{participant_id}", "intensity_loso_predictions.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "intensity_temporal_corrected" in df.columns:
                df["intensity_pred"] = df["intensity_temporal_corrected"]
            return df, path
    raise FileNotFoundError(f"No internal intensity file found for participant {participant_id} under {root}")


def read_external_intensity(root, mode, dataset_name, subject_id):
    path = os.path.join(root, mode, dataset_name, subject_id, "external_intensity_predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path), path


def mu_from_intensity_df(int_df):
    if "prob_modhigh" in int_df.columns:
        p_high = int_df["prob_modhigh"].to_numpy(dtype=float)
        return 0.02 * p_high + 0.06 * (1.0 - p_high)
    intensity = int_df["intensity_pred"].to_numpy(dtype=int)
    return np.where(intensity == 2, 0.02, 0.06).astype(float)


def plot_vo2(df, ekf, y_true, path, title):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    if y_true is not None:
        ax.plot(df["t"], y_true, color="black", linewidth=1.05, label="True VO2")
    ax.plot(df["t"], ekf["ekf_vo2_ml_s"], color="#0072B2", linewidth=1.0, label="EKF VO2")
    last = None
    if "activity" in df.columns:
        for t, act in zip(df["t"], df["activity"]):
            if act != last:
                ax.axvline(t, color="0.75", linestyle="--", linewidth=0.7, alpha=0.55)
                last = act
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("VO2 (ml/s)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_hr(df, pred_df, path, title):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    x = np.arange(len(pred_df))
    ax.plot(x, pred_df["hr_true"], color="black", linewidth=1.05, label="True HR")
    ax.plot(x, pred_df["hr_pred"], color="#0072B2", linewidth=1.0, label="PINN HR")
    ax.plot(x, pred_df["hr_whitebox"], color="#009E73", linewidth=0.9, alpha=0.75, label="White-box HR")
    if "original_index" in pred_df.columns and "activity" in df.columns:
        last = None
        for x_i, idx in enumerate(pred_df["original_index"].astype(int).to_numpy()):
            act = df.iloc[idx]["activity"]
            if act != last:
                ax.axvline(x_i, color="0.75", linestyle="--", linewidth=0.7, alpha=0.55)
                last = act
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("HR (bpm)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_part3(df, ekf_vo2, output_dir, subject_label, split_modes, epochs):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for split_mode, scenarios in split_modes.items():
        for scenario in scenarios:
            print(f"Part3 {subject_label} {split_mode} {scenario}")
            pred, whitebox, train, test, train_order, test_order, metrics, whitebox_metrics, lambdas, history = train_predict_pinn(
                df,
                ekf_vo2,
                split_mode,
                scenario,
                epochs=epochs,
                phys_weight=PHYS_WEIGHT,
            )
            row = {
                "subject": subject_label,
                "split_mode": split_mode,
                "scenario": scenario,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "phys_weight": PHYS_WEIGHT,
                **metrics,
                **{f"whitebox_{k}": v for k, v in whitebox_metrics.items()},
                **{f"lambda{i + 1}": lambdas[i] for i in range(6)},
            }
            rows.append(row)
            pd.DataFrame(history).to_csv(os.path.join(output_dir, f"part3_history_{split_mode}_{scenario}.csv"), index=False)
            pd.DataFrame([{f"lambda{i + 1}": lambdas[i] for i in range(6)}]).to_csv(
                os.path.join(output_dir, f"part3_final_lambdas_{split_mode}_{scenario}.csv"), index=False
            )
            pred_df = pd.DataFrame(
                {
                    "original_index": test_order,
                    "t": df.loc[test_order, "t"].to_numpy(),
                    "hr_true": df.loc[test_order, "hr"].to_numpy(),
                    "hr_pred": pred[test_order],
                    "hr_whitebox": whitebox[test_order],
                    "ekf_vo2_ml_s": np.asarray(ekf_vo2)[test_order],
                }
            )
            pred_df.to_csv(os.path.join(output_dir, f"part3_predictions_{split_mode}_{scenario}.csv"), index=False)
            train_df = pd.DataFrame(
                {
                    "original_index": train_order,
                    "t": df.loc[train_order, "t"].to_numpy(),
                    "hr_true": df.loc[train_order, "hr"].to_numpy(),
                    "ekf_vo2_ml_s": np.asarray(ekf_vo2)[train_order],
                }
            )
            train_df.to_csv(os.path.join(output_dir, f"part3_train_{split_mode}_{scenario}.csv"), index=False)
            plot_hr(df, pred_df, os.path.join(output_dir, f"part3_hr_{split_mode}_{scenario}.png"), f"{subject_label} {split_mode} {scenario}")
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, "part3_summary.csv"), index=False)
    return summary


def proxy_training_features(df, int_df, mode):
    feat = wrist_features(df, mode).copy()
    feat["intensity_pred"] = int_df["intensity_pred"].to_numpy(dtype=float)
    for col in ["prob_low", "prob_moderate", "prob_modhigh"]:
        feat[col] = int_df[col].to_numpy(dtype=float) if col in int_df.columns else 0.0
    return feat


def train_external_dot789_proxy(data_root, internal_intensity_root, mode):
    info = load_info(data_root)
    pids = participant_ids(data_root, info)
    xs, ys = [], []
    for pid in pids:
        df = load_subject(data_root, pid)
        int_df, _ = read_internal_intensity(internal_intensity_root, pid)
        n = min(len(df), len(int_df))
        df = df.iloc[:n].reset_index(drop=True)
        int_df = int_df.iloc[:n].reset_index(drop=True)
        mu = mu_from_intensity_df(int_df)
        xs.append(proxy_training_features(df, int_df, mode))
        ys.append(np.log1p(np.clip(dot789_mp_o2(df, info.loc[pid], mu), 0.0, 0.12)))
    X = pd.concat(xs).to_numpy(dtype=np.float32)
    y = np.concatenate(ys)
    scaler = StandardScaler().fit(X)
    models = [
        ExtraTreesRegressor(n_estimators=700, min_samples_leaf=1, max_features="sqrt", random_state=710, n_jobs=1),
        HistGradientBoostingRegressor(max_iter=420, learning_rate=0.035, l2_regularization=0.03, random_state=711),
    ]
    Xs = scaler.transform(X)
    for model in models:
        model.fit(Xs, y)
    return info, scaler, models


def predict_external_dot789_proxy(df, int_df, mode, scaler, models):
    feat = proxy_training_features(df, int_df, mode)
    X = scaler.transform(feat.to_numpy(dtype=np.float32))
    pred = np.mean([np.expm1(model.predict(X)) for model in models], axis=0)
    return pd.Series(np.clip(pred, 0.0, 0.12)).ewm(span=8, adjust=False).mean().to_numpy(dtype=float)


def run_internal(data_root, internal_intensity_root, output_root, mode, epochs):
    info = load_info(data_root)
    pids = participant_ids(data_root, info)
    all_part2, all_part3 = [], []
    for pid in pids:
        print(f"Internal Part2/3 participant {pid}")
        df = load_subject(data_root, pid).reset_index(drop=True)
        int_df, int_path = read_internal_intensity(internal_intensity_root, pid)
        n = min(len(df), len(int_df))
        df = df.iloc[:n].reset_index(drop=True)
        int_df = int_df.iloc[:n].reset_index(drop=True)
        intensity = int_df["intensity_pred"].to_numpy(dtype=int)
        mu = mu_from_intensity_df(int_df)
        mp_o2 = dot789_mp_o2(df, info.loc[pid], mu)
        ekf, ekf_r_value, ekf_n_value = safe_run_ekf(df, info.loc[pid], intensity, mp_o2)
        ekf["mu"] = mu
        ekf["intensity_pred"] = intensity
        ekf["mp_o2_dot789_l_min"] = mp_o2 * 60.0
        base = os.path.join(output_root, "internal", f"participant_{pid}")
        part2_dir = os.path.join(base, "part2_dot789_ekf")
        part3_dir = os.path.join(base, "part3_pinn_hr")
        os.makedirs(part2_dir, exist_ok=True)
        ekf.to_csv(os.path.join(part2_dir, "part2_ekf_outputs.csv"), index=False)
        p2 = {
            "dataset": "internal",
            "participant_id": pid,
            "intensity_path": int_path,
            "ekf_r_value": ekf_r_value,
            "ekf_n_value": ekf_n_value,
            **regression_metrics(df["vo2"].to_numpy(dtype=float), ekf["ekf_vo2_ml_s"].to_numpy(dtype=float)),
            "ekf_vo2_mean": float(ekf["ekf_vo2_ml_s"].mean()),
            "ekf_vo2_max": float(ekf["ekf_vo2_ml_s"].max()),
            "true_vo2_mean": float(df["vo2"].mean()),
            "true_vo2_max": float(df["vo2"].max()),
        }
        pd.DataFrame([p2]).to_csv(os.path.join(part2_dir, "part2_ekf_summary.csv"), index=False)
        plot_vo2(df, ekf, df["vo2"].to_numpy(dtype=float), os.path.join(part2_dir, "part2_ekf_vo2.png"), f"Internal P{pid} DOT789 EKF")
        all_part2.append(p2)
        part3 = run_part3(
            df,
            ekf["ekf_vo2_ml_s"].to_numpy(dtype=float),
            part3_dir,
            f"internal_P{pid}",
            {"chronological": CHRONOLOGICAL_SCENARIOS, "per_activity": PER_ACTIVITY_SCENARIOS},
            epochs,
        )
        all_part3.append(part3.assign(dataset="internal", participant_id=pid))
    return pd.DataFrame(all_part2), pd.concat(all_part3, ignore_index=True)


def run_external(data_root, external_intensity_root, internal_intensity_root, output_root, mode, epochs, weee_dir, ppg_dir):
    info, scaler, models = train_external_dot789_proxy(data_root, internal_intensity_root, mode)
    all_part2, all_part3 = [], []
    for dataset_name, dataset_dir in [("WEEE", weee_dir), ("PPG_DaLiA", ppg_dir)]:
        for subject_id, path in list_external_files(dataset_dir):
            print(f"External Part2/3 {dataset_name} {subject_id}")
            df = e4_file_to_dot6_dataframe(path).reset_index(drop=True)
            df["activity"] = "external"
            int_df, int_path = read_external_intensity(external_intensity_root, mode, dataset_name, subject_id)
            n = min(len(df), len(int_df))
            df = df.iloc[:n].reset_index(drop=True)
            int_df = int_df.iloc[:n].reset_index(drop=True)
            intensity = int_df["intensity_pred"].to_numpy(dtype=int)
            mu = mu_from_intensity_df(int_df)
            mp_o2 = predict_external_dot789_proxy(df, int_df, mode, scaler, models)
            # External demographics are not needed by the trained proxy, but the EKF
            # requires body size. Use representative internal averages as a neutral
            # fallback; downstream Part3 is calibrated on the external subject's HR.
            info_row = info.mean(numeric_only=True)
            info_row["Weight (kg)"] = float(info["Weight (kg)"].mean())
            info_row["Height (cm)"] = float(info["Height (cm)"].mean())
            info_row["Muscle (%)"] = float(info["Muscle (%)"].mean())
            info_row["Sex"] = "male"
            ekf, ekf_r_value, ekf_n_value = safe_run_ekf(df, info_row, intensity, mp_o2)
            ekf["mu"] = mu
            ekf["intensity_pred"] = intensity
            ekf["mp_o2_dot789_teacher_l_min"] = mp_o2 * 60.0
            base = os.path.join(output_root, "external", dataset_name, subject_id)
            part2_dir = os.path.join(base, "part2_dot789_teacher_ekf")
            part3_dir = os.path.join(base, "part3_pinn_hr")
            os.makedirs(part2_dir, exist_ok=True)
            ekf.to_csv(os.path.join(part2_dir, "part2_ekf_outputs.csv"), index=False)
            p2 = {
                "dataset": dataset_name,
                "subject_id": subject_id,
                "intensity_path": int_path,
                "ekf_r_value": ekf_r_value,
                "ekf_n_value": ekf_n_value,
                "ekf_vo2_mean": float(ekf["ekf_vo2_ml_s"].mean()),
                "ekf_vo2_max": float(ekf["ekf_vo2_ml_s"].max()),
                "mp_o2_mean_l_min": float(np.mean(mp_o2) * 60.0),
                "mp_o2_max_l_min": float(np.max(mp_o2) * 60.0),
                "hr_mean": float(df["hr"].mean()),
                "hr_max": float(df["hr"].max()),
            }
            pd.DataFrame([p2]).to_csv(os.path.join(part2_dir, "part2_ekf_summary.csv"), index=False)
            plot_vo2(df, ekf, None, os.path.join(part2_dir, "part2_ekf_vo2.png"), f"{dataset_name} {subject_id} DOT789-teacher EKF")
            all_part2.append(p2)
            part3 = run_part3(
                df,
                ekf["ekf_vo2_ml_s"].to_numpy(dtype=float),
                part3_dir,
                f"{dataset_name}_{subject_id}",
                {"chronological": CHRONOLOGICAL_SCENARIOS},
                epochs,
            )
            all_part3.append(part3.assign(dataset=dataset_name, subject_id=subject_id))
    return pd.DataFrame(all_part2), pd.concat(all_part3, ignore_index=True)


def run(args):
    os.makedirs(args.output_root, exist_ok=True)
    all_tables = {}
    if args.scope in ["both", "internal"]:
        internal_p2, internal_p3 = run_internal(args.data_root, args.internal_intensity_root, args.output_root, args.mode, args.epochs)
        internal_p2.to_csv(os.path.join(args.output_root, "internal_part2_summary.csv"), index=False)
        internal_p3.to_csv(os.path.join(args.output_root, "internal_part3_summary.csv"), index=False)
        all_tables["internal_p2"] = internal_p2
        all_tables["internal_p3"] = internal_p3
    if args.scope in ["both", "external"]:
        external_p2, external_p3 = run_external(
            args.data_root,
            args.external_intensity_root,
            args.internal_intensity_root,
            args.output_root,
            args.mode,
            args.epochs,
            args.weee_dir,
            args.ppg_dir,
        )
        external_p2.to_csv(os.path.join(args.output_root, "external_part2_summary.csv"), index=False)
        external_p3.to_csv(os.path.join(args.output_root, "external_part3_summary.csv"), index=False)
        all_tables["external_p2"] = external_p2
        all_tables["external_p3"] = external_p3
    with open(os.path.join(args.output_root, "README_part23_from_saved_intensity.md"), "w", encoding="utf-8") as f:
        f.write("# Part2/Part3 from saved intensity predictions\n\n")
        f.write(f"Mode: `{args.mode}`. PINN physics loss weight: `{PHYS_WEIGHT}`. Epochs: `{args.epochs}`.\n\n")
        f.write(f"Moderate-high intensity metabolic proxy scale: `{MODHIGH_PROXY_SCALE}`. This scale is applied only when `intensity_pred == 2` before EKF inference.\n\n")
        f.write("Internal Part2 computes DOT7+DOT8+DOT9 metabolic proxy directly from internal DOT789 velocity and saved internal intensity predictions. External Part2 trains an internal wrist-to-DOT789 proxy teacher and applies it to saved external wrist-intensity predictions.\n\n")
        if "internal_p2" in all_tables:
            f.write("## Internal Part2\n\n")
            f.write(all_tables["internal_p2"].round(4).to_markdown(index=False))
            f.write("\n\n")
        if "internal_p3" in all_tables:
            f.write("## Internal Part3 aggregate\n\n")
            cols = ["rmse", "mae", "corr", "r2", "whitebox_rmse", "whitebox_mae", "whitebox_corr"]
            f.write(all_tables["internal_p3"].groupby(["split_mode", "scenario"])[cols].agg(["median", "mean"]).round(4).to_markdown())
            f.write("\n\n")
        if "external_p2" in all_tables:
            f.write("## External Part2 aggregate\n\n")
            f.write(all_tables["external_p2"].groupby("dataset")[["ekf_vo2_mean", "ekf_vo2_max", "mp_o2_mean_l_min", "mp_o2_max_l_min"]].agg(["median", "mean"]).round(4).to_markdown())
            f.write("\n\n")
        if "external_p3" in all_tables:
            f.write("## External Part3 aggregate\n\n")
            cols = ["rmse", "mae", "corr", "r2", "whitebox_rmse", "whitebox_mae", "whitebox_corr"]
            f.write(all_tables["external_p3"].groupby(["dataset", "scenario"])[cols].agg(["median", "mean"]).round(4).to_markdown())
            f.write("\n")
