import os
from pathlib import Path

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
from run_part23_from_saved_intensity import dot789_mp_o2, plot_vo2
from train_saved_intensity_models_external import e4_file_to_dot6_dataframe, list_external_files


# ============================================================
# MANUAL PATH CONFIG
# ============================================================
WORKSPACE = os.environ.get("WRIST2TWIN_WORKSPACE", os.getcwd())
DATA_ROOT = os.path.join(WORKSPACE, "Data")
PART1_ROOT = os.path.join(WORKSPACE, "outputs_paper4_full_pipeline_5datasets", "part1_intensity")
OUTPUT_ROOT = os.path.join(WORKSPACE, "outputs_paper4_full_pipeline_5datasets", "part2_ekf")

EXTERNAL_DATASETS = {
    "PPG_DaLiA": os.path.join(WORKSPACE, "PPG_DaLiA"),
    "Stress_Aerobic": os.path.join(WORKSPACE, "Stress_Aerobic"),
    "BITS2_ADL_no_existing_health_issues": os.path.join(WORKSPACE, "BITS2_ADL_no_existing_health_issues"),
    "BITS2_ADL_other_health_conditions": os.path.join(WORKSPACE, "BITS2_ADL_other_health_conditions"),
    "PAMAP2": os.path.join(WORKSPACE, "pamap2"),
}

MODES_TO_RUN = ["e4_compatible_no_gyr", "dot_wrist_with_gyr"]
MU_LOW_MODERATE = 0.06
MU_MODERATE_HIGH = 0.012
MODHIGH_PROXY_SCALE = 1.0
# ============================================================


def intensity_to_mu(intensity):
    intensity = np.asarray(intensity, dtype=int)
    return np.where(intensity == 2, MU_MODERATE_HIGH, MU_LOW_MODERATE).astype(float)


def read_internal_intensity(part1_root, mode, pid):
    path = Path(part1_root) / "internal" / mode / f"participant_{pid}" / "intensity_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path), str(path)


def read_external_intensity(part1_root, mode, dataset_name, subject_id):
    path = Path(part1_root) / "external" / mode / dataset_name / subject_id / "intensity_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path), str(path)


def restore_activity(df, csv_path):
    raw = pd.read_csv(csv_path)
    n = min(len(df), len(raw))
    df = df.iloc[:n].reset_index(drop=True)
    raw = raw.iloc[:n].reset_index(drop=True)
    if "activity" in raw.columns:
        df["activity"] = raw["activity"].astype(str).to_numpy()
    if "markers" in raw.columns:
        df["markers"] = raw["markers"].astype(str).to_numpy()
    else:
        df["markers"] = df["activity"]
    return df


def safe_run_ekf_no_scale(df, info_row, intensity, mp_o2):
    mp_o2 = np.nan_to_num(np.asarray(mp_o2, dtype=float), nan=0.0, posinf=0.12, neginf=0.0)
    mp_o2 = np.clip(mp_o2, 0.0, 0.12)
    last_error = None
    for r_value, n_value in [(0.005, 0.01), (0.01, 0.01), (0.02, 0.02), (0.05, 0.05), (0.10, 0.10)]:
        try:
            ekf = run_custom_ekf(df, info_row, intensity, mp_o2, r_value=r_value, n_value=n_value)
            return ekf, r_value, n_value
        except np.linalg.LinAlgError as exc:
            last_error = exc
    raise last_error


def proxy_training_features(df, int_df, mode):
    feat = wrist_features(df, mode).copy()
    feat["intensity_pred"] = int_df["intensity_pred"].to_numpy(dtype=float)
    for col in ["prob_low", "prob_moderate", "prob_modhigh"]:
        feat[col] = int_df[col].to_numpy(dtype=float) if col in int_df.columns else 0.0
    return feat


def train_external_proxy(data_root, part1_root, mode):
    info = load_info(data_root)
    pids = participant_ids(data_root, info)
    xs, ys = [], []
    for pid in pids:
        df = load_subject(data_root, pid).reset_index(drop=True)
        int_df, _ = read_internal_intensity(part1_root, mode, pid)
        n = min(len(df), len(int_df))
        df = df.iloc[:n].reset_index(drop=True)
        int_df = int_df.iloc[:n].reset_index(drop=True)
        intensity = int_df["intensity_pred"].to_numpy(dtype=int)
        mu = intensity_to_mu(intensity)
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


def predict_external_proxy(df, int_df, mode, scaler, models):
    feat = proxy_training_features(df, int_df, mode)
    X = scaler.transform(feat.to_numpy(dtype=np.float32))
    pred = np.mean([np.expm1(model.predict(X)) for model in models], axis=0)
    return pd.Series(np.clip(pred, 0.0, 0.12)).ewm(span=8, adjust=False).mean().to_numpy(dtype=float)


def external_info_row(dataset_dir, record_id, internal_info):
    info_row = internal_info.mean(numeric_only=True)
    info_row["Weight (kg)"] = float(internal_info["Weight (kg)"].mean())
    info_row["Height (cm)"] = float(internal_info["Height (cm)"].mean())
    info_row["Muscle (%)"] = float(internal_info["Muscle (%)"].mean())
    info_row["Sex"] = "male"
    info_path = Path(dataset_dir) / "subject_info_summary.csv"
    if info_path.exists():
        info = pd.read_csv(info_path)
        id_col = None
        for candidate in ["record_id", "subject_id", "participant", "id"]:
            if candidate in info.columns:
                id_col = candidate
                break
        if id_col is not None:
            match = info[info[id_col].astype(str) == str(record_id)]
            if len(match):
                row = match.iloc[0]
                if "weight_kg" in row and pd.notna(row["weight_kg"]):
                    info_row["Weight (kg)"] = float(row["weight_kg"])
                if "height_cm" in row and pd.notna(row["height_cm"]):
                    info_row["Height (cm)"] = float(row["height_cm"])
                if "sex" in row and str(row["sex"]).lower() in ["male", "female"]:
                    info_row["Sex"] = str(row["sex"]).lower()
    return info_row


def run_internal_mode(data_root, part1_root, output_root, mode):
    info = load_info(data_root)
    pids = participant_ids(data_root, info)
    rows = []
    for pid in pids:
        print(f"Part2 internal {mode} participant {pid}")
        df = load_subject(data_root, pid).reset_index(drop=True)
        int_df, int_path = read_internal_intensity(part1_root, mode, pid)
        n = min(len(df), len(int_df))
        df = df.iloc[:n].reset_index(drop=True)
        int_df = int_df.iloc[:n].reset_index(drop=True)
        intensity = int_df["intensity_pred"].to_numpy(dtype=int)
        mu = intensity_to_mu(intensity)
        mp_o2 = dot789_mp_o2(df, info.loc[pid], mu)
        ekf, r_value, n_value = safe_run_ekf_no_scale(df, info.loc[pid], intensity, mp_o2)
        ekf["mu"] = mu
        ekf["intensity_pred"] = intensity
        ekf["mp_o2_dot789_l_min"] = mp_o2 * 60.0
        out_dir = Path(output_root) / mode / "internal" / f"participant_{pid}" / "part2_dot789_ekf"
        out_dir.mkdir(parents=True, exist_ok=True)
        ekf.to_csv(out_dir / "part2_ekf_outputs.csv", index=False)
        row = {
            "mode": mode,
            "dataset": "internal",
            "participant_id": pid,
            "intensity_path": int_path,
            "ekf_r_value": r_value,
            "ekf_n_value": n_value,
            **regression_metrics(df["vo2"].to_numpy(dtype=float), ekf["ekf_vo2_ml_s"].to_numpy(dtype=float)),
            "ekf_vo2_mean": float(ekf["ekf_vo2_ml_s"].mean()),
            "ekf_vo2_max": float(ekf["ekf_vo2_ml_s"].max()),
            "true_vo2_mean": float(df["vo2"].mean()),
            "true_vo2_max": float(df["vo2"].max()),
        }
        pd.DataFrame([row]).to_csv(out_dir / "part2_ekf_summary.csv", index=False)
        plot_vo2(df, ekf, df["vo2"].to_numpy(dtype=float), out_dir / "part2_ekf_vo2.png", f"Internal P{pid} {mode}")
        rows.append(row)
    return pd.DataFrame(rows)


def run_external_mode(data_root, part1_root, output_root, mode, datasets):
    internal_info, scaler, models = train_external_proxy(data_root, part1_root, mode)
    rows = []
    for dataset_name, dataset_dir in datasets.items():
        for subject_id, csv_path in list_external_files(dataset_dir):
            print(f"Part2 external {mode} {dataset_name} {subject_id}")
            df = e4_file_to_dot6_dataframe(csv_path).reset_index(drop=True)
            df = restore_activity(df, csv_path)
            int_df, int_path = read_external_intensity(part1_root, mode, dataset_name, subject_id)
            n = min(len(df), len(int_df))
            df = df.iloc[:n].reset_index(drop=True)
            int_df = int_df.iloc[:n].reset_index(drop=True)
            intensity = int_df["intensity_pred"].to_numpy(dtype=int)
            mu = intensity_to_mu(intensity)
            mp_o2 = predict_external_proxy(df, int_df, mode, scaler, models)
            info_row = external_info_row(dataset_dir, subject_id, internal_info)
            ekf, r_value, n_value = safe_run_ekf_no_scale(df, info_row, intensity, mp_o2)
            ekf["mu"] = mu
            ekf["intensity_pred"] = intensity
            ekf["mp_o2_dot789_teacher_l_min"] = mp_o2 * 60.0
            out_dir = Path(output_root) / mode / "external" / dataset_name / subject_id / "part2_dot789_teacher_ekf"
            out_dir.mkdir(parents=True, exist_ok=True)
            ekf.to_csv(out_dir / "part2_ekf_outputs.csv", index=False)
            row = {
                "mode": mode,
                "dataset": dataset_name,
                "subject_id": subject_id,
                "source_file": csv_path,
                "intensity_path": int_path,
                "ekf_r_value": r_value,
                "ekf_n_value": n_value,
                "ekf_vo2_mean": float(ekf["ekf_vo2_ml_s"].mean()),
                "ekf_vo2_max": float(ekf["ekf_vo2_ml_s"].max()),
                "mp_o2_mean_l_min": float(np.mean(mp_o2) * 60.0),
                "mp_o2_max_l_min": float(np.max(mp_o2) * 60.0),
                "hr_mean": float(df["hr"].mean()),
                "hr_max": float(df["hr"].max()),
                "n": int(len(df)),
            }
            pd.DataFrame([row]).to_csv(out_dir / "part2_ekf_summary.csv", index=False)
            plot_vo2(df, ekf, None, out_dir / "part2_ekf_vo2.png", f"{dataset_name} {subject_id} {mode}")
            rows.append(row)
    return pd.DataFrame(rows)


def run(data_root=DATA_ROOT, part1_root=PART1_ROOT, output_root=OUTPUT_ROOT, external_datasets=EXTERNAL_DATASETS, modes=MODES_TO_RUN):
    Path(output_root).mkdir(parents=True, exist_ok=True)
    internal_all, external_all = [], []
    for mode in modes:
        internal_all.append(run_internal_mode(data_root, part1_root, output_root, mode))
        external_all.append(run_external_mode(data_root, part1_root, output_root, mode, external_datasets))
    internal = pd.concat(internal_all, ignore_index=True)
    external = pd.concat(external_all, ignore_index=True)
    internal.to_csv(Path(output_root) / "internal_part2_summary.csv", index=False)
    external.to_csv(Path(output_root) / "external_part2_summary.csv", index=False)
    return internal, external
