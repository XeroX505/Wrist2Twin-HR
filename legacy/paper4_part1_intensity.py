import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from optimize_wrist_loso_part12 import build_items, predict_intensity, train_intensity, window_features, wrist_features
from train_saved_intensity_models_external import e4_file_to_dot6_dataframe, list_external_files


# ============================================================
# MANUAL PATH CONFIG
# ============================================================
WORKSPACE = os.environ.get("WRIST2TWIN_WORKSPACE", os.getcwd())
DATA_ROOT = os.path.join(WORKSPACE, "Data")
OUTPUT_ROOT = os.path.join(WORKSPACE, "outputs_paper4_full_pipeline_5datasets", "part1_intensity")

EXTERNAL_DATASETS = {
    "PPG_DaLiA": os.path.join(WORKSPACE, "PPG_DaLiA"),
    "Stress_Aerobic": os.path.join(WORKSPACE, "Stress_Aerobic"),
    "BITS2_ADL_no_existing_health_issues": os.path.join(WORKSPACE, "BITS2_ADL_no_existing_health_issues"),
    "BITS2_ADL_other_health_conditions": os.path.join(WORKSPACE, "BITS2_ADL_other_health_conditions"),
    "PAMAP2": os.path.join(WORKSPACE, "pamap2"),
}

MODES_TO_RUN = ["e4_compatible_no_gyr", "dot_wrist_with_gyr"]
# ============================================================


def external_item_from_df(df, mode):
    feat = wrist_features(df, mode)
    dummy_y = np.zeros(len(df), dtype=int)
    wx, _, ranges = window_features(feat, dummy_y)
    return {"df": df, "feat": feat, "wx": wx, "ranges": ranges}


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


def plot_intensity(pred_df, path, title, has_truth=False):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.3]})
    if has_truth and "intensity_true" in pred_df.columns:
        axes[0].plot(pred_df["t"], pred_df["intensity_true"], color="black", linewidth=0.9, label="True")
    axes[0].plot(pred_df["t"], pred_df["intensity_raw"], color="#D55E00", linewidth=0.6, alpha=0.45, label="Raw")
    axes[0].plot(pred_df["t"], pred_df["intensity_pred"], color="#0072B2", linewidth=1.0, label="Smoothed")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title(title)
    axes[0].legend(frameon=False, ncol=3)
    for col, lab in [("prob_low", "Low"), ("prob_moderate", "Moderate"), ("prob_modhigh", "Moderate-high")]:
        axes[1].plot(pred_df["t"], pred_df[col], linewidth=0.9, label=lab)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Posterior")
    axes[1].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def internal_metrics(y_true, raw, smooth):
    cm = confusion_matrix(y_true, smooth, labels=[0, 1, 2])
    return {
        "raw_accuracy": float(accuracy_score(y_true, raw)),
        "smoothed_accuracy": float(accuracy_score(y_true, smooth)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, smooth)),
        "low_recall": float(cm[0, 0] / max(cm[0].sum(), 1)),
        "moderate_recall": float(cm[1, 1] / max(cm[1].sum(), 1)),
        "modhigh_recall": float(cm[2, 2] / max(cm[2].sum(), 1)),
        "switch_rate": float(np.mean(smooth[1:] != smooth[:-1]) if len(smooth) > 1 else 0.0),
    }


def save_model(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def run_internal_loso(data_root, output_root, mode):
    _, pids, items = build_items(data_root, mode)
    rows = []
    for test_pid in pids:
        print(f"Part1 internal LOSO {mode} participant {test_pid}")
        train_items = [items[pid] for pid in pids if pid != test_pid]
        test_item = items[test_pid]
        model = train_intensity(train_items)
        raw, smooth, prob = predict_intensity(model, test_item)
        pred = pd.DataFrame(
            {
                "t": test_item["df"]["t"],
                "activity": test_item["df"]["activity"],
                "markers": test_item["df"]["activity"],
                "intensity_true": test_item["y"],
                "intensity_raw": raw,
                "intensity_pred": smooth,
                "prob_low": prob[:, 0],
                "prob_moderate": prob[:, 1],
                "prob_modhigh": prob[:, 2],
                "mu": np.where(smooth == 2, 0.012, 0.06),
            }
        )
        out_dir = Path(output_root) / "internal" / mode / f"participant_{test_pid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out_dir / "intensity_predictions.csv", index=False)
        plot_intensity(pred, out_dir / "intensity_predictions.png", f"Internal P{test_pid} {mode}", has_truth=True)
        rows.append({"mode": mode, "participant_id": test_pid, "n": len(pred), **internal_metrics(test_item["y"], raw, smooth)})
    summary = pd.DataFrame(rows)
    summary.to_csv(Path(output_root) / "internal" / mode / "intensity_loso_summary.csv", index=False)
    return summary


def train_final_model(data_root, output_root, mode):
    print(f"Part1 train final model {mode}")
    _, pids, items = build_items(data_root, mode)
    model = train_intensity([items[pid] for pid in pids])
    payload = {"mode": mode, "participant_ids": pids, "model": model}
    save_model(payload, Path(output_root) / "saved_models" / f"{mode}_best_intensity_model.pkl")
    return payload


def run_external_predictions(payload, output_root, datasets):
    mode = payload["mode"]
    model = payload["model"]
    rows = []
    for dataset_name, dataset_dir in datasets.items():
        for subject_id, csv_path in list_external_files(dataset_dir):
            print(f"Part1 external {mode} {dataset_name} {subject_id}")
            df = e4_file_to_dot6_dataframe(csv_path)
            df = restore_activity(df, csv_path)
            item = external_item_from_df(df, mode)
            raw, smooth, prob = predict_intensity({"mode": mode, "model": model}.get("model", model), item)
            pred = pd.DataFrame(
                {
                    "t": df["t"],
                    "unix_t": df["unix_t"],
                    "activity": df["activity"],
                    "markers": df["markers"],
                    "HR_bpm": df["HR_bpm"],
                    "velocity_DOT6": df["velocity_DOT6"],
                    "acceleration_DOT6": df["acceleration_DOT6"],
                    "gyration_DOT6": df["gyration_DOT6"],
                    "intensity_raw": raw,
                    "intensity_pred": smooth,
                    "prob_low": prob[:, 0],
                    "prob_moderate": prob[:, 1],
                    "prob_modhigh": prob[:, 2],
                    "mu": np.where(smooth == 2, 0.012, 0.06),
                }
            )
            out_dir = Path(output_root) / "external" / mode / dataset_name / subject_id
            out_dir.mkdir(parents=True, exist_ok=True)
            pred.to_csv(out_dir / "intensity_predictions.csv", index=False)
            plot_intensity(pred, out_dir / "intensity_predictions.png", f"{dataset_name} {subject_id} {mode}")
            rows.append(
                {
                    "mode": mode,
                    "dataset": dataset_name,
                    "subject_id": subject_id,
                    "n": len(pred),
                    "low_fraction": float(np.mean(smooth == 0)),
                    "moderate_fraction": float(np.mean(smooth == 1)),
                    "modhigh_fraction": float(np.mean(smooth == 2)),
                    "mean_prob_modhigh": float(prob[:, 2].mean()),
                    "mean_mu": float(pred["mu"].mean()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(Path(output_root) / "external" / mode / "external_intensity_summary.csv", index=False)
    return summary


def run(data_root=DATA_ROOT, output_root=OUTPUT_ROOT, external_datasets=EXTERNAL_DATASETS, modes=MODES_TO_RUN):
    Path(output_root).mkdir(parents=True, exist_ok=True)
    all_internal, all_external = [], []
    for mode in modes:
        all_internal.append(run_internal_loso(data_root, output_root, mode))
        payload = train_final_model(data_root, output_root, mode)
        all_external.append(run_external_predictions(payload, output_root, external_datasets))
    internal_summary = pd.concat(all_internal, ignore_index=True)
    external_summary = pd.concat(all_external, ignore_index=True)
    internal_summary.to_csv(Path(output_root) / "internal_intensity_summary.csv", index=False)
    external_summary.to_csv(Path(output_root) / "external_intensity_summary.csv", index=False)
    return internal_summary, external_summary
