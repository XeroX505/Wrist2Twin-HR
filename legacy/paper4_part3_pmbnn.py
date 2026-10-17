import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from hybrid_parts_common import load_subject, participant_ids, load_info, regression_metrics
from part3_pinn_hr import PINNHR, make_features, physics_de_loss, reconstructed_by_segments
from train_saved_intensity_models_external import e4_file_to_dot6_dataframe, list_external_files


# ============================================================
# MANUAL PATH CONFIG
# ============================================================
WORKSPACE = os.environ.get("WRIST2TWIN_WORKSPACE", os.getcwd())
DATA_ROOT = os.path.join(WORKSPACE, "Data")
PART2_ROOT = os.path.join(WORKSPACE, "outputs_paper4_full_pipeline_5datasets", "part2_ekf")
OUTPUT_ROOT = os.path.join(WORKSPACE, "outputs_paper4_full_pipeline_5datasets", "part3_pmbnn")

EXTERNAL_DATASETS = {
    "PPG_DaLiA": os.path.join(WORKSPACE, "PPG_DaLiA"),
    "Stress_Aerobic": os.path.join(WORKSPACE, "Stress_Aerobic"),
    "BITS2_ADL_no_existing_health_issues": os.path.join(WORKSPACE, "BITS2_ADL_no_existing_health_issues"),
    "BITS2_ADL_other_health_conditions": os.path.join(WORKSPACE, "BITS2_ADL_other_health_conditions"),
    "PAMAP2": os.path.join(WORKSPACE, "pamap2"),
}

MODES_TO_RUN = ["e4_compatible_no_gyr", "dot_wrist_with_gyr"]
SCENARIOS = ["300", "50", "80"]
PHYS_WEIGHT = 10000.0
EPOCHS = 5000
# ============================================================


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


def split_per_activity_exact(df, scenario):
    n = len(df)
    train = np.zeros(n, dtype=bool)
    train_segments, test_segments = [], []
    groups = [(name, np.asarray(list(idx), dtype=int)) for name, idx in df.groupby("activity", sort=False).groups.items()]
    if scenario == "50":
        desired_total = int(round(0.5 * n))
    elif scenario == "80":
        desired_total = int(round(0.8 * n))
    elif scenario == "300":
        desired_total = min(300, max(1, n - 1))
    else:
        raise ValueError(scenario)
    desired_total = min(desired_total, max(1, n - 1))
    ratio = desired_total / max(n, 1)
    cuts, fractional, current_total = {}, [], 0
    for name, pos in groups:
        if len(pos) < 3:
            cut = min(len(pos), max(0, desired_total - current_total))
            frac = 0.0
        else:
            raw = ratio * len(pos)
            cut = min(max(1, int(np.floor(raw))), len(pos) - 1)
            frac = raw - np.floor(raw)
        cuts[name] = cut
        current_total += cut
        fractional.append((frac, name))
    for _, name in sorted(fractional, reverse=True):
        if current_total >= desired_total:
            break
        pos_len = len(dict(groups)[name])
        if cuts[name] < pos_len - 1:
            cuts[name] += 1
            current_total += 1
    for _, name in sorted(fractional):
        if current_total <= desired_total:
            break
        if cuts[name] > 1:
            cuts[name] -= 1
            current_total -= 1
    for name, pos in groups:
        cut = min(cuts[name], len(pos))
        if len(pos) >= 3:
            cut = min(max(1, cut), len(pos) - 1)
        train[pos[:cut]] = True
        if cut > 0:
            train_segments.append(pos[:cut])
        if cut < len(pos):
            test_segments.append(pos[cut:])
    test = ~train
    train_order = np.concatenate([s for s in train_segments if len(s) > 0]) if train_segments else np.array([], dtype=int)
    test_order = np.concatenate([s for s in test_segments if len(s) > 0]) if test_segments else np.array([], dtype=int)
    return train, test, train_segments, test_segments, train_order, test_order


def train_predict_pmbnn_per_activity(df, ekf_vo2, scenario, epochs=EPOCHS, lr=5e-3, phys_weight=PHYS_WEIGHT):
    train, test, train_segments, test_segments, train_order, test_order = split_per_activity_exact(df, scenario)
    X = make_features(ekf_vo2)
    scaler = StandardScaler().fit(X[train])
    xs = scaler.transform(X).astype(np.float32)
    hr = df["hr"].to_numpy(dtype=np.float32)
    xt = torch.from_numpy(xs)
    hr_t = torch.from_numpy(hr)
    vo2_l_min_t = torch.from_numpy((np.asarray(ekf_vo2, dtype=np.float32) * 60.0 / 1000.0).astype(np.float32))
    train_idx = torch.from_numpy(np.where(train)[0])
    model = PINNHR()
    with torch.no_grad():
        model.net[-1].bias.fill_(float(hr[train].mean()))
    opt = torch.optim.RMSprop(
        [
            {"params": [p for name, p in model.named_parameters() if name != "raw_lambda"], "lr": lr},
            {"params": [model.raw_lambda], "lr": lr * 5.0},
        ],
        alpha=0.99,
        weight_decay=0.0,
    )
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred_hr_t = model(xt)
        mse = F.mse_loss(pred_hr_t[train_idx], hr_t[train_idx])
        lam = model.lambdas()
        de = physics_de_loss(pred_hr_t, vo2_l_min_t, train_segments, lam)
        loss = mse + phys_weight * de
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if epoch % 25 == 0 or epoch == epochs - 1:
            lam_values = [float(v) for v in model.lambdas().detach().numpy()]
            history.append({"epoch": epoch, "mse": float(mse.detach()), "de_loss": float(de.detach()), "phys_weight": phys_weight, "loss": float(loss.detach()), **{f"lambda{i + 1}": lam_values[i] for i in range(6)}})
    model.eval()
    with torch.no_grad():
        pred_hr = model(xt).numpy().astype(np.float32)
    lambdas = [float(v) for v in model.lambdas().detach().numpy()]
    vo2_l_min = np.asarray(ekf_vo2, dtype=np.float32) * 60.0 / 1000.0
    whitebox_hr = np.full(len(hr), np.nan, dtype=np.float32)
    whitebox_train = reconstructed_by_segments(vo2_l_min, hr, train_segments, lambdas)
    whitebox_test = reconstructed_by_segments(vo2_l_min, hr, test_segments, lambdas)
    whitebox_hr[train] = whitebox_train[train]
    whitebox_hr[test] = whitebox_test[test]
    metrics = regression_metrics(hr[test], pred_hr[test])
    whitebox_metrics = regression_metrics(hr[test], whitebox_hr[test])
    return pred_hr, whitebox_hr, train, test, train_order, test_order, metrics, whitebox_metrics, lambdas, history


def plot_hr(df, pred_df, path, title):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    x = np.arange(len(pred_df))
    ax.plot(x, pred_df["hr_true"], color="black", linewidth=1.0, label="True HR")
    ax.plot(x, pred_df["hr_pred"], color="#0072B2", linewidth=0.95, label="PMBNN HR")
    ax.plot(x, pred_df["hr_whitebox"], color="#009E73", linewidth=0.85, alpha=0.8, label="White-box HR")
    last = None
    for x_i, idx in enumerate(pred_df["original_index"].astype(int).to_numpy()):
        act = df.iloc[idx]["activity"]
        if act != last:
            ax.axvline(x_i, color="0.75", linestyle="--", linewidth=0.6, alpha=0.55)
            last = act
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("HR (bpm)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_one(df, ekf_vo2, output_dir, subject_label, mode, dataset, subject_id, epochs):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        print(f"Part3 {mode} {subject_label} per_activity {scenario}")
        pred, whitebox, train, test, train_order, test_order, metrics, whitebox_metrics, lambdas, history = train_predict_pmbnn_per_activity(df, ekf_vo2, scenario, epochs=epochs)
        row = {
            "mode": mode,
            "dataset": dataset,
            "subject_id": subject_id,
            "subject": subject_label,
            "split_mode": "per_activity",
            "scenario": scenario,
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "phys_weight": PHYS_WEIGHT,
            **metrics,
            **{f"whitebox_{k}": v for k, v in whitebox_metrics.items()},
            **{f"lambda{i + 1}": lambdas[i] for i in range(6)},
        }
        rows.append(row)
        pd.DataFrame(history).to_csv(Path(output_dir) / f"part3_history_per_activity_{scenario}.csv", index=False)
        pd.DataFrame([{f"lambda{i + 1}": lambdas[i] for i in range(6)}]).to_csv(Path(output_dir) / f"part3_final_lambdas_per_activity_{scenario}.csv", index=False)
        pred_df = pd.DataFrame(
            {
                "original_index": test_order,
                "t": df.loc[test_order, "t"].to_numpy(),
                "activity": df.loc[test_order, "activity"].to_numpy(),
                "hr_true": df.loc[test_order, "hr"].to_numpy(),
                "hr_pred": pred[test_order],
                "hr_whitebox": whitebox[test_order],
                "ekf_vo2_ml_s": np.asarray(ekf_vo2)[test_order],
            }
        )
        pred_df.to_csv(Path(output_dir) / f"part3_predictions_per_activity_{scenario}.csv", index=False)
        train_df = pd.DataFrame(
            {
                "original_index": train_order,
                "t": df.loc[train_order, "t"].to_numpy(),
                "activity": df.loc[train_order, "activity"].to_numpy(),
                "hr_true": df.loc[train_order, "hr"].to_numpy(),
                "ekf_vo2_ml_s": np.asarray(ekf_vo2)[train_order],
            }
        )
        train_df.to_csv(Path(output_dir) / f"part3_train_per_activity_{scenario}.csv", index=False)
        plot_hr(df, pred_df, Path(output_dir) / f"part3_hr_per_activity_{scenario}.png", f"{subject_label} per-activity {scenario}")
    summary = pd.DataFrame(rows)
    summary.to_csv(Path(output_dir) / "part3_summary.csv", index=False)
    return summary


def run_internal_mode(data_root, part2_root, output_root, mode, epochs):
    info = load_info(data_root)
    pids = participant_ids(data_root, info)
    rows = []
    for pid in pids:
        df = load_subject(data_root, pid).reset_index(drop=True)
        ekf_path = Path(part2_root) / mode / "internal" / f"participant_{pid}" / "part2_dot789_ekf" / "part2_ekf_outputs.csv"
        ekf = pd.read_csv(ekf_path)
        n = min(len(df), len(ekf))
        df = df.iloc[:n].reset_index(drop=True)
        ekf = ekf.iloc[:n].reset_index(drop=True)
        out_dir = Path(output_root) / mode / "internal" / f"participant_{pid}" / "part3_pmbnn"
        rows.append(run_one(df, ekf["ekf_vo2_ml_s"].to_numpy(dtype=float), out_dir, f"internal_P{pid}", mode, "internal", pid, epochs))
    return pd.concat(rows, ignore_index=True)


def run_external_mode(part2_root, output_root, mode, datasets, epochs):
    rows = []
    for dataset_name, dataset_dir in datasets.items():
        for subject_id, csv_path in list_external_files(dataset_dir):
            df = e4_file_to_dot6_dataframe(csv_path).reset_index(drop=True)
            df = restore_activity(df, csv_path)
            ekf_path = Path(part2_root) / mode / "external" / dataset_name / subject_id / "part2_dot789_teacher_ekf" / "part2_ekf_outputs.csv"
            ekf = pd.read_csv(ekf_path)
            n = min(len(df), len(ekf))
            df = df.iloc[:n].reset_index(drop=True)
            ekf = ekf.iloc[:n].reset_index(drop=True)
            out_dir = Path(output_root) / mode / "external" / dataset_name / subject_id / "part3_pmbnn"
            rows.append(run_one(df, ekf["ekf_vo2_ml_s"].to_numpy(dtype=float), out_dir, f"{dataset_name}_{subject_id}", mode, dataset_name, subject_id, epochs))
    return pd.concat(rows, ignore_index=True)


def run(data_root=DATA_ROOT, part2_root=PART2_ROOT, output_root=OUTPUT_ROOT, external_datasets=EXTERNAL_DATASETS, modes=MODES_TO_RUN, epochs=EPOCHS):
    Path(output_root).mkdir(parents=True, exist_ok=True)
    internal_all, external_all = [], []
    for mode in modes:
        internal_all.append(run_internal_mode(data_root, part2_root, output_root, mode, epochs))
        external_all.append(run_external_mode(part2_root, output_root, mode, external_datasets, epochs))
    internal = pd.concat(internal_all, ignore_index=True)
    external = pd.concat(external_all, ignore_index=True)
    internal.to_csv(Path(output_root) / "internal_part3_summary.csv", index=False)
    external.to_csv(Path(output_root) / "external_part3_summary.csv", index=False)
    return internal, external
