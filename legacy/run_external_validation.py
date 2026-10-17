import argparse
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from compare_dot6_ekf_vo2 import dot789_mp_o2, regression_metrics, run_custom_ekf
from compare_sensor_intensity import merge_short_segments, rolling_mode, sensor_features, transition_matrix, viterbi, window_features
from hybrid_parts_common import activity_to_intensity, load_info, load_subject, participant_ids
from part3_pinn_hr import train_predict_pinn


SCENARIOS = ["300", "50", "80"]


def velocity_only_features(df, sensors):
    full = df.copy()
    for sensor in sensors:
        col = f"velocity_DOT{sensor}"
        if col not in full.columns:
            full[col] = 0.0
    out = pd.DataFrame(index=full.index)
    for sensor in sensors:
        v = full[f"velocity_DOT{sensor}"].astype(float)
        out[f"v{sensor}"] = v
        out[f"v{sensor}_abs"] = v.abs()
        out[f"v{sensor}_sq"] = v**2
        out[f"v{sensor}_diff"] = v.diff().fillna(0.0)
    v_norm = np.sqrt(sum(full[f"velocity_DOT{sensor}"].astype(float) ** 2 for sensor in sensors))
    out["velocity_norm"] = v_norm
    out["velocity_norm_sq"] = v_norm**2
    out["velocity_norm_diff"] = out["velocity_norm"].diff().fillna(0.0)
    for col in list(out.columns):
        base = out[col]
        out[f"{col}_ema_5s"] = base.ewm(span=5, adjust=False).mean()
        out[f"{col}_ema_15s"] = base.ewm(span=15, adjust=False).mean()
        out[f"{col}_ema_45s"] = base.ewm(span=45, adjust=False).mean()
        out[f"{col}_std_15s"] = base.rolling(15, min_periods=1).std().fillna(0.0)
        out[f"{col}_std_45s"] = base.rolling(45, min_periods=1).std().fillna(0.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def velocity_only_window_features(df, sensors, window=30):
    base = velocity_only_features(df, sensors)
    labels = df["activity"].map(activity_to_intensity).fillna(0).astype(int).to_numpy()
    rows, y, ranges = [], [], []
    arr = base.to_numpy(dtype=np.float32)
    velocity_norm = base["velocity_norm"].to_numpy(dtype=float)
    for start in range(0, len(df), window):
        end = min(start + window, len(df))
        block = arr[start:end]
        feats = []
        feats.extend(block.mean(axis=0))
        feats.extend(block.std(axis=0))
        feats.extend(block.min(axis=0))
        feats.extend(block.max(axis=0))
        sig = velocity_norm[start:end]
        if len(sig) < 4:
            feats.extend([0.0, 0.0, 0.0])
        else:
            sig = sig - np.mean(sig)
            spec = np.abs(np.fft.rfft(sig))
            freqs = np.fft.rfftfreq(len(sig), d=1.0)
            if len(spec) > 1:
                peak_i = 1 + np.argmax(spec[1:])
                peak_freq = freqs[peak_i]
                peak_power = spec[peak_i] / (spec.sum() + 1e-8)
            else:
                peak_freq, peak_power = 0.0, 0.0
            zc = np.mean(np.diff(np.signbit(sig)) != 0) if len(sig) > 1 else 0.0
            feats.extend([peak_freq, peak_power, zc])
        rows.append(feats)
        y.append(Counter(labels[start:end]).most_common(1)[0][0])
        ranges.append((start, end))
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=int), ranges


def intensity_to_mu(intensity):
    intensity = np.asarray(intensity, dtype=int)
    return np.where(intensity == 2, 0.02, 0.06).astype(float)


def estimate_rmr_vo2_ml_s(weight_kg, height_cm, age, sex):
    sex_l = str(sex).lower()
    s = 5.0 if sex_l.startswith("m") else -161.0
    kcal_day = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age + s
    return kcal_day / 1440.0 / 4.825 * 1000.0 / 60.0


def internal_training_subjects(data_root):
    info = load_info(data_root)
    pids = participant_ids(data_root, info)
    subjects = {pid: load_subject(data_root, pid) for pid in pids}
    return info, pids, subjects


def internal_dot6_calibration(subjects):
    v, a = [], []
    for df in subjects.values():
        v.extend(np.abs(df["velocity_DOT6"].astype(float).to_numpy()))
        a.extend(np.abs(df["acceleration_DOT6"].astype(float).to_numpy()))
    return {"velocity_q95": float(np.quantile(v, 0.95)), "acceleration_q95": float(np.quantile(a, 0.95))}


def external_df_from_e4(path, calibration):
    raw = pd.read_csv(path)
    v_raw = raw["total_velocity_mps"].astype(float).to_numpy()
    v_q95 = np.quantile(np.abs(v_raw), 0.95)
    velocity_scale = calibration["velocity_q95"] / max(v_q95, 1e-8)
    v_cal = v_raw * np.clip(velocity_scale, 0.2, 10.0)
    a_raw = np.r_[0.0, np.abs(np.diff(v_cal))]
    a_q95 = np.quantile(np.abs(a_raw), 0.95)
    acceleration_scale = calibration["acceleration_q95"] / max(a_q95, 1e-12)
    a_cal = a_raw * np.clip(acceleration_scale, 0.001, 1000.0)
    df = pd.DataFrame()
    df["t"] = np.arange(len(raw), dtype=int)
    df["unix_t"] = raw["t"].to_numpy()
    df["hr"] = raw["HR_bpm"].astype(float).interpolate(limit_direction="both").to_numpy()
    df["HR_bpm"] = df["hr"]
    df["vo2"] = 0.0
    df["vco2"] = 0.0
    df["activity"] = "external"
    df["markers"] = "external"
    df["velocity_DOT6"] = v_cal
    df["acceleration_DOT6"] = a_cal
    df["gyration_DOT6"] = 0.0
    df.attrs["velocity_scale"] = float(velocity_scale)
    df.attrs["acceleration_scale"] = float(acceleration_scale)
    return df


def external_df_from_e4_no_calibration(path):
    raw = pd.read_csv(path)
    df = pd.DataFrame()
    df["t"] = np.arange(len(raw), dtype=int)
    df["unix_t"] = raw["t"].to_numpy()
    df["hr"] = raw["HR_bpm"].astype(float).interpolate(limit_direction="both").to_numpy()
    df["HR_bpm"] = df["hr"]
    df["vo2"] = 0.0
    df["vco2"] = 0.0
    df["activity"] = "external"
    df["markers"] = "external"
    df["velocity_DOT6"] = raw["total_velocity_mps"].astype(float).to_numpy()
    df["acceleration_DOT6"] = np.r_[0.0, np.abs(np.diff(df["velocity_DOT6"].to_numpy(dtype=float)))]
    df["gyration_DOT6"] = 0.0
    return df


def train_full_dot6_intensity(subjects, info):
    train_x, train_y, seqs, win_x, win_y = [], [], [], [], []
    for pid, df in subjects.items():
        y = df["activity"].map(activity_to_intensity).to_numpy()
        train_x.append(velocity_only_features(df, [6]))
        train_y.append(y)
        seqs.append(y)
        wx, wy, _ = velocity_only_window_features(df, [6])
        win_x.append(wx)
        win_y.append(wy)
    X = pd.concat(train_x).to_numpy(dtype=np.float32)
    y = np.concatenate(train_y)
    scaler = StandardScaler().fit(X)
    models = [
        RandomForestClassifier(n_estimators=240, max_depth=14, min_samples_leaf=5, class_weight="balanced_subsample", random_state=101, n_jobs=1),
        ExtraTreesClassifier(n_estimators=360, min_samples_leaf=3, class_weight="balanced", random_state=102, n_jobs=1),
        HistGradientBoostingClassifier(max_iter=220, learning_rate=0.055, l2_regularization=0.08, random_state=103),
    ]
    Xs = scaler.transform(X)
    for model in models:
        model.fit(Xs, y)
    Xw = np.vstack(win_x)
    yw = np.concatenate(win_y)
    w_scaler = StandardScaler().fit(Xw)
    w_model = ExtraTreesClassifier(n_estimators=420, min_samples_leaf=2, class_weight="balanced", random_state=104, n_jobs=1)
    w_model.fit(w_scaler.transform(Xw), yw)
    trans = transition_matrix(seqs)
    return {"scaler": scaler, "models": models, "w_scaler": w_scaler, "w_model": w_model, "trans": trans}


def predict_external_intensity(df, info_row, intensity_model):
    Xt = velocity_only_features(df, [6]).to_numpy(dtype=np.float32)
    probs = []
    for model in intensity_model["models"]:
        p = model.predict_proba(intensity_model["scaler"].transform(Xt))
        full = np.zeros((len(df), 3), dtype=float)
        for i, cls in enumerate(model.classes_):
            full[:, int(cls)] = p[:, i]
        probs.append(full)
    prob_point = np.mean(probs, axis=0)
    wx, _, ranges = velocity_only_window_features(df, [6])
    wp = intensity_model["w_model"].predict_proba(intensity_model["w_scaler"].transform(wx))
    prob_win = np.zeros((len(df), 3), dtype=float)
    for ri, (start, end) in enumerate(ranges):
        full = np.zeros(3, dtype=float)
        for j, cls in enumerate(intensity_model["w_model"].classes_):
            full[int(cls)] = wp[ri, j]
        prob_win[start:end] = full
    prob = 0.45 * prob_point + 0.55 * prob_win
    raw = prob.argmax(axis=1)
    smooth = merge_short_segments(rolling_mode(viterbi(prob, intensity_model["trans"]), 41), 30)
    if getattr(predict_external_intensity, "use_percentile_fallback", False) and np.mean(smooth > 0) < 0.01 and np.mean(prob[:, 0]) > 0.75:
        # External E4 wrist velocities can be shifted far from the DOT6 training
        # distribution. If the supervised model collapses to all-low, use an
        # unsupervised within-subject intensity prior matching internal class
        # proportions (low/moderate/mod-high ~= 53/32/15%).
        motion = pd.Series(np.abs(df["velocity_DOT6"].to_numpy(dtype=float))).ewm(span=30, adjust=False).mean().to_numpy()
        q_low, q_mod = np.quantile(motion, [0.53, 0.847])
        raw = np.where(motion >= q_mod, 2, np.where(motion >= q_low, 1, 0)).astype(int)
        smooth = merge_short_segments(rolling_mode(raw, 41), 30)
        prob = np.full((len(df), 3), 0.05, dtype=float)
        prob[np.arange(len(df)), smooth] = 0.90
    mu = 0.02 * prob[:, 2] + 0.06 * (1.0 - prob[:, 2])
    return pd.DataFrame(
        {
            "t": df["t"],
            "intensity_raw": raw,
            "intensity_pred": smooth,
            "prob_low": prob[:, 0],
            "prob_moderate": prob[:, 1],
            "prob_modhigh": prob[:, 2],
            "mu": mu,
        }
    )


predict_external_intensity.use_percentile_fallback = False


def train_full_dot6_teacher_proxy(subjects, info, sensor_root):
    xs, ys = [], []
    for pid, df in subjects.items():
        intensity789 = df["activity"].map(activity_to_intensity).to_numpy(dtype=int)
        mu789 = intensity_to_mu(intensity789)
        teacher = dot789_mp_o2(df, info.loc[pid], mu789)
        xs.append(velocity_only_features(df, [6]))
        ys.append(np.log1p(np.clip(teacher, 0.0, None)))
    X = pd.concat(xs).to_numpy(dtype=np.float32)
    y = np.concatenate(ys)
    scaler = StandardScaler().fit(X)
    model = HistGradientBoostingRegressor(max_iter=280, learning_rate=0.045, l2_regularization=0.05, random_state=181)
    model.fit(scaler.transform(X), y)
    return scaler, model


def predict_external_mp_o2(df, info_row, scaler, model):
    X = velocity_only_features(df, [6]).to_numpy(dtype=np.float32)
    pred = np.expm1(model.predict(scaler.transform(X)))
    pred = pd.Series(np.clip(pred, 0.0, 0.06)).ewm(span=10, adjust=False).mean().to_numpy(dtype=float)
    return pred


def info_row_from_external(dataset, subject_id, demo_row, internal_info):
    sex_means = internal_info.assign(_sex=internal_info["Sex"].str.lower().str[0]).groupby("_sex")["Muscle (%)"].mean()
    if dataset == "WEEE":
        sex = "Male" if str(demo_row["Gender"]).lower().startswith("m") else "Female"
        weight = float(demo_row["Weight"])
        height = float(demo_row["Height"])
        age = float(demo_row["Age"])
    else:
        sex = "Male" if str(demo_row["gender"]).lower().startswith("m") else "Female"
        weight = float(demo_row["weight_kg"])
        height = float(demo_row["height_cm"])
        age = float(demo_row["age"])
    sex_key = "m" if sex.lower().startswith("m") else "f"
    muscle = float(sex_means.get(sex_key, internal_info["Muscle (%)"].mean()))
    rmr = estimate_rmr_vo2_ml_s(weight, height, age, sex)
    return pd.Series(
        {
            "Participant_ID": subject_id,
            "Weight (kg)": weight,
            "Height (cm)": height,
            "Sex": sex.lower(),
            "Age (years)": age,
            "Muscle (%)": muscle,
            "RMR VO2": rmr,
        }
    )


def load_external_dataset(dataset_dir, dataset_name, internal_info):
    if dataset_name == "WEEE":
        demo = pd.read_csv(os.path.join(dataset_dir, "Demographics.csv")).set_index("Participant")
        files = sorted([f for f in os.listdir(dataset_dir) if f.endswith("_E4_ACC_HR_1Hz.csv")])
        items = []
        for file_name in files:
            sid = file_name.split("_")[0]
            if sid not in demo.index:
                continue
            items.append((sid, os.path.join(dataset_dir, file_name), info_row_from_external("WEEE", sid, demo.loc[sid], internal_info)))
        return items
    demo = pd.read_csv(os.path.join(dataset_dir, "subject_info_summary.csv")).set_index("subject_id")
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith("_E4_ACC_HR_1Hz.csv")])
    items = []
    for file_name in files:
        sid = file_name.split("_")[0]
        if sid not in demo.index:
            continue
        items.append((sid, os.path.join(dataset_dir, file_name), info_row_from_external("PPG_DaLiA", sid, demo.loc[sid], internal_info)))
    return items


def plot_external_series(df, y, ylabel, path, title):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(df["t"], y, color="#0072B2", linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_external_part3(df, ekf, out_dir, subject_id, dataset_name, epochs, phys_weight):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        pred, whitebox, train, test, train_order, test_order, metrics, whitebox_metrics, lambdas, history = train_predict_pinn(
            df, ekf["ekf_vo2_ml_s"].to_numpy(), "chronological", scenario, epochs=epochs, phys_weight=phys_weight
        )
        row = {
            "dataset": dataset_name,
            "subject_id": subject_id,
            "split_mode": "chronological",
            "scenario": scenario,
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "phys_weight": float(phys_weight),
            **metrics,
            **{f"whitebox_{k}": v for k, v in whitebox_metrics.items()},
            **{f"lambda{i + 1}": lambdas[i] for i in range(6)},
        }
        rows.append(row)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, f"part3_history_chronological_{scenario}.csv"), index=False)
        pd.DataFrame([{f"lambda{i + 1}": lambdas[i] for i in range(6)}]).to_csv(os.path.join(out_dir, f"part3_final_lambdas_chronological_{scenario}.csv"), index=False)
        pred_df = pd.DataFrame(
            {
                "original_index": test_order,
                "t": df.loc[test_order, "t"].to_numpy(),
                "hr_true": df.loc[test_order, "hr"].to_numpy(),
                "hr_pred": pred[test_order],
                "hr_whitebox": whitebox[test_order],
                "ekf_vo2_ml_s": ekf.loc[test_order, "ekf_vo2_ml_s"].to_numpy(),
            }
        )
        pred_df.to_csv(os.path.join(out_dir, f"part3_predictions_chronological_{scenario}.csv"), index=False)
        plot_hr_external(pred_df, os.path.join(out_dir, f"part3_hr_chronological_{scenario}.png"), f"{dataset_name} {subject_id} chronological {scenario}")
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "part3_hr_summary.csv"), index=False)
    return summary


def plot_hr_external(pred_df, path, title):
    fig, ax = plt.subplots(figsize=(14, 4.8))
    x = np.arange(len(pred_df))
    ax.plot(x, pred_df["hr_true"], color="black", linewidth=1.15, label="True HR")
    ax.plot(x, pred_df["hr_pred"], color="#0072B2", linewidth=1.05, label="PINN HR")
    ax.plot(x, pred_df["hr_whitebox"], color="#009E73", linewidth=0.95, alpha=0.8, label="White-box HR")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("HR (bpm)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run(output_root, data_root, sensor_root, weee_dir, ppg_dir, epochs, phys_weight):
    info, pids, subjects = internal_training_subjects(data_root)
    calibration = internal_dot6_calibration(subjects)
    os.makedirs(output_root, exist_ok=True)
    print("Training full DOT6 intensity model on internal Data")
    intensity_model = train_full_dot6_intensity(subjects, info)
    print("Training full DOT6 teacher-scaled metabolic proxy model on internal Data")
    proxy_scaler, proxy_model = train_full_dot6_teacher_proxy(subjects, info, sensor_root)
    all_part1, all_part2, all_part3, all_demo = [], [], [], []
    for dataset_name, dataset_dir in [("WEEE", weee_dir), ("PPG_DaLiA", ppg_dir)]:
        for subject_id, path, info_row in load_external_dataset(dataset_dir, dataset_name, info):
            print(f"Running {dataset_name} {subject_id}")
            subj_dir = os.path.join(output_root, dataset_name, subject_id)
            part1_dir = os.path.join(subj_dir, "part1_external_dot6_intensity")
            part2_dir = os.path.join(subj_dir, "part2_external_ekf_vo2")
            part3_dir = os.path.join(subj_dir, "part3_external_pinn_hr")
            os.makedirs(part1_dir, exist_ok=True)
            os.makedirs(part2_dir, exist_ok=True)
            df = external_df_from_e4_no_calibration(path)
            part1 = predict_external_intensity(df, info_row, intensity_model)
            part1.to_csv(os.path.join(part1_dir, "part1_external_intensity_mu_timeseries.csv"), index=False)
            all_part1.append(
                {
                    "dataset": dataset_name,
                    "subject_id": subject_id,
                    "n": len(df),
                    "pred_low_fraction": float(np.mean(part1["intensity_pred"] == 0)),
                    "pred_moderate_fraction": float(np.mean(part1["intensity_pred"] == 1)),
                    "pred_modhigh_fraction": float(np.mean(part1["intensity_pred"] == 2)),
                    "mean_mu": float(part1["mu"].mean()),
                    "velocity_scale": float(df.attrs.get("velocity_scale", 1.0)),
                    "acceleration_scale": float(df.attrs.get("acceleration_scale", 1.0)),
                }
            )
            mp_o2 = predict_external_mp_o2(df, info_row, proxy_scaler, proxy_model)
            ekf = run_custom_ekf(df, info_row, part1["intensity_pred"].to_numpy(dtype=int), mp_o2, r_value=0.005, n_value=0.01)
            ekf["mu"] = part1["mu"].to_numpy(dtype=float)
            ekf.to_csv(os.path.join(part2_dir, "part2_external_ekf_outputs.csv"), index=False)
            plot_external_series(df, ekf["ekf_vo2_ml_s"], "EKF VO2 (ml/s)", os.path.join(part2_dir, "part2_external_ekf_vo2.png"), f"{dataset_name} {subject_id} EKF VO2")
            all_part2.append(
                {
                    "dataset": dataset_name,
                    "subject_id": subject_id,
                    "ekf_vo2_min": float(ekf["ekf_vo2_ml_s"].min()),
                    "ekf_vo2_mean": float(ekf["ekf_vo2_ml_s"].mean()),
                    "ekf_vo2_max": float(ekf["ekf_vo2_ml_s"].max()),
                    "mp_o2_obs_mean_l_min": float(ekf["mp_o2_obs"].mean()),
                    "hr_mean": float(df["hr"].mean()),
                    "hr_min": float(df["hr"].min()),
                    "hr_max": float(df["hr"].max()),
                }
            )
            demo_row = info_row.to_dict()
            demo_row.update({"dataset": dataset_name, "subject_id": subject_id})
            all_demo.append(demo_row)
            part3 = run_external_part3(df, ekf, part3_dir, subject_id, dataset_name, epochs, phys_weight)
            all_part3.append(part3)
    part1_df = pd.DataFrame(all_part1)
    part2_df = pd.DataFrame(all_part2)
    part3_df = pd.concat(all_part3, ignore_index=True)
    demo_df = pd.DataFrame(all_demo)
    part1_df.to_csv(os.path.join(output_root, "external_part1_intensity_summary.csv"), index=False)
    part2_df.to_csv(os.path.join(output_root, "external_part2_ekf_vo2_summary.csv"), index=False)
    part3_df.to_csv(os.path.join(output_root, "external_part3_hr_summary.csv"), index=False)
    demo_df.to_csv(os.path.join(output_root, "external_demographics_used.csv"), index=False)
    write_external_manual(output_root, part1_df, part2_df, part3_df, epochs, phys_weight)


def write_external_manual(output_root, part1_df, part2_df, part3_df, epochs, phys_weight):
    lines = []
    lines.append("# External validation: WEEE and PPG_DaLiA\n\n")
    lines.append("## Pipeline\n\n")
    lines.append("Internal `Data` subjects were used to train the DOT6 wrist-intensity model and the DOT6-to-DOT789 teacher-scaled metabolic-proxy model. The current external setting restricts Part1 and Part2 to velocity-only IMU features: internal DOT6 velocity-derived features are used as model inputs, and the proxy teacher target is generated from internal DOT789 velocity. External WEEE/PPG_DaLiA subjects were not used for Part1/Part2 fitting. At test time, only E4 wrist total velocity, HR, and demographics were used.\n\n")
    lines.append("Part2 has no VO2 ground truth in the external datasets, so EKF VO2 is summarized by range and used as the input to Part3. Part3 uses chronological calibration only: 300 s, 50%, and 80% train splits.\n\n")
    lines.append(f"PINN epochs: {epochs}; physics weight: {phys_weight}.\n\n")
    lines.append("## Part1 intensity summary\n\n")
    lines.append(part1_df.groupby("dataset")[["pred_low_fraction", "pred_moderate_fraction", "pred_modhigh_fraction", "mean_mu"]].agg(["median", "mean"]).round(4).to_markdown())
    lines.append("\n\n## Part2 EKF VO2 summary\n\n")
    lines.append(part2_df.groupby("dataset")[["ekf_vo2_min", "ekf_vo2_mean", "ekf_vo2_max", "hr_mean"]].agg(["median", "mean", "min", "max"]).round(4).to_markdown())
    lines.append("\n\n## Part3 HR summary\n\n")
    lines.append(part3_df.groupby(["dataset", "scenario"])[["rmse", "mae", "corr", "r2", "whitebox_rmse", "whitebox_mae", "whitebox_corr"]].agg(["median", "mean"]).round(4).to_markdown())
    lines.append("\n\n## Notes\n\n")
    lines.append("RMR VO2 was estimated from demographics using a resting energy equation when not directly available; muscle percentage was imputed from internal Data sex-specific averages. These assumptions are saved in `external_demographics_used.csv` and should be reported as limitations.\n")
    with open(os.path.join(output_root, "external_validation_manual.md"), "w", encoding="utf-8") as f:
        f.write("".join(lines))
