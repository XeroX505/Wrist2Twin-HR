import itertools
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from compare_dot6_ekf_vo2 import dot789_mp_o2, regression_metrics, run_custom_ekf
from compare_sensor_intensity import merge_short_segments, rolling_mode, transition_matrix, viterbi
from hybrid_parts_common import activity_to_intensity, load_info, load_subject, participant_ids
from run_external_validation import intensity_to_mu


MODES = {
    "e4_compatible_no_gyr": {
        "signals": [
            "velocity_DOT6",
            "acceleration_DOT6",
            "linear_velocity_x_DOT6",
            "linear_velocity_y_DOT6",
            "linear_velocity_z_DOT6",
            "filtered_x_DOT6",
            "filtered_y_DOT6",
            "filtered_z_DOT6",
        ]
    },
    "dot_wrist_with_gyr": {
        "signals": [
            "velocity_DOT6",
            "acceleration_DOT6",
            "gyration_DOT6",
            "linear_velocity_x_DOT6",
            "linear_velocity_y_DOT6",
            "linear_velocity_z_DOT6",
            "filtered_x_DOT6",
            "filtered_y_DOT6",
            "filtered_z_DOT6",
            "filtered_gyr_x_DOT6",
            "filtered_gyr_y_DOT6",
            "filtered_gyr_z_DOT6",
        ]
    },
}


def robust_scale(sig):
    sig = np.asarray(sig, dtype=float)
    q = np.quantile(np.abs(sig), 0.95)
    return max(float(q), 1e-6)


def wrist_features(df, mode, normalize=True):
    cols = MODES[mode]["signals"]
    data = {}
    for col in cols:
        sig = df[col].astype(float).to_numpy() if col in df.columns else np.zeros(len(df))
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        clip = max(np.quantile(np.abs(sig), 0.999), 1.0)
        sig = np.clip(sig, -clip, clip)
        scale = robust_scale(sig) if normalize else 1.0
        base = {
            col: sig,
            f"{col}_abs": np.abs(sig),
            f"{col}_sq": sig**2,
            f"{col}_logabs": np.log1p(np.abs(sig)),
            f"{col}_norm": sig / scale,
            f"{col}_abs_norm": np.abs(sig) / scale,
            f"{col}_diff": np.r_[0.0, np.diff(sig)],
            f"{col}_absdiff": np.r_[0.0, np.abs(np.diff(sig))],
        }
        for name, val in base.items():
            data[name] = val

    if all(c in df.columns for c in ["linear_velocity_x_DOT6", "linear_velocity_y_DOT6", "linear_velocity_z_DOT6"]):
        lv = np.sqrt(sum(df[f"linear_velocity_{axis}_DOT6"].astype(float).to_numpy() ** 2 for axis in ["x", "y", "z"]))
        data["linear_velocity_axis_norm"] = lv
        data["linear_velocity_axis_norm_sq"] = lv**2
    if all(c in df.columns for c in ["filtered_x_DOT6", "filtered_y_DOT6", "filtered_z_DOT6"]):
        av = np.sqrt(sum(df[f"filtered_{axis}_DOT6"].astype(float).to_numpy() ** 2 for axis in ["x", "y", "z"]))
        data["acc_axis_norm"] = av
        data["acc_axis_norm_sq"] = av**2
    if mode == "dot_wrist_with_gyr" and all(c in df.columns for c in ["filtered_gyr_x_DOT6", "filtered_gyr_y_DOT6", "filtered_gyr_z_DOT6"]):
        gv = np.sqrt(sum(df[f"filtered_gyr_{axis}_DOT6"].astype(float).to_numpy() ** 2 for axis in ["x", "y", "z"]))
        data["gyr_axis_norm"] = gv
        data["gyr_axis_norm_sq"] = gv**2

    base_data = dict(data)
    for col, values in base_data.items():
        s = pd.Series(np.asarray(values, dtype=float))
        for span in [3, 5, 10, 20, 45, 90]:
            data[f"{col}_ema_{span}s"] = s.ewm(span=span, adjust=False).mean().to_numpy()
        for win in [15, 45, 90]:
            r = s.rolling(win, min_periods=1)
            data[f"{col}_mean_{win}s"] = r.mean().to_numpy()
            data[f"{col}_std_{win}s"] = r.std().fillna(0.0).to_numpy()
            data[f"{col}_max_{win}s"] = r.max().to_numpy()
    out = pd.DataFrame(data, index=df.index)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.clip(lower=-1e6, upper=1e6)


def window_features(base, labels, window=30):
    rows, y, ranges = [], [], []
    arr = base.to_numpy(dtype=np.float32)
    driver_col = "velocity_DOT6_abs_norm" if "velocity_DOT6_abs_norm" in base else base.columns[0]
    driver = base[driver_col].to_numpy(dtype=float)
    for start in range(0, len(base), window):
        end = min(start + window, len(base))
        block = arr[start:end]
        feats = []
        feats.extend(block.mean(axis=0))
        feats.extend(block.std(axis=0))
        feats.extend(block.min(axis=0))
        feats.extend(block.max(axis=0))
        sig = driver[start:end] - np.mean(driver[start:end])
        if len(sig) >= 4:
            spec = np.abs(np.fft.rfft(sig))
            freqs = np.fft.rfftfreq(len(sig), d=1.0)
            peak_i = 1 + np.argmax(spec[1:]) if len(spec) > 1 else 0
            peak_freq = freqs[peak_i] if peak_i else 0.0
            peak_power = spec[peak_i] / (spec.sum() + 1e-8) if peak_i else 0.0
            zc = np.mean(np.diff(np.signbit(sig)) != 0)
        else:
            peak_freq, peak_power, zc = 0.0, 0.0, 0.0
        feats.extend([peak_freq, peak_power, zc])
        rows.append(feats)
        y.append(Counter(labels[start:end]).most_common(1)[0][0])
        ranges.append((start, end))
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=int), ranges


def build_items(data_root, mode):
    info = load_info(data_root)
    pids = participant_ids(data_root, info)
    subjects = {pid: load_subject(data_root, pid) for pid in pids}
    items = {}
    for pid, df in subjects.items():
        y = df["activity"].map(activity_to_intensity).to_numpy(dtype=int)
        feat = wrist_features(df, mode)
        wx, wy, ranges = window_features(feat, y)
        mp = dot789_mp_o2(df, info.loc[pid], intensity_to_mu(y))
        extra_vo2 = np.maximum(df["vo2"].to_numpy(dtype=float) - float(info.loc[pid, "RMR VO2"]), 0.0) / 1000.0
        items[pid] = {
            "pid": pid,
            "df": df,
            "info": info.loc[pid],
            "y": y,
            "feat": feat,
            "wx": wx,
            "wy": wy,
            "ranges": ranges,
            "mp_dot789": mp,
            "extra_vo2": extra_vo2,
        }
    return info, pids, items


def train_point_models(train_items):
    X = pd.concat([it["feat"] for it in train_items]).to_numpy(dtype=np.float32)
    y = np.concatenate([it["y"] for it in train_items])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    models = [
        ExtraTreesClassifier(n_estimators=900, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", random_state=401, n_jobs=1),
        RandomForestClassifier(n_estimators=650, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", random_state=402, n_jobs=1),
        HistGradientBoostingClassifier(max_iter=450, learning_rate=0.03, l2_regularization=0.02, random_state=403),
    ]
    for model in models:
        model.fit(Xs, y)
    return scaler, models


def train_window_model(train_items):
    X = np.vstack([it["wx"] for it in train_items])
    y = np.concatenate([it["wy"] for it in train_items])
    scaler = StandardScaler().fit(X)
    model = ExtraTreesClassifier(n_estimators=900, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", random_state=404, n_jobs=1)
    model.fit(scaler.transform(X), y)
    return scaler, model


def point_prob(item, scaler, models):
    X = scaler.transform(item["feat"].to_numpy(dtype=np.float32))
    probs = []
    for model in models:
        p = model.predict_proba(X)
        full = np.zeros((len(item["df"]), 3), dtype=float)
        for j, cls in enumerate(model.classes_):
            full[:, int(cls)] = p[:, j]
        probs.append(full)
    return np.mean(probs, axis=0)


def window_prob(item, scaler, model):
    wp = model.predict_proba(scaler.transform(item["wx"]))
    prob = np.zeros((len(item["df"]), 3), dtype=float)
    for i, (start, end) in enumerate(item["ranges"]):
        full = np.zeros(3, dtype=float)
        for j, cls in enumerate(model.classes_):
            full[int(cls)] = wp[i, j]
        prob[start:end] = full
    return prob


def smooth_prob(prob, trans, weights=(1.0, 1.0, 1.0), roll=41, min_len=30):
    adjusted = prob * np.asarray(weights, dtype=float)[None, :]
    adjusted = adjusted / np.clip(adjusted.sum(axis=1, keepdims=True), 1e-9, None)
    raw = adjusted.argmax(axis=1)
    smooth = merge_short_segments(rolling_mode(viterbi(adjusted, trans), roll), min_len)
    return raw, smooth, adjusted


def tune_intensity(train_items, prob_point_by_pid, prob_win_by_pid):
    trans = transition_matrix([it["y"] for it in train_items], stay_boost=70.0)
    grid = []
    for win_weight, mod_boost, high_boost, roll, min_len in itertools.product(
        # [0.45, 0.60, 0.75],
        # [1.0, 1.25, 1.6, 2.0],
        # [1.0, 1.5, 2.5, 4.0, 6.0],
        # [31, 41, 61],
        # [20, 40, 60],
        [0.45],
        [1.0],
        [1.0],
        [31],
        [20],
    ):
        score_parts = []
        high_recalls = []
        for it in train_items:
            pid = it["pid"]
            prob = (1.0 - win_weight) * prob_point_by_pid[pid] + win_weight * prob_win_by_pid[pid]
            _, smooth, _ = smooth_prob(prob, trans, (1.0, mod_boost, high_boost), roll, min_len)
            score_parts.append(balanced_accuracy_score(it["y"], smooth))
            cm = confusion_matrix(it["y"], smooth, labels=[0, 1, 2])
            high_recalls.append(cm[2, 2] / max(cm[2].sum(), 1))
        score = float(np.mean(score_parts) + 0.12 * np.mean(high_recalls))
        grid.append((score, win_weight, mod_boost, high_boost, roll, min_len))
    best = max(grid, key=lambda x: x[0])
    return {
        "score": best[0],
        "win_weight": best[1],
        "mod_boost": best[2],
        "high_boost": best[3],
        "roll": int(best[4]),
        "min_len": int(best[5]),
        "trans": trans,
    }


def train_intensity(train_items):
    p_scaler, p_models = train_point_models(train_items)
    w_scaler, w_model = train_window_model(train_items)
    pp, wp = {}, {}
    for it in train_items:
        pp[it["pid"]] = point_prob(it, p_scaler, p_models)
        wp[it["pid"]] = window_prob(it, w_scaler, w_model)
    params = tune_intensity(train_items, pp, wp)
    return {
        "p_scaler": p_scaler,
        "p_models": p_models,
        "w_scaler": w_scaler,
        "w_model": w_model,
        "params": params,
    }


def predict_intensity(model, item):
    pp = point_prob(item, model["p_scaler"], model["p_models"])
    wp = window_prob(item, model["w_scaler"], model["w_model"])
    p = model["params"]
    prob = (1.0 - p["win_weight"]) * pp + p["win_weight"] * wp
    raw, smooth, adjusted = smooth_prob(prob, p["trans"], (1.0, p["mod_boost"], p["high_boost"]), p["roll"], p["min_len"])
    return raw, smooth, adjusted


def proxy_features(item, prob, intensity):
    feat = item["feat"].copy()
    feat["prob_low"] = prob[:, 0]
    feat["prob_moderate"] = prob[:, 1]
    feat["prob_modhigh"] = prob[:, 2]
    feat["intensity_pred"] = intensity.astype(float)
    return feat


def train_proxy(train_items, pred_cache, target):
    xs, ys = [], []
    for it in train_items:
        pid = it["pid"]
        xs.append(proxy_features(it, pred_cache[pid]["prob"], pred_cache[pid]["smooth"]))
        if target == "dot789_mp":
            y = it["mp_dot789"]
        elif target == "dot789_mp_scaled_train_vo2":
            mp = it["mp_dot789"]
            true = it["extra_vo2"]
            scale = np.quantile(true, 0.95) / max(np.quantile(mp, 0.95), 1e-6)
            y = mp * np.clip(scale, 0.5, 4.0)
        elif target == "true_extra_vo2_upper_bound":
            y = it["extra_vo2"]
        else:
            raise ValueError(target)
        ys.append(np.log1p(np.clip(y, 0.0, 0.12)))
    X = pd.concat(xs).to_numpy(dtype=np.float32)
    y = np.concatenate(ys)
    scaler = StandardScaler().fit(X)
    models = [
        ExtraTreesRegressor(n_estimators=900, min_samples_leaf=1, max_features="sqrt", random_state=501, n_jobs=1),
        HistGradientBoostingRegressor(max_iter=520, learning_rate=0.03, l2_regularization=0.02, random_state=502),
    ]
    for model in models:
        model.fit(scaler.transform(X), y)
    return scaler, models


def predict_proxy(item, pred, scaler, models):
    X = scaler.transform(proxy_features(item, pred["prob"], pred["smooth"]).to_numpy(dtype=np.float32))
    vals = [np.expm1(model.predict(X)) for model in models]
    out = np.mean(vals, axis=0)
    return pd.Series(np.clip(out, 0.0, 0.12)).ewm(span=8, adjust=False).mean().to_numpy(dtype=float)


def plot_lines(df, y_true, y_pred, path, title, ylabel):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(df["t"], y_true, color="black", linewidth=1.1, label="True")
    ax.plot(df["t"], y_pred, color="#0072B2", linewidth=1.0, label="Pred")
    last = None
    for t, act in zip(df["t"], df["activity"]):
        if act != last:
            ax.axvline(t, color="0.75", linestyle="--", linewidth=0.7, alpha=0.55)
            last = act
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_mode(data_root, output_root, mode):
    mode_dir = os.path.join(output_root, mode)
    os.makedirs(mode_dir, exist_ok=True)
    _, pids, items = build_items(data_root, mode)
    part1_rows, part2_rows = [], []
    proxy_targets = ["dot789_mp", "dot789_mp_scaled_train_vo2", "true_extra_vo2_upper_bound"]
    for test_pid in pids:
        print(f"{mode} LOSO participant {test_pid}")
        train_pids = [pid for pid in pids if pid != test_pid]
        train_items = [items[pid] for pid in train_pids]
        test_item = items[test_pid]
        clf = train_intensity(train_items)

        pred_cache = {}
        for pid in pids:
            raw, smooth, prob = predict_intensity(clf, items[pid])
            pred_cache[pid] = {"raw": raw, "smooth": smooth, "prob": prob}

        pred = pred_cache[test_pid]
        cm = confusion_matrix(test_item["y"], pred["smooth"], labels=[0, 1, 2])
        part1_rows.append(
            {
                "mode": mode,
                "participant_id": test_pid,
                "raw_accuracy": float(accuracy_score(test_item["y"], pred["raw"])),
                "smoothed_accuracy": float(accuracy_score(test_item["y"], pred["smooth"])),
                "balanced_accuracy": float(balanced_accuracy_score(test_item["y"], pred["smooth"])),
                "low_recall": float(cm[0, 0] / max(cm[0].sum(), 1)),
                "moderate_recall": float(cm[1, 1] / max(cm[1].sum(), 1)),
                "modhigh_recall": float(cm[2, 2] / max(cm[2].sum(), 1)),
                "win_weight": clf["params"]["win_weight"],
                "mod_boost": clf["params"]["mod_boost"],
                "high_boost": clf["params"]["high_boost"],
                "roll": clf["params"]["roll"],
                "min_len": clf["params"]["min_len"],
            }
        )
        p1_dir = os.path.join(mode_dir, f"participant_{test_pid}", "part1")
        os.makedirs(p1_dir, exist_ok=True)
        pd.DataFrame(
            {
                "t": test_item["df"]["t"],
                "activity": test_item["df"]["activity"],
                "intensity_true": test_item["y"],
                "intensity_raw": pred["raw"],
                "intensity_pred": pred["smooth"],
                "prob_low": pred["prob"][:, 0],
                "prob_moderate": pred["prob"][:, 1],
                "prob_modhigh": pred["prob"][:, 2],
            }
        ).to_csv(os.path.join(p1_dir, "part1_intensity_timeseries.csv"), index=False)
        plot_lines(test_item["df"], test_item["y"], pred["smooth"], os.path.join(p1_dir, "part1_intensity_true_pred.png"), f"{mode} P{test_pid} intensity", "Intensity")

        for target in proxy_targets:
            scaler, models = train_proxy(train_items, pred_cache, target)
            mp_o2 = predict_proxy(test_item, pred, scaler, models)
            ekf = run_custom_ekf(test_item["df"], test_item["info"], pred["smooth"], mp_o2, r_value=0.005, n_value=0.01)
            metrics = regression_metrics(test_item["df"]["vo2"].to_numpy(dtype=float), ekf["ekf_vo2_ml_s"].to_numpy(dtype=float))
            row = {
                "mode": mode,
                "participant_id": test_pid,
                "proxy_target": target,
                **metrics,
                "ekf_vo2_mean": float(ekf["ekf_vo2_ml_s"].mean()),
                "ekf_vo2_max": float(ekf["ekf_vo2_ml_s"].max()),
                "true_vo2_mean": float(test_item["df"]["vo2"].mean()),
                "true_vo2_max": float(test_item["df"]["vo2"].max()),
                "mp_o2_mean_l_min": float(np.mean(mp_o2) * 60.0),
                "mp_o2_max_l_min": float(np.max(mp_o2) * 60.0),
            }
            part2_rows.append(row)
            p2_dir = os.path.join(mode_dir, f"participant_{test_pid}", "part2", target)
            os.makedirs(p2_dir, exist_ok=True)
            ekf.to_csv(os.path.join(p2_dir, "part2_ekf_outputs.csv"), index=False)
            pd.DataFrame({"t": test_item["df"]["t"], "mp_o2_l_s": mp_o2}).to_csv(os.path.join(p2_dir, "part2_proxy_timeseries.csv"), index=False)
            plot_lines(test_item["df"], test_item["df"]["vo2"].to_numpy(dtype=float), ekf["ekf_vo2_ml_s"].to_numpy(dtype=float), os.path.join(p2_dir, "part2_ekf_vo2_true_pred.png"), f"{mode} P{test_pid} {target}", "VO2 (ml/s)")

    part1 = pd.DataFrame(part1_rows)
    part2 = pd.DataFrame(part2_rows)
    part1.to_csv(os.path.join(mode_dir, "part1_intensity_loso_summary.csv"), index=False)
    part2.to_csv(os.path.join(mode_dir, "part2_ekf_vo2_loso_summary.csv"), index=False)
    best = part2.sort_values(["participant_id", "rmse"]).groupby("participant_id", as_index=False).first()
    best.to_csv(os.path.join(mode_dir, "part2_best_by_subject.csv"), index=False)
    return part1, part2, best


def run(output_root="outputs_wrist_loso_part12_optimization", data_root="Data"):
    os.makedirs(output_root, exist_ok=True)
    all_part1, all_part2, all_best = [], [], []
    for mode in MODES:
        p1, p2, best = run_mode(data_root, output_root, mode)
        all_part1.append(p1)
        all_part2.append(p2)
        all_best.append(best)
    part1 = pd.concat(all_part1, ignore_index=True)
    part2 = pd.concat(all_part2, ignore_index=True)
    best = pd.concat(all_best, ignore_index=True)
    part1.to_csv(os.path.join(output_root, "all_modes_part1_intensity_summary.csv"), index=False)
    part2.to_csv(os.path.join(output_root, "all_modes_part2_ekf_summary.csv"), index=False)
    best.to_csv(os.path.join(output_root, "all_modes_part2_best_by_subject.csv"), index=False)
    lines = []
    lines.append("# Wrist LOSO Part1/Part2 optimization\n\n")
    lines.append("Two wrist-input settings were evaluated. `e4_compatible_no_gyr` uses DOT6 velocity and acceleration, including total and three-axis components, but no gyration. `dot_wrist_with_gyr` additionally uses DOT6 total and three-axis gyration. DOT789 is used freely as the metabolic-proxy teacher; test-subject labels/VO2 are never used for training that subject.\n\n")
    lines.append("## Part1 by mode\n\n")
    lines.append(part1.groupby("mode")[["smoothed_accuracy", "balanced_accuracy", "low_recall", "moderate_recall", "modhigh_recall"]].agg(["median", "mean", "min", "max"]).round(4).to_markdown())
    lines.append("\n\n## Part2 by mode and proxy target\n\n")
    lines.append(part2.groupby(["mode", "proxy_target"])[["rmse", "mae", "corr", "r2", "ekf_vo2_max", "true_vo2_max"]].agg(["median", "mean"]).round(4).to_markdown())
    lines.append("\n\n## Best Part2 per subject\n\n")
    lines.append(best[["mode", "participant_id", "proxy_target", "rmse", "mae", "corr", "r2", "ekf_vo2_max", "true_vo2_max"]].round(4).to_markdown(index=False))
    lines.append("\n")
    with open(os.path.join(output_root, "wrist_loso_part12_optimization_manual.md"), "w", encoding="utf-8") as f:
        f.write("".join(lines))
