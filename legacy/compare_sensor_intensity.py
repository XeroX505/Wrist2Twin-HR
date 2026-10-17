import argparse
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from hybrid_parts_common import INTENSITY_NAMES, activity_to_intensity, load_info, load_subject, participant_ids


SENSOR_SETS = {
    "DOT5": [5],
    "DOT6": [6],
    "DOT789": [7, 8, 9],
    "DOT5789": [5, 7, 8, 9],
}


def ensure_sensor_columns(df, sensors):
    out = df.copy()
    for s in sensors:
        for prefix in ["velocity", "acceleration", "gyration"]:
            col = f"{prefix}_DOT{s}"
            if col not in out.columns:
                out[col] = 0.0
    return out


def sensor_features(df, info_row, sensors):
    df = ensure_sensor_columns(df, sensors)
    cols = [f"{prefix}_DOT{s}" for s in sensors for prefix in ["velocity", "acceleration", "gyration"]]
    out = df[cols].astype(float).copy()
    v_norm = np.sqrt(sum(df[f"velocity_DOT{s}"].astype(float) ** 2 for s in sensors))
    a_norm = np.sqrt(sum(df[f"acceleration_DOT{s}"].astype(float) ** 2 for s in sensors))
    g_norm = np.sqrt(sum(df[f"gyration_DOT{s}"].astype(float) ** 2 for s in sensors))
    out["velocity_norm"] = v_norm
    out["acceleration_norm"] = a_norm
    out["gyration_norm"] = g_norm
    out["motion_power"] = v_norm * a_norm
    for col in ["velocity_norm", "acceleration_norm", "gyration_norm", "motion_power"]:
        out[f"{col}_ema_5s"] = out[col].ewm(span=5, adjust=False).mean()
        out[f"{col}_ema_15s"] = out[col].ewm(span=15, adjust=False).mean()
        out[f"{col}_ema_45s"] = out[col].ewm(span=45, adjust=False).mean()
        out[f"{col}_std_15s"] = out[col].rolling(15, min_periods=1).std().fillna(0.0)
        out[f"{col}_std_45s"] = out[col].rolling(45, min_periods=1).std().fillna(0.0)
        out[f"{col}_diff"] = out[col].diff().fillna(0.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def window_features(df, info_row, sensors, window=30):
    base = sensor_features(df, info_row, sensors)
    labels = df["activity"].map(activity_to_intensity).to_numpy()
    rows, y, ranges = [], [], []
    arr = base.to_numpy(dtype=np.float32)
    for start in range(0, len(df), window):
        end = min(start + window, len(df))
        block = arr[start:end]
        feats = []
        feats.extend(block.mean(axis=0))
        feats.extend(block.std(axis=0))
        feats.extend(block.min(axis=0))
        feats.extend(block.max(axis=0))
        for col in ["velocity_norm", "acceleration_norm", "gyration_norm", "motion_power"]:
            sig = base[col].to_numpy(dtype=float)[start:end]
            if len(sig) < 4:
                feats.extend([0.0, 0.0, 0.0])
                continue
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


def transition_matrix(label_sequences, stay_boost=35.0):
    mat = np.ones((3, 3), dtype=float)
    for y in label_sequences:
        for a, b in zip(y[:-1], y[1:]):
            mat[int(a), int(b)] += 1.0
    mat += np.eye(3) * stay_boost
    return mat / mat.sum(axis=1, keepdims=True)


def viterbi(prob, trans):
    logp = np.log(np.clip(prob, 1e-9, 1.0))
    logt = np.log(np.clip(trans, 1e-9, 1.0))
    n, k = logp.shape
    dp = np.zeros((n, k))
    ptr = np.zeros((n, k), dtype=int)
    dp[0] = logp[0] + np.log(np.ones(k) / k)
    for i in range(1, n):
        scores = dp[i - 1][:, None] + logt
        ptr[i] = scores.argmax(axis=0)
        dp[i] = scores.max(axis=0) + logp[i]
    y = np.zeros(n, dtype=int)
    y[-1] = dp[-1].argmax()
    for i in range(n - 2, -1, -1):
        y[i] = ptr[i + 1, y[i + 1]]
    return y


def rolling_mode(y, window=41):
    return pd.Series(y).rolling(window, center=True, min_periods=1).apply(lambda x: Counter(x.astype(int)).most_common(1)[0][0]).to_numpy(dtype=int)


def merge_short_segments(y, min_len=30):
    y = y.copy()
    changed = True
    while changed:
        changed = False
        starts = np.r_[0, np.where(y[1:] != y[:-1])[0] + 1]
        ends = np.r_[starts[1:], len(y)]
        for idx, (start, end) in enumerate(zip(starts, ends)):
            if end - start >= min_len or len(starts) == 1:
                continue
            left = y[start - 1] if start > 0 else None
            right = y[end] if end < len(y) else None
            fill = left if left is not None else right
            if left is not None and right is not None and left != right:
                left_len = start - starts[idx - 1] if idx > 0 else 0
                right_len = ends[idx + 1] - end if idx + 1 < len(ends) else 0
                fill = left if left_len >= right_len else right
            if fill is not None:
                y[start:end] = fill
                changed = True
                break
    return y


def loso_predict(subjects, info, test_pid, sensors, feature_cache=None, window_cache=None):
    train_x, train_y, seqs, win_x, win_y = [], [], [], [], []
    for pid, df in subjects.items():
        y = df["activity"].map(activity_to_intensity).to_numpy()
        if pid == test_pid:
            continue
        train_x.append(feature_cache[pid] if feature_cache is not None else sensor_features(df, info.loc[pid], sensors))
        train_y.append(y)
        seqs.append(y)
        wx, wy, _ = window_cache[pid] if window_cache is not None else window_features(df, info.loc[pid], sensors)
        win_x.append(wx)
        win_y.append(wy)
    X = pd.concat(train_x).to_numpy(dtype=np.float32)
    y = np.concatenate(train_y)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    models = [
        RandomForestClassifier(n_estimators=240, max_depth=14, min_samples_leaf=5, class_weight="balanced_subsample", random_state=11, n_jobs=1),
        ExtraTreesClassifier(n_estimators=360, min_samples_leaf=3, class_weight="balanced", random_state=12, n_jobs=1),
        HistGradientBoostingClassifier(max_iter=220, learning_rate=0.055, l2_regularization=0.08, random_state=13),
    ]
    for m in models:
        m.fit(Xs, y)
    Xw = np.vstack(win_x)
    yw = np.concatenate(win_y)
    w_scaler = StandardScaler().fit(Xw)
    w_model = ExtraTreesClassifier(n_estimators=420, min_samples_leaf=2, class_weight="balanced", random_state=14, n_jobs=1)
    w_model.fit(w_scaler.transform(Xw), yw)

    df_test = subjects[test_pid]
    Xt_df = feature_cache[test_pid] if feature_cache is not None else sensor_features(df_test, info.loc[test_pid], sensors)
    Xt = Xt_df.to_numpy(dtype=np.float32)
    probs = []
    for m in models:
        p = m.predict_proba(scaler.transform(Xt))
        full = np.zeros((len(df_test), 3))
        for i, cls in enumerate(m.classes_):
            full[:, int(cls)] = p[:, i]
        probs.append(full)
    prob_point = np.mean(probs, axis=0)
    wx, _, ranges = window_cache[test_pid] if window_cache is not None else window_features(df_test, info.loc[test_pid], sensors)
    wp = w_model.predict_proba(w_scaler.transform(wx))
    prob_win = np.zeros((len(df_test), 3))
    for ri, (start, end) in enumerate(ranges):
        full = np.zeros(3)
        for j, cls in enumerate(w_model.classes_):
            full[int(cls)] = wp[ri, j]
        prob_win[start:end] = full
    prob = 0.45 * prob_point + 0.55 * prob_win
    raw = prob.argmax(axis=1)
    pred = merge_short_segments(rolling_mode(viterbi(prob, transition_matrix(seqs)), 41), 30)
    true = df_test["activity"].map(activity_to_intensity).to_numpy()
    return true, raw, pred, prob


def summarize(sensor_set, pid, true, raw, pred):
    row = {
        "sensor_set": sensor_set,
        "participant_id": pid,
        "raw_accuracy": float(accuracy_score(true, raw)),
        "smoothed_accuracy": float(accuracy_score(true, pred)),
    }
    for cls, name in enumerate(["low", "moderate", "modhigh"]):
        mask = true == cls
        row[f"{name}_accuracy"] = float(np.mean(pred[mask] == cls)) if np.any(mask) else np.nan
    cm = confusion_matrix(true, pred, labels=[0, 1, 2])
    for i, name_i in enumerate(["low", "moderate", "modhigh"]):
        for j, name_j in enumerate(["low", "moderate", "modhigh"]):
            row[f"cm_{name_i}_as_{name_j}"] = int(cm[i, j])
    return row


def plot_sensor_comparison(summary, out_dir):
    agg = summary.groupby("sensor_set")["smoothed_accuracy"].agg(["median", "mean", "min", "max"]).reset_index()
    plt.figure(figsize=(9, 4.8))
    x = np.arange(len(agg))
    plt.bar(x, agg["median"], color="#1f77b4")
    plt.errorbar(x, agg["median"], yerr=[agg["median"] - agg["min"], agg["max"] - agg["median"]], fmt="none", ecolor="black", capsize=4)
    plt.xticks(x, agg["sensor_set"])
    plt.ylim(0, 1)
    plt.ylabel("LOSO smoothed accuracy")
    plt.title("Intensity Recognition by Sensor Set")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sensor_set_accuracy_comparison.png"), dpi=220)
    plt.close()


def plot_intensity_timeseries(timeseries, sensor_set, pid, out_dir):
    t = timeseries["t"].to_numpy(dtype=float)
    true = timeseries["intensity_true"].to_numpy(dtype=int)
    pred = timeseries["intensity_pred"].to_numpy(dtype=int)
    activity = timeseries["activity"].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    ax.step(t, true, where="post", color="#0072B2", linewidth=1.6, label="True intensity")
    ax.step(t, pred + 0.04, where="post", color="#D55E00", linewidth=1.25, alpha=0.88, label="Predicted intensity")

    change_idx = np.where(activity[1:] != activity[:-1])[0] + 1
    for idx in change_idx:
        ax.axvline(t[idx], color="0.78", linestyle="--", linewidth=0.8, alpha=0.75)

    acc = accuracy_score(true, pred)
    ax.set_title(f"{sensor_set} participant {pid}: intensity prediction (accuracy={acc:.3f})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low", "Moderate", "Moderate-high"])
    ax.set_ylim(-0.3, 2.35)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"participant_{pid}_intensity_true_vs_pred.png"), dpi=220)
    plt.close(fig)


def plot_existing_timeseries(output_dir):
    for set_name in SENSOR_SETS:
        set_dir = os.path.join(output_dir, set_name)
        if not os.path.isdir(set_dir):
            continue
        for file_name in sorted(os.listdir(set_dir)):
            if not file_name.startswith("participant_") or not file_name.endswith("_timeseries.csv"):
                continue
            pid = int(file_name.split("_")[1])
            path = os.path.join(set_dir, file_name)
            plot_intensity_timeseries(pd.read_csv(path), set_name, pid, set_dir)


def run(data_root, output_dir):
    info = load_info(data_root)
    pids = participant_ids(data_root, info)
    subjects = {pid: load_subject(data_root, pid) for pid in pids}
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for set_name, sensors in SENSOR_SETS.items():
        set_dir = os.path.join(output_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)
        print(f"Precomputing {set_name} features")
        feature_cache = {pid: sensor_features(df, info.loc[pid], sensors) for pid, df in subjects.items()}
        window_cache = {pid: window_features(df, info.loc[pid], sensors) for pid, df in subjects.items()}
        for pid in pids:
            print(f"{set_name} participant {pid}")
            true, raw, pred, prob = loso_predict(subjects, info, pid, sensors, feature_cache, window_cache)
            rows.append(summarize(set_name, pid, true, raw, pred))
            df = subjects[pid]
            out = pd.DataFrame(
                {
                    "t": df["t"],
                    "activity": df["activity"],
                    "intensity_true": true,
                    "intensity_raw": raw,
                    "intensity_pred": pred,
                    "prob_low": prob[:, 0],
                    "prob_moderate": prob[:, 1],
                    "prob_modhigh": prob[:, 2],
                }
            )
            out.to_csv(os.path.join(set_dir, f"participant_{pid}_timeseries.csv"), index=False)
            plot_intensity_timeseries(out, set_name, pid, set_dir)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, "sensor_set_intensity_summary.csv"), index=False)
    summary.groupby("sensor_set")[["raw_accuracy", "smoothed_accuracy", "low_accuracy", "moderate_accuracy", "modhigh_accuracy"]].agg(["median", "mean", "min", "max"]).to_csv(os.path.join(output_dir, "sensor_set_intensity_aggregate.csv"))
    plot_sensor_comparison(summary, output_dir)
    return summary
