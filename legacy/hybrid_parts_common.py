import os
from collections import Counter

import numpy as np
import pandas as pd


INTENSITY_NAMES = ["Low", "Moderate", "Moderate-High"]
LOW_ACTIVITIES = {"sitting", "reading", "standing", "survey", "dishwasher", "start"}
MODERATE_ACTIVITIES = {"mopping", "stacking shelves", "treadmill 3km", "treadmill 5km"}
MODHIGH_ACTIVITIES = {"stairs", "cycling"}


def load_info(data_root):
    return pd.read_excel(os.path.join(data_root, "Data Storage file.xlsx")).set_index("Participant_ID")


def participant_ids(data_root, info):
    return [
        int(pid)
        for pid in info.index
        if os.path.exists(os.path.join(data_root, f"Participant {int(pid)}", "ramon_windowed_data.xlsx"))
    ]


def load_subject(data_root, participant_id):
    df = pd.read_excel(os.path.join(data_root, f"Participant {participant_id}", "ramon_windowed_data.xlsx"))
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    df = df.sort_values("t").reset_index(drop=True)
    df["activity"] = pd.Series(df["markers"]).ffill().fillna("Unknown").astype(str)
    df["markers"] = df["activity"]
    for sensor in [7, 8, 9]:
        for prefix in ["velocity", "acceleration", "gyration"]:
            col = f"{prefix}_DOT{sensor}"
            if col not in df.columns:
                df[col] = 0.0
    return df


def activity_to_intensity(activity):
    a = str(activity).strip().lower()
    if a in MODHIGH_ACTIVITIES:
        return 2
    if a in MODERATE_ACTIVITIES:
        return 1
    return 0


def dot789_features(df, info_row):
    cols = [f"{prefix}_DOT{sensor}" for sensor in [7, 8, 9] for prefix in ["velocity", "acceleration", "gyration"]]
    out = df[cols].astype(float).copy()
    mass = float(info_row["Weight (kg)"])
    sex = str(info_row["Sex"]).lower()
    mass_torso = mass * (0.6664 if sex.startswith("m") else 0.6314)
    mass_leg = mass * (0.1668 if sex.startswith("m") else 0.1843)
    ke = 0.5 * (
        mass_torso * df["velocity_DOT7"].astype(float).to_numpy() ** 2
        + mass_leg * df["velocity_DOT8"].astype(float).to_numpy() ** 2
        + mass_leg * df["velocity_DOT9"].astype(float).to_numpy() ** 2
    )
    out["ke_dot789"] = ke
    out["velocity_norm"] = np.sqrt(sum(df[f"velocity_DOT{s}"].astype(float) ** 2 for s in [7, 8, 9]))
    out["acceleration_norm"] = np.sqrt(sum(df[f"acceleration_DOT{s}"].astype(float) ** 2 for s in [7, 8, 9]))
    out["gyration_norm"] = np.sqrt(sum(df[f"gyration_DOT{s}"].astype(float) ** 2 for s in [7, 8, 9]))
    for col in ["ke_dot789", "velocity_norm", "acceleration_norm", "gyration_norm"]:
        out[f"{col}_ema_5s"] = out[col].ewm(span=5, adjust=False).mean()
        out[f"{col}_ema_15s"] = out[col].ewm(span=15, adjust=False).mean()
        out[f"{col}_ema_45s"] = out[col].ewm(span=45, adjust=False).mean()
        out[f"{col}_roll_std_15s"] = out[col].rolling(15, min_periods=1).std().fillna(0.0)
        out[f"{col}_diff"] = out[col].diff().fillna(0.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def marker_change_indices(df):
    markers = df["markers"]
    idx = df.index[pd.notnull(markers)].tolist() if markers.isna().any() else df.index[df["activity"].ne(df["activity"].shift())].tolist()
    if not idx or idx[0] != 0:
        idx = [0] + idx
    return idx


def merge_short_segments(y, min_len=25):
    y = np.asarray(y).copy()
    changed = True
    while changed:
        changed = False
        starts = np.r_[0, np.where(y[1:] != y[:-1])[0] + 1]
        ends = np.r_[starts[1:], len(y)]
        for start, end in zip(starts, ends):
            if end - start >= min_len or len(starts) == 1:
                continue
            left = y[start - 1] if start > 0 else None
            right = y[end] if end < len(y) else None
            fill = left if left is not None else right
            if left is not None and right is not None and left != right:
                fill = left if (start - starts[np.where(starts == start)[0][0] - 1]) >= (ends[min(np.where(starts == start)[0][0] + 1, len(ends)-1)] - end) else right
            if fill is not None:
                y[start:end] = fill
                changed = True
                break
    return y


def rolling_mode(y, window=31):
    s = pd.Series(y)
    return s.rolling(window, center=True, min_periods=1).apply(lambda x: Counter(x.astype(int)).most_common(1)[0][0]).to_numpy(dtype=int)


def regression_metrics(y, pred):
    import math
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    corr = np.nan
    if np.std(y) > 1e-8 and np.std(pred) > 1e-8:
        corr = float(np.corrcoef(y, pred)[0, 1])
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "corr": corr,
    }
