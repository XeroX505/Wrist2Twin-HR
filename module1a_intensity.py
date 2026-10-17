"""Module 1A: frozen wrist intensity inference and subject-held-out training."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from _bootstrap import activate

activate()
from optimize_wrist_loso_part12 import predict_intensity
from paper4_part1_intensity import external_item_from_df, run_internal_loso, train_final_model


def load_model(path, mode):
    payload = joblib.load(path)
    if payload["mode"] != mode:
        raise ValueError("Intensity artifact does not match sensor mode")
    return payload


def predict(df, payload):
    mode = payload["mode"]
    raw, smooth, prob = predict_intensity(payload["model"], external_item_from_df(df, mode))
    return pd.DataFrame({
        "intensity_raw": raw, "intensity_pred": smooth,
        "prob_low": prob[:, 0], "prob_moderate": prob[:, 1],
        "prob_modhigh": prob[:, 2], "mu": np.where(smooth == 2, 0.012, 0.06),
    })


def train_loso(data_root, output_root, mode):
    """Never use all-participant pretrained artifacts to evaluate these participants."""
    Path(output_root).mkdir(parents=True, exist_ok=True)
    return run_internal_loso(str(data_root), str(output_root), mode)


def train_all(data_root, output_root, mode):
    return train_final_model(str(data_root), str(output_root), mode)
