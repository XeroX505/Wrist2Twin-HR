"""Module 1B: teacher construction and frozen wrist regression."""
from pathlib import Path
import platform
import joblib
import pandas as pd
import sklearn
from _bootstrap import activate

activate()
from paper4_part2_ekf import predict_external_proxy, proxy_training_features, train_external_proxy
from run_part23_from_saved_intensity import dot789_mp_o2 as teacher_observation
from hybrid_parts_common import load_subject, participant_ids


def fit(data_root, loso_intensity_root, mode, destination):
    info, scaler, models = train_external_proxy(str(data_root), str(loso_intensity_root), mode)
    pids = participant_ids(str(data_root), info)
    first = load_subject(str(data_root), pids[0]).reset_index(drop=True)
    intensity = pd.read_csv(Path(loso_intensity_root) / "internal" / mode /
                            f"participant_{pids[0]}" / "intensity_predictions.csv")
    if len(first) != len(intensity):
        raise ValueError("OOF intensity length differs from the original recording")
    names = list(proxy_training_features(first, intensity, mode).columns)
    defaults = {name: float(info[name].mean()) for name in
                ["Weight (kg)", "Height (cm)", "Muscle (%)", "RMR VO2"]}
    defaults["Sex"] = "male"
    payload = {
        "mode": mode, "participant_ids": pids, "scaler": scaler, "models": models,
        "feature_names": names, "demographic_defaults": defaults,
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "training_target": "log1p(clipped_multisegment_kinetic_proxy_L_per_s)",
        "mu_modhigh": 0.012, "mu_other": 0.06, "proxy_scale": 1.0,
        "intensity_source": "saved development LOSO predictions",
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, destination, compress=3)
    return payload


def load_model(path, mode):
    payload = joblib.load(path)
    if payload["mode"] != mode:
        raise ValueError("Proxy artifact does not match sensor mode")
    return payload


def predict(df, intensity, payload):
    if len(df) != len(intensity):
        raise ValueError("Intensity and wrist record must have identical lengths")
    actual = list(proxy_training_features(df, intensity, payload["mode"]).columns)
    if actual != payload["feature_names"]:
        raise ValueError("Proxy feature schema differs from training")
    return predict_external_proxy(df, intensity, payload["mode"],
                                  payload["scaler"], payload["models"])
