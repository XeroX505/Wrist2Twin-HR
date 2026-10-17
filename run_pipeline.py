"""One-command execution of the established Wrist2Twin-HR pipeline."""
import argparse
import hashlib
import json
import random
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import module1a_intensity as m1a
import module1b_metabolic_proxy as m1b
import module2_pm_ekf as m2
import module3_pmbnn as m3
from _bootstrap import activate

activate()
from hybrid_parts_common import load_info, load_subject, participant_ids
from paper4_part2_ekf import external_info_row, run_internal_mode
from train_saved_intensity_models_external import e4_file_to_dot6_dataframe, list_external_files

BASE = Path(__file__).resolve().parent
MODES = ("e4_compatible_no_gyr", "dot_wrist_with_gyr")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value, base=BASE):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def validate_artifacts(model_root, modes):
    manifest = json.loads((model_root / "manifest.json").read_text())
    for mode in modes:
        for module in ["module1a", "module1b"]:
            path = model_root / f"{mode}_{module}.joblib"
            expected = manifest["artifacts"][path.name]["sha256"]
            if file_hash(path) != expected:
                raise ValueError(f"Artifact checksum mismatch: {path}. Run git lfs pull if necessary.")
    return manifest


def load_record(path, mode):
    raw = pd.read_csv(path)
    required = ["HR_bpm", "vel_x_mps", "vel_y_mps", "vel_z_mps",
                "acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]
    if mode == "dot_wrist_with_gyr":
        required += ["gyr_x_rad_s", "gyr_y_rad_s", "gyr_z_rad_s"]
    missing = sorted(set(required) - set(raw))
    if missing:
        raise ValueError(f"{path}: missing channels {missing}")
    if not np.isfinite(raw[required].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: nonfinite input; preprocess explicitly before evaluation")
    if "t" in raw:
        delta = np.diff(raw["t"].to_numpy(dtype=float))
        if (delta < 0.95).any() or not np.isfinite(delta).all():
            raise ValueError(f"{path}: expected increasing processed 1-Hz timestamps")
        if not np.allclose(delta, 1.0, atol=0.05):
            warnings.warn(f"{path}: time gaps detected; established reproduction treats stored rows as successive 1-s samples")
    label = next((c for c in ["activity", "markers", "marker"] if c in raw), None)
    if label is None:
        raise ValueError(f"{path}: activity or markers column required for personalized splits")
    df = e4_file_to_dot6_dataframe(path)
    df["activity"] = raw[label].ffill().fillna("Unknown").astype(str)
    df["markers"] = df["activity"]
    return df.reset_index(drop=True)


def summary_tables(rows, destination):
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(destination / "subject_metrics.csv", index=False)
    group = ["mode", "dataset", "scenario"]
    metrics = ["rmse", "mae", "r2", "corr"]
    grouped = df.groupby(group, dropna=False)
    counts = grouped.size().rename("n_recordings")
    stats = grouped[metrics].agg(["median", "min", "max"])
    stats.columns = ["_".join(v) for v in stats.columns]
    counts.to_frame().join(stats).reset_index().to_csv(destination / "group_metrics.csv", index=False)


def run(config, action):
    modes = config.get("modes", [MODES[1]])
    if not modes or set(modes) - set(MODES):
        raise ValueError(f"Valid modes: {MODES}")
    scenarios = [str(x) for x in config.get("scenarios", ["80"])]
    if not scenarios or set(scenarios) - {"80", "50", "300"}:
        raise ValueError("Scenarios must be 80, 50 or 300")
    seed = int(config.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output = resolve(config.get("output_root", "outputs"))
    model_root = resolve(config.get("model_root", "pretrained"))
    dev = resolve(config.get("development_data_root", "Data"))
    intensity_root = (resolve(config["development_intensity_root"])
                      if config.get("development_intensity_root") else output / "part1_intensity")
    epochs = int(config.get("epochs", 5000))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if action == "validate":
        validate_artifacts(model_root, modes)
        print("Checksums verified. Use only trusted joblib/pickle files.")
        return
    output.mkdir(parents=True, exist_ok=True)
    if action == "train":
        model_root.mkdir(parents=True, exist_ok=True)
        for mode in modes:
            m1a.train_loso(dev, intensity_root, mode)
            payload = m1a.train_all(dev, output / "training", mode)
            joblib.dump(payload, model_root / f"{mode}_module1a.joblib", compress=3)
            m1b.fit(dev, intensity_root, mode, model_root / f"{mode}_module1b.joblib")
        write_manifest(model_root)
        return
    validate_artifacts(model_root, modes)
    rows = []
    for mode in modes:
        if config.get("development_enabled", False):
            # Refit held-out classifiers; never evaluate with all-participant weights.
            m1a.train_loso(dev, intensity_root, mode)
            run_internal_mode(str(dev), str(intensity_root), str(output / "part2_ekf"), mode)
            info = load_info(str(dev))
            for pid in participant_ids(str(dev), info):
                df = load_subject(str(dev), pid).reset_index(drop=True)
                ekf_path = output / "part2_ekf" / mode / "internal" / f"participant_{pid}" / "part2_dot789_ekf" / "part2_ekf_outputs.csv"
                ekf = pd.read_csv(ekf_path)
                target = output / "part3_pmbnn" / mode / "internal" / f"participant_{pid}"
                rows.append(m3.calibrate_and_evaluate(df, ekf["ekf_vo2_ml_s"].to_numpy(), target,
                            str(pid), mode, "internal", scenarios, epochs))
        a = m1a.load_model(model_root / f"{mode}_module1a.joblib", mode)
        b = m1b.load_model(model_root / f"{mode}_module1b.joblib", mode)
        for dataset, setting in config.get("datasets", {}).items():
            if setting["mode"] != mode:
                continue
            directory = resolve(setting["path"])
            files = list_external_files(str(directory))
            excluded = {str(x).casefold() for x in setting.get("exclude_records", [])}
            files = [(subject, path) for subject, path in files if subject.casefold() not in excluded]
            if not files:
                raise FileNotFoundError(f"No *_E4_ACC_HR_1Hz.csv records in {directory}")
            ids = [subject for subject, _ in files]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate legacy record IDs: use unique filename prefixes before the first underscore")
            for subject, path in files:
                print(f"{mode} {dataset} {subject}", flush=True)
                df = load_record(path, mode)
                intensity = m1a.predict(df, a)
                proxy = m1b.predict(df, intensity, b)
                defaults = pd.DataFrame([b["demographic_defaults"]])
                demographics = external_info_row(str(directory), subject, defaults)
                ekf = m2.estimate(df, demographics, intensity["intensity_pred"], proxy)
                target = output / mode / dataset / subject
                target.mkdir(parents=True, exist_ok=True)
                stage1 = pd.concat([df[["t", "activity"]], intensity], axis=1)
                stage1["proxy_l_s"] = proxy
                stage1.to_csv(target / "module1_outputs.csv", index=False)
                ekf.to_csv(target / "module2_outputs.csv", index=False)
                rows.append(m3.calibrate_and_evaluate(df, ekf["ekf_vo2_ml_s"].to_numpy(), target / "module3",
                            subject, mode, dataset, scenarios, epochs))
                summary_tables(rows, output)
    summary_tables(rows, output)
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")


def write_manifest(root):
    artifacts = {}
    for p in sorted(root.glob("*.joblib")):
        artifacts[p.name] = {"sha256": file_hash(p), "bytes": p.stat().st_size}
    data = {"format": 1, "sample_rate_hz": 1, "trusted_models_only": True, "artifacts": artifacts}
    (root / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(BASE / "config.example.json"))
    parser.add_argument("--action", choices=["run", "train", "validate"], default="run")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")), args.action)
