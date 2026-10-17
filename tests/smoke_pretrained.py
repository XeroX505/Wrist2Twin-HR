"""Local integration smoke check; the recording remains outside the repository."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import module1a_intensity as a
import module1b_metabolic_proxy as b
import module2_pm_ekf as ekf_module
import module3_pmbnn as hr_module
from run_pipeline import BASE, MODES, load_record, validate_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model-root", default=str(BASE / "pretrained"))
    args = parser.parse_args()
    root = Path(args.model_root)
    validate_artifacts(root, [args.mode])
    df = load_record(args.record, args.mode).iloc[:120].reset_index(drop=True)
    model_a = a.load_model(root / f"{args.mode}_module1a.joblib", args.mode)
    intensity = a.predict(df, model_a)
    del model_a
    model_b = b.load_model(root / f"{args.mode}_module1b.joblib", args.mode)
    proxy = b.predict(df, intensity, model_b)
    demographics = pd.Series(model_b["demographic_defaults"])
    del model_b
    assert len(proxy) == len(df) and np.isfinite(proxy).all()
    result = ekf_module.estimate(df, demographics, intensity["intensity_pred"], proxy)
    assert np.isfinite(result["ekf_vo2_ml_s"]).all()
    with tempfile.TemporaryDirectory() as directory:
        summary = hr_module.calibrate_and_evaluate(df, result["ekf_vo2_ml_s"].to_numpy(),
                  directory, "integration", args.mode, "local_smoke", ["80"], 2)
        assert np.isfinite(summary[["rmse", "mae", "r2", "corr"]].to_numpy()).all()
    print(json.dumps({"mode": args.mode, "samples": len(df), "stages": "1A,1B,2,3", "passed": True}))


if __name__ == "__main__":
    main()
