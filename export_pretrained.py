"""Export existing Module 1A weights and fit/save previously unsaved Module 1B."""
import argparse
import pickle
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits
import module1b_metabolic_proxy as m1b
from run_pipeline import MODES, write_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--intensity-root", required=True)
    p.add_argument("--destination", default=str(Path(__file__).resolve().parent / "pretrained"))
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = p.parse_args()
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        for mode in args.modes:
            print(f"Exporting Module 1A: {mode}", flush=True)
            source = Path(args.intensity_root) / "saved_models" / f"{mode}_best_intensity_model.pkl"
            with source.open("rb") as f:
                payload = pickle.load(f)
            if payload["mode"] != mode or len(payload["participant_ids"]) != 8:
                raise ValueError("Expected the established eight-participant final classifier")
            joblib.dump(payload, destination / f"{mode}_module1a.joblib", compress=3)
            del payload
            print(f"Training and exporting Module 1B: {mode}", flush=True)
            m1b.fit(args.data_root, args.intensity_root, mode, destination / f"{mode}_module1b.joblib")
            write_manifest(destination)
    print("Both modules exported", flush=True)


if __name__ == "__main__":
    main()
