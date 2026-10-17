"""Module 3 entry point; retain established equations and split allocation."""
from _bootstrap import activate

activate()
import paper4_part3_pmbnn as implementation

PHYSICS_WEIGHT = 10000.0
split_per_activity = implementation.split_per_activity_exact


def calibrate_and_evaluate(df, vo2_ml_s, output_dir, subject, mode, dataset,
                           scenarios=("80",), epochs=5000):
    if len(df) != len(vo2_ml_s):
        raise ValueError("VO2 and HR must have exactly matching lengths")
    train, test, *_ = split_per_activity(df, "80")
    if not train.any() or not test.any():
        raise ValueError("Recording must contain both calibration and evaluation samples")
    previous = implementation.SCENARIOS
    implementation.SCENARIOS = list(scenarios)
    try:
        return implementation.run_one(df, vo2_ml_s, output_dir, subject,
                                      mode, dataset, subject, epochs)
    finally:
        implementation.SCENARIOS = previous
