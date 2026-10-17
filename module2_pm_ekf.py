"""Module 2: physiological EKF with no manual amplitude scaling."""
import numpy as np
from _bootstrap import activate

activate()
from paper4_part2_ekf import safe_run_ekf_no_scale

MU_MODERATE_HIGH = 0.012
MODHIGH_PROXY_SCALE = 1.0


def estimate(df, demographics, intensity, proxy_l_s):
    intensity = np.asarray(intensity, dtype=int)
    proxy = np.asarray(proxy_l_s, dtype=float)
    if len(df) != len(intensity) or len(df) != len(proxy):
        raise ValueError("EKF input lengths must agree; truncation is not performed")
    if not np.isfinite(proxy).all() or not np.isin(intensity, [0, 1, 2]).all():
        raise ValueError("Invalid metabolic observation or intensity labels")
    output, r_value, n_value = safe_run_ekf_no_scale(df, demographics, intensity, proxy)
    output["mu"] = np.where(intensity == 2, MU_MODERATE_HIGH, 0.06)
    output["proxy_l_s"] = proxy
    output.attrs.update(ekf_r_value=r_value, ekf_n_value=n_value)
    return output
