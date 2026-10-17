import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from hybrid_parts_common import load_subject, regression_metrics


class PINNHR(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        init = torch.tensor([0.02, 0.10, -5.3, 10.5, 0.44, 0.3], dtype=torch.float32)
        lower = torch.tensor([0.01, 0.06, -6.0, 7.0, 0.1, -0.5], dtype=torch.float32)
        upper = torch.tensor([0.03, 0.15, -2.0, 20.0, 0.6, 0.5], dtype=torch.float32)
        scaled = (init - lower) / (upper - lower)
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.raw_lambda = nn.Parameter(torch.log(scaled / (1.0 - scaled)))

    def lambdas(self):
        s = torch.sigmoid(self.raw_lambda)
        return self.lower + (self.upper - self.lower) * s

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_features(vo2):
    vo2 = np.asarray(vo2, dtype=np.float32)
    d = np.diff(vo2, prepend=vo2[0])
    ema = pd.Series(vo2).ewm(span=20, adjust=False).mean().to_numpy(dtype=np.float32)
    return np.column_stack([vo2, d, ema]).astype(np.float32)


def split_train_test(df, mode, scenario, return_segments=False):
    n = len(df)
    train = np.zeros(n, dtype=bool)
    train_segments = []
    test_segments = []
    if mode == "chronological":
        cut = scenario_cut(n, scenario)
        train[:cut] = True
        train_segments.append(np.arange(0, cut, dtype=int))
        if cut < n:
            test_segments.append(np.arange(cut, n, dtype=int))
    elif mode == "per_activity":
        groups = [(name, np.asarray(list(idx), dtype=int)) for name, idx in df.groupby("activity", sort=False).groups.items()]
        ratio = {"300": 0.2, "20": 0.2, "50": 0.5, "80": 0.8}.get(scenario)
        cuts = {}
        if ratio is not None:
            desired_total = int(round(ratio * n))
            fractional = []
            current_total = 0
            for name, pos in groups:
                if len(pos) < 3:
                    cut = len(pos)
                    frac = 0.0
                else:
                    raw = ratio * len(pos)
                    cut = min(max(1, int(np.floor(raw))), len(pos) - 1)
                    frac = raw - np.floor(raw)
                cuts[name] = cut
                current_total += cut
                fractional.append((frac, name))
            for _, name in sorted(fractional, reverse=True):
                if current_total >= desired_total:
                    break
                pos_len = len(dict(groups)[name])
                if cuts[name] < pos_len - 1:
                    cuts[name] += 1
                    current_total += 1
            for _, name in sorted(fractional):
                if current_total <= desired_total:
                    break
                if cuts[name] > 1:
                    cuts[name] -= 1
                    current_total -= 1
        for name, pos in groups:
            if len(pos) < 3:
                train[pos] = True
                train_segments.append(pos)
                continue
            cut = cuts[name]
            train[pos[:cut]] = True
            train_segments.append(pos[:cut])
            test_segments.append(pos[cut:])
    else:
        raise ValueError(mode)
    test = ~train
    if return_segments:
        train_order = np.concatenate([s for s in train_segments if len(s) > 0]) if train_segments else np.array([], dtype=int)
        test_order = np.concatenate([s for s in test_segments if len(s) > 0]) if test_segments else np.array([], dtype=int)
        return train, test, train_segments, test_segments, train_order, test_order
    return train, test


def scenario_cut(n, scenario):
    if scenario == "300":
        return min(300, max(1, n - 1))
    if scenario == "20":
        return max(1, int(0.2 * n))
    if scenario == "50":
        return max(1, int(0.5 * n))
    if scenario == "80":
        return max(1, int(0.8 * n))
    raise ValueError(scenario)


def physics_de_loss(hr_pred, vo2_l_min, train_segments, lam):
    losses = []
    for seg in train_segments:
        if len(seg) < 2:
            continue
        idx = torch.as_tensor(seg, dtype=torch.long)
        hr_seg = hr_pred[idx]
        vo2_seg = torch.clamp(vo2_l_min[idx], min=1e-6)
        sv = lam[0] * torch.log(vo2_seg) + lam[1]
        tpr = lam[2] * torch.log(vo2_seg) + lam[3]
        map_proxy = hr_seg * sv * tpr
        hr_grad = hr_seg[1:] - hr_seg[:-1]
        map_grad = map_proxy[1:] - map_proxy[:-1]
        losses.append(torch.mean((hr_grad - (lam[4] * map_grad + lam[5])) ** 2))
    if not losses:
        return torch.zeros((), dtype=hr_pred.dtype)
    return torch.stack(losses).mean()


def solve_ode_numpy(vo2_l_min, init_hr, lambdas):
    vo2 = np.maximum(np.asarray(vo2_l_min, dtype=np.float64), 1e-6)
    lam = np.asarray(lambdas, dtype=np.float64)
    sv = lam[0] * np.log(vo2) + lam[1]
    tpr = lam[2] * np.log(vo2) + lam[3]
    out = [float(init_hr)]
    for i in range(1, len(vo2)):
        denom = 1.0 - lam[4] * sv[i] * tpr[i]
        if abs(denom) < 1e-6:
            denom = np.sign(denom) * 1e-6 if denom != 0 else 1e-6
        new_hr = (out[-1] - lam[4] * out[-1] * sv[i - 1] * tpr[i - 1] + lam[5]) / denom
        out.append(float(np.clip(new_hr, 35.0, 210.0)))
    return np.asarray(out, dtype=np.float32)


def reconstructed_by_segments(vo2_l_min, hr, segments, lambdas):
    out = np.full(len(vo2_l_min), np.nan, dtype=np.float32)
    for seg in segments:
        if len(seg) == 0:
            continue
        init_hr = hr[int(seg[0])]
        out[seg] = solve_ode_numpy(np.asarray(vo2_l_min)[seg], init_hr, lambdas)
    return out


def whitebox_segment(vo2, init_hr, lam, prev_vo2=None):
    vo2 = torch.clamp(vo2, min=1e-4)
    if prev_vo2 is None:
        prev_v = vo2[0]
    else:
        prev_v = torch.clamp(prev_vo2, min=1e-4)
    hr_values = [init_hr]
    prev_hr = init_hr
    prev_sv = lam[0] * torch.log(prev_v) + lam[1]
    prev_tpr = lam[2] * torch.log(prev_v) + lam[3]
    for i in range(1, len(vo2)):
        cur_sv = lam[0] * torch.log(vo2[i]) + lam[1]
        cur_tpr = lam[2] * torch.log(vo2[i]) + lam[3]
        prev_map = prev_hr * prev_sv * prev_tpr
        cur_map = prev_hr * cur_sv * cur_tpr
        next_hr = prev_hr + lam[4] * (cur_map - prev_map) + lam[5]
        next_hr = torch.clamp(next_hr, 35.0, 210.0)
        hr_values.append(next_hr)
        prev_hr = next_hr
        prev_sv = cur_sv
        prev_tpr = cur_tpr
    return torch.stack(hr_values)


def whitebox_predict_numpy(vo2, hr, segments, lambdas):
    out = np.full(len(vo2), np.nan, dtype=np.float32)
    lam = torch.tensor(lambdas, dtype=torch.float32)
    vo2t = torch.tensor(np.asarray(vo2, dtype=np.float32))
    hrt = torch.tensor(np.asarray(hr, dtype=np.float32))
    with torch.no_grad():
        for seg in segments:
            if len(seg) == 0:
                continue
            first = int(seg[0])
            if first > 0:
                prev_vo2 = vo2t[first - 1]
                init_hr = hrt[first - 1]
                seq = torch.cat([vo2t[first - 1:first], vo2t[seg]])
                pred = whitebox_segment(seq, init_hr, lam, prev_vo2=prev_vo2)[1:]
            else:
                pred = whitebox_segment(vo2t[seg], hrt[first], lam)
            out[seg] = pred.numpy()
    return out


def identify_lambdas(vo2, hr, train_segments, epochs=1600, lr=8e-3):
    vo2t = torch.tensor(np.asarray(vo2, dtype=np.float32))
    hrt = torch.tensor(np.asarray(hr, dtype=np.float32))
    prev_idx = []
    curr_idx = []
    for seg in train_segments:
        if len(seg) > 1:
            prev_idx.extend(seg[:-1])
            curr_idx.extend(seg[1:])
    prev_idx = torch.tensor(prev_idx, dtype=torch.long)
    curr_idx = torch.tensor(curr_idx, dtype=torch.long)
    model = LambdaWhiteBox()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        lam = model.lambdas()
        if len(curr_idx) > 0:
            sv_prev = lam[0] * torch.log(torch.clamp(vo2t[prev_idx], min=1e-4)) + lam[1]
            sv_curr = lam[0] * torch.log(torch.clamp(vo2t[curr_idx], min=1e-4)) + lam[1]
            tpr_prev = lam[2] * torch.log(torch.clamp(vo2t[prev_idx], min=1e-4)) + lam[3]
            tpr_curr = lam[2] * torch.log(torch.clamp(vo2t[curr_idx], min=1e-4)) + lam[3]
            map_prev = hrt[prev_idx] * sv_prev * tpr_prev
            map_curr = hrt[prev_idx] * sv_curr * tpr_curr
            pred_next = hrt[prev_idx] + lam[4] * (map_curr - map_prev) + lam[5]
            mse = F.mse_loss(pred_next, hrt[curr_idx])
        else:
            mse = torch.zeros(())
        center = torch.tensor([0.02, 0.10, -5.3, 10.5, 0.44, 0.0], dtype=torch.float32)
        reg = torch.mean(((lam - center) / torch.tensor([0.02, 0.08, 3.0, 8.0, 1.0, 1.0])) ** 2)
        loss = mse + 0.002 * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if epoch % 25 == 0 or epoch == epochs - 1:
            lam_values = [float(v) for v in model.lambdas().detach().numpy()]
            history.append({"epoch": epoch, "whitebox_mse": float(mse.detach()), "lambda_reg": float(reg.detach()), "loss": float(loss.detach()), **{f"lambda{i + 1}": lam_values[i] for i in range(6)}})
    return [float(v) for v in model.lambdas().detach().numpy()], history


def train_predict_pinn(df, ekf_vo2, mode, scenario, epochs=1600, lr=5e-3, phys_weight=100000.0):
    train, test, train_segments, test_segments, train_order, test_order = split_train_test(df, mode, scenario, return_segments=True)
    X = make_features(ekf_vo2)
    scaler = StandardScaler().fit(X[train])
    xs = scaler.transform(X).astype(np.float32)
    hr = df["hr"].to_numpy(dtype=np.float32)
    xt = torch.from_numpy(xs)
    hr_t = torch.from_numpy(hr)
    vo2_l_min_t = torch.from_numpy((np.asarray(ekf_vo2, dtype=np.float32) * 60.0 / 1000.0).astype(np.float32))
    train_idx = torch.from_numpy(np.where(train)[0])
    model = PINNHR()
    with torch.no_grad():
        model.net[-1].bias.fill_(float(hr[train].mean()))
    opt = torch.optim.RMSprop(
        [
            {"params": [p for n, p in model.named_parameters() if n != "raw_lambda"], "lr": lr},
            {"params": [model.raw_lambda], "lr": lr * 5.0},
        ],
        alpha=0.99,
        weight_decay=0.0,
    )
    history = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred_hr_t = model(xt)
        mse = F.mse_loss(pred_hr_t[train_idx], hr_t[train_idx])
        lam = model.lambdas()
        de = physics_de_loss(pred_hr_t, vo2_l_min_t, train_segments, lam)
        loss = mse + phys_weight * de
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if epoch % 25 == 0 or epoch == epochs - 1:
            lam_values = [float(v) for v in model.lambdas().detach().numpy()]
            history.append(
                {
                    "epoch": epoch,
                    "mse": float(mse.detach()),
                    "de_loss": float(de.detach()),
                    "phys_weight": phys_weight,
                    "loss": float(loss.detach()),
                    **{f"lambda{i + 1}": lam_values[i] for i in range(6)},
                    "stage": "pinn_joint_training",
                }
            )
    model.eval()
    with torch.no_grad():
        pred_hr = model(xt).numpy().astype(np.float32)
    lambdas = [float(v) for v in model.lambdas().detach().numpy()]
    vo2_l_min = np.asarray(ekf_vo2, dtype=np.float32) * 60.0 / 1000.0
    whitebox_train = reconstructed_by_segments(vo2_l_min, hr, train_segments, lambdas)
    whitebox_test = reconstructed_by_segments(vo2_l_min, hr, test_segments, lambdas)
    whitebox_hr = np.full(len(hr), np.nan, dtype=np.float32)
    whitebox_hr[train] = whitebox_train[train]
    whitebox_hr[test] = whitebox_test[test]
    metrics = regression_metrics(hr[test], pred_hr[test])
    whitebox_metrics = regression_metrics(hr[test], whitebox_hr[test])
    return pred_hr, whitebox_hr, train, test, train_order, test_order, metrics, whitebox_metrics, lambdas, history


def run_pinn_hr(data_root, participant_id, part2_dir, output_dir, epochs=1600):
    df = load_subject(data_root, participant_id)
    ekf = pd.read_csv(os.path.join(part2_dir, "part2_ekf_outputs.csv"))
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for mode in ["chronological", "per_activity"]:
        for scenario in ["300", "50", "80"]:
            pred, whitebox, train, test, train_order, test_order, metrics, whitebox_metrics, lambdas, history = train_predict_pinn(
                df, ekf["ekf_vo2_ml_s"].to_numpy(), mode, scenario, epochs=epochs, phys_weight=run_pinn_hr.phys_weight
            )
            rows.append(
                {
                    "participant_id": participant_id,
                    "split_mode": mode,
                    "scenario": scenario,
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "phys_weight": float(run_pinn_hr.phys_weight),
                    **metrics,
                    **{f"whitebox_{k}": v for k, v in whitebox_metrics.items()},
                    **{f"lambda{i + 1}": lambdas[i] for i in range(6)},
                }
            )
            pd.DataFrame(history).to_csv(os.path.join(output_dir, f"part3_history_{mode}_{scenario}.csv"), index=False)
            pd.DataFrame([{f"lambda{i + 1}": lambdas[i] for i in range(6)}]).to_csv(
                os.path.join(output_dir, f"part3_final_lambdas_{mode}_{scenario}.csv"), index=False
            )
            pd.DataFrame(
                {
                    "concat_index": np.arange(len(test_order)),
                    "original_index": test_order,
                    "t": df.loc[test_order, "t"].to_numpy(),
                    "activity": df.loc[test_order, "activity"].to_numpy(),
                    "hr_true": df.loc[test_order, "hr"].to_numpy(),
                    "hr_whitebox": whitebox[test_order],
                    "hr_pred": pred[test_order],
                    "ekf_vo2_ml_s": ekf.loc[test_order, "ekf_vo2_ml_s"].to_numpy(),
                }
            ).to_csv(os.path.join(output_dir, f"part3_predictions_{mode}_{scenario}.csv"), index=False)
            pd.DataFrame(
                {
                    "concat_index": np.arange(len(train_order)),
                    "original_index": train_order,
                    "t": df.loc[train_order, "t"].to_numpy(),
                    "activity": df.loc[train_order, "activity"].to_numpy(),
                    "hr_true": df.loc[train_order, "hr"].to_numpy(),
                    "hr_whitebox": whitebox[train_order],
                    "hr_pred": pred[train_order],
                    "ekf_vo2_ml_s": ekf.loc[train_order, "ekf_vo2_ml_s"].to_numpy(),
                }
            ).to_csv(os.path.join(output_dir, f"part3_train_{mode}_{scenario}.csv"), index=False)
            plot_hr_concat(df, pred, whitebox, test_order, os.path.join(output_dir, f"part3_hr_{mode}_{scenario}.png"), f"{mode} {scenario} test")
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, "part3_hr_summary.csv"), index=False)
    return summary


run_pinn_hr.phys_weight = 100000.0


def plot_hr(df, pred, test, path):
    plt.figure(figsize=(14, 5))
    plt.plot(df.loc[test, "t"], df.loc[test, "hr"], color="black", linewidth=1.2, label="True HR")
    plt.plot(df.loc[test, "t"], pred[test], color="#1f77b4", linewidth=1.1, label="Predicted HR")
    last = None
    for t, act in zip(df["t"], df["activity"]):
        if act != last:
            plt.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
            last = act
    plt.xlabel("Time (s)")
    plt.ylabel("HR (bpm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_hr_concat(df, pred, whitebox, order, path, title):
    x = np.arange(len(order))
    plt.figure(figsize=(14, 5))
    plt.plot(x, df.loc[order, "hr"], color="black", linewidth=1.2, label="True HR")
    plt.plot(x, whitebox[order], color="#009E73", linewidth=1.0, alpha=0.85, label="White-box HR")
    plt.plot(x, pred[order], color="#0072B2", linewidth=1.1, label="PINN HR")
    activity = df.loc[order, "activity"].astype(str).to_numpy()
    if len(activity) > 0:
        change_idx = np.where(activity[1:] != activity[:-1])[0] + 1
        for idx in change_idx:
            plt.axvline(idx, color="gray", linestyle="--", linewidth=0.8, alpha=0.45)
    plt.title(title)
    plt.xlabel("Concatenated sample index")
    plt.ylabel("HR (bpm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
