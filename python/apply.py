"""Apply trained neural Bayes estimators to the simulated test set and to the real cells.

Usage: python python/apply.py
Real data: the 39 TRAINING winters of each cell only (validation/test winters are not read).
Outputs: results/nbe_sim_seed<k>.csv, results/nbe_real_seed<k>.csv, results/nbe_apply.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ["q05", "q25", "q50", "q75", "q95"]


def split_of_winter(w):
    return np.where(w % 5 == 0, "test", np.where(w % 5 == 3, "val", "train"))


def training_matrix():
    wm = pd.read_csv(ROOT / "data" / "winter_maxima.csv")
    assert (wm["set"] == split_of_winter(wm["winter"].to_numpy())).all()
    tr = wm[wm["set"] == "train"]
    assert tr["winter"].nunique() == 39
    Y = tr.pivot(index="cell", columns="winter", values="max_mm").sort_index()
    assert Y.shape == (256, 39) and not Y.isna().any().any()
    return Y.index.to_numpy(), Y.to_numpy()


def to_frame(ids, q):
    cols = {"id": ids}
    for t in nbe.TARGETS:
        for j, lev in enumerate(LEVELS):
            cols[f"{t}_{lev}"] = q[t][:, j]
    return pd.DataFrame(cols)


def ood_check(x_real, n_sim=200_000):
    """Mahalanobis distance of each real feature vector from the prior-predictive feature
    distribution, and the share of simulated vectors that are at least as far."""
    rng = np.random.default_rng(424242)
    mu, sigma, xi = nbe.draw_prior(rng, n_sim)
    xs, _, _ = nbe.features(nbe.simulate(rng, mu, sigma, xi))
    c = xs.mean(axis=0); P = np.linalg.pinv(np.cov(xs, rowvar=False))
    d_sim = np.einsum("ij,jk,ik->i", xs - c, P, xs - c)
    d_real = np.einsum("ij,jk,ik->i", x_real - c, P, x_real - c)
    tail = np.array([(d_sim >= d).mean() for d in d_real])
    return {"n_sim": n_sim, "real_tail_prob_min": float(tail.min()),
            "n_cells_tail_prob_below_0.001": int((tail < 0.001).sum()),
            "n_cells_tail_prob_below_0.01": int((tail < 0.01).sum()),
            "sim_m_range": [float(xs[:, -2].min() * 100), float(xs[:, -2].max() * 100)],
            "real_m_range": [float(x_real[:, -2].min() * 100), float(x_real[:, -2].max() * 100)],
            "sim_s_range": [float(np.exp(xs[:, -1].min())), float(np.exp(xs[:, -1].max()))],
            "real_s_range": [float(np.exp(x_real[:, -1].min())), float(np.exp(x_real[:, -1].max()))]}


def main():
    torch.set_num_threads(1)                # timing on one core
    ts = pd.read_csv(ROOT / "results" / "sim" / "test_set.csv")
    Ysim = ts[[f"y{i}" for i in range(1, 40)]].to_numpy()
    cells, Yreal = training_matrix()
    info = {"timing_note": "one CPU thread, all datasets in one batch, model already loaded"}
    for seed in (1, 2, 3):
        f = ROOT / "models" / f"nbe_seed{seed}.pt"
        if not f.exists():
            continue
        model = nbe.NBE(); model.load_state_dict(torch.load(f)); model.eval()
        to_frame(ts["id"].to_numpy(), nbe.predict(model, Ysim)).to_csv(
            ROOT / "results" / f"nbe_sim_seed{seed}.csv", index=False)
        t0 = time.perf_counter()
        q = nbe.predict(model, Yreal)
        el = time.perf_counter() - t0
        to_frame(cells, q).to_csv(ROOT / "results" / f"nbe_real_seed{seed}.csv", index=False)
        info[f"seed{seed}_time_256_cells_s"] = el
    x_real, _, _ = nbe.features(Yreal)
    info["ood"] = ood_check(x_real)
    json.dump(info, open(ROOT / "results" / "nbe_apply.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
