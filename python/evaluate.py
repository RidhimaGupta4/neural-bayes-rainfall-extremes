"""All evaluation (PREREGISTRATION.md, "Evaluation" and "Criteria").

This is the ONLY script that reads validation and test winters. Every estimate it evaluates was
made from the 39 training winters of each cell (or from simulated data).
Usage: python python/evaluate.py
Outputs: results/evaluation.json, results/real_comparison.csv, results/heldout_by_cell.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MAIN_SEED = 1
LOGT = ["z10", "z100", "z1000"]          # assessed on the log scale


def qcols(df, t, prefix=""):
    return [df[f"{prefix}{t}_{l}"].to_numpy() for l in ("q05", "q25", "q50", "q75", "q95")]


def on_scale(t, v):
    return np.log(v) if t in LOGT else v


# ------------------------------------------------------------------------------------------
# 1. Simulation with known truth
# ------------------------------------------------------------------------------------------
def simulation_metrics():
    ts = pd.read_csv(RES / "sim" / "test_set.csv").set_index("id")
    st = pd.read_csv(RES / "mcmc_sim.csv").set_index("id").loc[ts.index]
    ml = pd.read_csv(RES / "mle_sim.csv").set_index("id").loc[ts.index]
    out = {"n": len(ts), "stan_converged": int(st["converged"].sum()), "by_target": {}}
    for t in ["xi", "z10", "z100", "z1000"]:
        truth = on_scale(t, ts[t].to_numpy())
        res = {}
        for name, df in [(f"nbe_seed{s}", pd.read_csv(RES / f"nbe_sim_seed{s}.csv")
                          .set_index("id").loc[ts.index]) for s in (1, 2, 3)
                         if (RES / f"nbe_sim_seed{s}.csv").exists()] + [("stan", st)]:
            q05, q25, q50, q75, q95 = [on_scale(t, v) for v in qcols(df, t)]
            c90 = (q05 <= truth) & (truth <= q95); c50 = (q25 <= truth) & (truth <= q75)
            res[name] = {"rmse_median": float(np.sqrt(np.mean((q50 - truth) ** 2))),
                         "cover90": float(c90.mean()),
                         "cover90_mcse": float(np.sqrt(c90.mean() * (1 - c90.mean()) / len(c90))),
                         "cover50": float(c50.mean()), "mean_width90": float(np.mean(q95 - q05))}
        key = "xi" if t == "xi" else f"log_{t}"
        est = ml[f"{key}_est"].to_numpy(); lo = ml[f"{key}_lo90"].to_numpy()
        hi = ml[f"{key}_hi90"].to_numpy()
        c90 = (lo <= truth) & (truth <= hi)
        res["mle"] = {"rmse_median": float(np.sqrt(np.mean((est - truth) ** 2))),
                      "cover90": float(c90.mean()), "mean_width90": float(np.mean(hi - lo))}
        # agreement of the NBE (main seed) with Stan on simulated data
        nb = pd.read_csv(RES / f"nbe_sim_seed{MAIN_SEED}.csv").set_index("id").loc[ts.index]
        a = [on_scale(t, v) for v in qcols(nb, t)]; b = [on_scale(t, v) for v in qcols(st, t)]
        res["nbe_vs_stan"] = {
            "corr_median": float(np.corrcoef(a[2], b[2])[0, 1]),
            "median_abs_diff_median": float(np.median(np.abs(a[2] - b[2]))),
            "median_width90_ratio": float(np.median((a[4] - a[0]) / (b[4] - b[0])))}
        out["by_target"][t] = res
    out["stan_cover90_converged_only"] = {
        t: float((((on_scale(t, st[f"{t}_q05"]) <= on_scale(t, ts[t])) &
                   (on_scale(t, ts[t]) <= on_scale(t, st[f"{t}_q95"])))[st["converged"]]).mean())
        for t in ["xi", "z100"]}
    return out


# ------------------------------------------------------------------------------------------
# 2. Real cells: NBE vs Stan, pre-registered criteria
# ------------------------------------------------------------------------------------------
def stan_draw_sd(cells):
    """Exact posterior SDs of xi and log z_T from the saved Stan draws."""
    rows = []
    for c in cells:
        d = pd.read_csv(RES / "draws" / f"real_cell{c:03d}.csv")
        mu, sg, xi = d["mu"].to_numpy(), d["sigma"].to_numpy(), d["xi"].to_numpy()
        r = {"id": c, "xi_sd_draws": xi.std(ddof=1)}
        for T in (10, 100, 1000):
            r[f"log_z{T}_sd_draws"] = np.log(nbe.return_level(T, mu, sg, xi)).std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows).set_index("id")


def real_metrics():
    st = pd.read_csv(RES / "mcmc_real.csv").set_index("id")
    cells = st.index.to_numpy()
    sd = stan_draw_sd(cells)
    comp = pd.DataFrame(index=cells)
    out = {"n_cells": len(cells), "stan_converged": int(st["converged"].sum()),
           "stan_total_time_s": float(st["time_s"].sum()), "criteria": {}, "seeds": {}}
    for s in (1, 2, 3):
        f = RES / f"nbe_real_seed{s}.csv"
        if not f.exists():
            continue
        nb = pd.read_csv(f).set_index("id").loc[cells]
        seed_res = {}
        for t in ["xi", "z10", "z100", "z1000"]:
            a = [on_scale(t, v) for v in qcols(nb, t)]; b = [on_scale(t, v) for v in qcols(st, t)]
            sdv = sd["xi_sd_draws"] if t == "xi" else sd[f"log_{t}_sd_draws"]
            dz = np.abs(a[2] - b[2]) / sdv.to_numpy()
            wr = (a[4] - a[0]) / (b[4] - b[0])
            seed_res[t] = {"share_median_within_0.25sd": float(np.mean(dz < 0.25)),
                           "median_abs_diff_in_sd": float(np.median(dz)),
                           "max_abs_diff_in_sd": float(np.max(dz)),
                           "share_width_ratio_0.8_1.25": float(np.mean((wr > 0.8) & (wr < 1.25))),
                           "width_ratio_median": float(np.median(wr)),
                           "width_ratio_range": [float(wr.min()), float(wr.max())]}
            if s == MAIN_SEED:
                comp[f"{t}_diff_in_sd"] = dz; comp[f"{t}_width_ratio"] = wr
                comp[f"{t}_nbe_q50"] = a[2]; comp[f"{t}_stan_q50"] = b[2]
                comp[f"{t}_nbe_q05"] = a[0]; comp[f"{t}_nbe_q95"] = a[4]
                comp[f"{t}_stan_q05"] = b[0]; comp[f"{t}_stan_q95"] = b[4]
        out["seeds"][f"seed{s}"] = seed_res
    m = out["seeds"][f"seed{MAIN_SEED}"]
    for t in ["xi", "z100"]:
        out["criteria"][t] = {
            "medians_within_0.25sd_share": m[t]["share_median_within_0.25sd"],
            "width_ratio_in_range_share": m[t]["share_width_ratio_0.8_1.25"],
            "pass": m[t]["share_median_within_0.25sd"] >= 0.9 and
                    m[t]["share_width_ratio_0.8_1.25"] >= 0.9}
    out["reproduces_mcmc"] = all(v["pass"] for v in out["criteria"].values())
    comp.index.name = "cell"
    comp.to_csv(RES / "real_comparison.csv")
    return out


# ------------------------------------------------------------------------------------------
# 3. Held-out winters (the only place they are read) and 4. Storm Desmond
# ------------------------------------------------------------------------------------------
def heldout_metrics(B=2000, seed=11):
    wm = pd.read_csv(ROOT / "data" / "winter_maxima.csv")
    tr = wm[wm["set"] == "train"]; ho = wm[wm["set"].isin(["val", "test"])].copy()
    assert ho["winter"].nunique() == 25 and tr["winter"].nunique() == 39
    st = pd.read_csv(RES / "mcmc_real.csv").set_index("id")
    nb = pd.read_csv(RES / f"nbe_real_seed{MAIN_SEED}.csv").set_index("id")
    ml = pd.read_csv(RES / "mle_real.csv").set_index("id")
    z10 = {"nbe": nb["z10_q50"], "stan": st["z10_q50"], "mle": np.exp(ml["log_z10_est"]),
           "empirical_training_q90": tr.groupby("cell")["max_mm"].quantile(0.9, interpolation="linear")}
    # type-7 empirical quantile; reference only
    for k, v in z10.items():
        ho[f"above_{k}"] = (ho["max_mm"].to_numpy() > v.loc[ho["cell"]].to_numpy()).astype(float)
        e = ho["max_mm"].to_numpy() - v.loc[ho["cell"]].to_numpy()
        ho[f"pin_{k}"] = np.where(e >= 0, 0.9 * e, -0.1 * e)
    rng = np.random.default_rng(seed)
    winters = ho["winter"].unique()
    by_w = {w: g for w, g in ho.groupby("winter")}
    boots = []
    for _ in range(B):
        g = pd.concat([by_w[w] for w in rng.choice(winters, len(winters), replace=True)])
        boots.append({**{k: g[f"above_{k}"].mean() for k in z10},
                      "pin_nbe_minus_stan": (g["pin_nbe"] - g["pin_stan"]).mean(),
                      "pin_mle_minus_stan": (g["pin_mle"] - g["pin_stan"]).mean()})
    bt = pd.DataFrame(boots)
    out = {"n_winters": int(len(winters)), "n_maxima": int(len(ho)), "rate_above_z10": {},
           "pinball_0.9": {k: float(ho[f"pin_{k}"].mean()) for k in ["nbe", "stan", "mle"]}}
    for k in z10:
        out["rate_above_z10"][k] = {"rate": float(ho[f"above_{k}"].mean()),
                                    "ci95": [float(bt[k].quantile(0.025)), float(bt[k].quantile(0.975))]}
    for d in ["pin_nbe_minus_stan", "pin_mle_minus_stan"]:
        a, b = d.split("_minus_")
        out[d] = {"mean": float((ho[f"pin_{a.split('_')[1]}"] - ho[f"pin_{b}"]).mean()),
                  "ci95": [float(bt[d].quantile(0.025)), float(bt[d].quantile(0.975))]}
    ho[["cell", "winter", "set", "max_mm"] + [c for c in ho.columns if c.startswith(("above_", "pin_"))]] \
        .to_csv(RES / "heldout_by_cell.csv", index=False)
    # Storm Desmond at its wettest cell
    des = pd.read_csv(ROOT / "data" / "desmond_field.csv")
    peak = int(des.loc[des["rain_mm"].idxmax(), "cell"]); y = float(des["rain_mm"].max())
    d = pd.read_csv(RES / "draws" / f"real_cell{peak:03d}.csv")
    mu, sg, xi = d["mu"].to_numpy(), d["sigma"].to_numpy(), d["xi"].to_numpy()
    zz = (y - mu) / sg; tt = 1 + xi * zz
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(tt > 0, np.exp(-np.maximum(tt, 1e-300) ** (-1 / xi)), np.where(xi > 0, 0.0, 1.0))
    rp = 1 / np.maximum(1 - F, 1e-300)
    out["desmond"] = {
        "cell": peak, "rain_mm": y,
        "training_max_at_cell": float(tr.loc[tr["cell"] == peak, "max_mm"].max()),
        "stan_return_period_median": float(np.median(rp)),
        "stan_return_period_ci90": [float(np.quantile(rp, 0.05)), float(np.quantile(rp, 0.95))],
        "z100": {"nbe": [float(nb.loc[peak, f"z100_{l}"]) for l in ("q05", "q50", "q95")],
                 "stan": [float(st.loc[peak, f"z100_{l}"]) for l in ("q05", "q50", "q95")]},
        "z1000": {"nbe": [float(nb.loc[peak, f"z1000_{l}"]) for l in ("q05", "q50", "q95")],
                  "stan": [float(st.loc[peak, f"z1000_{l}"]) for l in ("q05", "q50", "q95")]}}
    return out


if __name__ == "__main__":
    res = {"simulation": simulation_metrics(), "real": real_metrics(), "heldout": heldout_metrics()}
    tr = [json.load(open(RES / f"training_seed{s}.json")) for s in (1, 2, 3)
          if (RES / f"training_seed{s}.json").exists()]
    res["training"] = [{k: t[k] for k in ("seed", "best_val_loss", "epochs_run", "train_time_s")}
                       for t in tr]
    res["timing"] = {"stan_256_cells_s": res["real"]["stan_total_time_s"],
                     "nbe_256_cells_s": json.load(open(RES / "nbe_apply.json"))
                     [f"seed{MAIN_SEED}_time_256_cells_s"]}
    json.dump(res, open(RES / "evaluation.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
