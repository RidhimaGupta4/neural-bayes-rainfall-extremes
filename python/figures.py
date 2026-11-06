"""Figures from the saved results (no fitting here). Usage: python python/figures.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C_NBE, C_STAN, C_MLE = "#0072B2", "#D55E00", "#7f7f7f"
ev = json.load(open(RES / "evaluation.json"))
comp = pd.read_csv(RES / "real_comparison.csv")

# 1. Real cells: NBE vs Stan -------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for k, (t, lab) in enumerate([("xi", "shape ξ"), ("z100", "log 100-winter level")]):
    a = ax[k]
    a.errorbar(comp[f"{t}_stan_q50"], comp[f"{t}_nbe_q50"],
               xerr=[comp[f"{t}_stan_q50"] - comp[f"{t}_stan_q05"], comp[f"{t}_stan_q95"] - comp[f"{t}_stan_q50"]],
               fmt="none", ecolor=C_STAN, alpha=0.15, lw=0.6)
    a.errorbar(comp[f"{t}_stan_q50"], comp[f"{t}_nbe_q50"],
               yerr=[comp[f"{t}_nbe_q50"] - comp[f"{t}_nbe_q05"], comp[f"{t}_nbe_q95"] - comp[f"{t}_nbe_q50"]],
               fmt="none", ecolor=C_NBE, alpha=0.15, lw=0.6)
    a.scatter(comp[f"{t}_stan_q50"], comp[f"{t}_nbe_q50"], s=6, color="black", zorder=3)
    lo = min(comp[f"{t}_stan_q05"].min(), comp[f"{t}_nbe_q05"].min())
    hi = max(comp[f"{t}_stan_q95"].max(), comp[f"{t}_nbe_q95"].max())
    a.plot([lo, hi], [lo, hi], color="grey", lw=0.8)
    a.set_xlabel("Stan (MCMC) posterior median")
    a.set_ylabel("neural Bayes estimator posterior median")
    a.set_title(f"{lab[0].upper() + lab[1:]}, 256 cells")
a = ax[2]
bins = np.linspace(0.85, 1.15, 31)
a.hist(comp["xi_width_ratio"], bins=bins, alpha=0.6, color=C_NBE, label="ξ")
a.hist(comp["z100_width_ratio"], bins=bins, alpha=0.6, color=C_STAN, label="100-winter level")
for x in (0.8, 1.25):
    a.axvline(x, color="black", ls="--", lw=0.8)
a.set_xlim(0.78, 1.27)
a.set_xlabel("90% interval width, NBE / Stan"); a.set_ylabel("cells")
a.set_title("90% interval width ratio\n(pre-registered range 0.8–1.25, dashed)")
a.legend(frameon=False)
fig.suptitle("Real cells, 39 training winters each. Bars: 90% intervals (orange Stan, blue neural estimator)", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "nbe_vs_mcmc_real.png"); plt.close(fig)

# 2. Simulation: calibration and accuracy ------------------------------------------------
sim = ev["simulation"]["by_target"]
targets = [("xi", "ξ"), ("z10", "log z₁₀"), ("z100", "log z₁₀₀"), ("z1000", "log z₁₀₀₀")]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
x = np.arange(len(targets)); w = 0.26
for j, (m, c, lab) in enumerate([("nbe_seed1", C_NBE, "neural Bayes estimator"),
                                  ("stan", C_STAN, "Stan (MCMC)"), ("mle", C_MLE, "MLE, Wald interval")]):
    cov = [sim[t][m]["cover90"] for t, _ in targets]
    ax[0].bar(x + (j - 1) * w, cov, w, color=c, label=lab)
    rm = [sim[t][m]["rmse_median"] for t, _ in targets]
    ax[1].bar(x + (j - 1) * w, rm, w, color=c, label=lab)
ax[0].axhspan(0.87, 0.93, color="grey", alpha=0.15, lw=0)
ax[0].axhline(0.9, color="black", lw=0.8)
ax[0].set_ylim(0.75, 0.95); ax[0].set_ylabel("coverage of 90% intervals")
ax[0].set_title("Calibration on 1,000 simulated datasets\n(shaded: pre-registered range 0.87–0.93)")
ax[1].set_ylabel("RMSE of point estimate (posterior median / MLE)")
ax[1].set_title("Accuracy on the same datasets")
for a in ax:
    a.set_xticks(x); a.set_xticklabels([l for _, l in targets])
ax[1].legend(frameon=False, loc="upper left")
fig.savefig(FIG / "simulation_calibration.png"); plt.close(fig)

# 3. Maps: NBE return levels, and the NBE - Stan difference in posterior SDs ----------------
cells = pd.read_csv(ROOT / "data" / "cells.csv").set_index("cell")
nb = pd.read_csv(RES / "nbe_real_seed1.csv").set_index("id")
grid = lambda v: v.reindex(cells.index).to_numpy().reshape(16, 16)  # rows = iy (south first)
ext = [320, 400, 480, 560]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, v, title, cmap, kw in [
    (ax[0], nb["xi_q50"], "NBE posterior median of ξ\n(one cell at a time)", "PuOr_r", {"vmin": -0.15, "vmax": 0.35}),
    (ax[1], nb["z100_q50"], "NBE posterior median\n100-winter level (mm)", "viridis", {}),
    (ax[2], comp.set_index("cell")["z100_diff_in_sd"],
     "|NBE − Stan| median of log z₁₀₀\n(in Stan posterior SDs)", "Greys", {"vmin": 0, "vmax": 0.25})]:
    im = a.imshow(grid(v), origin="lower", extent=ext, cmap=cmap, **kw)
    a.set_title(title); a.set_xlabel("Easting (km)"); a.set_ylabel("Northing (km)")
    fig.colorbar(im, ax=a, shrink=0.85)
d = ev["heldout"]["desmond"]
ax[1].plot(*cells.loc[d["cell"], ["x_km", "y_km"]], marker="x", color="red", ms=8)
fig.savefig(FIG / "maps.png"); plt.close(fig)
print("figures written")
