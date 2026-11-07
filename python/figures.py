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


def check_no_overlap(fig, name, pad=1.0):
    """Fail if any two text elements overlap, if any text runs into another panel or colorbar,
    or if a legend covers plotted data. Run on the final layout, before saving."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [tx for tx in fig.texts if tx.get_visible() and tx.get_text().strip()]
    owner = {}
    for a in fig.axes:
        items = [a.title, a._left_title, a._right_title, a.xaxis.label, a.yaxis.label]
        # only tick labels that are actually drawn (inside the axis view limits)
        x0, x1 = sorted(a.get_xlim()); y0, y1 = sorted(a.get_ylim())
        eps = 1e-9 * max(1.0, abs(x1 - x0), abs(y1 - y0))
        items += [tl for tl in a.get_xticklabels() if x0 - eps <= tl.get_position()[0] <= x1 + eps]
        items += [tl for tl in a.get_yticklabels() if y0 - eps <= tl.get_position()[1] <= y1 + eps]
        leg = a.get_legend()
        if leg is not None:
            items += leg.get_texts()
        for tx in items:
            if tx.get_visible() and tx.get_text().strip():
                texts.append(tx); owner[id(tx)] = a
    boxes = [tx.get_window_extent(r).expanded(1.0, 1.0).padded(pad) for tx in texts]
    problems = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                problems.append(f"text '{texts[i].get_text()}' overlaps '{texts[j].get_text()}'")
        for a in fig.axes:
            if owner.get(id(texts[i])) is a:
                continue
            if boxes[i].overlaps(a.get_window_extent(r)):
                problems.append(f"text '{texts[i].get_text()}' runs into another panel")
    for a in fig.axes:
        leg = a.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(r)
        for art in list(a.patches) + list(a.lines) + list(a.collections) + list(a.images):
            try:
                if art.get_window_extent(r).overlaps(lb) and art.get_visible():
                    problems.append(f"legend covers data in panel '{a.get_title()}'")
                    break
            except Exception:
                pass
    if problems:
        raise SystemExit(f"{name}: " + "; ".join(sorted(set(problems))))
    print(f"{name}: no overlaps")

# 1. Real cells: NBE vs Stan -------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
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
a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.suptitle("Real cells, 39 training winters each. Bars: 90% intervals (orange Stan, blue neural estimator)")
check_no_overlap(fig, "nbe_vs_mcmc_real.png")
fig.savefig(FIG / "nbe_vs_mcmc_real.png"); plt.close(fig)

# 2. Simulation: calibration and accuracy ------------------------------------------------
sim = ev["simulation"]["by_target"]
targets = [("xi", "ξ"), ("z10", "log z₁₀"), ("z100", "log z₁₀₀"), ("z1000", "log z₁₀₀₀")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
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
ax[1].set_ylabel("RMSE of point estimate\n(posterior median; MLE)")
ax[1].set_title("Accuracy on the same datasets")
for a in ax:
    a.set_xticks(x); a.set_xticklabels([l for _, l in targets])
fig.legend(*ax[1].get_legend_handles_labels(), frameon=False, loc="outside lower center", ncol=3)
check_no_overlap(fig, "simulation_calibration.png")
fig.savefig(FIG / "simulation_calibration.png"); plt.close(fig)

# 3. Maps: NBE return levels, and the NBE - Stan difference in posterior SDs ----------------
cells = pd.read_csv(ROOT / "data" / "cells.csv").set_index("cell")
nb = pd.read_csv(RES / "nbe_real_seed1.csv").set_index("id")
grid = lambda v: v.reindex(cells.index).to_numpy().reshape(16, 16)  # rows = iy (south first)
ext = [320, 400, 480, 560]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
for a, v, title, cmap, kw in [
    (ax[0], nb["xi_q50"], "NBE posterior median of ξ\n(one cell at a time)", "PuOr_r", {"vmin": -0.15, "vmax": 0.35}),
    (ax[1], nb["z100_q50"], "NBE posterior median\n100-winter level (mm)", "viridis", {}),
    (ax[2], comp.set_index("cell")["z100_diff_in_sd"],
     "|NBE − Stan| median of log z₁₀₀\n(in Stan posterior SDs)", "Greys", {"vmin": 0, "vmax": 0.25})]:
    im = a.imshow(grid(v), origin="lower", extent=ext, cmap=cmap, **kw)
    a.set_title(title); a.set_xlabel("Easting (km)"); a.set_ylabel("Northing (km)")
    fig.colorbar(im, ax=a, shrink=0.85, pad=0.02)
d = ev["heldout"]["desmond"]
ax[1].plot(*cells.loc[d["cell"], ["x_km", "y_km"]], marker="x", color="red", ms=8)
check_no_overlap(fig, "maps.png")
fig.savefig(FIG / "maps.png"); plt.close(fig)
print("figures written")
