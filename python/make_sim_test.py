"""Simulated test set with known truth: 1,000 datasets of 39 maxima drawn from the prior.

Seed 20261106 is used only here (training uses seeds 1-3 and validation seed 999999).
Output: results/sim/test_set.csv (truth columns, then y1..y39)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(20261106)
mu, sigma, xi = nbe.draw_prior(rng, 1000)
y = nbe.simulate(rng, mu, sigma, xi)
df = pd.DataFrame({"id": np.arange(1, 1001), "mu": mu, "sigma": sigma, "xi": xi,
                   **{f"z{T}": nbe.return_level(T, mu, sigma, xi) for T in nbe.RETURN_PERIODS}})
ydf = pd.DataFrame(y, columns=[f"y{i}" for i in range(1, nbe.N_WINTERS + 1)])
(ROOT / "results" / "sim").mkdir(parents=True, exist_ok=True)
pd.concat([df, ydf], axis=1).to_csv(ROOT / "results" / "sim" / "test_set.csv", index=False)
print("wrote", len(df), "test datasets")
