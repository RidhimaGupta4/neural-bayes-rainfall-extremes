"""Train the neural Bayes estimator on simulations from the prior only.

Usage: python python/train.py <seed>
Outputs: models/nbe_seed<seed>.pt, results/training_seed<seed>.json

No rainfall data are read here. Fresh parameters and samples are simulated every epoch; early
stopping uses a fixed simulated validation set (its own seed, never used for training).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"width": 256, "depth": 3, "n_per_epoch": 200_000, "batch": 2048, "lr": 1e-3,
            "max_epochs": 200, "patience": 15, "lr_patience": 5, "lr_factor": 0.5,
            "n_val": 50_000, "val_seed": 999_999}


def simulate_set(rng, n):
    mu, sigma, xi = nbe.draw_prior(rng, n)
    y = nbe.simulate(rng, mu, sigma, xi)
    x, m, s = nbe.features(y)
    t = nbe.standardised_targets(mu, sigma, xi, m, s)
    return torch.from_numpy(x), torch.from_numpy(t)


def main(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    xv, tv = simulate_set(np.random.default_rng(SETTINGS["val_seed"]), SETTINGS["n_val"])
    taus = torch.tensor(nbe.TAUS, dtype=torch.float32)
    model = nbe.NBE(width=SETTINGS["width"], depth=SETTINGS["depth"])
    opt = torch.optim.Adam(model.parameters(), lr=SETTINGS["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=SETTINGS["lr_factor"], patience=SETTINGS["lr_patience"])
    best, best_state, bad, history = np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(1, SETTINGS["max_epochs"] + 1):
        model.train()
        x, t = simulate_set(rng, SETTINGS["n_per_epoch"])
        perm = torch.randperm(len(x))
        tot = 0.0
        for i in range(0, len(x), SETTINGS["batch"]):
            idx = perm[i:i + SETTINGS["batch"]]
            loss = nbe.pinball_loss(model(x[idx]), t[idx], taus)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = sum(nbe.pinball_loss(model(xv[i:i + 10000]), tv[i:i + 10000], taus).item()
                     * len(xv[i:i + 10000]) for i in range(0, len(xv), 10000)) / len(xv)
        sched.step(vl)
        history.append({"epoch": epoch, "train_loss": tot / len(x), "val_loss": vl,
                        "lr": opt.param_groups[0]["lr"]})
        print(f"seed {seed} epoch {epoch} train {tot / len(x):.5f} val {vl:.5f}", flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= SETTINGS["patience"]:
                break
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(best_state, ROOT / "models" / f"nbe_seed{seed}.pt")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump({"seed": seed, "settings": SETTINGS, "best_val_loss": best,
               "epochs_run": len(history), "train_time_s": time.time() - t0,
               "torch_version": torch.__version__, "history": history},
              open(ROOT / "results" / f"training_seed{seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]))
