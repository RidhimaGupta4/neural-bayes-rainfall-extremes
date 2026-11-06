"""Neural Bayes estimator (NBE) for the GEV distribution of 39 winter maxima.

The network maps a sample of n = 39 i.i.d. maxima to posterior quantiles (levels in TAUS) of
mu, log sigma, xi and the 10-, 100- and 1,000-winter return levels, under the prior in
PREREGISTRATION.md. It is trained only on samples simulated from that prior.

Equivariant set-up: the sorted sample is standardised by its mean m and SD s; the network
predicts quantiles of (mu - m)/s, log(sigma/s), xi and (z_T - m)/s, and m/100 and log s are
also inputs (the prior is not location-scale invariant). Every back-transformation is monotone,
so quantiles map to quantiles exactly.
"""
import numpy as np
import torch
from torch import nn

N_WINTERS = 39
TAUS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
RETURN_PERIODS = (10, 100, 1000)
TARGETS = ["mu", "log_sigma", "xi", "z10", "z100", "z1000"]
PRIOR = {"mu": (10.0, 150.0), "sigma": (2.0, 50.0), "xi_beta": (9.0, 6.0)}


# ------------------------------------------------------------------------------------------
# GEV and the prior
# ------------------------------------------------------------------------------------------
def gev_quantile(p, mu, sigma, xi):
    """GEV quantile; xi > 0 is a heavy tail; |xi| < 1e-6 uses the Gumbel limit."""
    p, mu, sigma, xi = np.broadcast_arrays(p, mu, sigma, xi)
    yp = -np.log(p)
    out = np.empty(np.shape(p), dtype=float)
    gum = np.abs(xi) < 1e-6
    out[gum] = mu[gum] - sigma[gum] * np.log(yp[gum])
    ng = ~gum
    out[ng] = mu[ng] + sigma[ng] / xi[ng] * (yp[ng] ** (-xi[ng]) - 1.0)
    return out


def return_level(T, mu, sigma, xi):
    return gev_quantile(1.0 - 1.0 / T, mu, sigma, xi)


def draw_prior(rng, size):
    mu = rng.uniform(*PRIOR["mu"], size)
    sigma = rng.uniform(*PRIOR["sigma"], size)
    xi = rng.beta(*PRIOR["xi_beta"], size) - 0.5
    return mu, sigma, xi


def simulate(rng, mu, sigma, xi, n=N_WINTERS):
    u = rng.uniform(size=(len(mu), n))
    return gev_quantile(u, mu[:, None], sigma[:, None], xi[:, None])


# ------------------------------------------------------------------------------------------
# Features and targets
# ------------------------------------------------------------------------------------------
def features(y):
    """y: (B, n) samples -> (B, n + 2) features, plus m and s for back-transformation."""
    ys = np.sort(y, axis=1)
    m = ys.mean(axis=1)
    s = ys.std(axis=1, ddof=1)
    z = (ys - m[:, None]) / s[:, None]
    x = np.concatenate([z, (m / 100.0)[:, None], np.log(s)[:, None]], axis=1)
    return x.astype(np.float32), m, s


def standardised_targets(mu, sigma, xi, m, s):
    cols = [(mu - m) / s, np.log(sigma / s), xi]
    cols += [(return_level(T, mu, sigma, xi) - m) / s for T in RETURN_PERIODS]
    return np.stack(cols, axis=1).astype(np.float32)


def back_transform(q, m, s):
    """q: (B, n_targets, n_taus) standardised quantiles -> dict of quantile arrays on the
    natural scale (mm for mu and z_T; log mm for log_sigma)."""
    m = m[:, None]; s = s[:, None]
    out = {"mu": m + s * q[:, 0, :], "log_sigma": np.log(s) + q[:, 1, :], "xi": q[:, 2, :]}
    for j, T in enumerate(RETURN_PERIODS):
        out[f"z{T}"] = m + s * q[:, 3 + j, :]
    return out


# ------------------------------------------------------------------------------------------
# Network: median plus positive increments, so quantiles never cross
# ------------------------------------------------------------------------------------------
class NBE(nn.Module):
    def __init__(self, n_in=N_WINTERS + 2, width=256, depth=3,
                 n_targets=len(TARGETS), n_taus=len(TAUS)):
        super().__init__()
        assert n_taus == 5
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_targets * n_taus)
        self.n_targets = n_targets

    def forward(self, x):
        h = self.head(self.body(x)).view(-1, self.n_targets, 5)
        sp = nn.functional.softplus
        med = h[..., 2]
        q25 = med - sp(h[..., 1]); q05 = q25 - sp(h[..., 0])
        q75 = med + sp(h[..., 3]); q95 = q75 + sp(h[..., 4])
        return torch.stack([q05, q25, med, q75, q95], dim=-1)


def pinball_loss(q, y, taus):
    """q: (B, K, L) predicted quantiles; y: (B, K) targets; mean over batch, sum over K, L."""
    diff = y[..., None] - q
    t = taus.view(1, 1, -1)
    return torch.maximum(t * diff, (t - 1.0) * diff).sum(dim=(1, 2)).mean()


def predict(model, y, batch=20000):
    """Posterior quantiles for samples y (B, 39), as a dict target -> (B, 5) array."""
    model.eval()
    x, m, s = features(np.asarray(y, dtype=float))
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(torch.from_numpy(x[i:i + batch])).numpy())
    return back_transform(np.concatenate(out, axis=0), m, s)
