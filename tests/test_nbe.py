"""Unit tests for the NBE building blocks. Run: python tests/test_nbe.py"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import genextreme

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import nbe  # noqa: E402


def test_gev_quantile_matches_scipy():
    # scipy's genextreme uses c = -xi
    for xi in [-0.3, -0.05, 1e-8, 0.05, 0.2, 0.45]:
        p = np.array([0.01, 0.3, 0.9, 0.999])
        ours = nbe.gev_quantile(p, 40.0, 12.0, xi)
        ref = genextreme.ppf(p, -xi, loc=40.0, scale=12.0)
        assert np.allclose(ours, ref, rtol=1e-6, atol=1e-6), (xi, ours, ref)


def test_return_level_definition():
    z = nbe.return_level(100, 40.0, 12.0, 0.1)
    assert np.isclose(genextreme.cdf(z, -0.1, loc=40.0, scale=12.0), 0.99)


def test_prior_ranges():
    mu, sigma, xi = nbe.draw_prior(np.random.default_rng(1), 100_000)
    assert mu.min() >= 10 and mu.max() <= 150
    assert sigma.min() >= 2 and sigma.max() <= 50
    assert xi.min() > -0.5 and xi.max() < 0.5
    assert abs(xi.mean() - 0.1) < 0.005          # Beta(9, 6) - 0.5 has mean 0.1


def test_features_permutation_invariant_and_targets_invert():
    rng = np.random.default_rng(2)
    mu, sigma, xi = nbe.draw_prior(rng, 50)
    y = nbe.simulate(rng, mu, sigma, xi)
    x1, m, s = nbe.features(y)
    x2, _, _ = nbe.features(y[:, ::-1].copy())
    assert np.allclose(x1, x2)
    t = nbe.standardised_targets(mu, sigma, xi, m, s)
    q = np.repeat(t[:, :, None], 5, axis=2)            # "quantiles" equal to the truth
    back = nbe.back_transform(q, m, s)
    assert np.allclose(back["mu"][:, 2], mu, rtol=1e-4)
    assert np.allclose(np.exp(back["log_sigma"][:, 2]), sigma, rtol=1e-4)
    assert np.allclose(back["z100"][:, 2], nbe.return_level(100, mu, sigma, xi), rtol=1e-4)


def test_quantiles_never_cross():
    torch.manual_seed(0)
    q = nbe.NBE()(torch.randn(500, nbe.N_WINTERS + 2)).detach().numpy()
    assert np.all(np.diff(q, axis=2) >= 0)


def test_pinball_minimiser_is_the_quantile():
    # the constant minimising mean pinball loss over a sample is its tau-quantile
    y = torch.tensor(np.random.default_rng(3).gamma(2.0, size=20001), dtype=torch.float32)
    taus = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    grid = torch.linspace(0, 10, 2001)
    for k, tau in enumerate(taus):
        losses = [nbe.pinball_loss(torch.full((len(y), 1, 1), float(c)), y[:, None],
                                   tau.view(1)).item() for c in grid]
        best = grid[int(np.argmin(losses))].item()
        assert abs(best - np.quantile(y.numpy(), tau.item())) < 0.02, (tau, best)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
