# figures/

Figures produced by `python/figures.py` from the saved results; no model is fitted there. Before
saving, every figure is checked automatically for overlapping text and for legends covering data.

| File | Shows |
|---|---|
| `nbe_vs_mcmc_real.png` | Neural Bayes estimator vs Stan posterior medians and 90% intervals for ξ and the log 100-winter level in all 256 cells, and the ratio of their interval widths |
| `simulation_calibration.png` | Coverage of 90% intervals and accuracy on 1,000 simulated datasets with known truth, for the estimator, Stan and maximum likelihood |
| `maps.png` | Maps of the estimator's posterior median ξ and 100-winter level, and its distance from Stan in posterior SDs; Storm Desmond's peak cell marked |
