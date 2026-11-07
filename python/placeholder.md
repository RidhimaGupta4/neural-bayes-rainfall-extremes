# python/

The neural Bayes estimator and the evaluation.

| File | Purpose |
|---|---|
| `nbe.py` | GEV quantile and return-level functions, the prior, input features, the network (median plus positive increments, so quantiles never cross) and the pinball loss |
| `train.py` | Trains one network on fresh prior simulations every epoch, with early stopping on a fixed simulated validation set; reads no rainfall data |
| `make_sim_test.py` | Creates the 1,000 simulated test datasets with known truth (own seed, never used in training) |
| `apply.py` | Applies the trained networks to the simulated test set and to the real cells' training winters; out-of-distribution check; timing |
| `evaluate.py` | All evaluation against the pre-registered criteria; the only code that reads validation and test winters |
| `figures.py` | Figures, with an automatic check for overlapping text |
