# Pre-registered analysis plan

Written and committed before the neural estimator was trained and before any model in this
repository was fitted to rainfall data. The author has analysed the same data before (the pilot
[`rainfall-extremes-pilot`](https://github.com/RidhimaGupta4/rainfall-extremes-pilot) and an
earlier Bayesian spatial analysis), so this is not a blind study of the data; it is a study of an
inference method, and every choice below that could favour the method is fixed here.

## Question

Can a **neural Bayes estimator** (NBE), trained only on Monte Carlo simulations from a GEV prior,
reproduce full Bayesian inference by MCMC for rare winter rainfall at single 5 km cells in
Cumbria: posterior medians and credible intervals for the GEV shape ξ and for 10-, 100- and
1,000-winter return levels? How much faster is it, and where does it fail?

## Data and split

* Winter (October–March) maxima of HadUK-Grid 5 km daily rainfall, 256 cells × 64 winters
  (1961/62–2024/25), built by `R/00_build_data.R` from the pilot's file.
* Split by winter start year (identical to the pilot): years ending 0 or 5 → test (12 winters,
  including Storm Desmond's 2015/16); 3 or 8 → validation (13); the rest → training (39).
* **Every estimate uses the 39 training winters of a cell only.** Validation and test winters are
  read only by the held-out evaluation script. No choice below is tuned on them, so they form one
  pre-specified held-out set of 25 winters.

## Model and prior (shared by every Bayesian method)

For one cell, the 39 training maxima are independent draws from GEV(μ, σ, ξ), ξ > 0 a heavy tail.

* μ ~ Uniform(10, 150) mm
* σ ~ Uniform(2, 50) mm
* ξ + 0.5 ~ Beta(9, 6) (Martins & Stedinger 2000; ξ in (−0.5, 0.5), mean 0.10)

The bounds are physical (UK winter daily maxima have locations of tens of mm and scales of a few
to tens of mm); they were not fitted to these data. The same prior is used for the NBE's training
simulations, for MCMC in Stan and for the JAGS cross-check, so the NBE is judged against the exact
posterior it is meant to approximate.

Targets: μ, log σ, ξ and the return levels z_T for T = 10, 100, 1,000 winters
(z_T = μ + σ[(−log(1 − 1/T))^(−ξ) − 1]/ξ).

## Neural Bayes estimator

* **Input:** the 39 maxima sorted (order statistics are sufficient for an i.i.d. sample, so the
  estimator is permutation-invariant), standardised by the sample mean m and SD s, plus m/100 and
  log s.
* **Output:** posterior quantiles at levels 0.05, 0.25, 0.5, 0.75, 0.95 of (μ − m)/s, log(σ/s),
  ξ and (z_T − m)/s, converted back exactly (the maps are monotone). Non-crossing by construction:
  the network outputs the median and positive increments.
* **Loss:** the quantile (pinball) loss summed over targets and levels. The minimiser of expected
  pinball loss is the posterior quantile, so a well-trained network approximates the posterior
  quantiles under the prior above (Sainsbury-Dale, Zammit-Mangion & Huser 2024; Richards,
  Sainsbury-Dale, Zammit-Mangion & Huser 2024).
* **Training:** multilayer perceptron, 3 hidden layers of 256 units. Parameters drawn from the
  prior and fresh samples simulated on the fly every epoch (no fixed training set to overfit).
  Early stopping on a fixed simulated validation set of 50,000 datasets. Three networks with
  different seeds are trained; the main results use seed 1, and the spread across seeds is
  reported. Architecture and optimiser settings may be changed only on the basis of the simulated
  validation loss, and any change is recorded below.
* **No real rainfall data enter training, validation or tuning.**

## Comparison methods

* **MCMC (reference):** Stan, the same likelihood and prior, 4 chains × (1,000 + 1,000)
  iterations. A fit counts as converged if R-hat < 1.01, bulk and tail ESS > 400 and there are no
  divergences; non-converged fits are reported, not dropped.
* **JAGS cross-check:** the same model (zeros trick) for 16 cells, one per region of 4 × 4 cells
  (the south-west cell of each region), to confirm the Stan implementation.
* **Maximum likelihood:** per-cell MLE with Wald (delta-method) 90% intervals, the classical
  baseline.

## Evaluation

1. **Simulation with known truth** (1,000 datasets drawn from the prior with seeds never used in
   training): RMSE of the posterior median for ξ and log z₁₀₀; coverage and width of 50% and 90%
   intervals; agreement of NBE and Stan quantiles; run time. MLE included.
2. **Real data, 256 cells (training winters):** agreement of NBE and Stan posterior medians,
   in units of the Stan posterior SD, and the ratio of 90% interval widths, for ξ and log z_T.
   An out-of-distribution check: each cell's (m, log s, standardised order statistics) compared
   with the range of the simulated training inputs.
3. **Held-out winters (25):** the proportion of held-out maxima above each method's posterior
   median 10-winter level (target 0.10) and the 0.9 quantile (pinball) score, with 95% intervals
   from a bootstrap over **winters** (cells within a winter are dependent). The training winters'
   own empirical 90% quantiles are reported as a reference, because the test winters are known
   from the pilot to be wetter than the training winters.
4. **Storm Desmond** (5 December 2015, a test winter): at its wettest cell, each method's
   posterior for z₁₀₀ and z₁,₀₀₀ compared with the observed 254.8 mm.

## Criteria (fixed now; reported whatever the outcome)

The NBE is said to **reproduce MCMC** on the real cells if, for log z₁₀₀ and for ξ, in at least
90% of the 256 cells: |NBE median − Stan median| < 0.25 Stan posterior SDs, and the NBE/Stan
ratio of 90% interval widths is between 0.8 and 1.25. Its intervals are said to be **calibrated**
if simulated 90% coverage is within 0.87–0.93 (about ±3 Monte Carlo SEs with 1,000 datasets)
for ξ and log z₁₀₀.

## Known limitations, stated in advance

* One cell at a time: no pooling between cells and no spatial dependence. This study is about the
  inference engine, not the spatial model.
* The NBE is only as good as its prior and training range; data outside the simulated range are
  flagged by the out-of-distribution check rather than trusted.

## Amendments

Each change is recorded here with whether it was made before or after the result it affects.

1. **Stan starting values** (computation only; recorded after 15 of 20 chunks of simulated fits
   and before any real-data MCMC). The random jitter on starting values could place μ just above
   150 or σ just above 50 when the starting point was clipped at the prior bound, so all chains of
   one simulated dataset were rejected before sampling. Starting values are now clipped after the
   jitter. The model, prior, sampler settings and the 750 completed fits are unchanged.
2. **Refits of flagged MCMC fits** (computation only; recorded after the MCMC diagnostics were
   seen and before any comparison with the neural estimator was computed). 80 of 256 real fits and
   188 of 1,000 simulated fits failed the convergence rule, all but 5 of them only through 1–13
   divergent transitions out of 4,000, which come from the GEV's hard support boundary. Every
   flagged fit is refitted once with `adapt_delta = 0.999`, same model, prior, chains and
   iterations (`R/10b_refit_flagged.R`). A refit replaces the original only if it passes the rule
   or has fewer divergences; fits that still fail stay flagged and are reported. Results are
   reported for all fits and, as a sensitivity check, for converged fits only.
