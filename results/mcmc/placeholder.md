# results/mcmc/

Intermediate Stan output saved in chunks of 50 fits by `R/10_mcmc.R`, so that interrupted runs
resume without repeating fits (`sim_chunkNN.rds`: simulated test set; `real_chunkNN.rds`: real
cells).

These hold the first-pass fits. The final values, including the refits of runs that failed the
convergence rule (pre-registration Amendment 2), are in `results/mcmc_sim.csv` and
`results/mcmc_real.csv`.
