# R/

R scripts for the data build and the reference methods (Stan, JAGS, maximum likelihood).

| File | Purpose |
|---|---|
| `utils.R` | GEV density, distribution, quantile and return-level functions; the pre-specified winter split; helper functions |
| `00_build_data.R` | Builds winter (October–March) maxima for 256 cells × 64 winters from the HadUK-Grid file in `data/raw/` and assigns each winter to training, validation or test |
| `10_mcmc.R` | Stan reference posterior for the 1,000 simulated test datasets (`sim`) and the 256 real cells (`real`, 39 training winters each); resumable in chunks |
| `10b_refit_flagged.R` | Refits MCMC runs that failed the convergence rule with `adapt_delta = 0.999` (pre-registration Amendment 2) |
| `11_jags.R` | JAGS cross-check of the Stan implementation on 16 cells |
| `12_mle.R` | Maximum-likelihood baseline with Wald (delta-method) 90% intervals |
| `13_export_draws.R` | Exports Stan posterior draws for the real cells to CSV for the Python evaluation |

Every script that touches rainfall data uses the 39 training winters only.
