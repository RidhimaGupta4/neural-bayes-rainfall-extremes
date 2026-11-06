# JAGS cross-check of the Stan implementation on 16 real cells (the south-west cell of each
# 4 x 4 region), 39 TRAINING winters each. Same likelihood and prior.
# Output: results/jags_check.csv (JAGS and Stan posterior quantiles side by side)

source(file.path("R", "utils.R"))
suppressPackageStartupMessages({ library(rjags); library(posterior) })
cells <- read.csv(repo_path("data", "cells.csv"))
wm <- read.csv(repo_path("data", "winter_maxima.csv")); check_split(wm)
tr <- wm[wm$set == "train", ]
stopifnot(all(tr$set == "train"), length(unique(tr$winter)) == 39)
pick <- cells$cell[(cells$ix - 1) %% 4 == 0 & (cells$iy - 1) %% 4 == 0]
stopifnot(length(pick) == 16)
stan <- read.csv(repo_path("results", "mcmc_real.csv"))

rows <- lapply(pick, function(s) {
  y <- tr$max_mm[tr$cell == s][order(tr$winter[tr$cell == s])]
  sg <- min(max(sd(y) * sqrt(6) / pi, 2.5), 49); m <- min(max(mean(y) - 0.5772 * sg, 10.5), 149.5)
  inits <- lapply(1:3, function(k) list(mu = m, sigma = sg, xi_raw = 0.55,
                                        .RNG.name = "base::Mersenne-Twister", .RNG.seed = s * 10 + k))
  jm <- jags.model(repo_path("jags", "gev_cell.jags"), data = list(y = y, W = 39, zero = 0),
                   inits = inits, n.chains = 3, quiet = TRUE)
  update(jm, 5000, progress.bar = "none")
  smp <- coda.samples(jm, c("xi", "z100"), n.iter = 10000, thin = 5, progress.bar = "none")
  dr <- as_draws_array(aperm(simplify2array(lapply(smp, as.matrix)), c(1, 3, 2)))
  sm <- summarise_draws(dr, "rhat", "ess_bulk")
  dm <- as_draws_matrix(dr)
  qx <- quantile(dm[, "xi"], c(0.05, 0.5, 0.95), names = FALSE)
  qz <- quantile(dm[, "z100"], c(0.05, 0.5, 0.95), names = FALSE)
  st <- stan[stan$id == s, ]
  data.frame(cell = s,
             jags_xi_q05 = qx[1], jags_xi_q50 = qx[2], jags_xi_q95 = qx[3],
             stan_xi_q05 = st$xi_q05, stan_xi_q50 = st$xi_q50, stan_xi_q95 = st$xi_q95,
             jags_z100_q05 = qz[1], jags_z100_q50 = qz[2], jags_z100_q95 = qz[3],
             stan_z100_q05 = st$z100_q05, stan_z100_q50 = st$z100_q50, stan_z100_q95 = st$z100_q95,
             jags_max_rhat = max(sm$rhat), jags_min_ess = min(sm$ess_bulk))
})
out <- do.call(rbind, rows)
write.csv(out, repo_path("results", "jags_check.csv"), row.names = FALSE)
print(out[, c("cell", "jags_xi_q50", "stan_xi_q50", "jags_z100_q50", "stan_z100_q50",
              "jags_max_rhat")])
