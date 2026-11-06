# Refit every MCMC fit that failed the convergence rule, with adapt_delta = 0.999 (smaller
# steps near the GEV's hard support boundary). Computation only: same model, prior, chains and
# iterations (PREREGISTRATION.md, Amendment 2). The refit replaces the original row only if it
# passes the rule; otherwise the better of the two (fewer divergences) is kept and stays flagged.
# Usage: Rscript R/10b_refit_flagged.R sim | real

source(file.path("R", "utils.R"))
suppressPackageStartupMessages({ library(cmdstanr); library(posterior) })
if (Sys.getenv("CMDSTAN") != "") set_cmdstan_path(Sys.getenv("CMDSTAN"))
mode <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(mode %in% c("sim", "real"))
TAUS <- c(0.05, 0.25, 0.5, 0.75, 0.95)
VARS <- c("mu", "log_sigma", "xi", "z10", "z100", "z1000")
res <- read.csv(repo_path("results", sprintf("mcmc_%s.csv", mode)))
if (!"refit" %in% names(res)) { res$refit <- FALSE; res$divergences_first_fit <- res$divergences }

if (mode == "sim") {
  ts <- read.csv(repo_path("results", "sim", "test_set.csv"))
  getY <- function(id) as.numeric(ts[ts$id == id, paste0("y", 1:39)])
} else {
  wm <- read.csv(repo_path("data", "winter_maxima.csv")); check_split(wm)
  tr <- wm[wm$set == "train", ]
  stopifnot(all(tr$set == "train"), length(unique(tr$winter)) == 39)
  getY <- function(id) tr$max_mm[tr$cell == id][order(tr$winter[tr$cell == id])]
}
mod <- cmdstan_model(repo_path("stan", "gev_cell.stan"))
flag <- which(!res$converged & !res$refit)
cat(length(flag), "fits to refit\n")
for (i in flag) {
  y <- getY(res$id[i]); seed <- 9000 + res$id[i]
  set.seed(seed)
  s <- min(max(sd(y) * sqrt(6) / pi, 2.5), 49); m <- min(max(mean(y) - 0.5772 * s, 10.5), 149.5)
  inits <- lapply(1:4, function(k) list(mu = min(max(m * exp(rnorm(1, 0, 0.01)), 10.5), 149.5),
                                        sigma = min(max(s * exp(rnorm(1, 0, 0.01)), 2.5), 49.5),
                                        xi = 0.05 + rnorm(1, 0, 0.005)))
  t0 <- Sys.time()
  fit <- mod$sample(data = list(n = 39, y = y), seed = seed, chains = 4, parallel_chains = 2,
                    iter_warmup = 1000, iter_sampling = 1000, adapt_delta = 0.999,
                    refresh = 0, show_messages = FALSE, show_exceptions = FALSE, init = inits)
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  dr <- fit$draws(variables = VARS)
  sm <- summarise_draws(dr, ~quantile(.x, TAUS, names = FALSE), "sd", "rhat", "ess_bulk", "ess_tail")
  dv <- sum(fit$diagnostic_summary(quiet = TRUE)$num_divergent)
  conv <- max(sm$rhat) < 1.01 && min(sm$ess_bulk) > 400 && min(sm$ess_tail) > 400 && dv == 0
  res$refit[i] <- TRUE
  if (conv || dv < res$divergences[i]) {
    res$time_s[i] <- res$time_s[i] + el
    res$max_rhat[i] <- max(sm$rhat); res$min_ess_bulk[i] <- min(sm$ess_bulk)
    res$min_ess_tail[i] <- min(sm$ess_tail); res$divergences[i] <- dv; res$converged[i] <- conv
    for (v in VARS) {
      r <- sm[sm$variable == v, ]
      res[i, paste0(v, "_q", c("05", "25", "50", "75", "95"))] <- unlist(r[, 2:6])
      res[i, paste0(v, "_sd")] <- r$sd
    }
    if (mode == "real") {
      dm <- as_draws_matrix(fit$draws(variables = c("mu", "sigma", "xi")))
      write.csv(data.frame(mu = as.numeric(dm[, "mu"]), sigma = as.numeric(dm[, "sigma"]),
                           xi = as.numeric(dm[, "xi"])),
                repo_path("results", "draws", sprintf("real_cell%03d.csv", res$id[i])),
                row.names = FALSE)
    }
  } else {
    res$time_s[i] <- res$time_s[i] + el
  }
}
write.csv(res, repo_path("results", sprintf("mcmc_%s.csv", mode)), row.names = FALSE)
cat(mode, ": converged", sum(res$converged), "of", nrow(res), "; still flagged",
    sum(!res$converged), "; max divergences among flagged",
    if (any(!res$converged)) max(res$divergences[!res$converged]) else 0, "\n")
