# MCMC reference (Stan): the same likelihood and prior as the neural Bayes estimator.
#
# Usage: Rscript R/10_mcmc.R sim     # 1,000 simulated test datasets (results/sim/test_set.csv)
#        Rscript R/10_mcmc.R real    # 256 cells, 39 TRAINING winters each
# Resumable: results are saved in chunks of 50 under results/mcmc/.
# Outputs: results/mcmc_<sim|real>.csv (posterior quantiles, SDs, diagnostics, time per fit);
#          results/draws/real_cell<NNN>.rds (posterior draws for real cells; not committed)

source(file.path("R", "utils.R"))
suppressPackageStartupMessages({ library(cmdstanr); library(posterior) })
if (Sys.getenv("CMDSTAN") != "") set_cmdstan_path(Sys.getenv("CMDSTAN"))

TAUS <- c(0.05, 0.25, 0.5, 0.75, 0.95)
VARS <- c("mu", "log_sigma", "xi", "z10", "z100", "z1000")
CHUNK <- 50
mode <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(mode %in% c("sim", "real"))
out_dir <- repo_path("results", "mcmc"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(repo_path("results", "draws"), recursive = TRUE, showWarnings = FALSE)

if (mode == "sim") {
  ts <- read.csv(repo_path("results", "sim", "test_set.csv"))
  Y <- as.matrix(ts[, paste0("y", 1:39)]); ids <- ts$id
} else {
  wm <- read.csv(repo_path("data", "winter_maxima.csv"))
  check_split(wm)
  tr <- wm[wm$set == "train", ]                      # TRAINING winters only
  stopifnot(all(tr$set == "train"), length(unique(tr$winter)) == 39)
  ids <- sort(unique(tr$cell))
  Y <- t(sapply(ids, function(s) tr$max_mm[tr$cell == s][order(tr$winter[tr$cell == s])]))
}
stopifnot(ncol(Y) == 39, !anyNA(Y))

mod <- cmdstan_model(repo_path("stan", "gev_cell.stan"))

inits_for <- function(y, chains, seed) {
  set.seed(seed)
  s <- min(max(sd(y) * sqrt(6) / pi, 2.5), 49)
  m <- min(max(mean(y) - 0.5772 * s, 10.5), 149.5)
  # jitter, then clip so every chain starts strictly inside the prior bounds
  lapply(seq_len(chains), function(k) list(
    mu = min(max(m * exp(rnorm(1, 0, 0.01)), 10.5), 149.5),
    sigma = min(max(s * exp(rnorm(1, 0, 0.01)), 2.5), 49.5),
    xi = 0.05 + rnorm(1, 0, 0.005)))
}

fit_one <- function(y, id, seed) {
  t0 <- Sys.time()
  fit <- mod$sample(data = list(n = length(y), y = y), seed = seed, chains = 4,
                    parallel_chains = 2, iter_warmup = 1000, iter_sampling = 1000,
                    adapt_delta = 0.95, refresh = 0, show_messages = FALSE,
                    show_exceptions = FALSE, init = inits_for(y, 4, seed))
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  dr <- fit$draws(variables = VARS)
  sm <- summarise_draws(dr, ~quantile(.x, TAUS, names = FALSE), "sd", "rhat", "ess_bulk",
                        "ess_tail")
  dg <- fit$diagnostic_summary(quiet = TRUE)
  row <- data.frame(id = id, time_s = el, max_rhat = max(sm$rhat), min_ess_bulk = min(sm$ess_bulk),
                    min_ess_tail = min(sm$ess_tail), divergences = sum(dg$num_divergent))
  for (v in VARS) {
    r <- sm[sm$variable == v, ]
    q <- unlist(r[, 2:6])
    row[paste0(v, "_q", c("05", "25", "50", "75", "95"))] <- as.list(q)
    row[paste0(v, "_sd")] <- r$sd
  }
  if (mode == "real") saveRDS(as_draws_matrix(fit$draws(variables = c("mu", "sigma", "xi"))),
                              repo_path("results", "draws", sprintf("real_cell%03d.rds", id)))
  row
}

chunks <- split(seq_along(ids), ceiling(seq_along(ids) / CHUNK))
for (k in seq_along(chunks)) {
  f <- file.path(out_dir, sprintf("%s_chunk%02d.rds", mode, k))
  if (file.exists(f)) next
  rows <- lapply(chunks[[k]], function(i) fit_one(Y[i, ], ids[i], seed = 5000 + ids[i]))
  saveRDS(do.call(rbind, rows), f)
  cat(mode, "chunk", k, "of", length(chunks), "done\n")
}
all_rows <- do.call(rbind, lapply(seq_along(chunks), function(k)
  readRDS(file.path(out_dir, sprintf("%s_chunk%02d.rds", mode, k)))))
all_rows$converged <- with(all_rows, max_rhat < 1.01 & min_ess_bulk > 400 & min_ess_tail > 400 &
                             divergences == 0)
write.csv(all_rows, repo_path("results", sprintf("mcmc_%s.csv", mode)), row.names = FALSE)
cat("MCMC", mode, ":", nrow(all_rows), "fits,", sum(all_rows$converged), "converged\n")
