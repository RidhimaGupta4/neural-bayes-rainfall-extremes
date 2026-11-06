# Shared helpers: GEV distribution, the pre-specified split, scores.
# Every function here is deterministic and is unit-tested in tests/testthat.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
repo_path <- function(...) {
  root <- Sys.getenv("BAYES_REPO_ROOT", unset = "")
  if (root == "") {
    root <- normalizePath(".", mustWork = TRUE)
    while (!file.exists(file.path(root, "PREREGISTRATION.md"))) {
      parent <- dirname(root)
      if (parent == root) stop("Run from inside the repository.")
      root <- parent
    }
  }
  file.path(root, ...)
}

# ---------------------------------------------------------------------------
# GEV distribution, xi > 0 = heavy tail. Vectorised over all arguments.
# |xi| < 1e-6 is treated as the Gumbel limit.
# ---------------------------------------------------------------------------
gev_eps <- 1e-6

dgev <- function(y, mu, sigma, xi, log = FALSE) {
  n <- max(length(y), length(mu), length(sigma), length(xi))
  y <- rep_len(y, n); mu <- rep_len(mu, n); sigma <- rep_len(sigma, n); xi <- rep_len(xi, n)
  z <- (y - mu) / sigma
  out <- rep(-Inf, n)
  gum <- abs(xi) < gev_eps
  out[gum] <- -log(sigma[gum]) - z[gum] - exp(-z[gum])
  ng <- !gum
  t <- 1 + xi[ng] * z[ng]
  ok <- t > 0
  idx <- which(ng)[ok]
  tt <- t[ok]
  out[idx] <- -log(sigma[idx]) - (1 + 1 / xi[idx]) * log(tt) - tt^(-1 / xi[idx])
  if (log) out else exp(out)
}

pgev <- function(q, mu, sigma, xi) {
  n <- max(length(q), length(mu), length(sigma), length(xi))
  q <- rep_len(q, n); mu <- rep_len(mu, n); sigma <- rep_len(sigma, n); xi <- rep_len(xi, n)
  z <- (q - mu) / sigma
  out <- numeric(n)
  gum <- abs(xi) < gev_eps
  out[gum] <- exp(-exp(-z[gum]))
  ng <- !gum
  t <- 1 + xi[ng] * z[ng]
  # outside the support: below the lower end (xi > 0) -> 0; above the upper end (xi < 0) -> 1
  val <- ifelse(t > 0, exp(-pmax(t, 0)^(-1 / xi[ng])), ifelse(xi[ng] > 0, 0, 1))
  out[ng] <- val
  out
}

qgev <- function(p, mu, sigma, xi) {
  n <- max(length(p), length(mu), length(sigma), length(xi))
  p <- rep_len(p, n); mu <- rep_len(mu, n); sigma <- rep_len(sigma, n); xi <- rep_len(xi, n)
  yp <- -log(p)
  out <- numeric(n)
  gum <- abs(xi) < gev_eps
  out[gum] <- mu[gum] - sigma[gum] * log(yp[gum])
  ng <- !gum
  out[ng] <- mu[ng] + sigma[ng] / xi[ng] * (yp[ng]^(-xi[ng]) - 1)
  out
}

rgev <- function(n, mu, sigma, xi) qgev(runif(n), mu, sigma, xi)

# Return level for a return period of T winters (exceeded once every T winters on average).
# Keeps the dimensions of mu (e.g. a draws x cells matrix).
return_level <- function(T, mu, sigma, xi) {
  out <- qgev(1 - 1 / T, mu, sigma, xi)
  if (!is.null(dim(mu))) dim(out) <- dim(mu)
  out
}

# Return period (winters) of a value y: 1 / P(Y > y). Keeps the dimensions of mu.
return_period <- function(y, mu, sigma, xi) {
  out <- 1 / pmax(1 - pgev(y, mu, sigma, xi), 1e-300)
  if (!is.null(dim(mu))) dim(out) <- dim(mu)
  out
}

# ---------------------------------------------------------------------------
# The pre-specified winter split (identical to the pilot's rainpilot.data.interleaved_split)
# ---------------------------------------------------------------------------
split_of_winter <- function(winter) {
  ifelse(winter %% 5 == 0, "test", ifelse(winter %% 5 == 3, "val", "train"))
}

# Covariate: standardised log mean daily rainfall per cell, from TRAINING days only.
# R: [days, cells] daily rainfall; set_day: split label per day.
compute_covariate <- function(R, set_day) {
  train_days <- which(set_day == "train")
  lm_train <- log(colMeans(R[train_days, , drop = FALSE]))
  center <- mean(lm_train); scale <- sd(lm_train)
  list(log_mean_train = lm_train, x_cov = (lm_train - center) / scale,
       center = center, scale = scale)
}

check_split <- function(df) {
  # df has columns winter and set; every winter must sit in exactly one set
  per_winter <- tapply(df$set, df$winter, function(s) length(unique(s)))
  stopifnot(all(per_winter == 1))
  stopifnot(all(df$set == split_of_winter(df$winter)))
  invisible(TRUE)
}

# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
# CRPS of observation y against a predictive sample x (unbiased "fair" version is not used:
# this is the standard sample estimator E|X - y| - 0.5 E|X - X'|, via the sorted-sample form).
crps_sample <- function(y, x) {
  m <- length(x)
  xs <- sort(x)
  e1 <- mean(abs(xs - y))
  e2 <- 2 * sum((2 * seq_len(m) - m - 1) * xs) / m^2
  e1 - 0.5 * e2
}

pinball <- function(y, q, tau) ifelse(y >= q, tau * (y - q), (1 - tau) * (q - y))

# log of the mean of exp(v), stable
log_mean_exp <- function(v) {
  m <- max(v)
  if (!is.finite(m)) return(m)          # every term is -Inf: the value has zero density
  m + log(mean(exp(v - m)))
}

# Winter-block summary of a per-observation score difference: sum within winter, then
# mean and SE over winters (winters are the independent units; cells within a winter are not).
winter_block_summary <- function(diff, winter) {
  per_w <- tapply(diff, winter, sum)
  n <- length(per_w)
  c(total = sum(per_w), mean_per_winter = mean(per_w), se_per_winter = sd(per_w) / sqrt(n),
    n_winters = n)
}

# Percentile bootstrap over winters of a statistic computed from a data frame with a `winter`
# column. `stat` takes a data frame and returns a numeric vector.
winter_bootstrap <- function(df, stat, B = 1000, seed = 1) {
  set.seed(seed)
  ws <- unique(df$winter)
  idx_by_w <- split(seq_len(nrow(df)), df$winter)
  t(replicate(B, {
    pick <- sample(ws, length(ws), replace = TRUE)
    stat(df[unlist(idx_by_w[as.character(pick)], use.names = FALSE), , drop = FALSE])
  }))
}

save_json <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(x, path, auto_unbox = TRUE, digits = NA, pretty = TRUE)
}
