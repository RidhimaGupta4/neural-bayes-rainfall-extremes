# Maximum-likelihood baseline: per-dataset GEV MLE with Wald (delta-method) 90% intervals.
# Usage: Rscript R/12_mle.R sim | real     (real: 39 TRAINING winters per cell)
# Output: results/mle_<sim|real>.csv

source(file.path("R", "utils.R"))
mode <- commandArgs(trailingOnly = TRUE)[1]
stopifnot(mode %in% c("sim", "real"))
if (mode == "sim") {
  ts <- read.csv(repo_path("results", "sim", "test_set.csv"))
  Y <- as.matrix(ts[, paste0("y", 1:39)]); ids <- ts$id
} else {
  wm <- read.csv(repo_path("data", "winter_maxima.csv")); check_split(wm)
  tr <- wm[wm$set == "train", ]
  stopifnot(all(tr$set == "train"), length(unique(tr$winter)) == 39)
  ids <- sort(unique(tr$cell))
  Y <- t(sapply(ids, function(s) tr$max_mm[tr$cell == s][order(tr$winter[tr$cell == s])]))
}

negll <- function(p, y) {
  v <- -sum(dgev(y, p[1], exp(p[2]), p[3], log = TRUE))
  if (!is.finite(v)) 1e10 else v
}
g_funs <- list(
  xi = function(p) p[3],
  log_z10 = function(p) log(return_level(10, p[1], exp(p[2]), p[3])),
  log_z100 = function(p) log(return_level(100, p[1], exp(p[2]), p[3])),
  log_z1000 = function(p) log(return_level(1000, p[1], exp(p[2]), p[3])))
num_grad <- function(f, p, h = 1e-5) sapply(seq_along(p), function(i) {
  e <- replace(numeric(length(p)), i, h); (f(p + e) - f(p - e)) / (2 * h)
})

rows <- lapply(seq_along(ids), function(i) {
  y <- Y[i, ]
  s0 <- sd(y) * sqrt(6) / pi
  st <- c(mean(y) - 0.5772 * s0, log(s0), 0.05)
  f1 <- optim(st, negll, y = y, method = "Nelder-Mead", control = list(maxit = 5000))
  f2 <- optim(f1$par, negll, y = y, method = "BFGS", hessian = TRUE, control = list(maxit = 2000))
  p <- f2$par
  V <- tryCatch(solve(f2$hessian), error = function(e) NULL)
  ok <- f2$convergence == 0 && !is.null(V) && all(eigen(V, only.values = TRUE)$values > 0)
  row <- data.frame(id = ids[i], converged = ok, mu = p[1], sigma = exp(p[2]), xi = p[3])
  for (g in names(g_funs)) {
    est <- g_funs[[g]](p)
    se <- if (ok) sqrt(as.numeric(t(num_grad(g_funs[[g]], p)) %*% V %*% num_grad(g_funs[[g]], p)))
          else NA
    row[[paste0(g, "_est")]] <- est
    row[[paste0(g, "_lo90")]] <- est - 1.645 * se
    row[[paste0(g, "_hi90")]] <- est + 1.645 * se
  }
  row
})
out <- do.call(rbind, rows)
write.csv(out, repo_path("results", sprintf("mle_%s.csv", mode)), row.names = FALSE)
cat("MLE", mode, ":", nrow(out), "datasets,", sum(out$converged), "with a usable Hessian\n")
