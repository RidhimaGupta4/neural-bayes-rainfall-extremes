# Export the saved Stan posterior draws for the real cells to CSV (read by python/evaluate.py).
source(file.path("R", "utils.R"))
fs <- list.files(repo_path("results", "draws"), pattern = "^real_cell[0-9]{3}\\.rds$", full.names = TRUE)
for (f in fs) {
  m <- readRDS(f)
  write.csv(data.frame(mu = as.numeric(m[, "mu"]), sigma = as.numeric(m[, "sigma"]),
                       xi = as.numeric(m[, "xi"])), sub("\\.rds$", ".csv", f), row.names = FALSE)
}
cat("exported", length(fs), "draw files\n")
