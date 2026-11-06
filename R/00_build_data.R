# Build winter maxima, cell table and the train-only covariate from the pilot's HadUK-Grid file.
#
# Input : data/raw/cumbria_winter_rainfall_5km.nc  (from the pilot repository; see data/README.md)
# Output: data/cells.csv, data/winter_maxima.csv, data/desmond_field.csv
#
# Leakage rules enforced here:
#   * the split is fixed by winter start year (identical to the pilot) and checked;
#   * the covariate and its centring/scaling use TRAINING winters only.

source(file.path("R", "utils.R"))
suppressPackageStartupMessages(library(ncdf4))

nc_path <- repo_path("data", "raw", "cumbria_winter_rainfall_5km.nc")
if (!file.exists(nc_path)) {
  stop("Missing ", nc_path, ". Download it with:\n  curl -L -o ", nc_path,
       " https://github.com/RidhimaGupta4/rainfall-extremes-pilot/raw/main/data/cumbria_winter_rainfall_5km.nc")
}

nc <- nc_open(nc_path)
rain <- ncvar_get(nc, "rainfall", collapse_degen = FALSE)   # dims as stored, reversed by ncdf4
dim_names <- sapply(nc$var$rainfall$dim, `[[`, "name")
x_m <- ncvar_get(nc, "x"); y_m <- ncvar_get(nc, "y")
lat <- ncvar_get(nc, "latitude"); lon <- ncvar_get(nc, "longitude")
time_units <- ncatt_get(nc, "time", "units")$value
tvals <- ncvar_get(nc, "time")
nc_close(nc)

stopifnot(identical(dim_names, c("x", "y", "time")))
nx <- length(x_m); ny <- length(y_m); nt <- length(tvals)
stopifnot(nx == 16, ny == 16, all(diff(x_m) > 0), all(diff(y_m) > 0))

# time -> dates (CF units "<unit> since <origin>")
unit <- sub(" since.*", "", time_units)
origin <- as.POSIXct(sub(".*since ", "", time_units), tz = "UTC")
mult <- switch(unit, days = 86400, hours = 3600, seconds = 1, stop("unknown time unit ", unit))
dates <- as.Date(origin + tvals * mult)
stopifnot(!anyNA(rain), all(rain >= 0), nt == 11664)

month <- as.integer(format(dates, "%m")); year <- as.integer(format(dates, "%Y"))
winter <- ifelse(month >= 10, year, year - 1)
stopifnot(all(month %in% c(10, 11, 12, 1, 2, 3)))
set_day <- split_of_winter(winter)

# cell table: cell = (iy - 1) * 16 + ix, iy south -> north, ix west -> east
cells <- expand.grid(ix = 1:nx, iy = 1:ny)
cells$cell <- (cells$iy - 1) * nx + cells$ix
cells <- cells[order(cells$cell), c("cell", "ix", "iy")]
cells$x_km <- x_m[cells$ix] / 1000
cells$y_km <- y_m[cells$iy] / 1000
cells$lat <- lat[cbind(cells$ix, cells$iy)]
cells$lon <- lon[cbind(cells$ix, cells$iy)]

# daily fields as [time, cell]
R <- matrix(NA_real_, nt, nx * ny)
for (k in seq_len(nrow(cells))) R[, k] <- rain[cells$ix[k], cells$iy[k], ]

# covariate: log mean daily winter rainfall per cell, TRAINING days only
cv <- compute_covariate(R, set_day)
cov_center <- cv$center; cov_scale <- cv$scale
cells$log_mean_train <- cv$log_mean_train
cells$x_cov <- cv$x_cov

# winter maxima for every winter and cell
winters <- sort(unique(winter))
stopifnot(length(winters) == 64, min(winters) == 1961, max(winters) == 2024)
wm <- do.call(rbind, lapply(winters, function(w) {
  d <- which(winter == w)
  data.frame(winter = w, cell = cells$cell, max_mm = apply(R[d, , drop = FALSE], 2, max),
             n_days = length(d))
}))
wm$set <- split_of_winter(wm$winter)
check_split(wm)

# Storm Desmond field (test winter 2015) for maps only
desmond_day <- which(dates == as.Date("2015-12-05"))
stopifnot(length(desmond_day) == 1)
desmond <- data.frame(cell = cells$cell, rain_mm = R[desmond_day, ])

dir.create(repo_path("data"), showWarnings = FALSE)
write.csv(cells, repo_path("data", "cells.csv"), row.names = FALSE)
write.csv(wm, repo_path("data", "winter_maxima.csv"), row.names = FALSE)
write.csv(desmond, repo_path("data", "desmond_field.csv"), row.names = FALSE)
save_json(list(cov_center = cov_center, cov_scale = cov_scale,
               n_train_winters = length(unique(wm$winter[wm$set == "train"])),
               n_val_winters = length(unique(wm$winter[wm$set == "val"])),
               n_test_winters = length(unique(wm$winter[wm$set == "test"])),
               covariate_source = "training winters only"),
          repo_path("data", "build_info.json"))

cat("Winters: train", sum(wm$set == "train") / 256, "val", sum(wm$set == "val") / 256,
    "test", sum(wm$set == "test") / 256, "\n")
