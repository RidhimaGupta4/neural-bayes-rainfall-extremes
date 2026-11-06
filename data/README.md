# Data

## Source

HadUK-Grid v1.3.2.ceda, 5 km daily rainfall (Met Office), cut by the pilot repository to a
16 × 16 grid over Cumbria (British National Grid eastings 320–400 km, northings 480–560 km),
October–March, 1 October 1961 to 31 March 2025 (64 winters, 11,664 days, no missing values).

The raw file is **not** stored here. `R/00_build_data.R` reads it from `data/raw/`:

```bash
mkdir -p data/raw
curl -L -o data/raw/cumbria_winter_rainfall_5km.nc \
  https://github.com/RidhimaGupta4/rainfall-extremes-pilot/raw/main/data/cumbria_winter_rainfall_5km.nc
```

## Derived files (committed)

| File | Contents |
|---|---|
| `winter_maxima.csv` | `winter` (start year), `cell` (1–256), `max_mm` (largest daily total that October–March), `n_days`, `set` (`train` / `val` / `test`) |
| `cells.csv` | cell index, grid position (`ix` west→east, `iy` south→north), centre in km and lat/lon, `log_mean_train` and the standardised covariate `x_cov` |
| `desmond_field.csv` | daily rainfall on 5 December 2015 (Storm Desmond), for maps only |
| `build_info.json` | covariate centring and scaling constants, and the number of winters per set |

**Split** (identical to the pilot, by winter start year): years ending 0 or 5 → test (12 winters,
including 2015/16), 3 or 8 → validation (13), otherwise → training (39).

**Covariate:** the log of each cell's mean daily winter rainfall over the **training winters
only**, standardised with constants also from the training winters (a SAAR-like covariate).
A unit test checks that changing held-out days leaves it unchanged.

## Licence and citation

Contains public sector information licensed under the
[Open Government Licence v3.0](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

Met Office; Hollis, D.; McCarthy, M.; Kendon, M.; Legg, T.; Simpson, I. (2018): HadUK-Grid
gridded and regional average climate observations for the UK. Centre for Environmental Data
Analysis. http://catalogue.ceda.ac.uk/uuid/4dc8450d889a491ebb20e724debe2dfb

HadUK-Grid is interpolated from rain gauges, so each 5 km value is an area average and is
smoother than single-gauge records (for example 341 mm at Honister on 5 December 2015).
