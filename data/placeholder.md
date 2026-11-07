# data/

Derived data from HadUK-Grid v1.3.2 5 km daily rainfall over Cumbria (Met Office, Open Government
Licence v3.0), built by `R/00_build_data.R`.

| File | Contents |
|---|---|
| `winter_maxima.csv` | Largest daily rainfall (mm) in each October–March winter for each of 256 cells, 1961/62–2024/25, with its split label (`train`, `val`, `test`) |
| `cells.csv` | Cell index, grid position and coordinates (British National Grid km, latitude/longitude) |
| `desmond_field.csv` | Daily rainfall on 5 December 2015 (Storm Desmond), used only in the held-out evaluation |
| `build_info.json` | Number of winters per split and build details |
| `raw/` | Where the raw HadUK-Grid file goes (not stored in the repository; see `raw/placeholder.md`) |

**Split (fixed by winter start year):** years ending 0 or 5 → test (12 winters), 3 or 8 →
validation (13), all others → training (39). Every estimate uses training winters only.
