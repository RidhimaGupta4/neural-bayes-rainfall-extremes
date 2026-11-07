# data/raw/

The raw HadUK-Grid file is not stored in this repository. Download it from the pilot repository:

    curl -L -o data/raw/cumbria_winter_rainfall_5km.nc \
      https://github.com/RidhimaGupta4/rainfall-extremes-pilot/raw/main/data/cumbria_winter_rainfall_5km.nc

Then run `Rscript R/00_build_data.R` to rebuild the files in `data/`.

Source: Met Office HadUK-Grid v1.3.2, CEDA Archive. Contains public sector information licensed
under the Open Government Licence v3.0.
