gdal_rasterize -ot Int16 -tr 0.000089831528412 0.000089831528412 -tap -te -61.6334116 -18.0415845 -50.2247177 -7.3490275 -a dissolve -co "COMPRESS=LZW" ~/research/reservoirs/backup/data/general/reservoir_gisfiles/general_borders/UpperXingu.shp ./xingu_10m.tif
