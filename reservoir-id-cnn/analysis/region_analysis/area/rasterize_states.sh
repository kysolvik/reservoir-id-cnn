gdal_rasterize -ot Byte -tr 10 10 -te -189160.00 738680.00 2085120.00 3404280.00 -tap -a GEOCODIGO -co "COMPRESS=LZW" ../../region_analysis/accuracy/data/shapefiles/brazil_states_aea.shp ./states_raster_aea.tif

