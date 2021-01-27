gdal_rasterize -ot Int16 -tr 10 10 -te -189160.00 738680.00 2085120.00 3404280.00 -tap -a NUNIVOTTO4 -co "COMPRESS=LZW" ../../region_analysis/accuracy/data/shapefiles/watersheds_4digit_aea.shp ./watersheds_raster_aea.tif

