gdal_rasterize -ot Int16 -tr 0.000089831528412 0.000089831528412 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 -tap -a NUNIVOTTO4 -co "COMPRESS=LZW" ../../region_analysis/accuracy/data/shapefiles/watersheds_4digit.shp ./data/watersheds_raster.tif

