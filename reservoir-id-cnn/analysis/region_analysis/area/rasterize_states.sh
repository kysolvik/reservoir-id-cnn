gdal_rasterize -tr 0.000089831528412 0.000089831528412 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 -tap -a GEOCODIGO -co "COMPRESS=LZW" ../../region_analysis/accuracy/data/shapefiles/brazil_states.shp ./states_raster.tif

