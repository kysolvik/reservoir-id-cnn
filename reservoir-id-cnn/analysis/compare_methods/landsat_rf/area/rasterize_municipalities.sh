gdal_rasterize -ot Int16 -tr 0.000089831528412 0.000089831528412 -tap -te -61.6334116 -18.0415845 -50.2247177 -7.3490275 -a PASTURE_UN -co "COMPRESS=LZW" ../../../region_analysis/accuracy/data/shapefiles/Municipal_Units_wgs84.shp ./data/municipal_units.tif

