gdalwarp --config GDAL_CACHEMAX 1000 -wm 1000 -t_srs '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' -tr 0.000089831528412 0.000089831528412 -tap -te -61.6334116 -18.0415845 -50.2247177 -7.3490275 -tap -co "COMPRESS=LZW" ../data/xingu_aster_10m.tif ./data/xingu_2007_reproj.tif