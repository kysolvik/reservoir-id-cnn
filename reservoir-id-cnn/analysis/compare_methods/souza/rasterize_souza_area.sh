gdal_rasterize -ot Int16 -tr 0.000089831528412 0.000089831528412 -tap -te -61.6478745 -16.2835815 -43.4423475 -2.4520414 -burn 1 -co "COMPRESS=LZW" ./data/souza_dams_concave_hull.shp ./data/souza_area.tif
