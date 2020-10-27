gdal_rasterize -ot Byte -tr 0.000089831528412 0.000089831528412 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 -tap -a BIOME -co "COMPRESS=LZW" ../../region_analysis/accuracy/data/shapefiles/biome_clip.shp ./biomes_raster.tif

