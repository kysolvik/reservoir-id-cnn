# 50m resolution
mkdir -p /mnt/disks/pred_data/vrts_50m/

# S2 10m
tsp gdalwarp --config GDAL_CACHEMAX 500 -wm 500 -co "BIGTIFF=YES" -co "TILED=YES" -co "COMPRESS=LZW" -r cubicspline -tap -tr 0.00044915764206 0.00044915764206 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 /mnt/disks/pred_data/vrts/s2_10m.vrt /mnt/disks/pred_data/vrts_50m/s2_10m.tif

# S1 10m
tsp gdalwarp --config GDAL_CACHEMAX 500 -wm 500 -co "COMPRESS=LZW" -co "BIGTIFF=YES" -co "TILED=YES" -r cubicspline -tap -tr 0.00044915764206 0.00044915764206 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 /mnt/disks/pred_data/vrts/s1_10m.vrt /mnt/disks/pred_data/vrts_50m/s1_10m.tif 

# S2 20m
tsp gdalwarp --config GDAL_CACHEMAX 500 -wm 500 -co "BIGTIFF=YES" -co "TILED=YES" -co "COMPRESS=LZW" -r cubicspline -tap -tr 0.00044915764206 0.00044915764206 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 /mnt/disks/pred_data/vrts/s2_20m.vrt /mnt/disks/pred_data/vrts_50m/s2_20m.tif 
