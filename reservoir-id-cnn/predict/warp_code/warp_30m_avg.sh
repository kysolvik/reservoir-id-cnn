# 30m resolution
mkdir -p /mnt/disks/pred_data/vrts_30m_avg/

# S2 10m
tsp gdalwarp --config GDAL_CACHEMAX 500 -wm 500 -co "BIGTIFF=YES" -co "TILED=YES" -co "COMPRESS=LZW" -r average -tap -tr 0.000269494585236 0.000269494585236 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 /mnt/disks/pred_data/vrts/s2_10m.vrt /mnt/disks/pred_data/vrts_30m_avg/s2_10m.tif

# S1 10m
tsp gdalwarp --config GDAL_CACHEMAX 500 -wm 500 -co "COMPRESS=LZW" -co "BIGTIFF=YES" -co "TILED=YES" -r average -tap -tr 0.000269494585236 0.000269494585236 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 /mnt/disks/pred_data/vrts/s1_10m.vrt /mnt/disks/pred_data/vrts_30m_avg/s1_10m.tif 

# S2 20m
tsp gdalwarp --config GDAL_CACHEMAX 500 -wm 500 -co "BIGTIFF=YES" -co "TILED=YES" -co "COMPRESS=LZW" -r average -tap -tr 0.000269494585236 0.000269494585236 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 /mnt/disks/pred_data/vrts/s2_20m.vrt /mnt/disks/pred_data/vrts_30m_avg/s2_20m.tif 
