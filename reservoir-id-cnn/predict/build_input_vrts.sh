mkdir -p /mnt/disks/pred_data/vrts/

# S2 10m
gdalbuildvrt /mnt/disks/pred_data/vrts/s2_10m.vrt -tap -tr 0.000089831528412 0.000089831528412 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 /mnt/disks/pred_data/sentinel/*.tif

# S1 10m
gdalbuildvrt /mnt/disks/pred_data/vrts/s1_10m.vrt -tap -tr 0.000089831528412 0.000089831528412 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 /mnt/disks/pred_data/sentinel1_10m_v2/*.tif

# S2 20m
gdalbuildvrt /mnt/disks/pred_data/vrts/s2_20m.vrt -tap -tr 0.000089831528412 0.000089831528412 -te -61.6823698 -24.7351113 -41.4721624 -2.2909735 /mnt/disks/pred_data/sentinel2_20m/*.tif

