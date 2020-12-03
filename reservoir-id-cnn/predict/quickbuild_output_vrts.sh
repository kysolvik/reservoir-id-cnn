gdalbuildvrt temp/out.vrt /mnt/disks/pred_data/out/*.tif
gdal_translate -co "COMPRESS=LZW" temp/out.vrt temp/out.tif

