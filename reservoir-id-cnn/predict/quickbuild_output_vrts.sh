num_start=${1}
gdalbuildvrt temp/$1.vrt /mnt/disks/pred_data/out/pred_$1*.tif
gdal_translate -co "COMPRESS=LZW" temp/$1.vrt temp/$1.tif

