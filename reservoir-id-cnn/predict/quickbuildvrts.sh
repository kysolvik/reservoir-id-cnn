d=$1
gdalbuildvrt temp/${d}.vrt ${d}/*.tif
gdal_translate -co "COMPRESS=LZW" temp/${d}.vrt temp/${d}.tif

