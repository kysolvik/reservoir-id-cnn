#!/bin/bash

# To run
# for f in $(gsutil ls gs://res-id/ee_exports/sentinel/*.tif);do tsp bash ./wrappers/extract_wrap.sh $f;done

f=$1
echo $f
gsutil cp $f ./temp_from_gs/
raster_index=$(awk -F'_' '{print $NF}' <<< $f)
raster_index="${raster_index%.*}"
echo $raster_index
mkdir out_ims/${raster_index}
python3 extract_subarrays.py ./temp_from_gs/$(basename $f) 80 500 500 ./out_ims/${raster_index} --out_prefix="im_${raster_index}_"
rm ./temp_from_gs/$(basename $f)

