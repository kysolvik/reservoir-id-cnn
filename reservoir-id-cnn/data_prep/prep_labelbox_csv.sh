#!/bin/bash
###
# Temporary script to record how to upload images to cloud storage and prep 
# labelbox csv.
#
# Will need more attention later.
###


# Set google cloud storage path
gs_dir=$1

# # Copy files to gs
# gsutil -m cp ./*.png $gs_dir
# 
# # Set permissions
# gsutil iam ch -r allUsers:legacyObjectReader $gs_dir

# Create csv
echo 'Image URL' > ./labelbox_urls.csv
gsutil ls $gs_dir >> ./labelbox_urls.csv
sed -i -e "s@${gs_dir}@https://storage.googleapis.com/@g" ./labelbox_urls.csv

