#!/bin/bash
###
# Temporary script to record how to upload images to cloud storage and prep 
# labelbox csv.
#
# Will need more attention later.
###

# Copy files to gs
gsutil -m cp ./*.png gs://res-id/cnn/training/raw_subsets

# Set permissions
gsutil iam ch allUsers:legacyObjectReader gs://res-id/cnn/training/raw_subsets/*

# Create csv
echo 'Image URL' > ./labelbox_urls.csv
gsutil ls gs://res-id/cnn/training/raw_subsets/ >> ./labelbox_urls.csv
sed -i 's-gs://-https://storage.googleapis.com/-g' ./labelbox_urls.csv

