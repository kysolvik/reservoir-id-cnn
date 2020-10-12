Get accuracy metrics for aster method for xingu region
Step 1: Clip the centers csv to the region of interest (file is in region comparison/accuracy)
Step 2: Run ./extract_matching_arrays.py (run with --help for guidance)
    Example: python3 extract_matching_arrays.py csvs/xingu_centers.csv ../data/xingu_aster_10m.tif training_images/ s2_20m_og
Step 3: Run ./prep_rf_npy.py (just by itself: python3 ./prep_rf_npy.py)
Step 4: Run ./calc_image_metrics.py (also just by itself)
Step 5; Run ./summarize_metrics.py to get final summarized metrics

