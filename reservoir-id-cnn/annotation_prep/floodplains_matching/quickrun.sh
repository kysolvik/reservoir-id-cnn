tsp python3 extract_floodplains_arrays.py csvs/extra_floodplain_points.csv gs://res-id/ee_exports/sentinel2_20m/ ./test_out/sen2_20m/ s2_20m_og
tsp python3 extract_floodplains_arrays.py csvs/extra_floodplain_points.csv gs://res-id/ee_exports/sentinel1_10m_v2/ ./test_out/sen1_10m_v2/ s1_v2_og
tsp python3 extract_floodplains_arrays.py csvs/extra_floodplain_points.csv gs://res-id/ee_exports/sentinel/ ./test_out/sen2/ og
tsp python3 make_empty_masks.py test_out/sen1_10m_v2/
