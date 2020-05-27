"""
Calculate accuracy metrics for each ecoregion, state, and biome
"""
import sys
import numpy as np
from skimage import transform
import pandas as pd

pred_npy_path = '../../train/logs/v23_fulltrain_good_68f1/predict/pred_test_masks.npy'
masks_npy_path = '../../train/data/prepped/imgs_mask_test.npy'
train_names_path = '../../train/data/prepped/train_names.csv'
test_names_path = '../../train/data/prepped/test_names.csv'
region_csv = './data/centers_ecoregions_states.csv'

pred = np.load(pred_npy_path)
pred = pred[:,:,:,0]
pred = transform.resize(pred, (pred.shape[0], 500, 500),
                        preserve_range=True)
pred[pred>0.5] = 255
pred[pred<0.5] = 0
pred = pred.astype(np.uint8)
test = np.load(masks_npy_path)

test = np.delete(test, -11,axis=0) # Remove HUGE reservoir)
pred = np.delete(pred, -11,axis=0)

test_names_df = pd.read_csv(test_names_path, header=None)

metrics_df = pd.DataFrame()
for i in range(pred.shape[0]):
    this_pred = pred[i]
    this_mask = test[i]
    temp_dict = {}
    temp_dict['filename'] = test_names_df.iloc[i, 0]

    temp_dict['true_pos'] = np.sum((this_pred==255)*(this_mask==255))
    temp_dict['false_pos'] = np.sum((this_pred==255)*(this_mask==0))
    temp_dict['true_neg'] = np.sum((this_pred==0)*(this_mask==0))
    temp_dict['false_neg'] = np.sum((this_pred==0)*(this_mask==255))

    temp_dict['precision'] = temp_dict['true_pos']/(temp_dict['true_pos'] + temp_dict['false_pos'])
    temp_dict['recall'] = temp_dict['true_pos']/(temp_dict['true_pos'] + temp_dict['false_neg'])
    temp_dict['f1'] = 2*(temp_dict['precision']*temp_dict['recall'])/(temp_dict['precision']+temp_dict['recall'])
    temp_dict['intersection'] = np.sum((this_pred/255)*(this_mask/255))
    temp_dict['j_sum'] = np.sum(this_pred + this_mask)/255
    temp_dict['ji'] = (temp_dict['intersection']) / (temp_dict['j_sum'] - temp_dict['intersection'])

    metrics_df = metrics_df.append(temp_dict, ignore_index=True)


# Join region, biome, etc.
region_df = pd.read_csv('./data/centers_ecoregions_states.csv')
metrics_df = pd.merge(metrics_df, region_df, how='inner', on='filename').drop(
    columns=['index_right', 'filename'])

metrics_df.to_csv('data/test_metrics_regions.csv', index=False)

# Assign training images to regions
train_names_df = pd.read_csv(train_names_path, header=None).rename(columns={0:'filename'})
train_regions = pd.merge(train_names_df, region_df, how='inner', on='filename').drop(
    columns=['index_right', 'filename'])

train_regions.to_csv('data/train_regions.csv', index=False)
