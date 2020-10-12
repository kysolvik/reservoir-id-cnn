"""
Calculate accuracy metrics for each ecoregion, state, and biome
"""
import sys
import numpy as np
from skimage import transform
import pandas as pd

rf_pred_test = './training_images/rf_test_predict.npy'
cnn_pred_test = '../../../../train/predict/pred_test_masks.npy'
masks_npy_path = '../../../../train/data/prepped/imgs_mask_test.npy'
cnn_test_names_path = '../../../../train/data/prepped/test_names.csv'
rf_test_names_path = './csvs/rf_test_names.csv'

pred = np.load(rf_pred_test)
pred = pred[:,:,:]
pred = transform.resize(pred, (pred.shape[0], 500, 500),
                        preserve_range=True)
pred[pred>0.5] = 255
pred[pred<0.5] = 0
pred = pred.astype(np.uint8)
test = np.load(masks_npy_path)

rf_test_names_df = pd.read_csv(rf_test_names_path)
cnn_test_names_df = pd.read_csv(cnn_test_names_path, names=['name'])
which_masks = np.isin(cnn_test_names_df['name'].values, rf_test_names_df['name'].values)

test = test[which_masks]


metrics_df = pd.DataFrame()
for i in range(pred.shape[0]):
    this_pred = pred[i]
    this_mask = test[i]
    temp_dict = {}
    temp_dict['filename'] = rf_test_names_df['name'].values[i]

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


# Write results to csv
metrics_df.to_csv('test_metrics.csv', index=False)

