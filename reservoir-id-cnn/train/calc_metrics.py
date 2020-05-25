import sys
import numpy as np
from skimage import transform

# test_npy_path = sys.argv[1]
# pred_npy_path = sys.argv[2]
pred_npy_path = './logs/v23_fulltrain_good_68f1/predict/pred_test_masks.npy'
test_npy_path = 'data/prepped/imgs_mask_test.npy'

pred = np.load(pred_npy_path)
pred = pred[:,:,:,0]
pred = transform.resize(pred, (pred.shape[0], 500, 500),
                        preserve_range=True)
pred[pred>0.5] = 255
pred[pred<0.5] = 0
pred = pred.astype(np.uint8)
test = np.load(test_npy_path)

test = np.delete(test, -11,axis=0) # Remove HUGE reservoir)
pred = np.delete(pred, -11,axis=0)

true_pos = np.sum((pred==255)*(test==255))
false_pos = np.sum((pred==255)*(test==0))
true_neg = np.sum((pred==0)*(test==0))
false_neg = np.sum((pred==0)*(test==255))

precision = true_pos/(true_pos + false_pos)
recall = true_pos/(true_pos + false_neg)
print(precision, recall, true_pos, false_pos, true_neg, false_neg)
