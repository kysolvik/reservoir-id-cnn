import sys
import numpy as np
from skimage import transform

# test_npy_path = sys.argv[1]
# pred_npy_path = sys.argv[2]
pred_npy_path = 'data/predict/pred_test_masks.npy'
test_npy_path = 'data/prepped/imgs_mask_test.npy'

pred = np.load(pred_npy_path)
pred = pred[:,:,:,0]
pred = transform.resize(pred, (pred.shape[0], 500, 500),
                        preserve_range=True)
pred[pred>0.5] = 255
pred[pred<0.5] = 0
pred = pred.astype(np.uint8)
test = np.load(test_npy_path)

# test = np.delete(test, -11,axis=0) # Remove HUGE reservoir)
# pred = np.delete(pred, -11,axis=0)

true_pos = np.sum((pred==255)*(test==255))
false_pos = np.sum((pred==255)*(test==0))
true_neg = np.sum((pred==0)*(test==0))
false_neg = np.sum((pred==0)*(test==255))

precision = true_pos/(true_pos + false_pos)
recall = true_pos/(true_pos + false_neg)
pfa = false_pos/(true_pos + false_pos)
pmd = false_neg/(true_pos + false_neg)
agree = true_pos + true_neg
total = (true_pos + false_pos + false_neg + true_neg)
pcc = (agree)/(total)
exp_a =  ((((true_pos+false_pos)*(true_pos+false_neg))+
           ((false_pos+true_neg)*(false_neg+true_neg)))/
          total
          )
kappa = (agree - exp_a)/(total - exp_a)
f1 = 2*(precision*recall)/(precision+recall)
iou = true_pos/(true_pos + false_pos + false_neg)
print('agree, exp_a, kappa',agree, exp_a, kappa)
print('pfa, pmd, pcc', pfa, pmd, pcc)
print('prec, rec, f1, iou, tp, fp, tn, fn',precision, recall, f1, iou, true_pos, false_pos, true_neg, false_neg)
