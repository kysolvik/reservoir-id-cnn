#!/usr/bin/env python3
"""Grid search over loss functions and learning rates

Example:
    python3 gridsearch.py

Notes:
    Must be run from reservoir-id-cnn/train/
    Prepped data should be in the: ./data/prepped/ directory

"""

import train
import json
import loss_functions as lf
from collections import OrderedDict

def main():
    lr_list = [1e-4, 2e-4, 6e-5]
    lf_list = [lf.dice_coef_wgt_loss]
    band_combo_dict = OrderedDict([
#         ('all',list(range(16))),
#         ('all_old', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15]),
        ('rgbn_nds_radar', [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]),
#         ('rgbn_nds', [0, 1, 2, 3, 12, 13, 14, 15]),
#         ('rgbn_nds_radar_old', [0, 1, 2, 3, 4, 5, 14, 15])
    ])


    out_dict = {}
    for lfunc in lf_list:
        if type(lfunc) == str:
            lf_name = lfunc
        else:
            lf_name = lfunc.__name__
        out_dict[lf_name] = {}
        for lr in lr_list:
            out_dict[lf_name][str(lr)] = {}
            for bc_name in sorted(band_combo_dict.keys()):
                print("STARTING:", bc_name, lf_name, lr)
                train_results = train.train(lr, lfunc, band_combo_dict[bc_name],
                                            test=True, val=True)
                out_dict[lf_name][str(lr)][bc_name] = train_results
                print("DONE:", bc_name, lf_name, lr)
                print("SO FAR:", out_dict)

    print(out_dict)

    with open('grid_results.json', 'w') as fp:
        json.dump(out_dict, fp, sort_keys=True, indent=4)


    return

if __name__=='__main__':
    main()

