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

def main():
    lr_list = [1e-5, 5e-5, 7.5e-5, 1e-4, 5e-4]
    lf_list = ['binary_cross_entropy', lf.dice_coef_loss, lf.jaccard_distance_loss]

    out_dict = {}
    for lf in lf_list:
        if type(lf) == str:
            lf_name = lf
        else:
            lf_name = lf.__name__
        out_dict[lf_name] = {}
        for lr in lr_list:
            train_results = train.train(lr, lf)
            out_dict[lf_name][str(lr)] = train_results

    with open('grid_results.json', 'w') as fp:
        json.dump(out_dict, fp, sort_keys=True, indent=4)

    print(out_dict)

    return

if __name__=='__main__':
    main()

