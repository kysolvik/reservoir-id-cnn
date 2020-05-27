"""
Takes the output from calc_image_metrics.py to calc summary metrics
per state, ecoregion, and biome
"""

import pandas as pd

train_regions_path = './data/train_regions.csv'
test_metrics_path = './data/test_metrics_regions.csv'

train_df = pd.read_csv(train_regions_path)
test_df = pd.read_csv(test_metrics_path)

def calc_grouped_metrics(test_df, train_df, groupby):
    grp = test_df[
        [groupby,'true_pos','true_neg','false_pos','false_neg', 'intersection', 'j_sum']
    ].groupby(groupby)
    new_df = grp.sum()
    new_df = new_df.assign(
        test_count=grp.size(),
        precision=new_df['true_pos']/(new_df['true_pos'] + new_df['false_pos']),
        recall=new_df['true_pos']/(new_df['true_pos'] + new_df['false_neg']),
        jaccard=new_df['intersection']/(new_df['j_sum']-new_df['intersection'])
    )
    new_df = new_df.assign(
        f1=2*(new_df['precision']*new_df['recall'])/(new_df['precision']+new_df['recall'])
    )
    new_df = new_df.assign(
        train_count=train_df.groupby(groupby).size().astype(int)
    )

    return new_df

for gb in ['ecoregion','biome','state']:
    new_df = calc_grouped_metrics(test_df, train_df, gb)
    new_df.to_csv('./data/summary_{}.csv'.format(gb))
