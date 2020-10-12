import pandas as pd

test_metrics_path = './test_metrics.csv'
test_df = pd.read_csv(test_metrics_path)

precision=test_df['true_pos'].sum()/(test_df['true_pos'].sum() + test_df['false_pos'].sum())
recall=test_df['true_pos'].sum()/(test_df['true_pos'].sum() + test_df['false_neg'].sum())
jaccard=test_df['intersection'].sum()/(test_df['j_sum'].sum()-test_df['intersection'].sum())

f1=2*(precision*recall)/(precision+recall)


print(precision, recall, jaccard, f1)

