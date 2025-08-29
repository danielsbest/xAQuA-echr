import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

if 'df' not in globals():
    raise RuntimeError("DataFrame 'df' not found. Run the data loading and preprocessing cells first.")
if 'recall@10' not in df.columns:
    raise RuntimeError("Column 'recall@10' not found. Ensure recall@10 has been computed in earlier cells.")

groups_df = (
    df.dropna(subset=['recall@10'])
      .groupby(['doc_db', 'query_embedding'])
      .size()
      .reset_index(name='n')
)
groups = [(row['doc_db'], row['query_embedding']) for _, row in groups_df.iterrows()]


def wilcoxon_for_groups(a, b):
    (doc_a, q_a) = a
    (doc_b, q_b) = b
    sub_a = df[(df['doc_db'] == doc_a) & (df['query_embedding'] == q_a)][['question_id', 'recall@10']].rename(columns={'recall@10': 'recall_a'})
    sub_b = df[(df['doc_db'] == doc_b) & (df['query_embedding'] == q_b)][['question_id', 'recall@10']].rename(columns={'recall@10': 'recall_b'})
    merged = pd.merge(sub_a, sub_b, on='question_id', how='inner').dropna()
    if merged.empty or len(merged) < 2:
        return np.nan, np.nan, len(merged)

    a_vals = merged['recall_a'].to_numpy()
    b_vals = merged['recall_b'].to_numpy()
    diffs = b_vals - a_vals

    if np.allclose(diffs, 0):
        return 0.0, 1.0, len(merged)

    try:
        stat, p = wilcoxon(a_vals, b_vals, zero_method='wilcox', alternative='two-sided')
    except ValueError:
        try:
            stat, p = wilcoxon(a_vals, b_vals, zero_method='pratt', alternative='two-sided')
        except Exception:
            stat, p = 0.0, 1.0
    return float(stat), float(p), len(merged)

pairs = list(itertools.combinations(groups, 2))
results = []
for (ga, gb) in pairs:
    stat, p, n_pairs = wilcoxon_for_groups(ga, gb)
    results.append({
        'A_doc_db': ga[0],
        'A_query_embedding': ga[1],
        'B_doc_db': gb[0],
        'B_query_embedding': gb[1],
        'n_pairs': n_pairs,
        'statistic': stat,
        'p_value': p,
    })

wilcoxon_results_df = pd.DataFrame(results)

try:
    from statsmodels.stats.multitest import multipletests
    valid_mask = wilcoxon_results_df['p_value'].notna()
    pvals = wilcoxon_results_df.loc[valid_mask, 'p_value'].to_numpy()
    reject, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    wilcoxon_results_df.loc[valid_mask, 'p_value_adj'] = pvals_adj
    wilcoxon_results_df.loc[valid_mask, 'significant_BH_0.05'] = reject
except Exception:
    valid_mask = wilcoxon_results_df['p_value'].notna()
    pvals = wilcoxon_results_df.loc[valid_mask, 'p_value'].to_numpy()
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty_like(order)
    ranked[order] = np.arange(1, m + 1)  # ranks start at 1 for smallest p-value
    
    p_adj = pvals * m / ranked
    
    p_adj_sorted = p_adj[order]
    for i in range(m - 2, -1, -1):
        p_adj_sorted[i] = min(p_adj_sorted[i], p_adj_sorted[i + 1])
    
    p_adj_final = np.empty_like(p_adj)
    p_adj_final[order] = np.minimum(p_adj_sorted, 1.0)
    wilcoxon_results_df.loc[valid_mask, 'p_value_adj'] = p_adj_final
    wilcoxon_results_df.loc[valid_mask, 'significant_BH_0.05'] = p_adj_final < 0.05

wilcoxon_results_df.sort_values(by=['p_value_adj', 'p_value'], inplace=True, na_position='last')

print(f"Total groups: {len(groups)}; total unordered pairs tested: {len(pairs)}")
sig_count = int(wilcoxon_results_df['significant_BH_0.05'].fillna(False).sum())
print(f"Significant pairs after BH (alpha=0.05): {sig_count}")

print("Top 20 pairs by BH-adjusted p-value:")
cols = ['A_doc_db','A_query_embedding','B_doc_db','B_query_embedding','n_pairs','statistic','p_value','p_value_adj','significant_BH_0.05']

wilcoxon_results_df = wilcoxon_results_df[~wilcoxon_results_df['A_query_embedding'].str.contains('translation') & ~wilcoxon_results_df['B_query_embedding'].str.contains('translation')]

wilcoxon_results_df[wilcoxon_results_df['A_doc_db'] == wilcoxon_results_df['B_doc_db']]