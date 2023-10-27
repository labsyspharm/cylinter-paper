import os
import pandas as pd
import matplotlib.pyplot as plt

out = os.path.join(
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/last_minute_figures/'
    'CRC_preQC_cluster21'
)
if not os.path.exists(out):
    os.makedirs(out)


df = pd.read_parquet(
    '/Volumes/T7 Shield/cylinter_input/sardana-097/output/checkpoints/clustering.parquet'
)

selected_cells = pd.read_csv(
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/scripts/scripts_for_last_minute_figures/'
    'CRC_preQC_cluster21/gated_cell_ids.csv', index_col=0
)

myids = [i for i in selected_cells['0']]

plt.scatter(df['emb1'], df['emb2'], s=1.0, lw=0.0, c='gainsboro')

plt.scatter(
    df['emb1'][(df['cluster_2d'] == 21) & (~df.index.isin(myids))],
    df['emb2'][(df['cluster_2d'] == 21) & (~df.index.isin(myids))],
    s=1.0, lw=0.0, c='tab:red'
)

plt.scatter(
    df['emb1'][df['cluster_2d'] == 1],
    df['emb2'][df['cluster_2d'] == 1],
    s=1.0, lw=0.0, c='tab:green'
)

plt.scatter(
    df['emb1'][df['cluster_2d'] == 17],
    df['emb2'][df['cluster_2d'] == 17],
    s=1.0, lw=0.0, c='tab:blue'
)

plt.scatter(
    df['emb1'][df.index.isin(myids)],
    df['emb2'][df.index.isin(myids)],
    s=1.0, lw=0.0, c='tab:orange'
)

plt.tight_layout()
plt.savefig(os.path.join(out, 'UMAP2.png'), dpi=800)
plt.close('all')
