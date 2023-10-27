import pandas as pd
import matplotlib.pyplot as plt

# save_dir = (
#     '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/last_minute_figures'
#     '/pts_per_cluster_topacio_raw')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

df = pd.read_parquet(
    '/Users/greg/Dropbox (HMS)/topacio/cylinter_output/TOPACIO_FINAL/'
    'output_raw/checkpoints/clustering.parquet'
)

plt.scatter(df['emb1'], df['emb2'], s=0.1, c='gainsboro')
plt.scatter(df['emb1'][df['cluster_2d'] == -1], df['emb2'][df['cluster_2d'] == -1], s=0.1, c='r')
plt.show()