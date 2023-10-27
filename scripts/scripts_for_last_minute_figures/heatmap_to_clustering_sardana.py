import os
import pandas as pd
import matplotlib.pyplot as plt

save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/'
    'last_minute_figures/heatmap_to_clustering'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_parquet(
    '/Volumes/My Book/cylinter_input/sardana-097/' +
    'output_raw/checkpoints/clustering.parquet'
    )
# df = df[df['cluster_2d'] != -1]
coi = [-1]
###############################################################################
# by channels heatmap

# coi1 = [185, 143, 79, 78, 183, 52, 96, 18, 42, 137, 22, 37, 36, 195, 69,
#         17, 93, 130, 19, 39, 58, 72, 73, 1, 166, 74, 64, 86, 23, 159, 156, 57,
#         163, 158, 146, 75, 127, 77, 152, 147, 126, 184, 11, 148, 157, 128, 9,
#         12, 6, 24, 10, 136, 104, 21, 38, 5, 145, 165, 140, 135, 48, 80, 3, 65,
#         20, 70, 105, 94, 92, 61, 2, 192, 161, 111, 89, 119, 118, 100, 59, 122,
#         155, 49, 8, 7, 173, 47, 88, 46, 172, 108, 123, 168, 68, 67, 87, 124,
#         170, 30, 142, 4, 103, 134, 51, 131, 33, 107, 162, 40, 167, 13, 121,
#         129, 98, 0, 85, 84, 76, 144, 139, 90, 50, 154, 29, 109, 54, 53, 14, 60,
#         106, 44, 95, 56, 55, 160, 82, 97, 35, 16, 133, 101, 41, 32, 164, 43,
#         83, 81, 180, 26, 196, 31, 28, 114, 63, 141, 25, 193, 45, 138, 191, 120,
#         91, 34, 149, 132, 102, 27, 71
#         ]
#
# coi2 = [197]
#
# coi3 = [i for i in df['cluster_2d'] if i not in coi + coi1 + coi2]
#
# plt.scatter(df['emb1'], df['emb2'], c='gainsboro', s=0.05, linewidth=0.0)
#
# for c, cluster_group in zip(
#   ['k', 'tab:red', 'tab:green', 'tab:blue'], [coi, coi1, coi2, coi3]):
#     test = df[df['cluster_2d'].isin(cluster_group)]
#     plt.scatter(test['emb1'], test['emb2'], c=c, s=0.05, linewidth=0.0)
# plt.savefig(os.path.join(save_dir, 'channels_to_clustering.png'), dpi=800)
# plt.close('all')

###############################################################################
# by clusters heatmap

coi1 = [2, 8, 19]
coi2 = [18, 11, 1, 16, 21, 9, 17, 0, 13]
coi3 = [6, 4, 20, 10, 14, 15]
coi4 = [12, 5, 3, 7]

plt.scatter(df['emb1'], df['emb2'], c='gainsboro', s=0.05, linewidth=0.0)

for c, cluster_group in zip(
  ['k', 'tab:orange', 'tab:red', 'tab:blue', 'tab:green'],
  [coi, coi1, coi2, coi3, coi4]):
    test = df[df['cluster_2d'].isin(cluster_group)]
    plt.scatter(test['emb1'], test['emb2'], c=c, s=0.05, linewidth=0.0)
plt.savefig(
    os.path.join(save_dir, 'clusters_to_clustering_sardana.png'), dpi=800
    )
plt.close('all')
