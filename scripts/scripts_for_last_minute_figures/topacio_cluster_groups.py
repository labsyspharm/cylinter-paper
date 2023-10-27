import os
import pandas as pd
import matplotlib.pyplot as plt

save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/'
    'last_minute_figures/topacio_cluster_groups'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_parquet(
    '/Users/greg/Dropbox (HMS)/topacio/cylinter_output/TOPACIO_FINAL/' +
    'output_raw/checkpoints/clustering.parquet'
    )

coi = [-1]
###############################################################################
# by channels heatmap

coi1 = [185, 143, 79, 78, 183, 52, 96, 18, 42, 137, 22, 37, 36, 195, 69,
        17, 93, 130, 19, 39, 58, 72, 73, 1, 166, 74, 64, 86, 23, 159, 156, 57,
        163, 158, 146, 75, 127, 77, 152, 147, 126, 184, 11, 148, 157, 128, 9,
        12, 6, 24, 10, 136, 104, 21, 38, 5, 145, 165, 140, 135, 48, 80, 3, 65,
        20, 70, 105, 94, 92, 61, 2, 192, 161, 111, 89, 119, 118, 100, 59, 122,
        155, 49, 8, 7, 173, 47, 88, 46, 172, 108, 123, 168, 68, 67, 87, 124,
        170, 30, 142, 4, 103, 134, 51, 131, 33, 107, 162, 40, 167, 13, 121,
        129, 98, 0, 85, 84, 76, 144, 139, 90, 50, 154, 29, 109, 54, 53, 14, 60,
        106, 44, 95, 56, 55, 160, 82, 97, 35, 16, 133, 101, 41, 32, 164, 43,
        83, 81, 180, 26, 196, 31, 28, 114, 63, 141, 25, 193, 45, 138, 191, 120,
        91, 34, 149, 132, 102, 27, 71
        ]

coi2 = [197]

coi3 = [i for i in df['cluster_2d'] if i not in coi + coi1 + coi2]

plt.scatter(df['emb1'], df['emb2'], c='gainsboro', s=0.05, linewidth=0.0)

for c, cluster_group in zip(
  ['k', 'tab:red', 'tab:green', 'tab:blue'], [coi, coi1, coi2, coi3]):
    test = df[df['cluster_2d'].isin(cluster_group)]
    plt.scatter(test['emb1'], test['emb2'], c=c, s=0.05, linewidth=0.0)
plt.savefig(os.path.join(save_dir, 'by_channels_heatmap.png'), dpi=800)
plt.close('all')

###############################################################################
# by clusters heatmap

coi1 = [12, 6, 194, 136, 24, 130, 19, 142, 65, 9, 37, 36, 195, 133, 17, 93,
        137, 22, 119, 111, 118, 89, 49, 155, 59, 122, 100, 51, 23, 58, 14, 53,
        128, 162, 121, 167, 13, 105, 129, 193, 138, 104, 21, 5, 38, 95, 39, 2,
        42, 60, 103, 173, 140, 150, 192, 61, 8, 7, 108, 159, 20, 44, 54, 92,
        94, 134, 0, 165, 106, 70, 144, 139, 90, 50, 109, 29, 56, 55, 82, 154,
        131, 107, 33, 87, 30, 170, 83, 64, 1, 72, 166, 73, 74, 86, 163, 158,
        156, 79, 78, 68, 67, 185, 143, 183, 52, 145, 96, 18, 81, 101, 41, 32,
        97, 35, 16, 88, 46, 34, 27, 71, 91, 69, 47, 28, 25, 63, 141, 114, 149,
        132, 102, 57, 31, 161, 80, 3, 172, 135, 48, 10, 85, 84, 76, 98,
        ]

coi2 = [15]

coi3 = [
    11, 77, 127, 126, 146, 75, 147, 152, 148, 184,
        ]

coi4 = [157, 4, 124]

coi5 = [
    180, 66, 26, 62, 43, 164, 160, 196, 45, 120, 191, 197,
    40, 123, 168
    ]

coi6 = [
    i for i in df['cluster_2d'] if i not in coi + coi1 + coi2 + coi3 + coi4 + coi5
    ]

plt.scatter(df['emb1'], df['emb2'], c='gainsboro', s=0.05, linewidth=0.0)

for c, cluster_group in zip(
  ['k', 'tab:green', 'tab:pink', 'tab:red', 'tab:orange', 'tab:purple', 'tab:gray'],
  [coi, coi1, coi2, coi3, coi4, coi5, coi6]):
    test = df[df['cluster_2d'].isin(cluster_group)]
    plt.scatter(test['emb1'], test['emb2'], c=c, s=0.05, linewidth=0.0)
plt.savefig(os.path.join(save_dir, 'by_clusters_heatmap.png'), dpi=800)
plt.close('all')
