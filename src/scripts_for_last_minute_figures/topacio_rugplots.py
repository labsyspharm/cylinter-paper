import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/last_minute_figures'
    '/topacio_rugplots')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read clustered raw TOPACIO data
df = pd.read_parquet(
    '/Users/greg/Dropbox (HMS)/topacio/cylinter_output/TOPACIO_FINAL/'
    'output_raw/checkpoints/clustering.parquet'
    )

# all channels
# channels = [
#     'CD3', 'PDL1', 'ECadherin', 'PD1', 'CD8a', 'CD45', 'GrB',
#     'CD163', 'CD68', 'CD20', 'CD4', 'FOXP3', 'CD11b', 'CD57',
#     'panCK', 'yH2AX', '53BP1', 'SMA', 'MHCII'
#     ]

# channels = [
#     'panCK', 'yH2AX', 'CD3', 'PDL1', 'ECadherin', 'PD1', 'CD8a', 'CD45', 'GrB',
#     'CD163', 'CD68', 'CD20', 'CD4', 'FOXP3', 'CD11b', 'CD57', '53BP1', 'SMA',
#     'MHCII'
#     ]

# channels = ['SMA', 'panCK', 'yH2AX', 'CD3', 'panCK']
# clusters = [4, 14, 77, 197, 404]
# colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:blue']

channels = ['panCK', 'panCK', 'panCK']
clusters = [197, 174, 14]
colors = ['tab:green', 'tab:blue', 'tab:red']
heights = [0.015, 0.045, 0.075]

handles = []
for channel, cluster, color, height in zip(
  channels, clusters, colors, heights):

    print(channel)

    g = plt.hist(
        df[channel], bins=600, density=True, histtype='step',
        linewidth=2.0, alpha=1.0, color='gainsboro'
        )

    # add rug plot for specific cluster/channel
    y_max = plt.gca().get_ylim()[1]
    test = df[df['cluster_2d'] == cluster]
    test = test.sample(n=50)
    for i in test.index:
        x = df[channel][df.index == i]
        if x.values == 0.0:
            zorder = 2.0
        else:
            zorder = 0.0
        plt.plot(
            x, y_max*height, marker='|', color=color, zorder=zorder
        )
    plt.title(channel)

    # color = g[2][0].get_facecolor()[:-1]
    # handles.append(
    #     Rectangle((0, 0), 1, 1, color=color, alpha=1.0, linewidth=0.0)
    #     )
    # plt.legend(handles, channels, fontsize=5.0, bbox_to_anchor=(1.01, 1.0))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'topacio_rugplots.pdf'))
plt.close('all')
