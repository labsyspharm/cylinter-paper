import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/last_minute_figures'
    '/topacio_samples_per_cluster')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_parquet(
    '/Users/greg/Dropbox (HMS)/topacio/cylinter_output/TOPACIO_FINAL/output_raw/checkpoints/'
    'clustering.parquet'
)
df = df[df['cluster_2d'] != -1]

plot_input = df.groupby(['cluster_2d', 'Sample']).size()

plot_input = plot_input.unstack()
plot_input = plot_input.fillna(value=0.0)

plot_input.columns = [i.split('_')[1] for i in plot_input.columns]

# have to convert cluster integers to strings to get plot to sort by bar height
plot_input.index = [str(i) for i in plot_input.index]

# sort clusters by the degree of their sample representation
plot_input = plot_input.reindex(plot_input.sum(axis=1).sort_values().index)

fig, ax = plt.subplots(figsize=(12, 5))

indexes = np.argsort(plot_input.values).T
heights = np.sort(plot_input.values).T
order = -1
bottoms = heights[::order].cumsum(axis=0)
bottoms = np.insert(bottoms, 0, np.zeros(len(bottoms[0])), axis=0)

# List of 30 tab-like hexadecimal colors
hex_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#5254a3', '#6b4c9a', '#9c9ede', '#637939', '#8ca252',
              '#b5cf6b', '#bd9e39', '#e7ba52', '#843c39', '#ad494a',
              '#d6616b', '#e7969c', '#7b4173', '#a55194', '#ce6dbd',
              '#de9ed6', '#ff9888', '#8e6d8e', '#c49c94', '#d7b5a6']

# Convert hexadecimal colors to RGB values
cmap = [mcolors.to_rgba(color) for color in hex_colors]

mpp_colors = dict(zip(plot_input.columns, cmap))

for btms, (idxs, vals) in enumerate(list(zip(indexes, heights))[::order]):
    mps = np.take(np.array(plot_input.columns), idxs)
    ax.bar(
        x=plot_input.index, height=vals, width=0.9, lw=0.0,
        bottom=bottoms[btms], color=[mpp_colors[m] for m in mps]
    )
ax.set_xlabel('Cluster', size=10, labelpad=10)
ax.set_ylabel('Cells per Sample', size=10, labelpad=13, c='k')
ax.tick_params(axis='x', which='major', labelsize=2, rotation=90)
ax.tick_params(axis='y', which='major', labelsize=7)
ax.margins(x=0)

markers = [
    Patch(facecolor=color, edgecolor=None) for color in mpp_colors.values()
]
plt.legend(
    markers, mpp_colors.keys(), title='TNBC Sample', prop={'size': 5.5},
    labelspacing=0.01, loc='upper left', bbox_to_anchor=(1.0, 1.01)
)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'topacio_samples_per_cluster.pdf'))
plt.close('all')
