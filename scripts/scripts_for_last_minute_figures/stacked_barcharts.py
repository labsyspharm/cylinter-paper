import os
import pandas as pd
import matplotlib.pyplot as plt

save_dir = (
    '/Users/greg/Dropbox (HMS)/topacio/cylinter_output/' +
    'TOPACIO_FINAL/output_orig/clustering/2d/cluster_barcharts'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_parquet(
    '/Users/greg/Dropbox (HMS)/topacio/cylinter_output/TOPACIO_FINAL/' +
    'output_orig/checkpoints/clustering.parquet'
    )

df = df[df['cluster_2d'] != -1]

for cluster in sorted(df['cluster_2d'].unique()):
    test = df[df['cluster_2d'] == cluster]
    plot = test.groupby('Sample').size().sort_values()
    plot = pd.DataFrame(plot/plot.sum()).T
    plot.columns = [i.split('_')[1].lstrip('0') for i in plot.columns]

    ax = plot.plot(
        kind='bar', width=0.1, stacked=True, color='w', edgecolor='k'
        )

    pos = -1
    for i, bar in enumerate(ax.patches):
        if i % len(plot.index) == 0:
            pos += 1
        xloc = bar.get_x() + 0.05
        yloc = bar.get_y() + bar.get_height()/2
        if bar.get_height() > 0.08:
            ax.annotate(
                str(plot.columns[pos]), xy=(xloc, yloc), fontname='Arial',
                va='center', ha='center', size=20
                )

    ax.get_legend().remove()
    ax.axis('off')

    plt.savefig(os.path.join(save_dir, f'{cluster}_pie.pdf'))
    plt.close('all')

###############################################################################

save_dir = (
    '/Users/greg/Dropbox (HMS)/topacio/cylinter_output/' +
    'TOPACIO_FINAL/output_raw/clustering/2d/cluster_barcharts'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_parquet(
    '/Users/greg/Dropbox (HMS)/topacio/cylinter_output/TOPACIO_FINAL/' +
    'output_raw/checkpoints/clustering.parquet'
    )

df = df[df['cluster_2d'] != -1]

my_clusters = [
    0, 1, 2, 4, 6, 11, 14, 15, 19, 21, 28, 77, 79, 89, 110, 118,
    166, 174, 177, 187, 197, 205, 215, 237, 268, 269, 272, 309, 312, 324, 328,
    336, 341, 383, 404, 405, 408, 415, 424, 427, 437, 444, 465, 466, 477, 478,
    480, 491
    ]

df = df[df['cluster_2d'].isin(my_clusters)]

for cluster in sorted(df['cluster_2d'].unique()):
    test = df[df['cluster_2d'] == cluster]
    plot = test.groupby('Sample').size().sort_values()
    plot = pd.DataFrame(plot/plot.sum()).T
    plot.columns = [i.split('_')[1].lstrip('0') for i in plot.columns]

    ax = plot.plot(
        kind='bar', width=0.1, stacked=True, color='w', edgecolor='k'
        )

    pos = -1
    for i, bar in enumerate(ax.patches):
        if i % len(plot.index) == 0:
            pos += 1
        xloc = bar.get_x() + 0.05
        yloc = bar.get_y() + bar.get_height()/2
        if bar.get_height() > 0.08:
            ax.annotate(
                str(plot.columns[pos]), xy=(xloc, yloc), fontname='Arial',
                va='center', ha='center', size=20
                )

    ax.get_legend().remove()
    ax.axis('off')

    plt.savefig(os.path.join(save_dir, f'{cluster}_pie.pdf'))
    plt.close('all')
