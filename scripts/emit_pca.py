import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from natsort import natsorted
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as norm


save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'emit_pca'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read raw, clustered EMIT single-cell data
raw = pd.read_parquet(
    '/Volumes/My Book/cylinter_input/emit22_full/output_raw/' +
    'checkpoints/clustering.parquet'
    )

# read clean, clustered single-cell data
clean = pd.read_parquet(
    '/Volumes/My Book/cylinter_input/emit22_full/output/' +
    'checkpoints/clustering.parquet'
    )

# read markers.csv
markers = pd.read_csv(
    '/Volumes/My Book/cylinter_input/emit22_full/markers.csv'
    )

# create a list of antibodies of interest (AOI)
aoi = [
    i for i in list(markers['marker_name']) if not
    any(x in i for x in ['DNA', 'IgG', 'CD56', 'CD13', 'pAUR',
        'CCNE', 'CDKN2A', 'PCNA_1', 'CDKN1B_2'])
        ]

# isolate AOI columns from EMIT data
abx_channels = [
    i for i in clean.columns if i.split('_cellMask')[0] in aoi
    ]

# mean-center raw data
# raw_copy = raw.copy()
# raw_centered = pd.DataFrame()
# for name, group in raw_copy.groupby(['Sample']):
#     for c in group[abx_channels].columns:
#         group[c] = group[c]-group[c].mean()
#     raw_centered = raw_centered.append(group, ignore_index=False)
# raw_centered = raw_centered.sort_index()

# mean-center clean data
# clean_copy = clean.copy()
# clean_centered = pd.DataFrame()
# for name, group in clean_copy.groupby(['Sample']):
#     for c in group[abx_channels].columns:
#         group[c] = group[c]-group[c].mean()
#     clean_centered = clean_centered.append(group, ignore_index=False)
# clean_centered = clean_centered.sort_index()

# raw_medians = raw_centered.groupby(['Sample']).median()[abx_channels]
# raw_medians = raw_medians.reindex(natsorted(raw_medians.index))

# clean_medians = clean_centered.groupby(['Sample']).median()[abx_channels]
# clean_medians = clean_medians.reindex(natsorted(clean_medians.index))

raw_medians = raw.groupby(['Sample']).median()[abx_channels]
raw_medians = raw_medians.reindex(natsorted(raw_medians.index))

clean_medians = clean.groupby(['Sample']).median()[abx_channels]
clean_medians = clean_medians.reindex(natsorted(clean_medians.index))

for name, data in zip(['raw', 'clean'], [raw_medians, clean_medians]):

    print(f'Analyzing {name} data...')

    evr = pd.DataFrame()

    for l in range(1, 101):
        shuffled = data.copy()
        for e, col in enumerate(shuffled.columns):
            shuffled[col] = shuffled[col].sample(
                frac=1, random_state=e+l).values

        # specify PCA parameters
        pca = PCA(n_components=20, random_state=1)

        idx = shuffled.index

        # normalize signal intensities across samples (axis=0)
        shuffled = norm(
            shuffled, norm='l2', axis=0, copy=True, return_norm=False
            )

        # for c in shuffled.columns:
        #     shuffled[c] = shuffled[c]-shuffled[c].mean()

        # apply PCA parameters to data
        projected = pca.fit_transform(shuffled)

        evr[l] = pca.explained_variance_ratio_

    # specify PCA parameters
    pca = PCA(n_components=20, random_state=1)

    idx = data.index

    # normalize signal intensities across samples (axis=0)
    data = norm(
        data, norm='l2', axis=0, copy=True, return_norm=False
        )

    # for c in data.columns:
    #     data[c] = data[c]-data[c].mean()

    # apply PCA parameters to data
    projected = pca.fit_transform(data)

    # generate dataframe for plot input
    scatter_input = pd.DataFrame(data=projected, index=idx)
    scatter_input.rename(columns={0: 'PC1', 1: 'PC2'}, inplace=True)

    sns.set_style('darkgrid')
    fig1, ax1 = plt.subplots()

    ax1.plot(pca.explained_variance_ratio_)
    ax1.plot(evr.mean(axis=1))

    plt.xticks(np.arange(0, 20, 1.0))

    ax1.set_xlabel(
        'PC', fontsize=10, labelpad=7.0)
    ax1.set_ylabel(
        'Explained Variance Ratio', fontsize=10, labelpad=4.0)

    legend_handles = []
    legend_handles.append(
        Line2D([0], [0], marker=None, color='tab:blue',
               label='unshuffled', markeredgewidth=0.7,
               markersize=5.0)
               )
    legend_handles.append(
        Line2D([0], [0], marker=None, color='tab:orange',
               label='average shuffled', markeredgewidth=0.7,
               markersize=5.0)
               )
    ax1.legend(
        handles=legend_handles,
        prop={'size': 10.0},
        bbox_to_anchor=[0.95, 1.0]
        )
    fig1.savefig(
        os.path.join(save_dir, f'{name}_variance.pdf'),
        bbox_inches='tight'
        )
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        data=scatter_input, x='PC1', y='PC2', color='tab:blue',
        linewidth=0.0, s=90, alpha=0.3, legend=False
        )

    ax2.set_xlabel(
        f'PC1 ({round((pca.explained_variance_ratio_[0] * 100), 2)}'
        '% of variance)', fontsize=10, labelpad=7.0)
    ax2.set_ylabel(
        f'PC2 ({round((pca.explained_variance_ratio_[1] * 100), 2)}'
        '% of variance)', fontsize=10, labelpad=4.0)
    ax2.tick_params(axis='both', which='major', labelsize=7.0)

    legend_handles = []
    legend_handles.append(
        Line2D([0], [0], marker='o', color='white',
               label='unshuffled', markeredgewidth=0.0,
               markerfacecolor='tab:blue',
               markersize=8.0)
               )
    ax2.legend(
        handles=legend_handles,
        prop={'size': 10.0},
        loc='upper right'
        )

    fig2.savefig(
        os.path.join(
            save_dir, f'{name}_pcaScoresPlot.png'), dpi=600,
        bbox_inches='tight'
        )
    plt.close(fig2)
