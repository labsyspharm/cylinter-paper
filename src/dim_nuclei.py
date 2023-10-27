import os
import pandas as pd
import seaborn as sns
import numpy as np
import math
from matplotlib import colors
from matplotlib.colors import ListedColormap
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from tifffile import imread
from umap import UMAP
import hdbscan
from joblib import Memory

# core = '12'
core = '840069_0048'
# dna1 = DNA1_cellMask'
dna1 = 'DNA0'
# dna_intensity_gate = (0, 3500)
dna_intensity_gate = (0, 15000)
channels = [f'DNA1:{core}']
vmax = 18000


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def categorical_cmap(numUniqueSamples, numCatagories, cmap='tab10', continuous=False):

    numSubcatagories = math.ceil(numUniqueSamples/numCatagories)

    if numCatagories > plt.get_cmap(cmap).N:
        raise ValueError('Too many categories for colormap.')
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, numCatagories))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(numCatagories, dtype=int))
        # rearrange hue order to taste
        cd = {
            'B': 0, 'O': 1, 'G': 2, 'R': 3, 'Pu': 4,
            'Br': 5, 'Pi': 6, 'Gr': 7, 'Y': 8, 'Cy': 9,
            }
        myorder = [
            cd['B'], cd['O'], cd['G'], cd['Pu'], cd['Y'],
            cd['R'], cd['Cy'], cd['Br'], cd['Gr'], cd['Pi']
            ]
        ccolors = [ccolors[i] for i in myorder]

        # use Okabe and Ito color-safe palette for first 6 colors
        # ccolors[0] = np.array([0.91, 0.29, 0.235]) #E84A3C
        # ccolors[1] = np.array([0.18, 0.16, 0.15]) #2E2926
        ccolors[0] = np.array([0.0, 0.447, 0.698, 1.0])  # blue
        ccolors[1] = np.array([0.902, 0.624, 0.0, 1.0])  # orange
        ccolors[2] = np.array([0.0, 0.620, 0.451, 1.0])  # bluish green
        ccolors[3] = np.array([0.8, 0.475, 0.655, 1.0])  # reddish purple
        ccolors[4] = np.array([0.941, 0.894, 0.259, 1.0])  # yellow
        ccolors[5] = np.array([0.835, 0.369, 0.0, 1.0])  # vermillion

    cols = np.zeros((numCatagories * numSubcatagories, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, numSubcatagories).reshape(numSubcatagories, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, numSubcatagories)
        arhsv[:, 2] = np.linspace(chsv[2], 1, numSubcatagories)
        rgb = colors.hsv_to_rgb(arhsv)
        cols[i * numSubcatagories:(i + 1) * numSubcatagories, :] = rgb
    cmap = colors.ListedColormap(cols)

    # trim colors if necessary
    if len(cmap.colors) > numUniqueSamples:
        trim = len(cmap.colors) - numUniqueSamples
        cmap_colors = cmap.colors[:-trim]
        cmap = colors.ListedColormap(cmap_colors, name='from_list', N=None)

    return cmap


save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'dim_nuclei'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read raw single-cell data

# data = pd.read_csv(
#     f'/Volumes/My Book/cylinter_input/emit22_full/csv/unmicst-{core}.csv'
#     )

data = pd.read_csv(
    f'/Volumes/My Book/cylinter_input/topacio_mcupdated_cellring/csv/' +
    f'unmicst-{core}_cellRingMask.csv', index_col=0
    )

# crop
# ymin = 1775
# ymax = 1925
# xmin = 1250
# xmax = 1400

ymin = 6700
ymax = 7100
xmin = 11300
xmax = 11700

# read DNA1 image of selected tissue and crop to target size

# dna = imread(
#     f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif', key=0
#     )

dna = imread(
    '/Volumes/My Book/cylinter_input/topacio_mcupdated_cellring/tif/' +
    f'{core}.ome.tif', key=0
    )

dna_crop = dna[ymin:ymax, xmin:xmax]

# read segmentation outlines for target tissue sample and crop ROI

# seg = imread(
#     f'/Volumes/My Book/cylinter_input/emit22_full/seg/{core}.ome.tif', key=0
#     )

seg = imread(
    '/Volumes/My Book/cylinter_input/topacio_mcupdated_cellring/seg/' +
    f'{core}.ome.tif', key=0
    )

seg_crop = seg[ymin:ymax, xmin:xmax]

# plot cropped image channels, segmentation outlines, and analysis rectangle
plt.imshow(seg_crop, alpha=1.0, cmap='Greys_r')
plt.imshow(dna_crop, alpha=0.6, cmap='magma', vmax=vmax)
plt.grid(False)

cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
cbar.set_alpha(1.0)
cbar.set_label('Hoechst per cell', labelpad=20, rotation=270)
cbar.draw_all()

plt.savefig(os.path.join(save_dir, 'core_12.png'), dpi=1000)
plt.close('all')

# isolate data for cells within ROI bounds (original coordinates)
data_roi = data[['Y_centroid', 'X_centroid', dna1]][
    (data['X_centroid'].between(xmin, xmax, inclusive='both')) &
    (data['Y_centroid'].between(ymin, ymax, inclusive='both'))
    ]

data_gated = data[['Y_centroid', 'X_centroid', dna1]][
    (data[dna1].between(dna_intensity_gate[0], dna_intensity_gate[1], inclusive='both'))
    ]

# color nuclei according to brightness
dim_ids = data['CellID'][
    data[dna1].between(dna_intensity_gate[0], dna_intensity_gate[1], inclusive='both')
    ]
dim = data[data['CellID'].isin(dim_ids)]
dim.loc[:, 'color'] = [[0.13, 1.0, 0.02] for i in dim.index]
bright = data[~data['CellID'].isin(dim_ids)]
bright.loc[:, 'color'] = [[0.9, 0.9, 0.9] for i in bright.index]
data = dim.append(bright)

sns.set_style('whitegrid')
plt.hist(data[dna1], bins=40, color='k', density=False, lw=0.0)
plt.axvline(dna_intensity_gate[1], c='tab:red')
plt.title(f'Core {core}')
plt.xlabel('DNA1')
plt.ylabel('cell count')
plt.savefig(os.path.join(save_dir, 'gated_hist.pdf'))
plt.close('all')


#########################################################
plt.imshow(dna)
plt.scatter(data['X_centroid'], data['Y_centroid'], c=data['color'], s=1)
plt.grid(False)
plt.show()

#########################################################
# transform x, y coordinates of cells within ROI bounds
data_gated['X_centroid'] = data_roi['X_centroid']-xmin
data_gated['Y_centroid'] = data_roi['Y_centroid']-ymin

# plot channel images and segmentation outlines
plt.imshow(seg_crop, alpha=1.0, cmap='Greys_r')
plt.imshow(dna_crop, alpha=0.6, cmap='magma', vmax=vmax)
plt.grid(False)

cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
cbar.set_alpha(1.0)
cbar.set_label('Hoechst per cell', labelpad=20, rotation=270)
cbar.draw_all()

# overlay cell centorids
plt.scatter(
    data_gated['X_centroid'],
    data_gated['Y_centroid'],
    s=4.5, lw=0.0, color='lime'
    )
plt.title(f'Core {core}')
plt.savefig(os.path.join(save_dir, 'dim_nuclei.png'), dpi=1000)

# save as a pdf, too
plt.savefig(os.path.join(save_dir, 'dim_nuclei.pdf'))
plt.close('all')

# cluster
# antibodies only
# cols_to_include = [
#     'CD63_cellMask', 'PCNA_2_cellMask', 'cPARP_cellMask', 'KI67_cellMask',
#     'CCND1_cellMask', 'CD107B_cellMask', 'CDKN1A_cellMask', 'CDK2_cellMask',
#     'CDKN1C_cellMask', 'ECAD_cellMask', 'CD32_cellMask', 'MART1_cellMask',
#     'aSMA_cellMask', 'CCNB1_cellMask', 'pCREB_cellMask', 'pan-CK_cellMask',
#     'CD45_cellMask', 'CCNA2_cellMask', 'CDKN1B_1_cellMask', 'CD73_cellMask',
#     ]
cols_to_include = [
    'CD3', 'PDL1', '53BP1', 'ECadherin', 'panCK', 'PD1', 'CD8a',
    'CD45', 'GrB', 'CD163', 'CD68', 'CD20', 'CD4', 'FOXP3',
    'SMA', 'CD11b', 'yH2AX', 'CD57', 'MHCII',
    ]

# cols_to_include = [
#     'DNA1_cellMask', 'Eccentricity', 'Area'
#     ]

# isolate channel columns for clustering and log-transform data
transformed_data = np.log10(data[cols_to_include] + 0.00001)
data.update(transformed_data)

# slice dataframe to generate plot input
clus_data = data[cols_to_include + ['CellID', 'color']].copy()
clus_data = clus_data.sample(frac=1.0, random_state=3)

# UMAP
if os.path.exists(os.path.join(save_dir, 'embedding.npy')):

    embedding = np.load(os.path.join(save_dir, 'embedding.npy'))
    clus_data['emb1'] = embedding[:, 0]
    clus_data['emb2'] = embedding[:, 1]

else:
    print('Performing UMAP embedding...')
    embedding = UMAP(
        random_state=5, min_dist=0.2,
        repulsion_strength=2.0).fit_transform(clus_data[cols_to_include])
    clus_data['emb1'] = embedding[:, 0]
    clus_data['emb2'] = embedding[:, 1]

    np.save(os.path.join(save_dir, 'embedding'), embedding)

# HDBSCAN
clustering = hdbscan.HDBSCAN(
    min_cluster_size=50).fit(clus_data[['emb1', 'emb2']])
clus_data['cluster'] = clustering.labels_


# generate categorical cmap for clusters
cmap = categorical_cmap(
    numUniqueSamples=len(clus_data['cluster'].unique()),
    numCatagories=10,
    cmap='tab10', continuous=False)

cmap = ListedColormap(
    np.insert(
        arr=cmap.colors, obj=0,
        values=[0, 0, 0], axis=0)
        )
trim = (
    len(cmap.colors) - len(
        clus_data['cluster'].unique())
    )
cmap = ListedColormap(
    cmap.colors[:-trim]
    )
sample_dict = dict(
    zip(
        natsorted(
            clus_data['cluster'].unique()),
        list(range(len(clus_data['cluster']
             .unique()))))
        )
c = [sample_dict[i] for i
     in clus_data['cluster']]

# plot
sns.set_style('white')
fig, ax = plt.subplots()
ax.scatter(
    clus_data['emb1'], clus_data['emb2'], c=clus_data['color'],
    s=9.5, ec=[0.0, 0.0, 0.0], lw=0.1, alpha=1.0)
ax.set_aspect('equal')
ax.set_aspect('equal')
plt.savefig(os.path.join(save_dir, 'clustering_gate.png'), dpi=1000)
plt.show()
plt.close('all')

fig, ax = plt.subplots()
ax.scatter(
    clus_data['emb1'], clus_data['emb2'], c=c, cmap=cmap,
    s=8.0, ec=[0.0, 0.0, 0.0], lw=0.01, alpha=1.0)
ax.set_aspect('equal')
ax.set_aspect('equal')
plt.savefig(os.path.join(save_dir, 'clustering_hdbscan.png'), dpi=1000)
plt.show()
plt.close('all')

# fig, ax = plt.subplots()
# ax.scatter(
#     clus_data['DNA1_cellMask'], clus_data['Area'], c=clus_data['color'],
#     s=8.0, ec=[0.0, 0.0, 0.0], lw=0.01, alpha=1.0)
# ax.set_aspect('equal')
# plt.savefig(os.path.join(save_dir, 'scatter.png'), dpi=1000)
# plt.show()
# plt.close('all')
