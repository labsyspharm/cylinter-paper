import os
import pickle

import numpy as np
import pandas as pd
import math
from natsort import natsorted

# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import colors
from matplotlib.colors import ListedColormap

from tifffile import imread

from umap import UMAP
import hdbscan
from joblib import Memory

import napari

# core #
core = '68'

# lower and upper gates on CD63
gate = (2.585, 3.0)


# function for formatting color bar values
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


# where to save output
save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'aggregate_artifact'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read unfiltered EMIT dataset
data = pd.read_parquet(
    '/Volumes/My Book/cylinter_input/emit22_full/output/' +
    'checkpoints/getSingleCellData.parquet'
    )
# filter data to isolate data from core of interest
core_data = data[data['Sample'] == core].copy()

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
clustering_cols = [
    i for i in core_data.columns if i.split('_cellMask')[0] in aoi
    ]

# log-transform antibody column data
transformed_data = np.log10(core_data[clustering_cols])

# update EMIT dataframe with log-transformed values
core_data.update(transformed_data)

# define crop window for EMIT core image
ymin = 205
ymax = 2795
xmin = 290
xmax = 2730

# read DNA1 image of selected core and crop to target size
dna = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif', key=0
    )
dna_crop = dna[ymin:ymax, xmin:xmax]

# read CD63 image of selected core and crop to target size
cd63 = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif', key=18
    )
cd63_crop = cd63[ymin:ymax, xmin:xmax]

# read segmentation outlines of selected core and crop to target size
seg = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/seg/{core}.ome.tif', key=0
    )
seg_crop = seg[ymin:ymax, xmin:xmax]

# plot cropped image: DNA+CD63
plt.imshow(dna_crop, alpha=1.0, cmap='Greys_r', vmin=0, vmax=25000)
plt.imshow(cd63_crop, cmap='magma', alpha=0.5, vmin=0, vmax=10000)
plt.grid(False)
# plt.colorbar(format=ticker.FuncFormatter(fmt))
cbar = plt.colorbar()
cbar.set_alpha(1.0)
cbar.set_label('CD63 signal per pixel', labelpad=20, rotation=270)
cbar.draw_all()
plt.savefig(
    os.path.join(
        save_dir,
        'original.png'), dpi=600,
        )
plt.savefig(
    os.path.join(
        save_dir,
        'original.pdf')
        )
plt.close('all')

# plot CD63 histogram
# sns.set_style('whitegrid')
plt.hist(
    core_data['CD63_cellMask'], bins=60, density=False, color='k', lw=0.0
    )
plt.axvline(gate[0], color='r')
plt.title(f'Core {core}')
plt.xlabel('log(CD63) per cell')
plt.ylabel('cell count')
plt.savefig(
    os.path.join(
        save_dir,
        'histogram.pdf')
        )
plt.close('all')

if os.path.exists(os.path.join(save_dir, 'roi.npy')):
    roi = np.load(os.path.join(save_dir, 'roi.npy'))
else:
    # gate on anti-CD63 antibody aggregate in Napari
    viewer = napari.view_image(
        dna, opacity=0.5, rgb=False, name='DNA1'
        )

    viewer.add_image(
        seg, rgb=False, blending='additive',
        opacity=0.5, colormap='green', visible=False,
        name='segmentation'
        )

    viewer.add_image(
        cd63, rgb=False, blending='additive',
        colormap='magma', visible=True,
        name='CD63'
        )

    selection_layer = viewer.add_shapes(
        shape_type='polygon',
        ndim=2,
        face_color=[1.0, 1.0, 1.0, 0.2],
        edge_color=[0.0, 0.66, 1.0, 1.0],
        edge_width=10.0,
        name='ROI(s)'
        )

    napari.run()

    np.save(os.path.join(save_dir, 'roi.npy'), selection_layer.data[0])
    roi = selection_layer.data[0]

# isolate core data within ROI bounds
roi_data = core_data[['X_centroid', 'Y_centroid', 'CellID']].astype(int)
roi_data['tuple'] = list(zip(roi_data['Y_centroid'], roi_data['X_centroid']))

columns, rows = np.meshgrid(
    np.arange(dna.shape[1]),
    np.arange(dna.shape[0])
    )
columns, rows = columns.flatten(), rows.flatten()

pixel_coords = np.vstack((rows, columns)).T
cell_coords = set([i for i in roi_data['tuple']])

cell_ids = set()
mask_coords = set()

selection_verts = np.round(roi).astype(int)
polygon = Path(selection_verts)
grid = polygon.contains_points(pixel_coords)
mask = grid.reshape(
    dna.shape[0], dna.shape[1])

mask_coords.update(
    [tuple(i) for i in np.argwhere(mask)]
    )

inter = mask_coords.intersection(cell_coords)

cell_ids.update(
    [i[1]['CellID'] for i in roi_data.iterrows() if
     i[1]['tuple'] in inter]
     )

in_roi = core_data[roi_data['CellID'].isin(cell_ids)]
out_roi = core_data[~roi_data['CellID'].isin(cell_ids)]

###############################################################################
# get CD63 bright cells within ROI
hi_roi = in_roi[['Y_centroid', 'X_centroid', 'CD63_cellMask', 'CellID']][
    (in_roi['X_centroid'].between(xmin, xmax, inclusive=True)) &
    (in_roi['Y_centroid'].between(ymin, ymax, inclusive=True)) &
    (in_roi['CD63_cellMask'] > gate[1])
    ]
# transform x, y coordinates
hi_roi['X_centroid'] = hi_roi['X_centroid']-xmin
hi_roi['Y_centroid'] = hi_roi['Y_centroid']-ymin

# get CD63 dim cells within ROI
low_roi = in_roi[['Y_centroid', 'X_centroid', 'CD63_cellMask', 'CellID']][
    (in_roi['X_centroid'].between(xmin, xmax, inclusive=True)) &
    (in_roi['Y_centroid'].between(ymin, ymax, inclusive=True)) &
    (in_roi['CD63_cellMask'].between(gate[0], gate[1], inclusive=True))
    ]
# transform x, y coordinates
low_roi['X_centroid'] = low_roi['X_centroid']-xmin
low_roi['Y_centroid'] = low_roi['Y_centroid']-ymin

# get CD63 bright cells outside of ROI
hi_not_roi = out_roi[['Y_centroid', 'X_centroid', 'CD63_cellMask', 'CellID']][
    (out_roi['X_centroid'].between(xmin, xmax, inclusive=True)) &
    (out_roi['Y_centroid'].between(ymin, ymax, inclusive=True)) &
    (out_roi['CD63_cellMask'] > gate[1])
    ]
# transform x, y coordinates
hi_not_roi['X_centroid'] = hi_not_roi['X_centroid']-xmin
hi_not_roi['Y_centroid'] = hi_not_roi['Y_centroid']-ymin

# get CD63 dim cells outside of ROI
low_not_roi = out_roi[['Y_centroid', 'X_centroid', 'CD63_cellMask', 'CellID']][
    (out_roi['X_centroid'].between(xmin, xmax, inclusive=True)) &
    (out_roi['Y_centroid'].between(ymin, ymax, inclusive=True)) &
    (out_roi['CD63_cellMask'].between(gate[0], gate[1], inclusive=True))
    ]
# transform x, y coordinates
low_not_roi['X_centroid'] = low_not_roi['X_centroid']-xmin
low_not_roi['Y_centroid'] = low_not_roi['Y_centroid']-ymin

# plot DNA channel with scatter points
plt.imshow(dna_crop, alpha=1.0, cmap='Greys_r', vmin=0, vmax=43056)
plt.grid(False)

# overlay cell centorids
plt.scatter(
    low_roi['X_centroid'],
    low_roi['Y_centroid'], s=0.5, color='tab:blue'
    )
plt.scatter(
    hi_roi['X_centroid'],
    hi_roi['Y_centroid'], s=0.5, color='tab:red'
    )
plt.scatter(
    low_not_roi['X_centroid'],
    low_not_roi['Y_centroid'], s=0.5, color='tab:green'
    )
plt.scatter(
    hi_not_roi['X_centroid'],
    hi_not_roi['Y_centroid'], s=0.5, color='tab:orange'
    )
plt.title(f'Core {core}')
plt.savefig(
    os.path.join(
        save_dir,
        'scatter.pdf')
        )
plt.close('all')
###############################################################################
# assign colors columns tp core data
core_data['color'] = [
    'tab:red' if i in list(hi_roi['CellID']) else
    'tab:blue' if i in list(low_roi['CellID']) else
    'tab:green' if i in list(hi_not_roi['CellID']) else
    'tab:orange' if i in list(low_not_roi['CellID']) else
    'k' for i in core_data['CellID']]

# perform clustering
print('Performing UMAP embedding...')
cldata1 = core_data[clustering_cols + ['color']].copy()
cldata1 = cldata1.sample(frac=1.0, random_state=3)
embedding1 = UMAP(
    random_state=1, min_dist=0.1, repulsion_strength=1.0).fit_transform(
        cldata1[clustering_cols])
cldata1['emb1'] = embedding1[:, 0]
cldata1['emb2'] = embedding1[:, 1]

clustering1 = hdbscan.HDBSCAN(
    min_cluster_size=16).fit(cldata1[['emb1', 'emb2']])
cldata1['cluster'] = clustering1.labels_

# generate categorical cmap for clusters
cmap = categorical_cmap(
    numUniqueSamples=len(cldata1['cluster'].unique()),
    numCatagories=10,
    cmap='tab10', continuous=False)

cmap = ListedColormap(
    np.insert(
        arr=cmap.colors, obj=0,
        values=[0, 0, 0], axis=0)
        )

# trim cmap to # unique samples
trim = (
    len(cmap.colors) - len(
        cldata1['cluster'].unique())
    )
cmap = ListedColormap(
    cmap.colors[:-trim]
    )

sample_dict = dict(
    zip(
        natsorted(
            cldata1['cluster'].unique()),
        list(range(len(cldata1['cluster']
             .unique()))))
        )

c = [sample_dict[i] for i
     in cldata1['cluster']]

# plot clustering
fig, ax = plt.subplots(2)

ax[0].scatter(
    cldata1['emb1'], cldata1['emb2'],
    c=cldata1['CD63_cellMask'], cmap='magma', s=0.6, lw=0.0
    )
# c=cldata1['color']
ax[0].set_aspect('equal')

ax[1].scatter(
    cldata1['emb1'], cldata1['emb2'],
    c=c, cmap=cmap, s=0.6, lw=0.0
    )
ax[1].set_aspect('equal')

plt.savefig(
    os.path.join(
        save_dir,
        'clustering.pdf')
        )
plt.close('all')
###############################################################################

# core_data['color'] = [
#     'k' if i < gate[0] else 'tab:blue' if gate[0] < i < gate[1] else
#     'tab:red' for i in core_data['CD63_cellMask']
#     ]
#
# core_data['color'] = [
#     'k' if i < gate[0] else 'lime' for i in core_data['CD63_cellMask']
#     ]
#
# cluster_colors = {
#     -1: '#000000', 0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c',
#     3: '#d62728', 4: '#9467bd', 5: '#8c564b', 6: '#e377c2',
#     7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf'
#      }
# cldata1['color_cluster'] = [cluster_colors[i] for i in cldata1['cluster']]
