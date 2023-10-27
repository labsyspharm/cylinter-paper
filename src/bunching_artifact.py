import os
import pickle
import glob

import numpy as np
import pandas as pd
import math
from natsort import natsorted

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
import dask.array as da
import zarr
import tifffile

# tissue #
tissue = '840063_0110'


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


def single_channel_pyramid(tiff_path, channel):

    target_filepath = tiff_path
    tiff = tifffile.TiffFile(target_filepath, is_ome=False)

    pyramid = [
        zarr.open(s[channel].aszarr())
        for s in tiff.series[0].levels
        ]

    pyramid = [
        da.from_zarr(z)
        for z in pyramid
        ]

    return pyramid


# where to save output
save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'bunching_artifact'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read markers.csv
markers = pd.read_csv(
    '/Volumes/My Book/cylinter_input/topacio_cellring/markers.csv'
    )

# read unfiltered TOPACIO dataset
tissue_data = pd.read_csv(
    '/Volumes/My Book/cylinter_input/bunching_artifact/' +
    'unmicst2-840063_0110.csv'
    )
mask_object = 'cellRingMask'

# tissue_data = pd.read_csv(
#     '/Volumes/My Book/cylinter_input/bunching_artifact/' +
#     '840063_0110.csv'
#     )
# mask_object = 'cellMask'

# tissue_data = pd.read_csv(
#     '/Volumes/My Book/cylinter_input/bunching_artifact/' +
#     'unmicst-840063_0110.csv'
#     )
# mask_object = 'cellRingMask'

# tissue_data = pd.read_csv(
#     '/Volumes/My Book/cylinter_input/bunching_artifact/' +
#     'unmicst_old-840063_0110.csv'
#     )
# mask_object = 'cellMask'

# tissue_data = pd.read_csv(
#     '/Volumes/My Book/cylinter_input/bunching_artifact/' +
#     'unmicst_new-840063_0110.csv'
#     )
# mask_object = 'cellMask'

# create a list of antibodies of interest (AOI)
aoi = [
    i for i in list(markers['marker_name']) if not
    any(x in i for x in ['DNA', 'antiRat', 'antiRabbit', 'antiGoat', 'pSTAT1',
        'Ki67', 'STING', 'pTBK1', 'pSTAT3', 'PCNA', 'HLAA', 'cPARP'])
        ]
# isolate AOI columns from EMIT data
clustering_cols = [
    i for i in tissue_data.columns if i.split(f'_{mask_object}')[0] in aoi
    ]

# drop very-low intensity cells
lower_quantiles = tissue_data[clustering_cols].quantile(q=0.01)
upper_quantiles = tissue_data[clustering_cols].quantile(q=0.99)
for lq, uq in zip(lower_quantiles.items(), upper_quantiles.items()):
    tissue_data = tissue_data[tissue_data[lq[0]] > lq[1]]
    tissue_data = tissue_data[tissue_data[uq[0]] < uq[1]]

# log-transform antibody column data
transformed_data = np.log10(tissue_data[clustering_cols])

# update EMIT dataframe with log-transformed values
tissue_data.update(transformed_data)

# assign file path to OME-TIFF as a variable
img_path = ('/Volumes/My Book/cylinter_input/bunching_artifact/' +
            f'{tissue}.ome.tif')

# read DNA1 image of selected tissue
dna = single_channel_pyramid(
    glob.glob(img_path)[0], channel=0
    )

# read CD8a image of selected tissue
channel_number = markers['channel_number'][
            markers['marker_name']
            == 'CD8a']
cd8a = single_channel_pyramid(
    glob.glob(img_path)[0], channel=channel_number.item() - 1
    )

# read segmentation outlines of selected tissue and crop to target size
seg_path = ('/Volumes/My Book/cylinter_input/topacio_cellring/seg/' +
            '840063_0110-median_r_50.ome.tif')
seg = single_channel_pyramid(
    glob.glob(seg_path)[0], channel=0
    )

# read DNA1 image for matplotlib plot
dna_imread = imread(
    '/Volumes/My Book/cylinter_input/bunching_artifact/'
    f'{tissue}.ome.tif', key=0
    )

# read CD8a image for matplotlib plot
cd8a_imread = imread(
    '/Volumes/My Book/cylinter_input/bunching_artifact/'
    f'{tissue}.ome.tif', key=13
    )

# plot DNA+CD8a image
plt.imshow(dna_imread, alpha=1.0, cmap='Greys_r', vmin=0, vmax=65535)
plt.imshow(cd8a_imread, cmap='magma', alpha=0.5,  vmin=2000, vmax=6725)
plt.grid(False)
# plt.colorbar(format=ticker.FuncFormatter(fmt))
cbar = plt.colorbar()
cbar.set_alpha(1.0)
cbar.set_label('CD8a signal per pixel', labelpad=20, rotation=270)
cbar.draw_all()
plt.savefig(
    os.path.join(
        save_dir,
        'cd8a.png'), dpi=1000,
        )
plt.savefig(
    os.path.join(
        save_dir,
        'cd8a.pdf')
        )
plt.close('all')

# plot CD8a histogram
gate = (4.0, tissue_data[f'CD8a_{mask_object}'].max())
plt.hist(
    tissue_data[f'CD8a_{mask_object}'], bins=60,
    density=False, color='k', lw=0.0
    )
plt.axvline(gate[0], color='r')
plt.axvline(gate[1], color='r')
plt.title(f'Sample {tissue}')
plt.xlabel('log(CD8a) per cell')
plt.ylabel('cell count')
plt.savefig(
    os.path.join(
        save_dir,
        'histogram.pdf')
        )
plt.close('all')

centroids = tissue_data[['Y_centroid', 'X_centroid']][
    (tissue_data[f'CD8a_{mask_object}'].between(gate[0], gate[1],
     inclusive=True))
    ]

if os.path.exists(os.path.join(save_dir, 'polygon_dict.pkl')):
    with open(os.path.join(save_dir, 'polygon_dict.pkl'), 'rb') as handle:
        polygon_dict = pickle.load(handle)
else:

    polygon_dict = {}
    polygons = []

    # visualize tissue in Napari
    viewer = napari.view_image(
        dna, opacity=0.5, rgb=False, name='DNA1'
        )

    viewer.add_image(
        seg, rgb=False, blending='additive',
        opacity=0.5, colormap='green', visible=False,
        name='segmentation'
        )

    viewer.add_image(
        cd8a, rgb=False, blending='additive',
        colormap='magma', visible=True,
        name='cd8a'
        )

    viewer.add_points(
        centroids,
        name='centroids',
        properties=None,
        face_color='yellow',
        edge_color='k',
        edge_width=0.0, size=4.0
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

    if selection_layer.data:

        polygon_dict[tissue] = selection_layer.data

        os.chdir(save_dir)
        f = open(os.path.join(save_dir, 'polygon_dict.pkl'), 'wb')
        pickle.dump(polygon_dict, f)
        f.close()

if not os.path.exists(os.path.join(save_dir, 'in_roi.parquet')):

    # isolate core data within ROI bounds
    roi_data = tissue_data[['X_centroid', 'Y_centroid', 'CellID']].astype(int)
    roi_data['tuple'] = list(
        zip(roi_data['Y_centroid'], roi_data['X_centroid']))

    columns, rows = np.meshgrid(
        np.arange(dna_imread.shape[1]),
        np.arange(dna_imread.shape[0])
        )
    columns, rows = columns.flatten(), rows.flatten()

    pixel_coords = np.vstack((rows, columns)).T
    cell_coords = set([i for i in roi_data['tuple']])

    cell_ids = set()
    mask_coords = set()
    for poly in polygon_dict[tissue]:
        selection_verts = np.round(poly).astype(int)
        polygon = Path(selection_verts)
        grid = polygon.contains_points(pixel_coords)
        mask = grid.reshape(
            dna_imread.shape[0], dna_imread.shape[1])

        mask_coords.update(
            [tuple(i) for i in np.argwhere(mask)]
            )

    inter = mask_coords.intersection(cell_coords)

    cell_ids.update(
        [i[1]['CellID'] for i in roi_data.iterrows() if
         i[1]['tuple'] in inter]
         )

    in_roi = tissue_data[roi_data['CellID'].isin(cell_ids)]

    in_roi.to_parquet(os.path.join(save_dir, 'in_roi.parquet'))

else:
    in_roi = pd.read_parquet(os.path.join(save_dir, 'in_roi.parquet'))

plt.imshow(dna_imread, cmap='Greys_r')
plt.savefig(
    os.path.join(
        save_dir,
        'original.png'), dpi=1000,
        )
plt.close('all')

plt.imshow(dna_imread, cmap='Greys_r')
plt.scatter(in_roi['X_centroid'], in_roi['Y_centroid'], s=0.01)
plt.savefig(
    os.path.join(
        save_dir,
        'gate.png'), dpi=1000,
        )
plt.close('all')
###############################################################################
# assign cells inside ROI bounds the color yellow
tissue_data['color'] = [
    'tab:blue' if i in in_roi.index
    else 'gray' for i in tissue_data.index]

# perform clustering
cldata1 = tissue_data[clustering_cols + ['color']].copy()
cldata1 = cldata1.sample(frac=1.0, random_state=3)

if os.path.exists(
  os.path.join(save_dir, 'embedding.npy')):

    embedding = np.load(
      os.path.join(save_dir, 'embedding.npy')
      )
    cldata1['emb1'] = embedding[:, 0]
    cldata1['emb2'] = embedding[:, 1]

else:
    print('Performing UMAP embedding...')
    embedding = UMAP(
        random_state=3, min_dist=0.1, repulsion_strength=1.0).fit_transform(
            cldata1[clustering_cols])

    cldata1['emb1'] = embedding[:, 0]
    cldata1['emb2'] = embedding[:, 1]

    np.save(
        os.path.join(save_dir, 'embedding'),
        embedding
        )

clustering1 = hdbscan.HDBSCAN(
    min_cluster_size=100).fit(cldata1[['emb1', 'emb2']])
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
fig, ax = plt.subplots(3)

ax[0].scatter(
    cldata1['emb1'], cldata1['emb2'],
    c=cldata1[f'CD8a_{mask_object}'], cmap='magma', s=0.01, lw=0.0
    )
ax[0].set_aspect('equal')

ax[1].scatter(
    cldata1['emb1'], cldata1['emb2'],
    c=cldata1['color'], cmap='magma', s=0.01, lw=0.0
    )
ax[1].set_aspect('equal')

ax[2].scatter(
    cldata1['emb1'], cldata1['emb2'],
    c=c, cmap=cmap, s=0.01, lw=0.0
    )
ax[2].set_aspect('equal')

plt.savefig(
    os.path.join(
        save_dir,
        'clustering.png'), dpi=1000,
        )
plt.close('all')
