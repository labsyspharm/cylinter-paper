import os
import yaml

import re
import glob

from math import ceil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

import tifffile
from lazy_ops import DatasetView
import napari
import dask.array as da
import zarr

from datetime import datetime
startTime = datetime.now()


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


sample_name = 'WD-76845-097'

cylinter_input_path = (
    '/Volumes/T7 Shield/cylinter_input/sardana-097/'
    )

tif_path = cylinter_input_path + f'tif/{sample_name}.ome.tif'

he_path = cylinter_input_path + f'hema_eosin.ome.tif'

seg_path = cylinter_input_path + f'seg/{sample_name}.ome.tif'

sc_data_path = (
    '/Volumes/T7 Shield/cylinter_input/sardana-097/output_raw/'
    'checkpoints/clustering.parquet'
    )

for_channels_path = (
    '/Volumes/T7 Shield/cylinter_input/clean_quant/output_3d_v2/'
    'consensus_clustering.parquet'
    )

cylinter_config_path = (
    '/Volumes/T7 Shield/cylinter_input/clean_quant/config.yml'
    )

markers_path = (
    '/Volumes/T7 Shield/cylinter_input/clean_quant/markers.csv'
    )

image_contrast_path = (
    '/Volumes/T7 Shield/cylinter_input/clean_quant/output_3d_v2/'
    'contrast/contrast_limits.yml'
    )

# import clustered single-cell data
data = pd.read_parquet(sc_data_path)
for_channels = pd.read_parquet(for_channels_path)

# drop noisy cells (cluster=-1) from VAE clusters
data = data[data['CD45_PE'] <= 0.25]
print(data['CD45_PE'])

# import config.yml
with open(cylinter_config_path) as f:
    config = yaml.safe_load(f)
markers_to_exclude = config['markersToExclude']  # channelExclusionsClustering

# import markers.csv
markers = pd.read_csv(markers_path)

# get first name of first DNA channel
dna1 = markers['marker_name'][markers['channel_number'] == 1][0]
dna_moniker = str(re.search(r'[^\W\d]+', dna1).group())

# isolate antibodies of interest
abx_channels = [
    i for i in for_channels.columns if 'nucleiRingMask' in i
    if 'Hoechst' not in i
    if i not in markers_to_exclude]

# import image contrast settings
with open(image_contrast_path) as f:
    contrast_limits = yaml.safe_load(f)

# add H&E image (single channel image)
# tiff = tifffile.TiffFile(he_path, is_ome=False)
# pyramid = [
#     zarr.open(tiff.series[0].levels[0].aszarr())[i] for i in
#     list(range(len(tiff.series[0].levels)))
#     ]
# pyramid = [DatasetView(i).lazy_transpose([1, 2, 0]) for i in pyramid]
# pyramid = [da.from_zarr(z) for z in pyramid]
#
# viewer = napari.view_image(pyramid, rgb=True, name='H&E')

# or add H&E image (separate RGB channels)
for color, channel in zip(['red', 'green', 'blue'], [0, 1, 2]):

    img = single_channel_pyramid(
        glob.glob(he_path)[0], channel=channel
        )

    if channel == 0:
        viewer = napari.view_image(
            img, rgb=False, colormap=color, blending='additive',
            visible=False, name=f'H&E_{color}'
            )
    else:
        viewer.add_image(
            img, rgb=False, colormap=color, blending='additive',
            visible=False, name=f'H&E_{color}'
            )

# read DNA1 channel
dna = single_channel_pyramid(
    glob.glob(tif_path)[0], channel=0
    )

# add DNA1 channel to Napari image viewer
viewer.add_image(
    dna, rgb=False, blending='additive',
    colormap='gray', visible=True, opacity=1.0,
    name='DNA1'
    )

# loop over antibodies of interest and add them to Napari image viewer
for ch in abx_channels:
    ch = ch.rsplit('_', 1)[0]
    channel_number = markers['channel_number'][
                markers['marker_name'] == ch]
    img = single_channel_pyramid(
        glob.glob(tif_path)[0],
        channel=(channel_number.item() - 1)
        )

    viewer.add_image(
        img, rgb=False, blending='additive',
        colormap='green', visible=False,
        name=ch
        )

# apply previously defined contrast limits
for ch in abx_channels:
    ch = ch.rsplit('_', 1)[0]
    viewer.layers[ch].contrast_limits = (
        contrast_limits[ch][0], contrast_limits[ch][1])

# add centroids of selected CD4 T cells
num_colors = len(list(cm.tab20.colors))
num_clusters = len(data['CellID'].unique())
palette_multiplier = ceil(num_clusters/num_colors)
colors = list(cm.tab20.colors)*palette_multiplier
colors = colors[0:num_clusters]

for c, id in zip(
  colors, sorted(data['CellID'].unique(), reverse=True)
  ):
    centroids = data[
        ['Y_centroid', 'X_centroid']][data['CellID'] == id]
    viewer.add_points(
        centroids, name=f'CD45_{id}',
        face_color='lime',
        edge_color='lime',
        edge_width=0.0, size=500.0, opacity=1.0, blending='translucent',
        visible=True
        )

# read segmentation outlines
seg = single_channel_pyramid(
    glob.glob(seg_path)[0], channel=0
    )
viewer.add_image(
    seg, rgb=False, blending='additive',
    colormap='gray', visible=False,
    name='segmentation', opacity=0.3
    )

viewer.scale_bar.visible = True
viewer.scale_bar.unit = 'um'

# run Napari image viewer
napari.run()
print('Completed in ' + str(datetime.now() - startTime))
