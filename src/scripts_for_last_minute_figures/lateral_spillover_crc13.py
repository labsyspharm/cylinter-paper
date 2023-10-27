import os
import yaml

import re
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tifffile
import napari
import dask.array as da
import zarr


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

clusters_to_show = [13]

cylinter_input_path = (
    '/Volumes/My Book/cylinter_input/sardana-097/'
    )

sc_data_path = os.path.join(
    cylinter_input_path, 'output/checkpoints/clustering.parquet'
    )

cylinter_config_path = os.path.join(cylinter_input_path, 'config.yml')

markers_path = os.path.join(cylinter_input_path, 'markers.csv')

image_contrast_path = os.path.join(
    cylinter_input_path, 'output/contrast/contrast_limits.yml'
    )

tif_path = cylinter_input_path + f'tif/{sample_name}.ome.tif'
seg_path = cylinter_input_path + f'seg/{sample_name}.ome.tif'

# import clustered single-cell data
data = pd.read_parquet(sc_data_path)

# grab header for cluster column of SC data
cluster_col = [i for i in data.columns if 'cluster_' in i][0]

# drop noisy cells (cluster = -1)
data = data[data[cluster_col] != -1]

# select data from sample of interest
data = data[data['Sample'] == sample_name]

# import config.yml
with open(cylinter_config_path) as f:
    config = yaml.safe_load(f)
markers_to_exclude = config['channelExclusionsClustering']

# import markers.csv
markers = pd.read_csv(markers_path)

# get first name of first DNA channel
dna1 = markers['marker_name'][markers['channel_number'] == 1][0]
dna_moniker = str(re.search(r'[^\W\d]+', dna1).group())

# isolate antibodies of interest
abx_channels = [
    i for i in data.columns if i in list(markers['marker_name'])
    if dna_moniker not in i if i not in markers_to_exclude]

# import image contrast settings
with open(image_contrast_path) as f:
    contrast_limits = yaml.safe_load(f)

# read DNA1 channel
dna = single_channel_pyramid(
    glob.glob(tif_path)[0], channel=0
    )

# add DNA1 channel to Napari image viewer
viewer = napari.view_image(
    dna, rgb=False, blending='additive',
    colormap='gray', visible=True, opacity=0.2,
    name='DNA1'
    )

# loop over antibodies of interest and add them to Napari image viewer
for ch in abx_channels:
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
    viewer.layers[ch].contrast_limits = (
        contrast_limits[ch][0], contrast_limits[ch][1])

# add centroids of cells for each cluster to Napari image viewer

for cluster in sorted(clusters_to_show, reverse=True):  # data[cluster_col].unique()
    centroids = data[
        ['Y_centroid', 'X_centroid']][data[cluster_col] == cluster]
    viewer.add_points(
        centroids, name=f'cluster{cluster}',
        face_color='white',
        edge_color='white',
        edge_width=0.0, size=4.0,
        visible=False
        )

# read segmentation outlines
seg = single_channel_pyramid(
    glob.glob(seg_path)[0], channel=0
    )
viewer.add_image(
    seg, rgb=False, blending='additive',
    colormap='red', visible=False,
    name='segmentation', opacity=0.3
    )

viewer.scale_bar.visible = True
viewer.scale_bar.unit = "um"

# run Napari image viewer
napari.run()
