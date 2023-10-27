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
tissue = 'WD-76845-097'


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
    'sardana_artifact'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read markers.csv
markers = pd.read_csv(
    '/Volumes/My Book/cylinter_input/sardana-097/markers.csv'
    )

# assign file path to OME-TIFF as a variable
img_path = ('/Volumes/My Book/cylinter_input/sardana-097/tif/' +
            f'{tissue}.ome.tif')

# read DNA1 image of selected tissue
dna = single_channel_pyramid(
    glob.glob(img_path)[0], channel=0
    )

# read segmentation outlines of selected tissue and crop to target size
seg_path = ('/Volumes/My Book/cylinter_input/topacio_cellring/seg/' +
            '840063_0110-median_r_50.ome.tif')
seg = single_channel_pyramid(
    glob.glob(seg_path)[0], channel=0
    )

# visualize tissue in Napari
viewer = napari.view_image(
    dna, opacity=0.5, rgb=False, name='0'
    )

for i in range(4, 38, 4):

    dna = single_channel_pyramid(
        glob.glob(img_path)[0], channel=i
        )

    viewer.add_image(
        dna, rgb=False, blending='additive',
        colormap='gray', visible=False,
        name=str(i)
        )

viewer.add_image(
    seg, rgb=False, blending='additive',
    opacity=0.5, colormap='green', visible=False,
    name='segmentation'
    )

napari.run()
