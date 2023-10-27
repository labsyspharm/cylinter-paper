import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from tifffile import imread

core = '95'
channels = [f'DNA1:{core}']


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


# OPEN TARGET TISSUE SAMPLE IN NAPARI USING CYLINTER'S "setContrast" MODULE.
# THEN RUN THIS CODE FROM NAPARI

save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'optical_artifact'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read image channels for target tissue sample and crop ROI
ymin = 1934
ymax = 2197
xmin = 2000
xmax = 2310

# read unfiltered single-cell data
data = pd.read_parquet(
    '/Volumes/My Book/cylinter_output/emit22_full_less_cc/' +
    'checkpoints/getSingleCellData.parquet'
    )

# read DNA1 image of selected tissue and crop to target size
dna = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif', key=0
    )
dna_crop = dna[ymin:ymax, xmin:xmax]

# read pCREB image of selected tissue and crop to target size
pcreb = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif', key=33
    )
pcreb_crop = pcreb[ymin:ymax, xmin:xmax]

# read segmentation outlines for selected tissue and crop to target size
seg = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/seg/{core}.ome.tif', key=0
    )
seg_crop = seg[ymin:ymax, xmin:xmax]

# plot cropped image channels, segmentation outlines, and analysis rectangle
# plt.imshow(dna_crop*2, cmap='Greys_r')
plt.imshow(seg_crop, alpha=1.0, cmap='Greys_r')
plt.imshow(pcreb_crop, cmap='magma', alpha=0.75)
rect = patches.Rectangle(
    (10, 100), 290, 50, linewidth=1, edgecolor='r', facecolor='none'
    )
plt.gca().add_patch(rect)
plt.grid(False)

# plt.colorbar(format=ticker.FuncFormatter(fmt))
cbar = plt.colorbar()
cbar.set_label('pCREB per pixel', labelpad=20, rotation=270)

# grab data for cells within ROI bounds (original coordinates)
data_roi = data[['Y_centroid', 'X_centroid', 'pCREB_cellMask']][
    (data['Sample'] == core) &
    (data['X_centroid'].between(xmin, xmax, inclusive=True)) &
    (data['Y_centroid'].between(ymin, ymax, inclusive=True))
    ]

# transform x, y coordinates of cells within ROI bounds
data_roi_transform = data_roi.copy()
data_roi_transform['X_centroid'] = data_roi_transform['X_centroid']-2000
data_roi_transform['Y_centroid'] = data_roi_transform['Y_centroid']-1934

# grab data for cells within analysis rectangle (transformed coordinates)
data_rec = data_roi_transform[
    (data_roi_transform['X_centroid'].between(15, 293, inclusive=True)) &
    (data_roi_transform['Y_centroid'].between(100, 150, inclusive=True))
    ]

# sort data points by x coordinate and set row index as X centroids so
# scatter plot x-axis values are monotonically increasing integers
# associated with successive cells from left to right in the analysis rectangle
data_rec = data_rec.sort_values(by='X_centroid')
data_rec.set_index('X_centroid', drop=False, inplace=True)

plt.scatter(
    data_rec['X_centroid'],
    data_rec['Y_centroid'], s=1.0, color='w'
    )
plt.title(f'Core {core}')
plt.savefig(
    os.path.join(
        save_dir,
        'optical_artifact_img.pdf')
        )
plt.close('all')

sns.set_style('whitegrid')
plt.plot(np.log10(data_rec['pCREB_cellMask']), marker='o', lw=2.0, c='k')
plt.xlabel('X coordinate')
plt.ylabel('log(pCREB) per cell')
plt.savefig(
    os.path.join(
        save_dir,
        'optical_artifact_trace.pdf')
        )
plt.close('all')
